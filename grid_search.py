""" """

import gzip
import json
import multiprocessing as mp
import os
import pickle
import sys
import traceback
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid
from tqdm.auto import tqdm

from compat import plot_confusion_matrix, plot_roc_curve
from config import CFG, DEFAULTS, DataPreprocessConfig, FeatureConfig, GridSearchConfig
from data_processing import get_training_data
from metrics import SeizureClassificationMetrics
from models import _MODEL_MAP, get_model
from utils import CrossPlatformUnpickler, ReprMixin, timeout

__all__ = [
    "GridSearch",
    "perform_grid_search_no_cv",
    "perform_grid_search_cv",
    "gather_grid_search_results",
]


warnings.filterwarnings("ignore")
_NUM_CPUS = mp.cpu_count()


class GridSearch(ReprMixin):
    """ """

    __name__ = "GridSearch"

    def __init__(
        self,
        grid_search_config: Optional[CFG] = None,
        preprocess_config: Optional[CFG] = None,
        feature_config: Optional[CFG] = None,
        cv: Optional[int] = None,
        strict_glioma: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        grid_search_config : CFG, optional
            Grid search configuration.
        preprocess_config : CFG, optional
            Data preprocessing configuration.
        feature_config : CFG, optional
            Feature engineering configuration.
        cv : int, optional
            Number of folds for cross validation.
        strict_glioma : bool, default True
            Whether to strictly filter glioma types.
            If True, exclude data samples with non-glioma types,
            e.g., those in `DataPreprocessConfig.exclude_types_zh` or
            in `DataPreprocessConfig.exclude_types_en`.

        """
        if grid_search_config is None:
            self._grid_search_config = deepcopy(GridSearchConfig)
        else:
            _grid_search_config = deepcopy(grid_search_config)
            self._grid_search_config = deepcopy(GridSearchConfig)
            self._grid_search_config.update(_grid_search_config)

        if preprocess_config is None:
            self._preprocess_config = deepcopy(DataPreprocessConfig)
        else:
            _preprocess_config = deepcopy(preprocess_config)
            self._preprocess_config = deepcopy(DataPreprocessConfig)
            self._preprocess_config.update(_preprocess_config)

        if feature_config is None:
            self._feature_config = deepcopy(FeatureConfig)
        else:
            _feature_config = deepcopy(feature_config)
            self._feature_config = deepcopy(FeatureConfig)
            self._feature_config.update(_feature_config)
        self.cv = cv
        self.strict_glioma = strict_glioma

        self.__cache = {}
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        self.__feature_list = None
        self.best_clf, self.best_params, self.best_score = None, None, None
        self.test_score = None
        self._no = 1

    def search(
        self,
        model_name: str = "rf",
        feature_set: str = "TDSB",
        cv: Optional[int] = None,
        experiment_tag: Optional[str] = None,
        strict_glioma: Optional[bool] = None,
    ) -> tuple:
        """
        Perform grid search.

        Parameters
        ----------
        model_name : str, default "rf"
            Model name.
        feature_set : str, default "TDSB"
            Feature set name.
        cv : int, optional
            Number of folds for cross validation.
            If None, no cross validation will be performed,
            and the grid search will be performed on a fixed train-test split.
        experiment_tag : str, optional
            Tag for the experiment, used to store the results in the internal cache.
        strict_glioma : bool, optional
            Whether to use strict glioma filtering during data preprocessing.
            If not provided, the value of `self.strict_glioma` will be used.

        Returns
        -------
        best_clf : BaseEstimator
            Best classifier.
        best_params : dict
            Best parameters for the classifier.
        best_score : float
            Best score for the classifier on the test set.
        test_score : float, optional
            The score of the best classifier on the left out test set.
            If cv is None, this will not be returned.

        """
        feature_config = deepcopy(self.feature_config)
        feature_config.set_name = feature_set
        if strict_glioma is None:
            strict_glioma = self.strict_glioma

        (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.__feature_list,
        ) = get_training_data(  # type: ignore
            self.preprocess_config, feature_config, feature_set, strict_glioma=strict_glioma
        )  # type: ignore
        feature_config.feature_list = self.feature_list

        cache_key = self._get_cache_key(model_name, feature_set, cv, experiment_tag)

        if cv is None:
            (
                self.best_clf,
                self.best_params,
                self.best_score,
            ) = _perform_grid_search_no_cv(
                model_name,
                self.grid_search_config[model_name],
                self.X_train,  # type: ignore
                self.y_train,  # type: ignore
                self.X_test,  # type: ignore
                self.y_test,  # type: ignore
            )

            self.best_clf.preprocess_config = self.preprocess_config
            self.best_clf.feature_config = feature_config

            # save in self.__cache
            self.__cache[cache_key] = dict(
                best_clf=deepcopy(self.best_clf),
                best_params=deepcopy(self.best_params),
                best_score=self.best_score,
                X_train=deepcopy(self.X_train),
                y_train=deepcopy(self.y_train),
                X_test=deepcopy(self.X_test),
                y_test=deepcopy(self.y_test),
            )

            self._no += 1

            return self.best_clf, self.best_params, self.best_score
        else:
            (
                self.best_clf,
                self.best_params,
                self.best_score,
                self.test_score,
            ) = _perform_grid_search_cv(
                model_name,
                self.grid_search_config[model_name],
                self.X_train,  # type: ignore
                self.y_train,  # type: ignore
                self.X_test,  # type: ignore
                self.y_test,  # type: ignore
                cv,
            )

            self.best_clf.preprocess_config = self.preprocess_config
            self.best_clf.feature_config = feature_config

            # save in self.__cache
            self.__cache[cache_key] = dict(
                best_clf=deepcopy(self.best_clf),
                best_params=deepcopy(self.best_params),
                best_score=self.best_score,
                test_score=self.test_score,
                X_train=deepcopy(self.X_train),
                y_train=deepcopy(self.y_train),
                X_test=deepcopy(self.X_test),
                y_test=deepcopy(self.y_test),
            )

            self._no += 1

            return self.best_clf, self.best_params, self.best_score, self.test_score

    @property
    def grid_search_config(self) -> CFG:
        return self._grid_search_config

    @property
    def preprocess_config(self) -> CFG:
        return self._preprocess_config

    @property
    def feature_config(self) -> CFG:
        return self._feature_config

    @property
    def feature_list(self) -> List[str]:
        return self.__feature_list  # type: ignore

    def get_cache(
        self,
        key: Optional[str] = None,
        model_name: str = "rf",
        feature_set: str = "TDSB",
        cv: Optional[int] = None,
        name: Optional[str] = None,
    ) -> dict:
        """
        Get the cached grid search results from the internal cache.

        Parameters
        ----------
        key : str, optional
            Key of the cached results.
            If None, the key will be generated from the other parameters.
        model_name : str, default "rf"
            Model name.
        feature_set : str, default "TDSB"
            Feature set name.
        cv : int, optional
            Number of folds for cross validation.
        name : str, optional
            Extra tag for the experiment.

        Returns
        -------
        cache : dict
            Cached grid search results.

        """
        if key is None:
            key = self._get_cache_key(model_name, feature_set, cv, name)
        return self.__cache[key]

    def _get_cache_key(
        self,
        model_name: str = "rf",
        feature_set: str = "TDSB",
        cv: Optional[int] = None,
        name: Optional[str] = None,
    ) -> str:
        """
        Generate a key for the internal cache.

        Parameters
        ----------
        model_name : str, default "rf"
            Model name.
        feature_set : str, default "TDSB"
            Feature set name.
        cv : int, optional
            Number of folds for cross validation.
        name : str, optional
            Extra tag for the experiment.

        Returns
        -------
        key : str
            Key for the internal cache.

        """
        key = f"{model_name}_{feature_set}"
        if cv is not None:
            key += f"_{cv}"
        if name is None:
            name = f"ex{self._no}"
        key += f"_{name}"
        return key

    def list_cache(self) -> List[str]:
        """List all the keys in the internal cache"""
        return list(self.__cache)

    def update_feature_config(self, config: CFG) -> None:
        """
        Update the feature config.

        Parameters
        ----------
        config : CFG
            New (partial) feature config.

        """
        self._feature_config.update(config)

    def accuracy_score(self, cached_item: Optional[dict] = None) -> float:
        """
        Compute the accuracy score of the cached grid search results.

        Parameters
        ----------
        cached_item : dict, optional
            Cached grid search results.
            If not provided, the current best model will be used.

        Returns
        -------
        float
            Accuracy score.

        """
        if cached_item is not None:
            y_true = cached_item["y_test"]
            y_pred = cached_item["best_clf"].predict(cached_item["X_test"])
        elif self.best_clf is not None:
            y_true = self.y_test
            y_pred = self.best_clf.predict(self.X_test)
        else:
            raise ValueError("No trained classifier!")
        return accuracy_score(y_true, y_pred)  # type: ignore

    def confusion_matrix(self, cached_item: Optional[dict] = None) -> np.ndarray:
        """
        Compute the confusion matrix of the cached grid search results.

        Parameters
        ----------
        cached_item : dict, optional
            Cached grid search results.
            If not provided, the current best model will be used.

        Returns
        -------
        np.ndarray
            The confusion matrix.

        """
        if cached_item is not None:
            y_true = cached_item["y_test"]
            y_pred = cached_item["best_clf"].predict(cached_item["X_test"])
        elif self.best_clf is not None:
            y_true = self.y_test
            y_pred = self.best_clf.predict(self.X_test)
        else:
            raise ValueError("No trained classifier!")
        return confusion_matrix(y_true, y_pred)  # type: ignore

    def roc_auc_score(self, cached_item: Optional[dict] = None) -> float:
        """
        Compute the ROC AUC score of the cached grid search results.

        Parameters
        ----------
        cached_item : dict, optional
            Cached grid search results.
            If not provided, the current best model will be used.

        Returns
        -------
        float
            The ROC AUC score.

        """
        if cached_item is not None:
            y_true = cached_item["y_test"]
            y_score = cached_item["best_clf"].predict_proba(cached_item["X_test"])
        elif self.best_clf is not None:
            y_true = self.y_test
            y_score = self.best_clf.predict_proba(self.X_test)
        else:
            raise ValueError("No trained classifier!")
        return roc_auc_score(y_true, y_score[:, 1])  # type: ignore

    def f1_score(self, cached_item: Optional[dict] = None) -> float:
        """
        Compute the F1 score of the cached grid search results.

        Parameters
        ----------
        cached_item : dict, optional
            Cached grid search results.
            If not provided, the current best model will be used.

        Returns
        -------
        float
            The F1 score.

        """
        if cached_item is not None:
            y_true = cached_item["y_test"]
            y_pred = cached_item["best_clf"].predict(cached_item["X_test"])
        elif self.best_clf is not None:
            y_true = self.y_test
            y_pred = self.best_clf.predict(self.X_test)
        else:
            raise ValueError("No trained classifier!")
        return f1_score(y_true, y_pred)  # type: ignore

    def classification_report(self, cached_item: Optional[dict] = None) -> None:
        """
        Print the `sklearn` classification report of the cached grid search results.

        Parameters
        ----------
        cached_item : dict, optional
            Cached grid search results.
            If not provided, the current best model will be used.

        """
        if cached_item is not None:
            y_true = cached_item["y_test"]
            y_pred = cached_item["best_clf"].predict(cached_item["X_test"])
        elif self.best_clf is not None:
            y_true = self.y_test
            y_pred = self.best_clf.predict(self.X_test)
        else:
            raise ValueError("No trained classifier!")
        print(classification_report(y_true, y_pred))  # type: ignore

    def plot_roc_curve(self, cached_item: Optional[dict] = None) -> RocCurveDisplay:
        """
        Plot the ROC curve of the cached grid search results.

        Parameters
        ----------
        cached_item : dict, optional
            Cached grid search results.
            If not provided, the current best model will be used.

        Returns
        -------
        RocCurveDisplay
            The `sklearn` ROC curve display object.

        """
        if cached_item is not None:
            clf = cached_item["best_clf"]
            X_test, y_test = cached_item["X_test"], cached_item["y_test"]
        elif self.best_clf is not None:
            clf = self.best_clf
            X_test, y_test = self.X_test, self.y_test
        else:
            raise ValueError("No trained classifier!")
        return plot_roc_curve(clf, X_test, y_test)

    def plot_confusion_matrix(self, cached_item: Optional[dict] = None) -> ConfusionMatrixDisplay:
        """
        Plot the confusion matrix of the cached grid search results.

        Parameters
        ----------
        cached_item : dict, optional
            Cached grid search results.
            If not provided, the current best model will be used.

        Returns
        -------
        ConfusionMatrixDisplay
            The `sklearn` confusion matrix display object.

        """
        if cached_item is not None:
            clf = cached_item["best_clf"]
            X_test, y_test = cached_item["X_test"], cached_item["y_test"]
        elif self.best_clf is not None:
            clf = self.best_clf
            X_test, y_test = self.X_test, self.y_test
        else:
            raise ValueError("No trained classifier!")
        return plot_confusion_matrix(clf, X_test, y_test)

    def list_model_names(self) -> List[str]:
        """List the names of the model names to perform grid search."""
        return list(self.grid_search_config)

    def list_feature_sets(self, with_details: bool = False) -> Union[Dict[str, List[str]], List[str]]:
        """
        List the feature sets to perform grid search.

        Parameters
        ----------
        with_details : bool, default False
            Whether to return the feature set details.
            If False, only the feature set names will be returned.

        """
        feature_sets = deepcopy(self._feature_config.sets)
        feature_sets.pop("TDSB_ext", None)
        if with_details:
            return feature_sets
        else:
            return list(feature_sets.keys())

    def full_search(
        self,
        parallel: bool = False,
        save_cache: bool = True,
        time_limit: Union[int, float] = 4 * 3600,
        strict_glioma: Optional[bool] = None,
    ) -> None:
        """
        Perform grid search on all models and feature sets.

        Parameters
        ----------
        parallel : bool, default False
            Whether to perform grid search in parallel.
        save_cache : bool, default True
            Whether to save the grid search results to cache.
        time_limit : int or float, default 4 * 3600
            The time limit for each grid search.
        strict_glioma : bool, optional
            Whether to use strict glioma filtering during data preprocessing.
            If not provided, the value of `self.strict_glioma` will be used.

        """
        if parallel:
            # sk_mlp already parallelizes
            raise Exception("Parallel search not tested yet!")
            self._full_search_parallel(time_limit, strict_glioma=strict_glioma)
        else:
            self._full_search_sequential(time_limit, strict_glioma=strict_glioma)

        # save results
        df_results = pd.DataFrame()
        for idx, key in enumerate(self.list_cache()):
            split_key = key.split("_")
            feature, strategy = split_key[-2:]
            model = "_".join(split_key[:-2])
            cache = self.get_cache(key=key)
            if "B" not in feature:
                strategy = "NA"
            df_results = pd.concat(
                (
                    df_results,
                    pd.DataFrame(
                        dict(
                            model=model,
                            feature=feature,
                            BIO_na=strategy,
                            best_params=json.dumps(cache["best_params"], ensure_ascii=False),
                            best_score=cache["best_score"],
                        ),
                        index=[idx],
                    ),
                )
            )
        save_path = DEFAULTS.SAVE_DIR / f"""full_search_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv"""
        df_results.to_csv(save_path, index=False)
        if save_cache:
            save_path = save_path.with_suffix(".pkl.gz")
            with gzip.GzipFile(save_path, "wb") as gz_file:
                pickle.dump(self.__cache, gz_file)

    def _full_search_sequential(self, time_limit: Union[int, float], strict_glioma: Optional[bool] = None) -> None:
        """
        Perform the full grid search sequentially.

        Parameters
        ----------
        time_limit : int or float
            The time limit for each grid search, with unit in seconds.
        strict_glioma : bool, optional
            Whether to use strict glioma filtering during data preprocessing.
            If not provided, the value of `self.strict_glioma` will be used.

        """
        for feature_set in self.list_feature_sets():
            for model_name in self.list_model_names():
                if model_name == "svc" and self.feature_config.over_sampler is not None:
                    # searching got stuck for SVC with SMOTENC
                    # https://stackoverflow.com/questions/62341001/svm-stuck-in-fitting-the-model
                    continue
                strategies = ["keep", "drop"] if "B" in feature_set else ["keep"]
                for strategy in strategies:
                    feature_config = {
                        "BIO_na_strategy": strategy,
                        "binarize_variables": False,
                    }
                    if model_name == "lr":
                        feature_config["binarize_variables"] = True
                    self.update_feature_config(config=feature_config)  # type: ignore
                    try:
                        with timeout(time_limit) as t:
                            self.search(model_name, feature_set, experiment_tag=strategy, strict_glioma=strict_glioma)
                    except TimeoutError:
                        pass

    def _full_search_parallel(self, time_limit: Union[int, float], strict_glioma: Optional[bool] = None) -> None:
        """NOT tested,

        Perform the full grid search in parallel.

        Parameters
        ----------
        time_limit : int or float
            The time limit for each grid search, with unit in seconds.
        strict_glioma : bool, optional
            Whether to use strict glioma filtering during data preprocessing.
            If not provided, the value of `self.strict_glioma` will be used.

        """
        if strict_glioma is None:
            strict_glioma = self.strict_glioma
        iterable, cache_keys = [], []
        for feature_set in self.list_feature_sets():
            for model_name in self.list_model_names():
                strategies = ["keep", "drop"] if "B" in feature_set else ["keep"]
                for strategy in strategies:
                    feature_config = deepcopy(self.feature_config)
                    feature_config.update(
                        {
                            "BIO_na_strategy": strategy,
                            "binarize_variables": False,
                        }
                    )
                    if model_name == "lr":
                        feature_config["binarize_variables"] = True
                    feature_config.set_name = feature_set
                    grid_search_config = deepcopy(self.grid_search_config)
                    param_grid = grid_search_config[model_name]
                    preprocess_config = deepcopy(self.preprocess_config)
                    (
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        feature_list,
                    ) = get_training_data(  # type: ignore
                        preprocess_config, feature_config, feature_set, strict_glioma=strict_glioma
                    )  # type: ignore
                    feature_config.feature_list = feature_list
                    iterable.append((model_name, param_grid, X_train, y_train, X_test, y_test))
                    cache_keys.append(self._get_cache_key(model_name, feature_set, cv=None, name=strategy))
        with mp.Pool(processes=max(1, _NUM_CPUS // 2)) as pool:
            # results consists of tuples (best_clf, best_params, best_score)
            results = pool.star_map(_perform_grid_search_no_cv, iterable)
        for idx, result in enumerate(results):
            cache_key = cache_keys[idx]
            self.__cache[cache_key] = dict(
                best_clf=result[0],
                best_params=result[1],
                best_score=result[2],
                X_train=iterable[idx][2],
                y_train=iterable[idx][3],
                X_test=iterable[idx][4],
                y_test=iterable[idx][5],
            )


def perform_grid_search_no_cv(
    model_name: str,
    grid_search_config: Optional[CFG] = None,
    preprocess_config: Optional[CFG] = None,
    feature_config: Optional[CFG] = None,
    feature_set: str = "TDSB",
    strict_glioma: bool = True,
) -> Tuple[BaseEstimator, dict, float]:
    """
    Perform grid search without cross validation on a fixed data split.

    Parameters
    ----------
    model_name : str
        The name of the model.
    grid_search_config : CFG, optional
        The configuration for grid search.
        If not provided, the default configuration will be used.
    preprocess_config : CFG, optional
        The configuration for preprocessing.
        If not provided, the default configuration will be used.
    feature_config : CFG, optional
        The configuration for feature engineering.
        If not provided, the default configuration will be used.
    feature_set : str, default "TDSB"
        The name of the feature set.
    strict_glioma : bool, default True
        Whether to strictly filter glioma types.
        If True, exclude data samples with non-glioma types,
        e.g., those in `DataPreprocessConfig.exclude_types_zh` or
        in `DataPreprocessConfig.exclude_types_en`.

    Returns
    -------
    best_clf : BaseEstimator
        The best classifier.
    best_params : dict
        The best parameters.
    best_score : float
        The best score on the test set.

    """
    if grid_search_config is None:
        grid_search_config = deepcopy(GridSearchConfig)
    else:
        _grid_search_config = deepcopy(grid_search_config)
        grid_search_config = deepcopy(GridSearchConfig)
        grid_search_config.update(_grid_search_config)

    if preprocess_config is None:
        preprocess_config = deepcopy(DataPreprocessConfig)
    else:
        _preprocess_config = deepcopy(preprocess_config)
        preprocess_config = deepcopy(DataPreprocessConfig)
        preprocess_config.update(_preprocess_config)

    if feature_config is None:
        feature_config = deepcopy(FeatureConfig)
    else:
        _feature_config = deepcopy(feature_config)
        feature_config = deepcopy(FeatureConfig)
        feature_config.update(_feature_config)

    feature_config.set_name = feature_set

    X_train, y_train, X_test, y_test, feature_list = get_training_data(preprocess_config, feature_config, feature_set, strict_glioma=strict_glioma)  # type: ignore
    feature_config.feature_list = feature_list

    best_clf, best_params, best_score = _perform_grid_search_no_cv(
        model_name,
        grid_search_config[model_name],
        X_train,  # type: ignore
        y_train,  # type: ignore
        X_test,  # type: ignore
        y_test,  # type: ignore
    )

    best_clf.preprocess_config = preprocess_config
    best_clf.feature_config = feature_config

    return best_clf, best_params, best_score


def perform_grid_search_cv(
    model_name: str,
    grid_search_config: Optional[CFG] = None,
    preprocess_config: Optional[CFG] = None,
    feature_config: Optional[CFG] = None,
    feature_set: str = "TDSB",
    cv: int = 5,
    strict_glioma: bool = True,
) -> Tuple[BaseEstimator, dict, float, float]:
    """
    Perform grid search with cross validation.

    Parameters
    ----------
    model_name : str
        The name of the model.
    grid_search_config : CFG, optional
        The configuration for grid search.
        If not provided, the default configuration will be used.
    preprocess_config : CFG, optional
        The configuration for preprocessing.
        If not provided, the default configuration will be used.
    feature_config : CFG, optional
        The configuration for feature engineering.
        If not provided, the default configuration will be used.
    feature_set : str, default "TDSB"
        The name of the feature set.
    cv : int, default 5
        The number of folds for cross validation.
    strict_glioma : bool, default True
        Whether to strictly filter glioma types.
        If True, exclude data samples with non-glioma types,
        e.g., those in `DataPreprocessConfig.exclude_types_zh` or
        in `DataPreprocessConfig.exclude_types_en`.

    Returns
    -------
    best_clf : BaseEstimator
        The best classifier.
    best_params : dict
        The best parameters.
    best_score : float
        The best score for the cross validation.
    test_score : float
        The score of the best classifier on the left out test set.

    """
    if grid_search_config is None:
        grid_search_config = deepcopy(GridSearchConfig)
    else:
        _grid_search_config = deepcopy(grid_search_config)
        grid_search_config = deepcopy(GridSearchConfig)
        grid_search_config.update(_grid_search_config)

    if preprocess_config is None:
        preprocess_config = deepcopy(DataPreprocessConfig)
    else:
        _preprocess_config = deepcopy(preprocess_config)
        preprocess_config = deepcopy(DataPreprocessConfig)
        preprocess_config.update(_preprocess_config)

    if feature_config is None:
        feature_config = deepcopy(FeatureConfig)
    else:
        _feature_config = deepcopy(feature_config)
        feature_config = deepcopy(FeatureConfig)
        feature_config.update(_feature_config)

    feature_config.set_name = feature_set

    X_train, y_train, X_test, y_test, feature_list = get_training_data(preprocess_config, feature_config, feature_set, strict_glioma=strict_glioma)  # type: ignore
    feature_config.feature_list = feature_list

    best_clf, best_params, best_score, test_score = _perform_grid_search_cv(
        model_name, grid_search_config[model_name], X_train, y_train, X_test, y_test, cv  # type: ignore
    )

    best_clf.preprocess_config = preprocess_config
    best_clf.feature_config = feature_config

    return best_clf, best_params, best_score, test_score


def _perform_grid_search_no_cv(
    model_name: str,
    param_grid: ParameterGrid,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_jobs: Optional[int] = None,
) -> Tuple[BaseEstimator, dict, float]:
    """
    Performs a grid search on the given model and parameters without cross validation.
    """
    best_score = 0
    best_clf = None
    best_params = None
    with tqdm(param_grid, desc=model_name, dynamic_ncols=True, mininterval=1.0) as pbar:
        for params in pbar:
            updated_params = deepcopy(params)
            # use half of the CPU cores
            updated_params["n_jobs"] = n_jobs or max(1, _NUM_CPUS // 2)
            try:
                clf_gs = get_model(model_name, updated_params)
                clf_gs.fit(X_train, y_train)
            except Exception as e:
                if os.environ.get("DEBUG_MODE", "0") == "1":
                    print(f"Error training model {model_name} with params {updated_params}: {e}")
                    traceback.print_exc()
                continue
            y_prob = clf_gs.predict_proba(X_val)
            metric_score = roc_auc_score(y_true=y_val, y_score=y_prob[:, 1])
            if metric_score > best_score:
                best_score = metric_score
                best_params = params
                best_clf = clf_gs
    return best_clf, best_params, best_score  # type: ignore


def _perform_grid_search_cv(
    model_name: str,
    param_grid: ParameterGrid,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cv: int = 5,
    n_jobs: Optional[int] = None,
) -> Tuple[BaseEstimator, dict, float, float]:
    """
    Performs a grid search on the given model and parameters with cross validation.
    """
    gscv = GridSearchCV(
        estimator=get_model(model_name),
        param_grid=param_grid.param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs or max(1, _NUM_CPUS // 2),  # use half of the CPU cores
        verbose=1,
    )
    gscv.fit(X_train, y_train)
    best_clf = gscv.best_estimator_
    best_params = gscv.best_params_
    best_score = gscv.best_score_
    test_score = roc_auc_score(y_val, best_clf.predict_proba(X_val)[:, 1])

    return best_clf, best_params, best_score, test_score  # type: ignore


def gather_grid_search_results(
    raw: bool = False,
    model_name_map: Optional[Dict[str, str]] = None,
    sub_dirs: Optional[Union[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """
    Gather historical results of grid search from the saved csv files that store the results.

    Parameters
    ----------
    raw : bool, default False
        Whether to return the raw results.
    model_name_map : dict, optional
        The mapping from the model name to the model name to be displayed in the table.
        If not provided, the default mapping will be used.
    sub_dirs : str or sequence of str, optional
        The sub directories to store the results.

    Returns
    -------
    results_all : pd.DataFrame
        The table of all the historical grid search results.

    """
    if model_name_map is None:
        model_name_map = {k: v.__name__.replace("Classifier", "") for k, v in _MODEL_MAP.items()}
    folder = DEFAULTS.SAVE_DIR
    if sub_dirs is not None:
        if isinstance(sub_dirs, str):
            sub_dirs = [sub_dirs]
        folders = [folder / sub_dir for sub_dir in sub_dirs]
    else:
        folders = [folder]
    results_csv_files = []
    for folder in folders:
        results_csv_files.extend(folder.rglob("*.csv"))
    results = [pd.read_csv(file) for file in results_csv_files]
    print(f"Found {len(results)} results files.")
    if raw:
        results_all = pd.concat(results).reset_index(drop=True)
        results_all.loc[:, "model"] = results_all.model.map(model_name_map)
        return results_all
    results_all = results[0].copy(deep=True).drop(columns="best_params")
    results_all.best_score = ""
    results_all["filename"] = ""
    for idx, row in results_all.iterrows():
        results_all.at[idx, "best_score"] = np.array([df.loc[idx, "best_score"] for df in results]).tolist()  # type: ignore
        results_all.at[idx, "filename"] = [str(file) for file in results_csv_files]  # type: ignore
    results_all.loc[:, "best_score_mean"] = results_all.best_score.apply(lambda s: np.mean(s))
    results_all.loc[:, "best_score_std"] = results_all.best_score.apply(lambda s: np.std(s))
    results_all.loc[:, "best_score_max"] = results_all.best_score.apply(lambda s: np.max(s))
    results_all.loc[:, "best_score_min"] = results_all.best_score.apply(lambda s: np.min(s))
    results_all.loc[:, "model"] = results_all.model.map(model_name_map)
    return results_all


def gather_grid_search_metrics(
    model_name_map: Optional[Dict[str, str]] = None,
    sub_dirs: Optional[Union[str, Sequence[str]]] = None,
    metrics: Optional[List[str]] = None,
    save_filename: Optional[Union[str, Path]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Gather historical results of grid search from the saved csv files that store the results.

    Parameters
    ----------
    model_name_map : dict, optional
        The mapping from the model name to the model name to be displayed in the table.
        If not provided, the default mapping will be used.
    sub_dirs : str or sequence of str, optional
        The sub directories to store the results.
    metrics : list of str, optional
        The list of metrics to compute.
        If not provided, the default metrics will be used.
    save_filename : str or Path, optional
        The filename (ONLY the filename, not the path) to save the results.
        If provided, the results will be saved to the specified file.
        Only JSON format is supported (Note: the file extension will be changed to .json).
        This file will be saved in the DEFAULTS.SAVE_DIR / "metrics" directory.
        If not provided, the results will not be saved.

    Returns
    -------
    results_all : dict of pd.DataFrame
        The table of all the historical grid search results.

    """
    if model_name_map is None:
        model_name_map = {k: v.__name__.replace("Classifier", "") for k, v in _MODEL_MAP.items()}
    if metrics is None:
        metrics = "sens,spec,prec,npv,jac,acc,phi,fnr,fpr,fdr,for,plr,nlr,pt,ba,f1,fm,bm,mk,dor,auc".split(",")  # type: ignore

    folder = DEFAULTS.SAVE_DIR
    if sub_dirs is not None:
        if isinstance(sub_dirs, str):
            sub_dirs = [sub_dirs]
        folders = [folder / sub_dir for sub_dir in sub_dirs]
    else:
        folders = [folder]

    gs_files = []
    for folder in folders:
        gs_files.extend(folder.rglob("*.pkl.gz"))
    print(f"Found {len(gs_files)} grid search cache files.")

    metrics_manager = SeizureClassificationMetrics(subset=metrics)
    results_all = {}
    for gs_file in tqdm(gs_files, desc="Gathering metrics", dynamic_ncols=True, mininterval=1.0):
        with gzip.open(gs_file, "rb") as gf:
            gs_cache = CrossPlatformUnpickler(gf, path_resolution="string").load()  # type: ignore
            for model_key in gs_cache:
                if model_key not in results_all:
                    results_all[model_key] = []
                cached_item = gs_cache[model_key]
                y_true = cached_item["y_test"]
                y_pred = cached_item["best_clf"].predict_proba(cached_item["X_test"])
                metrics_result = metrics_manager(y_true, y_pred, thr=0.5)
                results_all[model_key].append(metrics_result)
    for model_key in results_all:
        # convert the list of dicts to a DataFrame
        df_model = pd.DataFrame(results_all[model_key])
        # and compute median, mean, std, ci_lower, ci_upper of each column (metric)
        df_summary = pd.DataFrame(columns=df_model.columns)
        for col in df_model.columns:
            df_summary.loc["median", col] = df_model[col].median()
            df_summary.loc["mean", col] = df_model[col].mean()
            df_summary.loc["std", col] = df_model[col].std()
            ci_lower = np.nanpercentile(df_model[col][~np.isinf(df_model[col])], 2.5)
            ci_upper = np.nanpercentile(df_model[col][~np.isinf(df_model[col])], 97.5)
            df_summary.loc["ci_lower", col] = ci_lower
            df_summary.loc["ci_upper", col] = ci_upper
        # concatenate the summary statistics to the original DataFrame
        results_all[model_key] = pd.concat([df_model, df_summary], ignore_index=False)

    if save_filename is not None:
        save_path = DEFAULTS.SAVE_DIR / "metrics" / Path(save_filename).with_suffix(".json").name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        write_content = {k: v.to_dict() for k, v in results_all.items()}
        save_path.write_text(json.dumps(write_content, ensure_ascii=False))

    return results_all


def make_clf_report(
    feature_set: Literal["TDSB", "TDB", "TDS", "TD"] = "TDSB",
    BIO_na_strategy: Literal["keep", "drop"] = "drop",
    metrics_file: Optional[Union[str, Path]] = None,
    sub_dirs: Optional[Union[str, Sequence[str]]] = None,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Make a classification report for all the models
    with the given feature set and BIO_na_strategy.

    Parameters
    ----------
    feature_set : str, default "TDSB"
        The feature set to use.
    BIO_na_strategy : str, default "drop"
        The BIO_na_strategy to use.
    metrics_file : str or Path, optional
        The metrics file to use.
        If provided, the metrics will be loaded from the file.
        If not provided, the metrics will be computed from the grid search cache files.
    sub_dirs : str or sequence of str, optional
        The sub directories to store the results.
        `sub_dirs` will be ignored if `metrics_file` is provided.
        At least one of `metrics_file` and `sub_dirs` should be provided.
    metrics : list of str, optional
        The list of metrics to compute.
        If not provided, the default metrics will be used.

    """
    if metrics_file is not None:
        save_path = Path(metrics_file).with_suffix(".json")
        if not save_path.exists():
            save_path = DEFAULTS.SAVE_DIR / "metrics" / save_path.name
        if not save_path.exists():
            raise FileNotFoundError(f"Metrics file '{metrics_file}' not found!")
        metrics_dict = {k: pd.DataFrame(v) for k, v in json.loads(save_path.read_text()).items()}
    elif sub_dirs is not None:
        metrics_dict = gather_grid_search_metrics(
            sub_dirs=sub_dirs,
            metrics=metrics,
        )
    else:
        raise ValueError("At least one of `metrics_file` and `sub_dirs` should be provided.")

    suffix = f"_{feature_set}_{BIO_na_strategy}"
    clf_report = pd.DataFrame(columns="MetricFullNames,RF,GDBT,Bagging,MLP,XGB,LR".split(","))
    model_name_map = {k.lower(): k for k in "RF,GDBT,Bagging,XGB,LR".split(",")}
    model_name_map["sk_mlp"] = "MLP"
    metrics_name_mapping = {
        "auc": "area under the receiver-operater characteristic curve",
        "sens": "sensitivity, recall, hit rate, true positive rate",
        "spec": "specificity, selectivity, true negative rate",
        "prec": "precision, positive predictive value",
        "npv": "negative predictive value",
        "jac": "jaccard index, threat score, critical success index",
        "acc": "accuracy",
        "phi": "phi coefficient, matthews correlation coefficient",
        "fnr": "false negative rate, miss rate",
        "fpr": "false positive rate, fall-out",
        "fdr": "false discovery rate",
        "for": "false omission rate",
        "plr": "positive likelihood ratio",
        "nlr": "negative likelihood ratio",
        "pt": "prevalence threshold",
        "ba": "balanced accuracy",
        "f1": "f1-measure",
        "fm": "fowlkes-mallows index",
        "bm": "bookmaker informedness",
        "mk": "markedness",
        "dor": "diagnostic odds ratio",
    }

    for model_key, df_metrics in metrics_dict.items():
        if not model_key.endswith(suffix):
            continue
        model_name = model_name_map.get("_".join(model_key.split("_")[:-2]).lower(), model_key)
        for metric in metrics_name_mapping.keys():
            if metric not in df_metrics.columns:
                continue
            mean_value = df_metrics.loc["mean", metric]
            ci_lower = df_metrics.loc["ci_lower", metric]
            ci_upper = df_metrics.loc["ci_upper", metric]
            clf_report.loc[metric, "MetricFullNames"] = metrics_name_mapping.get(metric, metric)
            clf_report.loc[metric, model_name] = f"{mean_value:.3f} ({ci_lower:.3f}, {ci_upper:.3f})"
    return clf_report


if __name__ == "__main__":
    # nohup command:
    # run without logging output:
    # nohup python grid_search.py [n] > /dev/null 2>&1 & echo $! > ./log/gs.pid
    # run with logging output:
    # nohup python grid_search.py [n] > ./log/gs.log 2>&1 & echo $! > ./log/gs.pid

    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print(f"Error: '{sys.argv[1]}' is not a valid number")
            print("Usage: python grid_search.py [n]")
            print("\nnohup examples:")
            print("  nohup python grid_search.py 10 > /dev/null 2>&1 & echo $! > ./log/gs.pid")
            print("  nohup python grid_search.py 10 > ./log/gs.log 2>&1 & echo $! > ./log/gs.pid")
            sys.exit(1)
    else:
        n = 1

    for experiment_num in range(1, n + 1):
        print(f"\n{'='*60}")
        print(f"Starting experiment {experiment_num}/{n}")
        print(f"{'='*60}")

        gs = GridSearch()
        gs.full_search()

        print(f"\nCompleted experiment {experiment_num}/{n}")

    print(f"\nAll {n} experiments completed!")
