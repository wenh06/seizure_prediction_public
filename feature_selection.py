""" """

import gzip
import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import (  # noqa: F401
    RFE,
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    VarianceThreshold,
    chi2,
    f_classif,
)
from tqdm.auto import tqdm

from config import CFG, DEFAULTS, DataPreprocessConfig, FeatureConfig, FeatureSelectionConfig
from data_processing import get_training_data
from models import get_model
from utils import ReprMixin

__all__ = ["FeatureSelector"]


class FeatureSelector(ReprMixin):
    """ """

    __name__ = "FeatureSelector"

    def __init__(
        self,
        feature_selection_config: Optional[CFG] = None,
        preprocess_config: Optional[CFG] = None,
        feature_config: Optional[CFG] = None,
    ) -> None:
        """
        Parameters
        ----------
        feature_selection_config: CFG, optional,
            Feature selection configuration.
            If not provided, the default configuration will be used.
        preprocess_config: CFG, optional,
            Data preprocessing configuration.
            If not provided, the default configuration will be used.
        feature_config: CFG, optional,
            Feature engeering configuration.
            If not provided, the default configuration will be used.

        """
        if feature_selection_config is None:
            self._feature_selection_config = deepcopy(FeatureSelectionConfig)
        else:
            _feature_selection_config = deepcopy(feature_selection_config)
            self._feature_selection_config = deepcopy(FeatureSelectionConfig)
            self._feature_selection_config.update(_feature_selection_config)

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

        self.__feature_list = None
        # self.__selector = None

    def perform_selection(
        self,
        selector: str = "rfe",
        parameters: Optional[CFG] = None,
        feature_set: str = "TDSB",
    ) -> List[Dict[str, Union[List[str], BaseEstimator]]]:
        """
        Perform feature selection.

        Parameters
        ----------
        selector: str, default "rfe",
            Name of the feature selection method.
        parameters: CFG, optional,
            Parameters for the feature selection method.
            If not provided, the default parameters will be used.
        feature_set: str, default "TDSB",
            Name of the feature set to be used.

        Returns
        -------
        a list of dictionaries, each of which contains the following keys:
            - "selected": a list of selected feature names
            - "dropped": a list of dropped feature names
            - "selector": the feature selection object

        """
        if parameters is None:
            parameters = deepcopy(self.feature_selection_config[selector])
        else:
            _parameters = deepcopy(self.feature_selection_config[selector])
            parameters.update(_parameters)

        estimator = parameters.pop("model", None)
        if estimator is not None:
            estimator_params = parameters.pop("model_params", {})
            parameters["estimator"] = get_model(estimator, estimator_params)

        if parameters.get("score_func", None) is not None:
            parameters["score_func"] = eval(parameters["score_func"])

        feature_config = deepcopy(self.feature_config)
        feature_config.set_name = feature_set

        (
            X_train,
            y_train,
            X_test,
            y_test,
            self.__feature_list,
        ) = get_training_data(self.preprocess_config, feature_config, feature_set)
        feature_config.feature_list = self.__feature_list

        total_feature_num = len(self.__feature_list)
        n_features_to_select = parameters.pop("n_features_to_select", None)
        n_features_to_select_key = parameters.pop("n_features_to_select_key", None)

        if n_features_to_select is not None:
            n_features_to_select = [int(round(ratio * total_feature_num)) for ratio in n_features_to_select]
        else:
            n_features_to_select = [None]

        l_results = []

        for num in n_features_to_select:
            param = deepcopy(parameters)
            if n_features_to_select_key is not None:
                param[n_features_to_select_key] = num
            if estimator is not None:
                param_estimator = param.pop("estimator", None)
                selector_obj = self.selector_map[selector](param_estimator, **param)
            else:
                selector_obj = self.selector_map[selector](**param)
            result = {}

            if isinstance(selector_obj, VarianceThreshold):
                selector_obj.fit(X_train)
            else:
                selector_obj.fit(X_train, y_train)

            result["selected"] = [
                feature for feature, is_selected in zip(self.feature_list, selector_obj.get_support()) if is_selected
            ]
            result["dropped"] = [
                feature for feature, is_selected in zip(self.feature_list, selector_obj.get_support()) if not is_selected
            ]
            result["selector"] = selector_obj
            l_results.append(result)

        return l_results

    def full_selections(
        self,
        feature_set: str = "TDSB",
    ) -> Dict[str, List[Dict[str, Union[List[str], BaseEstimator]]]]:
        """
        Perform feature selection for all feature selection methods.

        Parameters
        ----------
        feature_set: str, default "TDSB",
            Name of the feature set to be used.

        Returns
        -------
        a dictionary of feature selection results,
        with additional "feature_config" and "preprocess_config" keys.
        Other keys are the names of feature selection methods,
        values are lists of dictionaries,
        each of which contains the following keys:
            - "selected": a list of selected feature names
            - "dropped": a list of dropped feature names
            - "selector": the feature selection object

        """
        sel_res = {}
        with tqdm(
            self.feature_selection_config.items(),
            total=len(self.feature_selection_config),
        ) as pbar:
            for selector, parameters in pbar:
                sel_res[selector] = self.perform_selection(selector, parameters)
        save_path = DEFAULTS.SAVE_DIR / f"""feature_selection_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.tar.gz"""
        sel_res["feature_config"] = self.feature_config
        sel_res["preprocess_config"] = self.preprocess_config
        with gzip.GzipFile(save_path, "wb") as gz_file:
            pickle.dump(sel_res, gz_file)
        return sel_res

    @staticmethod
    def load_selections(path: Union[str, Path]) -> Dict[str, List[Dict[str, Union[List[str], np.ndarray, BaseEstimator]]]]:
        """
        Load feature selection results from a file.

        Parameters
        ----------
        path: str or Path,
            Path to the file that contains the feature selection results.

        Returns
        -------
        a dictionary of feature selection results

        """
        with gzip.GzipFile(path, "rb") as gz_file:
            sel_res = pickle.load(gz_file)
        for k, l_res in sel_res.items():
            if k in ["feature_config", "preprocess_config"]:
                continue
            if not hasattr(l_res[0]["selector"], "estimator"):
                continue
            for idx, res in enumerate(l_res):
                X_train, y_train, X_test, y_test, _ = get_training_data(
                    sel_res["preprocess_config"],
                    sel_res["feature_config"],
                    feature_set="TDSB",
                    feature_list=res["selected"],
                )
                sel_res[k][idx]["model"] = res["selector"].estimator
                sel_res[k][idx]["X_train"] = X_train
                sel_res[k][idx]["y_train"] = y_train
                sel_res[k][idx]["X_test"] = X_test
                sel_res[k][idx]["y_test"] = y_test
                sel_res[k][idx]["model"].fit(X_train, y_train)
        return sel_res

    def to_feature_config(
        self,
        sel_res: Dict[str, Any],
        method: str = "rfe",
        extra_config: Optional[CFG] = None,
    ) -> FeatureConfig:
        """
        Get the feature configuration used when doing the feature selection.

        Parameters
        ----------
        sel_res: dict,
            The feature selection results.
        method: str, default "rfe",
            The feature selection method to be used.
        extra_config: CFG, optional,
            Extra configuration to be added to the FeatureConfig object.

        Returns
        -------
        feature_config: FeatureConfig,
            The feature configuration used when doing the feature selection,
            updated by `extra_config` if provided.

        """
        feature_config = deepcopy(sel_res["feature_config"])
        feature_config.update(extra_config or {})
        feature_config.sets = {}
        for item in sel_res[method]:
            name = f"""{method.upper()}_{len(item["selected"])}"""
            feature_config.sets[name] = item["selected"]
        feature_config.set_list = list(feature_config.sets)
        return feature_config

    def transform(self, X: np.ndarray, feature_list: List[str]) -> np.ndarray:
        """
        Perform the transformation on the given data.

        Parameters
        ----------
        X: np.ndarray,
            The data to be transformed.
        feature_list: list of str,
            The list of features that correspond to `X`.

        Returns
        -------
        np.ndarray,
            The transformed data.

        """
        assert self.selector is not None, "Please perform feature selection first"
        assert self.feature_list == feature_list, "Feature list mismatch"
        return self.selector.transform(X)

    @property
    def feature_selection_config(self) -> CFG:
        """ """
        return self._feature_selection_config

    @property
    def preprocess_config(self) -> CFG:
        """ """
        return self._preprocess_config

    @property
    def feature_config(self) -> CFG:
        """ """
        return self._feature_config

    @property
    def feature_list(self) -> List[str]:
        """ """
        return self.__feature_list

    # @property
    # def selector(self) -> BaseEstimator:
    #     """ """
    #     return self.__selector

    @property
    def selector_map(self) -> Dict[str, BaseEstimator]:
        """ """
        return {
            "rfe": RFE,
            "kbest": SelectKBest,
            "vt": VarianceThreshold,
            "var_thr": VarianceThreshold,
            "sfm": SelectFromModel,
            "sequential": SequentialFeatureSelector,
            "sfs": SequentialFeatureSelector,
        }


if __name__ == "__main__":
    fs = FeatureSelector()
    sel_res = fs.full_selections()
    del sel_res
