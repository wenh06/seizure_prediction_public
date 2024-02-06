"""
"""

import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import CFG
from data_processing import get_features
from metrics import SeizureClassificationMetrics
from utils import ReprMixin

try:
    from nn import SeizureMLP
except ModuleNotFoundError:  # torch not installed
    SeizureMLP = Any


__all__ = [
    "get_model",
    "load_model",
    "find_optimal_threshold",
    "SeizurePredictionModel",
]


LogisticRegression.__name__ = "LR"
GradientBoostingClassifier.__name__ = "GBDT"
RandomForestClassifier.__name__ = "RF"
_MODEL_MAP = {
    "logistic_regression": LogisticRegression,
    "lr": LogisticRegression,
    # "ridge_classifier": RidgeClassifier,
    # "ridge": RidgeClassifier,
    "svm": SVC,
    "svc": SVC,
    "random_forest": RandomForestClassifier,
    "rf": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "gdbt": GradientBoostingClassifier,
    "gb": GradientBoostingClassifier,
    "bagging": BaggingClassifier,
    "xgboost": XGBClassifier,
    "xgb": XGBClassifier,
    "sk_mlp": MLPClassifier,
}


def get_model(model_name: str, params: Optional[dict] = None) -> BaseEstimator:
    """
    Returns a model instance.

    Parameters
    ----------
    model_name: str,
        The name of the model.
    params: dict, optional,
        Parameters for the model.

    Returns
    -------
    model: BaseEstimator,
        The model instance.

    """
    model_cls = _MODEL_MAP[model_name]
    if model_cls in [GradientBoostingClassifier, SVC, MLPClassifier]:
        params.pop("n_jobs", None)
    return model_cls(**(params or {}))


def load_model(model_path: Union[str, Path]) -> BaseEstimator:
    """
    Loads a model from a file.

    Parameters
    ----------
    model_path: str or Path,
        The path to the model file.

    Returns
    -------
    model: BaseEstimator,
        The model instance.

    """
    return pickle.loads(Path(model_path).read_bytes())


_cm = SeizureClassificationMetrics(subset=["sens", "spec", "prec", "acc", "npv"])
_full_cm = SeizureClassificationMetrics(subset=None)


def find_optimal_threshold(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    precision: int = 3,
    return_metrics: bool = True,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """ """
    assert isinstance(precision, int) and precision > 0
    y_prob = model.predict_proba(X_train)
    interval = [0, 1]
    thr = 0.5
    for p in range(1, precision + 1):
        try:
            thr = _find_optimal_threshold(y_train, y_prob, interval, p)
        except ValueError:
            break
        half_len = pow(0.1, p)  # 0.5 * pow(0.1, p)
        interval = [max(0, thr - half_len), min(1, thr + half_len)]
    if not return_metrics:
        return thr
    return thr, _full_cm(y_train, y_prob, thr)


def _find_optimal_threshold(y_train: np.ndarray, y_prob: np.ndarray, interval: Sequence[float], precision: int) -> float:
    """ """
    thresholds = np.linspace(
        interval[0],
        interval[1],
        round((interval[1] - interval[0]) / pow(0.1, precision)) + 1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = np.array([[v for k, v in _cm(y_train, y_prob, thr).items()] for thr in thresholds])
    # the most balanced threshold
    _scores = scores.copy()
    _scores[(_scores < 0.5) | np.isnan(_scores)] = np.nan
    _scores = -_scores.std(axis=1)
    if not np.isnan(_scores).all():
        thr = thresholds[np.nanargmax(_scores)]
    else:  # All-NaN slice encountered for np.nanargmax
        scores = -scores.std(axis=1)
        thr = thresholds[np.nanargmax(scores)]
    return thr


class SeizurePredictionModel(ReprMixin):
    __name__ = "SeizurePredictionModel"

    def __init__(
        self,
        base_model: Union[BaseEstimator, SeizureMLP],
        thr: float = 0.5,
        preprocess_config: Optional[CFG] = None,
        feature_config: Optional[CFG] = None,
    ) -> None:
        """ """
        self.base_model = base_model
        self.thr = thr
        if not hasattr(self.base_model, "preprocess_config"):
            assert preprocess_config is not None
            self.base_model.preprocess_config = preprocess_config
        if not hasattr(self.base_model, "feature_config"):
            assert feature_config is not None
            self.base_model.feature_config = feature_config

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """ """
        return self.base_model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, thr: Optional[float] = None) -> np.ndarray:
        """ """
        return (self.predict_proba(X) > (thr or self.thr)).astype(int)

    def extra_repr_keys(self) -> List[str]:
        """ """
        return ["base_model", "thr"]

    def save(self, model_path: Union[str, Path]) -> None:
        """ """
        Path(model_path).write_bytes(pickle.dumps(self))

    @classmethod
    def from_file(cls, model_path: Union[str, Path]) -> "SeizurePredictionModel":
        """ """
        return pickle.loads(Path(model_path).read_bytes())

    def adjust_threshold(self, X_train: np.ndarray, y_train: np.ndarray, precision: int = 3) -> None:
        """ """
        self.thr = find_optimal_threshold(self.model, X_train, y_train, precision=precision, return_metrics=False)

    def pipeline(self, feed_data: Union[str, dict, Sequence[dict], pd.DataFrame]) -> List[Dict[str, Union[int, float]]]:
        """ """
        if isinstance(feed_data, str):
            feed_data = json.loads(feed_data)

        if isinstance(feed_data, dict):
            feed_data = [feed_data]

        feed_data = pd.DataFrame(feed_data)
        X = get_features(
            preprocess_config=self.base_model.preprocess_config,
            feature_config=self.base_model.feature_config,
            feature_set=self.base_model.feature_config.set_name,
            feature_list=self.base_model.feature_config.feature_list,
            data=feed_data,
            ensure_y=False,
            inference=True,
        )[0].values
        proba = self.predict_proba(X)
        pred = (proba > self.thr).astype(int)
        # re-scale the probability
        # lagrangian polyn of (0,0), (1,1), (thr, 0.5)
        proba = proba * (self.thr * (proba - self.thr) + 0.5 * (1 - proba)) / self.thr / (1 - self.thr)
        ret = [{"prediction": int(p), "probability": float(q)} for p, q in zip(pred, proba)]
        return ret
