# !/usr/bin/env python
# coding=UTF-8

"""
"""

import json
import pickle
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, List, Union, Sequence, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import (  # noqa: F401
    Binarizer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    MinMaxScaler,
    StandardScaler,
)
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, SMOTEN, RandomOverSampler
from imblearn.over_sampling.base import BaseOverSampler
from diff_binom_confint import (
    compute_confidence_interval,
    compute_difference_confidence_interval,
)
from deprecate_kwargs import deprecate_kwargs

from config import CFG, DEFAULTS, DataPreprocessConfig, FeatureConfig
from utils import list_sum, stratified_train_test_split  # noqa: F401


__all__ = [
    "load_raw_data",
    "preprocess_data",
    "get_features",
    "get_training_data",
    "get_seizure_risk",
    "get_seizure_risk_difference",
]


_DATA_FP = DEFAULTS.DATA_DIR / "胶质瘤20220609.xlsx"


def load_raw_data(data_fp: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Loads the raw data.

    Parameters
    ----------
    data_fp: str or Path, optional,
        The file path of the raw data.
        If not provided, the default file path will be used.

    Returns
    -------
    df_data: pd.DataFrame,
        The raw data.

    """
    df_data = pd.read_excel(data_fp or _DATA_FP, engine="openpyxl")
    df_data.set_index("患者编号", inplace=True)
    return df_data


def preprocess_data(
    config: Optional[CFG] = None, data: Optional[Union[str, Path, pd.DataFrame]] = None
) -> pd.DataFrame:
    """
    Perform data preprocessing on the raw data with the given configuration.

    Parameters
    ----------
    config: CFG, optional,
        The configuration of the data preprocessing.
        If not provided, the default configuration will be used.
    data: str or Path or DataFrame, optional,
        The file path of the raw data, or the DataFrame of the raw data.
        If not provided, the default file path will be used.

    Returns
    -------
    df_refined: pd.DataFrame,
        The preprocessed data.

    """
    if config is None:
        config = deepcopy(DataPreprocessConfig)
    else:
        _config = deepcopy(config)
        config = deepcopy(DataPreprocessConfig)
        config.update(_config)

    if data is None or isinstance(data, (str, Path)):
        df_data = load_raw_data(data)
    elif isinstance(data, pd.DataFrame):
        df_data = data
    else:
        raise ValueError(f"`data` of type `{type(data)}` is not supported")

    original_cols = df_data.columns
    y_col = config.y_col
    x_cols = config.x_cols

    # Drop useless columns
    dropped_cols = [c for c in df_data.columns if c in config.dropped_cols]
    df_refined = df_data.drop(columns=dropped_cols)

    for c in ["肿瘤分区", "C肿瘤分区1"]:
        if c in df_refined.columns:
            df_refined.loc[df_refined.index, c] = df_refined[c].fillna("其它")

    # Converting categorical feature to numeric
    if y_col in df_refined.columns:
        df_refined.loc[df_refined.index, y_col] = df_refined[y_col].map(
            config.y_col_mapping
        )

    # BIO features
    BIO_cate_var = [c for c in df_refined.columns if c in config.BIO_cate_var]
    df_refined.loc[df_refined.index, BIO_cate_var] = df_refined[BIO_cate_var].fillna(
        config.BIO_na_fillvalue
    )
    for c in BIO_cate_var:
        df_refined.loc[df_refined.index, c] = df_refined[c].map(config.BIO_mapping)

    # use OneHotEncoding to refine tumor zone features
    # 肿瘤分区 = list(set(list_sum([cell.split(",") for cell in df_refined.肿瘤分区])))
    肿瘤分区 = [
        item.replace("肿瘤分区_", "")
        for item in config.tumor_cate_var
        if item.startswith("肿瘤分区_")
    ]
    if "肿瘤分区" in df_refined.columns:
        for item in 肿瘤分区:
            df_refined.loc[df_refined.index, f"肿瘤分区_{item}"] = df_refined["肿瘤分区"].apply(
                lambda s: int(item in s.split(","))
            )
        # df_refined.loc[df_refined.index, "肿瘤分区_额或颞"] = (
        #     df_refined["肿瘤分区_额"] | df_refined["肿瘤分区_颞"]
        # )
        df_refined.drop(columns=["肿瘤分区"], inplace=True)

    # ordinal mapping
    for c in config.categorical_var:
        if c not in df_refined.columns:
            continue
        if set(df_refined[c].unique()) == set([0, 1]) or c in BIO_cate_var:
            continue
        if c not in config.x_col_mappings:
            continue
        df_refined.loc[df_refined.index, c] = df_refined[c].map(
            config.x_col_mappings[c]
        )

    # normalize features via sklearn.preprocessing
    age_scaler_path = Path(config.age_scaler_path)
    age_scaler_path = DEFAULTS.SAVE_DIR / age_scaler_path.name
    if age_scaler_path.is_file():
        age_scaler = pickle.loads(age_scaler_path.read_bytes())
    else:
        age_scaler = MinMaxScaler()
    if "年龄" in df_refined.columns:
        df_refined.loc[df_refined.index, "年龄"] = age_scaler.fit_transform(
            df_refined["年龄"].values.reshape(-1, 1)
        ).flatten()
        if not age_scaler_path.is_file():
            age_scaler_path.write_bytes(pickle.dumps(age_scaler))

    return df_refined


def get_features(
    preprocess_config: Optional[CFG] = None,
    feature_config: Optional[CFG] = None,
    feature_set: Optional[str] = None,
    feature_list: Optional[List[str]] = None,
    data: Optional[Union[str, Path, pd.DataFrame]] = None,
    ensure_y: bool = False,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Get the features from the preprocessed data.

    Parameters
    ----------
    preprocess_config: CFG, optional,
        The configuration of the data preprocessing.
        If not provided, the default configuration will be used.
    feature_config: CFG, optional,
        The configuration of the feature engineering.
        If not provided, the default configuration will be used.
    feature_set: str, optional,
        The feature set to be used.
        If provided, overrides ``feature_config.set_name``.
        If not provided, `feature_config.set_name` should be provided.
    feature_list: list of str, optional,
        The feature list to be used.
        If provided, `feature_list` should be a subset
        of the features produced by the given `feature_set`.
        If not provided, the default feature list that
        corresponds to `feature_set` will be used.
    data: str or Path or DataFrame, optional,
        The file path of the raw data, or the DataFrame of the raw data.
        If not provided, the default file path will be used.
    ensure_y: bool, default False,
        Whether to ensure that the returned DataFrame contains the target variable.

    Returns
    -------
    df_refined: pd.DataFrame,
        The DataFrame of the features, optionally containing the target variable.
    feature_list: list of str,
        The feature list.

    """
    if preprocess_config is None:
        preprocess_config = deepcopy(DataPreprocessConfig)
    else:
        _preprocess_config = deepcopy(preprocess_config)
        preprocess_config = deepcopy(DataPreprocessConfig)
        preprocess_config.update(_preprocess_config)
    df_refined = preprocess_data(preprocess_config, data)

    if feature_config is None:
        feature_config = deepcopy(FeatureConfig)
    else:
        _feature_config = deepcopy(feature_config)
        feature_config = deepcopy(FeatureConfig)
        feature_config.update(_feature_config)

    if feature_set is not None:
        feature_config.set_name = feature_set
    elif feature_config.set_name is None:
        raise ValueError(
            "Either `feature_set` or `feature_config.set_name` should be provided."
        )
    else:
        feature_set = feature_config.set_name
    assert feature_set in feature_config.sets

    feature_cols = feature_config.sets[feature_set]
    assert set(feature_cols).issubset(
        df_refined.columns
    ), f"processed data table lack columns: {set(feature_cols) - set(df_refined.columns)}"
    if preprocess_config.y_col in df_refined.columns:
        df_refined = df_refined[[preprocess_config.y_col] + feature_cols]
    elif not ensure_y:
        df_refined = df_refined[feature_cols]
    else:
        raise ValueError(
            f"y_col `{preprocess_config.y_col}` is not in the refined data"
        )
    bio_cate_cols = [
        item for item in feature_cols if item in preprocess_config.BIO_cate_var
    ]
    bio_cont_cols = [
        item for item in feature_cols if item in preprocess_config.BIO_cont_var
    ]
    if feature_config.BIO_na_strategy == "drop":
        if kwargs.get("inference", False):
            # in inference mode, skip dropping rows with missing values
            # so that errors can be raised
            pass
        else:
            if len(bio_cont_cols) > 0:
                df_refined.drop(columns=bio_cont_cols, inplace=True)
            if len(bio_cate_cols) > 0:
                series = (
                    df_refined[bio_cate_cols]
                    == preprocess_config.BIO_mapping[preprocess_config.BIO_na_fillvalue]
                ).any(axis=1)
                df_refined = df_refined[~series]
    elif feature_config.BIO_na_strategy == "random":
        raise NotImplementedError
    elif feature_config.BIO_na_strategy == "knn":
        raise NotImplementedError
    elif feature_config.BIO_na_strategy == "keep":
        pass
    else:
        raise ValueError(f"{feature_config.BIO_na_strategy} is not supported")

    if feature_config.binarize_variables:
        l_dummy_df, l_dummy_cols = [], []
        for c in preprocess_config.categorical_var:
            if c not in df_refined.columns or len(df_refined[c].unique()) == 2:
                continue
            df_dummy = pd.get_dummies(df_refined[c], prefix=c)
            l_dummy_df.append(df_dummy)
            l_dummy_cols.append(c)
            feature_cols.remove(c)
            feature_cols.extend(df_dummy.columns.tolist())
        df_refined = pd.concat([df_refined] + l_dummy_df, axis=1)
        df_refined.drop(columns=l_dummy_cols, inplace=True)

    if feature_list is not None:
        assert set(feature_list).issubset(df_refined.columns.tolist())
        if preprocess_config.y_col in df_refined.columns:
            df_refined = df_refined[[preprocess_config.y_col] + feature_list]
        else:
            df_refined = df_refined[feature_list]
        feature_cols = feature_list
    else:
        feature_list = feature_cols

    # check nan cells in the refined data
    if not kwargs.get("allow_missing", False) and df_refined.isnull().values.any():
        err = {
            row_idx
            + 1: [c for c in df_refined.columns if pd.isnull(df_refined[c][row_idx])]
            for row_idx in range(df_refined.shape[0])
            if df_refined.isnull().iloc[row_idx].any()
        }
        err = "; ".join(
            [
                f"the {row_idx}-th piece of data has nan values in columns `{cols}`"
                for row_idx, cols in err.items()
            ]
        )
        raise ValueError(f"nan cells detected in the refined data: {err}")

    return df_refined, feature_list


def get_training_data(
    preprocess_config: Optional[CFG] = None,
    feature_config: Optional[CFG] = None,
    feature_set: str = "TDS",
    feature_list: Optional[List[str]] = None,
    data: Optional[Union[str, Path, pd.DataFrame]] = None,
    return_dtype: str = "np",
    **kwargs: Any,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]],
    Tuple[pd.DataFrame, pd.DataFrame, List[str]],
]:
    """
    Get the training data with the given configurations.

    Parameters
    ----------
    preprocess_config: CFG, optional,
        The configuration of the data preprocessing.
        If not provided, the default configuration will be used.
    feature_config: CFG, optional,
        The configuration of the feature engineering.
        If not provided, the default configuration will be used.
    feature_set: str, default "TDS",
        The feature set to be used.
    feature_list: list of str, optional,
        The feature list to be used.
        If provided, `feature_list` should be a subset
        of the features produced by the given `feature_set`.
        If not provided, the default feature list that
        corresponds to `feature_set` will be used.
    data: str or Path or DataFrame, optional,
        The file path of the raw data, or the DataFrame of the raw data.
        If not provided, the default file path will be used.
    return_dtype: str, default "np",
        The data type of the returned data, can be either "np" or "pd".

    Returns
    -------
    (tuple of `np.ndarray` or tuple of `pd.DataFrame`), list of str,
    if `return_dtype` is "np":
        X_train, X_test, y_train, y_test, feature_list
    if `return_dtype` is "pd":
        df_train, df_test, feature_list

    """
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

    assert kwargs.get("inference", False) is False, "inference mode is not supported"
    df_refined, feature_cols = get_features(
        preprocess_config,
        feature_config,
        feature_set,
        feature_list,
        data,
        ensure_y=True,
        **kwargs,
    )
    feature_list = feature_cols

    preprocess_config.split_file_path = Path(preprocess_config.split_file_path)
    preprocess_config.split_file_path = (
        DEFAULTS.DATA_DIR / preprocess_config.split_file_path.name
    )
    if preprocess_config.split_file_path.is_file():
        train_test_split = json.loads(preprocess_config.split_file_path.read_text())
        df_train = df_refined[df_refined.index.isin(train_test_split["train"])]
        df_test = df_refined[df_refined.index.isin(train_test_split["test"])]
    else:
        df_train, df_test = stratified_train_test_split(
            df_refined,
            stratified_cols=[preprocess_config.y_col, "年龄段", "性别"],
            test_ratio=0.2,
            reset_index=False,
        )
        preprocess_config.split_file_path.write_text(
            json.dumps(
                {"train": df_train.index.tolist(), "test": df_test.index.tolist()},
                ensure_ascii=False,
            )
        )

    if return_dtype.lower() == "np":
        X_train = df_train[feature_cols].values
        y_train = df_train[preprocess_config.y_col].values
        X_test = df_test[feature_cols].values
        y_test = df_test[preprocess_config.y_col].values
        over_sampler = feature_config.get("over_sampler", None)
        if over_sampler is not None:
            over_sampler = over_sampler.lower()
            if over_sampler == "smotenc":
                over_sampler_kw = {
                    "categorical_features": [
                        idx
                        for idx, col in enumerate(feature_cols)
                        if col not in preprocess_config.continuous_var
                    ]
                }
                if len(over_sampler_kw["categorical_features"]) == len(feature_cols):
                    over_sampler = "smoten"
                    warnings.warn(
                        "No categorical features are provided, switched to `SMOTEN`"
                    )
                    over_sampler_kw.pop("categorical_features")
            else:
                over_sampler_kw = {}
            over_sampler = _get_over_sampler(over_sampler, **over_sampler_kw)
            X_train, y_train = over_sampler.fit_resample(X_train, y_train)
            # no over-sampling for test set
            # X_test, y_test = over_sampler.fit_resample(X_test, y_test)
        return X_train, y_train, X_test, y_test, feature_list
    elif return_dtype.lower() == "pd":
        if feature_config.get("over_sampler", None) is not None:
            warnings.warn(
                "Over sampling is not supported when return_dtype is `pd.DataFrame`"
            )
        return df_train, df_test, feature_list
    else:
        raise ValueError(f"{return_dtype} is not supported")


@deprecate_kwargs([["conf_level", "ci"], ["confint_method", "ci_type"]])
def get_seizure_risk(
    col: str,
    classes: Optional[Union[str, int, Sequence[Union[str, int]]]] = None,
    ci: float = 0.95,
    ci_type: str = "wilson",
    df_data: Optional[pd.DataFrame] = None,
    data_fp: Optional[Union[str, Path]] = None,
    y_col: Optional[str] = None,
    positive_class: Optional[Union[str, int]] = None,
    negative_class: Optional[Union[str, int]] = None,
) -> Dict[Union[str, int], Dict[str, Union[float, Tuple[float, float]]]]:
    """
    Compute the seizure risk of a variable.

    Parameters
    ----------
    col: str
        The column name of the variable.
    classes: str or int or sequence of str or int, optional,
        The classes of the variable. If None, all classes of the variable will be used.
    ci: float, default 0.95,
        The confidence level of the confidence interval.
    ci_type: str, default "wilson",
        The method to compute the confidence interval.
    df_data: pd.DataFrame, optional,
        The data frame containing the variable and the target variable.
        If provided, `data_fp` will be ignored.
    data_fp: str or Path, optional,
        The file path of the data file.
        If not provided, the default data file will be used.
        Ignored if `df_data` is provided.
    y_col: str, optional,
        The column name of the target variable.
        If not provided, the default column name will be used.
    positive_class: str or int, optional,
        The positive class of the target variable.
        If not provided, the default positive class will be used.
    negative_class: str or int, optional,
        The negative class of the target variable.
        If not provided, the default negative class will be used.

    Returns
    -------
    dict of the form {class: {"risk": float, "confidence_interval": (float, float)}}

    """
    if y_col is None:
        y_col = DataPreprocessConfig.y_col
    if positive_class is None:
        positive_class = DataPreprocessConfig.positive_class
    if negative_class is None:
        negative_class = DataPreprocessConfig.negative_class
    if df_data is None:
        df_data = load_raw_data(data_fp)[[y_col, col]]
    df_risk = df_data.groupby(col)[y_col].value_counts().unstack()
    if df_risk.columns.tolist() == [negative_class, positive_class]:
        df_risk = df_risk[df_risk.columns[::-1]]
    ret_val = {}
    if classes is None:
        classes = df_risk.index.tolist()
    if isinstance(classes, (str, int)):
        classes = [classes]
    for c in classes:
        df_risk.loc["其他"] = df_risk.loc[df_risk.index != c].sum(axis=0)
        mat = df_risk.loc[[c, "其他"]].values
        n_positive, n_negative = mat[0, :]
        risk = n_positive / (n_positive + n_negative)
        confidence_interval = compute_confidence_interval(
            n_positive, n_positive + n_negative, ci, ci_type
        ).astuple()
        ret_val[c] = {
            "risk": risk,
            "confidence_interval": confidence_interval,
        }

    return ret_val


@deprecate_kwargs([["conf_level", "ci"], ["confint_method", "ci_type"]])
def get_seizure_risk_difference(
    col: str,
    ref_class: Union[str, int],
    ci: float = 0.95,
    ci_type: str = "wilson",
    df_data: Optional[pd.DataFrame] = None,
    data_fp: Optional[Union[str, Path]] = None,
    y_col: Optional[str] = None,
    positive_class: Optional[Union[str, int]] = None,
    negative_class: Optional[Union[str, int]] = None,
) -> Dict[Union[str, int], Dict[str, Union[float, Tuple[float, float]]]]:
    """
    Compute the seizure risk of a variable.

    Parameters
    ----------
    col: str
        The column name of the variable.
    ref_class: str or int,
        The class to be used as the reference class.
    ci: float, default 0.95,
        The confidence level of the confidence interval.
    ci_type: str, default "wilson",
        The method to compute the confidence interval.
    df_data: pd.DataFrame, optional,
        The data frame containing the variable and the target variable.
        If provided, `data_fp` will be ignored.
    data_fp: str or Path, optional,
        The file path of the data file.
        If not provided, the default data file will be used.
        Ignored if `df_data` is provided.
    y_col: str, optional,
        The column name of the target variable.
        If not provided, the default column name will be used.
    positive_class: str or int, optional,
        The positive class of the target variable.
        If not provided, the default positive class will be used.
    negative_class: str or int, optional,
        The negative class of the target variable.
        If not provided, the default negative class will be used.

    Returns
    -------
    dict of the form {class: {"risk_difference": float, "confidence_interval": (float, float)}}

    """
    if y_col is None:
        y_col = DataPreprocessConfig.y_col
    if positive_class is None:
        positive_class = DataPreprocessConfig.positive_class
    if negative_class is None:
        negative_class = DataPreprocessConfig.negative_class
    if df_data is None:
        df_data = load_raw_data(data_fp)[[y_col, col]]
    df_risk = df_data.groupby(col)[y_col].value_counts().unstack()
    if df_risk.columns.tolist() == [negative_class, positive_class]:
        df_risk = df_risk[df_risk.columns[::-1]]
    ret_val = {
        f"{ref_class} (Ref.)": {
            "risk_difference": 0,
            "confidence_interval": (0, 0),
        }
    }
    classes = df_risk.index.tolist()
    classes.remove(ref_class)
    ref_positive, ref_negative = df_risk.loc[ref_class].values
    ref_risk = get_seizure_risk(
        col,
        ref_class,
        ci,
        ci_type,
        df_data=df_data,
        y_col=y_col,
        positive_class=positive_class,
        negative_class=negative_class,
    )[ref_class]["risk"]
    for c in classes:
        n_positive, n_negative = df_risk.loc[c].values
        confidence_interval = compute_difference_confidence_interval(
            n_positive,
            n_positive + n_negative,
            ref_positive,
            ref_positive + ref_negative,
            ci,
            ci_type,
        ).astuple()
        risk_difference = (
            get_seizure_risk(
                col,
                c,
                ci,
                ci_type,
                df_data=df_data,
                y_col=y_col,
                positive_class=positive_class,
                negative_class=negative_class,
            )[c]["risk"]
            - ref_risk
        )
        ret_val[c] = {
            "risk_difference": risk_difference,
            "confidence_interval": confidence_interval,
        }

    return ret_val


def _get_over_sampler(name: str, **kwargs) -> BaseOverSampler:
    """ """
    if name.lower() == "smotenc":
        assert "categorical_features" in kwargs
        return SMOTENC(**kwargs)
    elif name.lower() == "smote":
        return SMOTE(**kwargs)
    elif name.lower() == "smoten":
        return SMOTEN(**kwargs)
    elif name.lower() == "randomoversampler":
        return RandomOverSampler(**kwargs)
    elif name.lower() == "adasyn":
        return ADASYN(**kwargs)
    else:
        raise ValueError(f"{name} is not supported")
