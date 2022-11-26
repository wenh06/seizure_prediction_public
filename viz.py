"""
"""

from pathlib import Path
from typing import Tuple, Sequence, Optional, Union, Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc
from sklearn.base import BaseEstimator
import shap

from config import DEFAULTS, DataPreprocessConfig
from data_processing import load_raw_data
from grid_search import gather_grid_search_results
from feature_selection import FeatureSelector
from risk_diff import gen_seizure_risk_diff_TDSB_ext
from utils import list_sum, separate_by_capital_letters


__all__ = [
    "plot_age_distribution",
    "plot_sex_distribution",
    "plot_feature_importance",
    "plot_feature_permutation_importance",
    "plot_roc_curve",
    "plot_grid_search_agg_boxplot",
    "plot_seizure_risk_difference",
    "plot_feature_selection_results",
    "plot_shap_summary",
]


# sns.set()
# plt.rcParams["xtick.labelsize"] = 18
# plt.rcParams["ytick.labelsize"] = 18
# plt.rcParams["axes.labelsize"] = 24
# plt.rcParams["legend.fontsize"] = 20
# plt.rcParams["legend.title_fontsize"] = 22

# mpl.use("pgf")  # stwich backend to pgf

font_dirs = [str(DEFAULTS.FONT_DIR)]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
_font_names = [item.name for item in font_manager.fontManager.ttflist]
# if "Times New Roman" in _font_names:
#     # times.ttf
#     # setting "Times New Roman" would cause LatexError
#     plt.rcParams["font.family"] = "times"
# else:
#     plt.rcParams["font.family"] = "JDLangZhengTi"
del font_dirs, font_files, font_file


_linestyle_tuple = [  # name, linestyle (offset, on-off-seq)
    ("solid", (0, ())),
    ("densely dashed", (0, (5, 1))),
    ("densely dotted", (0, (1, 1))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
    ("dashed", (0, (5, 5))),
    ("dotted", (0, (1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("loosely dotted", (0, (1, 10))),
    ("long dash with offset", (5, (10, 3))),
    ("loosely dashed", (0, (5, 10))),
]


_df_colors = pd.read_csv(DEFAULTS.DATA_DIR / "named-colors.csv")


def _get_color(name: str) -> str:
    """
    Get the html (hex) color of the name.

    Parameters
    ----------
    name: str,
        The name of the color.

    Returns
    -------
    color: str,
        The html color of the name.

    """
    return _df_colors.loc[_df_colors.name == name, "hex"].values[0]


def plot_age_distribution() -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the age distribution of the subjects.

    Returns
    -------
    fig: matplotlib.figure.Figure,
        The figure object.
    ax: matplotlib.axes.Axes,
        The axes object.

    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.set()
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["axes.labelsize"] = 26
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["legend.title_fontsize"] = 24
    if "Times New Roman" in _font_names:
        # times.ttf
        if mpl.get_backend() == "pgf":
            # setting "Times New Roman" would cause LatexError
            plt.rcParams["font.family"] = "times"
        else:
            plt.rcParams["font.family"] = "Times New Roman"
    else:
        plt.rcParams["font.family"] = "JDLangZhengTi"

    df = load_raw_data()
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.hist(
        [df[df.整体有癫痫 == "否"].年龄, df[df.整体有癫痫 == "是"].年龄],
        stacked=True,
        bins=40,
        label=[
            "Seizure Free",
            "Postoperative Seizure",
        ],
        color=["#1f77b4", "#d62728"],
        # color=[_get_color("International Klein Blue"), _get_color("Chinese red")],
    )
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of Subjects (n.u.)")
    rect = patches.Rectangle(
        (28, 24),
        10.5,
        20,
        linewidth=3,
        linestyle="dotted",
        edgecolor="b",
        facecolor="g",
        alpha=0.3,
    )
    # ax.add_patch(rect)
    ax.legend(loc="best")
    return fig, ax


def plot_sex_distribution() -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the sex distribution of the subjects.

    Returns
    -------
    fig: matplotlib.figure.Figure,
        The figure object.
    ax: matplotlib.axes.Axes,
        The axes object.

    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.set()
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["axes.labelsize"] = 26
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["legend.title_fontsize"] = 24
    if "Times New Roman" in _font_names:
        # times.ttf
        if mpl.get_backend() == "pgf":
            # setting "Times New Roman" would cause LatexError
            plt.rcParams["font.family"] = "times"
        else:
            plt.rcParams["font.family"] = "Times New Roman"
    else:
        plt.rcParams["font.family"] = "JDLangZhengTi"

    df = load_raw_data()
    fig, ax = plt.subplots(figsize=(8, 8))
    coords = ["Male", "Female"]
    df_tmp = df[df.整体有癫痫 == "否"]
    heights1 = [(df_tmp.性别 == "男").sum(), (df_tmp.性别 == "女").sum()]
    ax.bar(coords, heights1, width=0.35, label="Seizure Free", color="#1f77b4")
    df_tmp = df[df.整体有癫痫 == "是"]
    heights2 = [(df_tmp.性别 == "男").sum(), (df_tmp.性别 == "女").sum()]
    ax.bar(
        coords,
        heights2,
        width=0.35,
        label="Postoperative Seizure",
        bottom=heights1,
        color="#d62728",
    )
    ax.set_xlabel("Gender")
    ax.set_ylabel("Number of Subjects (n.u.)")
    ax.legend(loc="best")
    return fig, ax


def plot_feature_importance(
    clf: BaseEstimator,
    sort: bool = True,
    zh2en: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the feature importance of the given classifier.

    Parameters
    ----------
    clf: sklearn.base.BaseEstimator,
        The classifier.
    sort: bool, default True,
        Whether to sort the features by their importance.
    zh2en: bool, default True,
        If True, convert the feature names from Chinese to English.

    Returns
    -------
    fig: matplotlib.figure.Figure,
        The figure object.
    ax: matplotlib.axes.Axes,
        The axes object.

    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.set()
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["axes.labelsize"] = 26
    plt.rcParams["legend.fontsize"] = 22
    plt.rcParams["legend.title_fontsize"] = 24
    if "Times New Roman" in _font_names and zh2en:
        # times.ttf
        if mpl.get_backend() == "pgf":
            # setting "Times New Roman" would cause LatexError
            plt.rcParams["font.family"] = "times"
        else:
            plt.rcParams["font.family"] = "Times New Roman"
    else:
        plt.rcParams["font.family"] = "JDLangZhengTi"

    feature_list = clf.feature_config.feature_list
    if zh2en:
        feature_list = [
            DataPreprocessConfig.zh2en_mapping[item].replace("_", "-")
            if not item.startswith("BIO")
            else item
            for item in feature_list
        ]
    fig, ax = plt.subplots(figsize=(12, 0.5 * len(feature_list)))
    if sort:
        indices = np.argsort(clf.feature_importances_)
        ax.barh(np.array(feature_list)[indices], clf.feature_importances_[indices])
    else:
        ax.barh(feature_list, clf.feature_importances_)
    return fig, ax


def plot_feature_permutation_importance(
    clf: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sort: bool = True,
    merge_split_variables: bool = True,
    zh2en: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the feature permutation importance of the given classifier.

    Parameters
    ----------
    clf: sklearn.base.BaseEstimator,
        The classifier.
    X_test: numpy.ndarray,
        The test data.
    y_test: numpy.ndarray,
        The test labels.
    sort: bool, default True,
        Whether to sort the features by their importance.
    zh2en: bool, default True,
        If True, convert the feature names from Chinese to English.
    merge_split_variables: bool, default True,
        If True, merge the split variables into one variable.

    Returns
    -------
    fig: matplotlib.figure.Figure,
        The figure object.
    ax: matplotlib.axes.Axes,
        The axes object.

    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.set()
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["axes.labelsize"] = 26
    plt.rcParams["legend.fontsize"] = 22
    plt.rcParams["legend.title_fontsize"] = 24
    if "Times New Roman" in _font_names and zh2en:
        # times.ttf
        if mpl.get_backend() == "pgf":
            # setting "Times New Roman" would cause LatexError
            plt.rcParams["font.family"] = "times"
        else:
            plt.rcParams["font.family"] = "Times New Roman"
    else:
        plt.rcParams["font.family"] = "JDLangZhengTi"

    feature_list = clf.feature_config.feature_list
    perm_importance = permutation_importance(clf, X_test, y_test, scoring="roc_auc")
    importances_mean = perm_importance.importances_mean

    if merge_split_variables:
        tumor_zone_inds = [
            idx for idx, item in enumerate(feature_list) if item.startswith("肿瘤分区")
        ]
        comorbidity_inds = [
            idx for idx, item in enumerate(feature_list) if item.startswith("合并症")
        ]
        mask = np.ones_like(importances_mean, dtype=bool)
        mask[tumor_zone_inds + comorbidity_inds] = False
        tmp_values, tmp_features = [], []
        if len(tumor_zone_inds) > 0:
            tmp_values.append(importances_mean[tumor_zone_inds].sum())
            tmp_features.append("肿瘤分区")
        if len(comorbidity_inds) > 0:
            tmp_values.append(importances_mean[comorbidity_inds].sum())
            tmp_features.append("合并症")
        importances_mean = importances_mean[mask]
        importances_mean = np.append(importances_mean, tmp_values)
        feature_list = np.append(np.array(feature_list)[mask], tmp_features)

    if zh2en:
        zh2en_mapping = {
            k: separate_by_capital_letters(v, capitalize=True, drop=["cate"])
            for k, v in DataPreprocessConfig.zh2en_mapping.items()
        }
        zh2en_mapping.update(
            {
                "病理分级": "WHO grade",
            }
        )
        feature_list = [
            zh2en_mapping[item].replace("_", "-").replace(" (cm)", "")
            if not item.startswith("BIO_")
            else item.replace("BIO_", "")
            for item in feature_list
        ]

    fig, ax = plt.subplots(figsize=(12, 0.5 * len(feature_list)))
    if sort:
        indices = np.argsort(importances_mean)
        ax.barh(np.array(feature_list)[indices], importances_mean[indices])
    else:
        ax.barh(feature_list, importances_mean)
    ax.set_xlabel("Permutation Importance Mean on AUC")
    return fig, ax


def plot_roc_curve(
    estimators: Union[BaseEstimator, Sequence[BaseEstimator]],
    X: Union[np.ndarray, Sequence[np.ndarray]],
    y: Union[np.ndarray, Sequence[np.ndarray]],
    estimator_names: Optional[Union[str, Sequence[str]]] = None,
    sort: bool = True,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the ROC curve of the given estimators.

    Parameters
    ----------
    estimators: sklearn.base.BaseEstimator or list of sklearn.base.BaseEstimator,
        The estimator(s).
    X: numpy.ndarray or list of numpy.ndarray,
        The data.
    y: numpy.ndarray or list of numpy.ndarray,
        The labels.
    estimator_names: str or list of str, default None,
        The name(s) of the estimator(s).
        If None, the name (`.__class__.__name__`) of the estimator will be used.
    sort: bool, default True,
        Whether to sort the estimators by their AUC in descending order.
    **kwargs: dict,
        Additional keyword arguments for `ax.plot`.

    Returns
    -------
    fig: matplotlib.figure.Figure,
        The figure object.
    ax: matplotlib.axes.Axes,
        The axes object.

    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.set()
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["axes.labelsize"] = 26
    plt.rcParams["legend.fontsize"] = 22
    plt.rcParams["legend.title_fontsize"] = 24
    if "Times New Roman" in _font_names:
        # times.ttf
        if mpl.get_backend() == "pgf":
            # setting "Times New Roman" would cause LatexError
            plt.rcParams["font.family"] = "times"
        else:
            plt.rcParams["font.family"] = "Times New Roman"
    else:
        plt.rcParams["font.family"] = "JDLangZhengTi"

    fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (16, 12)))
    if isinstance(estimators, BaseEstimator):
        estimators = [estimators]
    if estimator_names is None:
        estimator_names = [
            estimator.__class__.__name__.replace("Classifier", "")
            for estimator in estimators
        ]
    elif isinstance(estimator_names, str):
        estimator_names = [estimator_names]
    assert len(estimators) == len(estimator_names)
    kwargs["linewidth"] = kwargs.get("linewidth", 2.5)
    plot_tuples = []
    for idx, estimator in enumerate(estimators):
        name = estimator_names[idx]
        if X[0].ndim == 1:
            X_ = X
        else:
            X_ = X[idx]
        if y[0].ndim == 0:
            y_ = y
        else:
            y_ = y[idx]
        y_pred = estimator.predict_proba(X_)[:, 1]
        fpr, tpr, _ = roc_curve(y_, y_pred)
        auc_score = auc(fpr, tpr)
        plot_tuples.append((fpr, tpr, auc_score, name))
    if sort:  # sort by auc score
        plot_tuples = sorted(plot_tuples, key=lambda x: x[2], reverse=True)
    for fpr, tpr, auc_score, name in plot_tuples:
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})", **kwargs)

    ax.plot([0, 1], [0, 1], ls=":", lw=1.5, c="k")

    xlabel = "False Positive Rate"
    ylabel = "True Positive Rate"
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc="lower right")

    return fig, ax


def plot_grid_search_agg_boxplot(
    feature_sets: Union[Sequence[str], str],
    bio_na: str = "keep",
    sub_dirs: Optional[Union[str, Sequence[str]]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the aggregated boxplot of the grid search results.

    Parameters
    ----------
    feature_sets: list of str or str,
        feature_sets, one of the following:
        "TD", "TDS", "TDB", "TDSB"
    bio_na: str, default="keep",
        how to handle missing values in the bio feature,
        valid only when feature_sets is a sequence
    sub_dirs: str or sequence of str, optional,
        The sub directories to store the results.

    Returns
    -------
    fig: matplotlib.figure.Figure,
        The figure object.
    ax: matplotlib.axes.Axes,
        The axes object.

    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.set()
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["legend.fontsize"] = 18
    plt.rcParams["legend.title_fontsize"] = 20
    if "Times New Roman" in _font_names:
        # times.ttf
        if mpl.get_backend() == "pgf":
            # setting "Times New Roman" would cause LatexError
            plt.rcParams["font.family"] = "times"
        else:
            plt.rcParams["font.family"] = "Times New Roman"
    else:
        plt.rcParams["font.family"] = "JDLangZhengTi"

    gs_results = gather_grid_search_results(raw=True, sub_dirs=sub_dirs)
    num_models = len(gs_results.model.unique())
    if isinstance(feature_sets, str) and "B" not in feature_sets:
        df_data = gs_results[gs_results.feature == feature_sets].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax = sns.boxplot(ax=ax, x="model", y="best_score", data=df_data)
        ax.set_ylabel("AUC")
        ax.set_xlabel("Model")
        for loc in np.arange(0.5, 0.5 + num_models, 1):
            ax.axvline(loc, linestyle="dashed", linewidth=1, color="lightgray")
    elif isinstance(feature_sets, str):
        df_data = gs_results[gs_results.feature == feature_sets].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(16, 8))
        ax = sns.boxplot(ax=ax, x="model", y="best_score", hue="BIO_na", data=df_data)
        ax.set_ylabel("AUC")
        ax.set_xlabel("Model")
        for loc in np.arange(0.5, 0.5 + num_models, 1):
            ax.axvline(loc, linestyle="dashed", linewidth=1, color="lightgray")
        ax.legend(loc="best", title="BIO-NA")
    else:  # feature_sets is a sequence of feature sets
        df_data = gs_results[
            (gs_results.feature.isin(feature_sets))
            & ((gs_results.BIO_na.isna()) | (gs_results.BIO_na == bio_na))
        ].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(16, 8))
        ax = sns.boxplot(ax=ax, x="model", y="best_score", hue="feature", data=df_data)
        ax.set_ylabel("AUC")
        ax.set_xlabel("Model")
        for loc in np.arange(0.5, 0.5 + num_models, 1):
            ax.axvline(loc, linestyle="dashed", linewidth=1, color="lightgray")
        ax.legend(loc="lower left", title="Feature Set", ncol=len(feature_sets))
    return fig, ax


def plot_seizure_risk_difference(
    seizure_risk_dict: Optional[dict] = None,
    comorbidity_type: int = 0,
    biomarker_type: int = 0,
    zh2en: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the seizure risk difference.

    Parameters
    ----------
    seizure_risk_dict: dict, optional,
        The dictionary of seizure risk difference.
        If None, use the default dictionary generated by
        `get_seizure_risk_difference`.
    comorbidity_type: int, default 0,
        valid only when seizure_risk_dict is None.
        The manifistation of type of the comorbidity variables.
        0 for comparison of "Yes" and "No" for each comorbidity variable.
        1 for comparisons for the positive part of each comorbidity variable.
    biomarker_type: int, default 0,
        the manifistation of type of the biomarker variables.
        0 for merging the classes other than "-" into one class.
        1 for keeping the classes other than "-" as they are.
    zh2en: bool, default True,
        If True, convert the feature names from Chinese to English.

    Returns
    -------
    fig: matplotlib.figure.Figure,
        The figure object.
    ax: matplotlib.axes.Axes,
        The axes object.

    """
    if seizure_risk_dict is None:
        seizure_risk_dict = gen_seizure_risk_diff_TDSB_ext(
            return_type="dict", comorbidity_type=comorbidity_type, zh2en=zh2en
        )

    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.set_style("white")
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["axes.labelsize"] = 22
    plt.rcParams["legend.fontsize"] = 18
    plt.rcParams["legend.title_fontsize"] = 20
    mpl.use("pgf")  # stwich backend to pgf
    plt.rcParams.update(
        {
            "text.usetex": True,  # use default xelatex
            "pgf.rcfonts": False,  # turn off default matplotlib fonts properties
            "pgf.preamble": [
                r"\usepackage{amssymb}",
                r"\usepackage{fontspec}",
                # r"\setmainfont{Times New Roman}",# EN fonts Romans
                r"\usepackage{xeCJK}",  # import xeCJK
                # r"\setCJKmainfont[Path = fonts/]{JDLangZhengTi.TTF}",# set CJK fonts as SimSun
                r"\xeCJKsetup{CJKecglue=}",  # turn off one space between CJK and EN fonts
                r"\usepackage{relsize}",
            ],
        }
    )
    if "Times New Roman" in _font_names:
        # times.ttf
        # setting "Times New Roman" would cause LatexError
        plt.rcParams["font.family"] = "times"
    else:
        plt.rcParams["font.family"] = "JDLangZhengTi"

    scatter_values = list_sum(
        [
            [val["seizure_risk_difference"]["risk_difference"] for val in v.values()]
            + [np.nan, np.nan]
            for k, v in seizure_risk_dict.items()
        ]
    )
    confints = list_sum(
        [
            [
                val["seizure_risk_difference"]["confidence_interval"]
                for val in v.values()
            ]
            + [(0, 0), (0, 0)]
            for k, v in seizure_risk_dict.items()
        ]
    )
    group_sep = r"$" + r"\cdots" * 11 + r"$"
    names = list_sum(
        [
            [
                group_sep,
                r"\smaller[1]\textbf{" + k.replace("_", r"\_").replace(" ", "~") + r"}",
            ]
            + [
                r"\smaller[2]\textbf{"
                + key.replace("~", r"$\mathbf{\sim}$")
                .replace("<=", r"$\mathbf{\leqslant}$")
                .replace(">=", r"$\mathbf{\geqslant}$ ")
                .replace(">", r"$\mathbf{>}$")
                .replace("<", r"$\mathbf{<}$ ")
                + r"}"
                for key in v.keys()
            ]
            for k, v in seizure_risk_dict.items()
        ]
    )
    x_lower = 0.1 * np.floor(10 * np.min([itv[0] for itv in confints]))
    x_upper = 0.1 * np.ceil(10 * np.max([itv[1] for itv in confints]))

    fig, ax = plt.subplots(
        figsize=((x_upper - x_lower + 0.2) * 8, 0.5 * len(scatter_values))
    )
    # plot the risk as diamonds
    ax.scatter(
        scatter_values[::-1],
        range(len(scatter_values)),
        marker="D",
        color="green",
        s=60,
    )
    ax.set_ylim(1, len(confints) + 1)
    for i, (start, end) in enumerate(confints[::-1]):
        if start == end:
            continue
        # plot the confidence interval as lines
        ax.plot([start, end], [i, i], color="black", lw=1.2)
        # plot the two ends of the confidence interval
        ax.plot(start - 0.003, i, marker="|", color="black", markersize=10, lw=0.8)
        ax.plot(end + 0.003, i, marker="|", color="black", markersize=10, lw=0.8)
    ax.set_xlim(x_lower - 0.13, x_upper + 0.13)
    x_step = 0.1
    x_ticks = np.arange(x_lower - 0.1, x_upper + 0.1 + x_step, x_step)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [f"$\\mathbf{{{num}}}$" for num in np.round(100 * x_ticks).astype(int)],
        fontsize=16,
    )
    for x in x_ticks:
        # plot the vertical dashed lines at the x ticks
        if abs(x) < 1e-3:  # x approx 0
            continue
        ax.axvline(x, ls=":", color="gray", lw=0.7, dash_capstyle="round")
    # plot the red dashed vertical line at x=0
    ax.axvline(0, ls=":", color="red", dash_capstyle="round")
    ax.set_yticks(range(2, len(names) + 1))
    ax.set_yticklabels(names[::-1][:-1])
    ax.tick_params(axis="both", which="both", length=0)
    for y in np.where(np.array(names[::-1]) == group_sep)[0][:-1]:
        # plot the horizontal dashed lines at the group separators
        # ax.axhline(y+2, xmin=-0.4, color="black")
        ax.plot(
            [x_lower - 1.5, x_upper + 0.1],
            [y + 2 - 0.051, y + 2 - 0.051],
            color="black",
            ls=(0, (0.1, 4.3)),
            dash_capstyle="round",
            lw=1.6,
        )

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    ax.set_xlabel(r"\textbf{Seizure Risk Difference (95\% CI)}", fontsize=22)

    return fig, ax


def plot_feature_selection_results(
    sel_res: Union[Dict[str, Any], str, Path],
    method: str = "rfe",
    method_map: Optional[Dict[str, str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot feature selection results.

    Parameters
    ----------
    sel_res: dict or str or Path,
        The dictionary of feature selection results,
        or the path to the file that stores the feature selection results.
    method: str, default "rfe",
        The method used for feature selection to plot.
    method_map: dict, default None,
        The mapping from the method name to the name to be displayed in the plot.
        If not provided, the default mapping will be used.

    Returns
    -------
    fig: matplotlib.figure.Figure,
        The figure object.
    ax: matplotlib.axes.Axes,
        The axes object.

    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.set()
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["legend.title_fontsize"] = 22
    if "Times New Roman" in _font_names:
        # times.ttf
        if mpl.get_backend() == "pgf":
            # setting "Times New Roman" would cause LatexError
            plt.rcParams["font.family"] = "times"
        else:
            plt.rcParams["font.family"] = "Times New Roman"
    else:
        plt.rcParams["font.family"] = "JDLangZhengTi"

    if isinstance(sel_res, (str, Path)):
        print(f"Loading feature selection results from {sel_res}")
        sel_res = FeatureSelector.load_selections(path=sel_res)
    valid_methods = ["rfe", "sequential", "sfm"]
    _method_map = method_map or {"rfe": "RFE", "sfm": "SFM", "sequential": "SFS"}
    assert (
        method in sel_res
    ), f"`method` should be one of {valid_methods}, but got \042{method}\042"
    return plot_roc_curve(
        [item["model"] for item in sel_res[method]],
        [item["X_test"] for item in sel_res[method]],
        [item["y_test"] for item in sel_res[method]],
        estimator_names=[
            f"{_method_map[method]}-{item['model'].__class__.__name__}-{item['X_test'].shape[1]}"
            for item in sel_res[method]
        ],
    )


def plot_shap_summary(
    shap_values: List[np.ndarray],
    X_test: np.ndarray,
    feature_list: Optional[List[str]] = None,
    class_idx: Optional[int] = 1,
    zh2en: bool = True,
    max_display: int = 10,
    **kwargs,
) -> Dict[str, plt.Figure]:
    """
    Plot SHAP summary plots.

    Parameters
    ----------
    shap_values: list of np.ndarray (of length 2 for binary classification),
        List of shap values for the model.
        can be obtained via, for example,

        ```python
        model = RandomForestClassifier()
        # explainer = shap.TreeExplainer(model)
        explainer = shap.KernelExplainer(model.predict_proba, X_train)
        shap_values = explainer.shap_values(X_test)
        ```
    X_test: np.ndarray,
        Test data (features) used for `shap.summary_plot`.
    feature_list: list of str, optional,
        List of feature names.
    class_idx: int, default 1,
        The index of the class to plot the summary plot for.
        None for multi(2)-class output.
    zh2en: bool, default True,
        If True, convert the feature names from Chinese to English.
    max_display: int, default 10,
        The maximum number of features to display in the summary plot.
    kwargs: dict,
        Other keyword arguments for `shap.summary_plot`.

    Returns
    -------
    figs: dict of plt.Figure,
        Dictionary of figures, including
        - `dot`: beeswarm plot of shap values of dot type
        - `violin`: beeswarm plot of shap values of violin type
        - `bar`: bar plot of shap values

    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.set()
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["legend.title_fontsize"] = 22
    if "Times New Roman" in _font_names and zh2en:
        # times.ttf
        if mpl.get_backend() == "pgf":
            # setting "Times New Roman" would cause LatexError
            plt.rcParams["font.family"] = "times"
        else:
            plt.rcParams["font.family"] = "Times New Roman"
    else:
        plt.rcParams["font.family"] = "JDLangZhengTi"

    # remove prefixes of "合并症_"
    feature_list = [f.replace("合并症_", "") for f in feature_list]

    row_height = 0.4
    fig_height = min(max_display, len(feature_list)) * row_height + 1.5

    if class_idx is None:
        shap_values_ = shap_values
    else:
        assert class_idx in [0, 1], f"`class_idx` should be 0 or 1, but got {class_idx}"
        shap_values_ = shap_values[class_idx]

    fig_beeswarm, _ = plt.subplots()
    fig_beeswarm = plt.gcf()
    if zh2en:
        zh2en_mapping = {
            k: separate_by_capital_letters(v, capitalize=True, drop=["cate"])
            for k, v in DataPreprocessConfig.zh2en_mapping.items()
        }
        zh2en_mapping.update(
            {
                "病理分级": "WHO grade",
            }
        )
        feature_list = [
            zh2en_mapping[item].replace("_ ", "-").replace("(cm)", "")
            if not item.startswith("BIO_")
            else item.replace("BIO_", "")
            for item in feature_list
        ]
    shap.summary_plot(
        shap_values_,
        X_test,
        feature_names=feature_list,
        show=False,
        max_display=min(max_display, len(feature_list)),
        plot_size=(16, fig_height),
        **kwargs,
    )
    fig_violin, _ = plt.subplots()
    fig_violin = plt.gcf()
    shap.summary_plot(
        shap_values_,
        X_test,
        feature_names=feature_list,
        show=False,
        plot_type="violin",
        max_display=min(max_display, len(feature_list)),
        plot_size=(8, fig_height),  # default width is 8
        **kwargs,
    )
    fig_bar, _ = plt.subplots()
    fig_bar = plt.gcf()
    shap.summary_plot(
        shap_values_,
        X_test,
        feature_names=feature_list,
        show=False,
        plot_type="bar",
        max_display=min(max_display, len(feature_list)),
        plot_size=(8, fig_height),  # default width is 8
        **kwargs,
    )
    figs = {"dot": fig_beeswarm, "violin": fig_violin, "bar": fig_bar}
    return figs
