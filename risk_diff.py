""" """

import pickle
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from diff_binom_confint import compute_confidence_interval, compute_difference_confidence_interval

from config import DEFAULTS, DataPreprocessConfig
from data_processing import get_seizure_risk, get_seizure_risk_difference, get_training_data
from utils import separate_by_capital_letters

__all__ = ["gen_seizure_risk_diff_TDSB_ext"]


_CONFINT_METHOD = "wilson"
_BIO_na_fillvalue = "NA"


def gen_seizure_risk_diff_TDSB_ext(
    save_path: Optional[Union[Path, str]] = None,
    return_type: str = "pd",
    zh2en: bool = True,
    comorbidity_type: int = 0,
    biomarker_type: int = 0,
    overwrite: bool = False,
) -> Union[pd.DataFrame, str, dict]:
    """
    Generate the seizure risk difference table for the feature set of extended TDSB.

    Parameters
    ----------
    save_path: Path or str, optional,
        The path to save the table.
    return_type: str, default "pd",
        The type of the returned table.
        Can be one of "pd", "latex", "md", "markdown", "html", "dict".
    zh2en: bool, default True,
        Whether to convert the column names from Chinese to English.
    comorbidity_type: int, default 0,
        the manifistation of type of the comorbidity variables.
        0 for comparison of "Yes" and "No" for each comorbidity variable.
        1 for comparisons for the positive part of each comorbidity variable.
    biomarker_type: int, default 0,
        the manifistation of type of the biomarker variables.
        0 for merging the classes other than "-" into one class.
        1 for keeping the classes other than "-" as they are.
    overwrite: bool, default False,
        Whether to overwrite the existing saved files.

    Returns
    -------
    pd.DataFrame or str or dict,
        The seizure risk difference table.

    """
    zh2en_mapping = {
        k: separate_by_capital_letters(v, capitalize=True, drop=["cate"]) for k, v in DataPreprocessConfig.zh2en_mapping.items()
    }
    zh2en_mapping.update(
        {
            "病理分级": "WHO grade",
            "WHOI级": "WHO grade I",
            "WHOII级": "WHO grade II",
            "WHOIII级": "WHO grade III",
            "WHOIV级": "WHO grade IV",
        }
    )
    Ref_indicator = " (Ref.)"

    df_train, df_test, _ = get_training_data(
        feature_config=dict(over_sampler=None, BIO_na_strategy="keep"),
        feature_set="TDSB_ext",
        return_dtype="pd",
        allow_missing=True,
    )
    df_data = pd.concat((df_train, df_test))

    rows = []
    ret_dict = {}

    # row 1 - 2
    rows.extend(
        [
            [
                "Feature",
                "",
                "Affected",
                "",
                "",
                "Seizure Risk (95% CI)",
                "",
                "Seizure Risk Difference (95% CI)",
            ],
            ["", "", "n", "%", "t/v", "n", "%", ""],
        ]
    )

    # row 3: overall statitics
    n_positive = df_data[df_data[DataPreprocessConfig.y_col] == 1].shape[0]
    rows.append(
        [
            "Total",
            "",
            f"{len(df_data)}",
            "100%",
            f"{len(df_train)}/{len(df_test)}",
            f"{n_positive}",
            f"{n_positive / len(df_data):.1%}",
            "-",
        ]
    )

    # age
    age_scaler = pickle.loads(DataPreprocessConfig.age_scaler_path.read_bytes())
    df_data.年龄 = age_scaler.inverse_transform(df_data.年龄.values.reshape(-1, 1)).flatten()
    df_train.年龄 = age_scaler.inverse_transform(df_train.年龄.values.reshape(-1, 1)).flatten()
    df_test.年龄 = age_scaler.inverse_transform(df_test.年龄.values.reshape(-1, 1)).flatten()

    def age_group_mapping(age):
        if age <= 20:
            return "<= 20"
        elif age <= 30:
            return "20 ~ 30"
        elif age <= 40:
            return "30 ~ 40"
        elif age <= 50:
            return "40 ~ 50"
        elif age <= 60:
            return "50 ~ 60"
        elif age <= 70:
            return "60 ~ 70"
        else:
            return "> 70"

    df_data["年龄段"] = df_data.年龄.apply(age_group_mapping)
    df_train["年龄段"] = df_train.年龄.apply(age_group_mapping)
    df_test["年龄段"] = df_test.年龄.apply(age_group_mapping)
    ref_group = "<= 20"
    n_affected = {
        age_group: {
            "total": df_data[df_data.年龄段 == age_group].shape[0],
            "train": df_train[df_train.年龄段 == age_group].shape[0],
            "test": df_test[df_test.年龄段 == age_group].shape[0],
        }
        for age_group in df_data.年龄段.unique()
    }
    n_positive = {
        age_group: df_data[df_data.年龄段 == age_group][DataPreprocessConfig.y_col].sum()
        for age_group in df_data.年龄段.unique()
    }
    seizure_risk = get_seizure_risk(df_data=df_data, col="年龄段", positive_class=1, negative_class=0)
    seizure_risk_diff = get_seizure_risk_difference(
        df_data=df_data,
        col="年龄段",
        ref_class=ref_group,
        positive_class=1,
        negative_class=0,
    )
    rows.append(["Age", "", "", "", "", "", "", ""])
    ret_dict["Age"] = {}
    for age_group in [
        "<= 20",
        "20 ~ 30",
        "30 ~ 40",
        "40 ~ 50",
        "50 ~ 60",
        "60 ~ 70",
        "> 70",
    ]:
        rows.append(
            [
                "",
                age_group,
                f"{n_affected[age_group]['total']}",
                f"{n_affected[age_group]['total'] / len(df_data):.1%}",
                f"{n_affected[age_group]['train']}/{n_affected[age_group]['test']}",
                f"{n_positive[age_group]}",
                f"{seizure_risk[age_group]['risk']:.1%} (from {seizure_risk[age_group]['confidence_interval'][0]:.1%} to {seizure_risk[age_group]['confidence_interval'][1]:.1%})",
                (
                    f"{seizure_risk_diff[age_group]['risk_difference']:.1%} (from {seizure_risk_diff[age_group]['confidence_interval'][0]:.1%} to {seizure_risk_diff[age_group]['confidence_interval'][1]:.1%})"
                    if age_group != ref_group
                    else "REF"
                ),
            ]
        )
        ret_dict["Age"][age_group + (Ref_indicator if age_group == ref_group else "")] = {
            "Affected": {
                "n": n_affected[age_group]["total"],
                "percent": n_affected[age_group]["total"] / len(df_data),
                "t/v": f"{n_affected[age_group]['train']}/{n_affected[age_group]['test']}",
            },
            "seiuzre_risk": {
                "n": n_positive[age_group],
                "percent": seizure_risk[age_group]["risk"],
                "confidence_interval": seizure_risk[age_group]["confidence_interval"],
            },
            "seizure_risk_difference": {
                "risk_difference": seizure_risk_diff[age_group]["risk_difference"] if age_group != ref_group else 0,
                "confidence_interval": (
                    seizure_risk_diff[age_group]["confidence_interval"] if age_group != ref_group else (0, 0)
                ),
            },
        }

    # gender
    ref_group = "男"
    inv_dict = {v: k for k, v in DataPreprocessConfig.x_col_mappings.性别.items()}
    df_data.loc[:, "性别"] = df_data.性别.map(inv_dict)
    df_train.loc[:, "性别"] = df_train.性别.map(inv_dict)
    df_test.loc[:, "性别"] = df_test.性别.map(inv_dict)
    n_affected = {
        gender: {
            "total": df_data[df_data.性别 == gender].shape[0],
            "train": df_train[df_train.性别 == gender].shape[0],
            "test": df_test[df_test.性别 == gender].shape[0],
        }
        for gender in df_data.性别.unique()
    }
    n_positive = {gender: df_data[df_data.性别 == gender][DataPreprocessConfig.y_col].sum() for gender in df_data.性别.unique()}
    seizure_risk = get_seizure_risk(df_data=df_data, col="性别", positive_class=1, negative_class=0)
    seizure_risk_diff = get_seizure_risk_difference(
        df_data=df_data,
        col="性别",
        ref_class=ref_group,
        positive_class=1,
        negative_class=0,
    )
    rows.append(["Gender", "", "", "", "", "", "", ""])
    ret_dict["Gender"] = {}
    for gender in ["男", "女"]:
        rows.append(
            [
                "",
                gender,
                f"{n_affected[gender]['total']}",
                f"{n_affected[gender]['total'] / len(df_data):.1%}",
                f"{n_affected[gender]['train']}/{n_affected[gender]['test']}",
                f"{n_positive[gender]}",
                f"{seizure_risk[gender]['risk']:.1%} (from {seizure_risk[gender]['confidence_interval'][0]:.1%} to {seizure_risk[gender]['confidence_interval'][1]:.1%})",
                (
                    f"{seizure_risk_diff[gender]['risk_difference']:.1%} (from {seizure_risk_diff[gender]['confidence_interval'][0]:.1%} to {seizure_risk_diff[gender]['confidence_interval'][1]:.1%})"
                    if gender != ref_group
                    else "REF"
                ),
            ]
        )
        ret_dict["Gender"][gender + (Ref_indicator if gender == ref_group else "")] = {
            "Affected": {
                "n": n_affected[gender]["total"],
                "percent": n_affected[gender]["total"] / len(df_data),
                "t/v": f"{n_affected[gender]['train']}/{n_affected[gender]['test']}",
            },
            "seiuzre_risk": {
                "n": n_positive[gender],
                "percent": seizure_risk[gender]["risk"],
                "confidence_interval": seizure_risk[gender]["confidence_interval"],
            },
            "seizure_risk_difference": {
                "risk_difference": seizure_risk_diff[gender]["risk_difference"] if gender != ref_group else 0,
                "confidence_interval": seizure_risk_diff[gender]["confidence_interval"] if gender != ref_group else (0, 0),
            },
        }

    # tumor zone
    tumor_zones = [
        "枕",
        "额",
        "颞",
        "顶",
        "小脑",
        "脑室",
        "岛",
        "丘脑",
        "胼胝体",
        "基底节",
        "鞍区",
        "脑干",
        "其它",
    ]
    prefix = "肿瘤分区_"
    ref_group = "枕"  # "额"
    n_affected = {
        tumor_zone: {
            "total": df_data[df_data[prefix + tumor_zone] == 1].shape[0],
            "train": df_train[df_train[prefix + tumor_zone] == 1].shape[0],
            "test": df_test[df_test[prefix + tumor_zone] == 1].shape[0],
        }
        for tumor_zone in tumor_zones
    }
    n_positive = {
        tumor_zone: df_data[df_data[prefix + tumor_zone] == 1][DataPreprocessConfig.y_col].sum() for tumor_zone in tumor_zones
    }
    rows.append(["肿瘤分区", "", "", "", "", "", "", ""])
    ret_dict["肿瘤分区"] = {}
    for tumor_zone in tumor_zones:
        seizure_risk_confint = compute_confidence_interval(
            n_positive[tumor_zone],
            n_affected[tumor_zone]["total"],
            method=_CONFINT_METHOD,
        ).astuple()
        seizure_risk_diff_confint = compute_difference_confidence_interval(
            n_positive[tumor_zone],
            n_affected[tumor_zone]["total"],
            n_positive[ref_group],
            n_affected[ref_group]["total"],
            method=_CONFINT_METHOD,
        ).astuple()
        rows.append(
            [
                "",
                tumor_zone,
                f"{n_affected[tumor_zone]['total']}",
                f"{n_affected[tumor_zone]['total'] / len(df_data):.1%}",
                f"{n_affected[tumor_zone]['train']}/{n_affected[tumor_zone]['test']}",
                f"{n_positive[tumor_zone]}",
                f"{n_positive[tumor_zone] / n_affected[tumor_zone]['total']:.1%} (from {seizure_risk_confint[0]:.1%} to {seizure_risk_confint[1]:.1%})",
                (
                    f"{n_positive[tumor_zone] / n_affected[tumor_zone]['total'] - n_positive[ref_group] / n_affected[ref_group]['total']:.1%} (from {seizure_risk_diff_confint[0]:.1%} to {seizure_risk_diff_confint[1]:.1%})"
                    if tumor_zone != ref_group
                    else "REF"
                ),
            ]
        )
        ret_dict["肿瘤分区"][tumor_zone + (Ref_indicator if tumor_zone == ref_group else "")] = {
            "Affected": {
                "n": n_affected[tumor_zone]["total"],
                "percent": n_affected[tumor_zone]["total"] / len(df_data),
                "t/v": f"{n_affected[tumor_zone]['train']}/{n_affected[tumor_zone]['test']}",
            },
            "seiuzre_risk": {
                "n": n_positive[tumor_zone],
                "percent": n_positive[tumor_zone] / n_affected[tumor_zone]["total"],
                "confidence_interval": seizure_risk_confint,
            },
            "seizure_risk_difference": {
                "risk_difference": (
                    n_positive[tumor_zone] / n_affected[tumor_zone]["total"]
                    - n_positive[ref_group] / n_affected[ref_group]["total"]
                    if tumor_zone != ref_group
                    else 0
                ),
                "confidence_interval": seizure_risk_diff_confint if tumor_zone != ref_group else (0, 0),
            },
        }

    # WHO grade
    ref_group = "WHOI级"
    inv_dict = {v: k for k, v in DataPreprocessConfig.x_col_mappings.病理分级.items()}
    df_data.loc[:, "病理分级"] = df_data.病理分级.map(inv_dict)
    df_train.loc[:, "病理分级"] = df_train.病理分级.map(inv_dict)
    df_test.loc[:, "病理分级"] = df_test.病理分级.map(inv_dict)
    n_affected = {
        who_grade: {
            "total": df_data[df_data.病理分级 == who_grade].shape[0],
            "train": df_train[df_train.病理分级 == who_grade].shape[0],
            "test": df_test[df_test.病理分级 == who_grade].shape[0],
        }
        for who_grade in df_data.病理分级.unique()
    }
    n_positive = {
        who_grade: df_data[df_data.病理分级 == who_grade][DataPreprocessConfig.y_col].sum()
        for who_grade in df_data.病理分级.unique()
    }
    seizure_risk = get_seizure_risk(df_data=df_data, col="病理分级", positive_class=1, negative_class=0)
    seizure_risk_diff = get_seizure_risk_difference(
        df_data=df_data,
        col="病理分级",
        ref_class=ref_group,
        positive_class=1,
        negative_class=0,
    )
    rows.append(["病理分级", "", "", "", "", "", "", ""])
    ret_dict["病理分级"] = {}
    for who_grade in ["WHOI级", "WHOII级", "WHOIII级", "WHOIV级"]:
        rows.append(
            [
                "",
                who_grade,
                f"{n_affected[who_grade]['total']}",
                f"{n_affected[who_grade]['total'] / len(df_data):.1%}",
                f"{n_affected[who_grade]['train']}/{n_affected[who_grade]['test']}",
                f"{n_positive[who_grade]}",
                f"{seizure_risk[who_grade]['risk']:.1%} (from {seizure_risk[who_grade]['confidence_interval'][0]:.1%} to {seizure_risk[who_grade]['confidence_interval'][1]:.1%})",
                (
                    f"{seizure_risk_diff[who_grade]['risk_difference']:.1%} (from {seizure_risk_diff[who_grade]['confidence_interval'][0]:.1%} to {seizure_risk_diff[who_grade]['confidence_interval'][1]:.1%})"
                    if who_grade != ref_group
                    else "REF"
                ),
            ]
        )
        ret_dict["病理分级"][str(who_grade) + (Ref_indicator if who_grade == ref_group else "")] = {
            "Affected": {
                "n": n_affected[who_grade]["total"],
                "percent": n_affected[who_grade]["total"] / len(df_data),
                "t/v": f"{n_affected[who_grade]['train']}/{n_affected[who_grade]['test']}",
            },
            "seiuzre_risk": {
                "n": n_positive[who_grade],
                "percent": n_positive[who_grade] / n_affected[who_grade]["total"],
                "confidence_interval": seizure_risk[who_grade]["confidence_interval"],
            },
            "seizure_risk_difference": {
                "risk_difference": seizure_risk_diff[who_grade]["risk_difference"] if who_grade != ref_group else 0,
                "confidence_interval": (
                    seizure_risk_diff[who_grade]["confidence_interval"] if who_grade != ref_group else (0, 0)
                ),
            },
        }

    # pathological classification
    ref_group = "分型胶质母"
    inv_dict = {v: k for k, v in DataPreprocessConfig.x_col_mappings.病理分型粗.items()}
    df_data.loc[:, "病理分型粗"] = df_data.病理分型粗.map(inv_dict)
    df_train.loc[:, "病理分型粗"] = df_train.病理分型粗.map(inv_dict)
    df_test.loc[:, "病理分型粗"] = df_test.病理分型粗.map(inv_dict)
    n_affected = {
        patho_class: {
            "total": df_data[df_data.病理分型粗 == patho_class].shape[0],
            "train": df_train[df_train.病理分型粗 == patho_class].shape[0],
            "test": df_test[df_test.病理分型粗 == patho_class].shape[0],
        }
        for patho_class in df_data.病理分型粗.unique()
    }
    n_positive = {
        patho_class: df_data[df_data.病理分型粗 == patho_class][DataPreprocessConfig.y_col].sum()
        for patho_class in df_data.病理分型粗.unique()
    }
    seizure_risk = get_seizure_risk(df_data=df_data, col="病理分型粗", positive_class=1, negative_class=0)
    seizure_risk_diff = get_seizure_risk_difference(
        df_data=df_data,
        col="病理分型粗",
        ref_class=ref_group,
        positive_class=1,
        negative_class=0,
    )
    rows.append(["病理分型粗", "", "", "", "", "", "", ""])
    ret_dict["病理分型粗"] = {}
    pathological_classes = [
        "分型胶质母",
        "分型间变型星形",
        "分型少突星形",
        "分型星形",
        "分型毛细胞星形",
        "分型中枢神经",
        "分型混合",
        "分型其它",
    ]
    for patho_class in pathological_classes:
        rows.append(
            [
                "",
                patho_class,
                f"{n_affected[patho_class]['total']}",
                f"{n_affected[patho_class]['total'] / len(df_data):.1%}",
                f"{n_affected[patho_class]['train']}/{n_affected[patho_class]['test']}",
                f"{n_positive[patho_class]}",
                f"{seizure_risk[patho_class]['risk']:.1%} (from {seizure_risk[patho_class]['confidence_interval'][0]:.1%} to {seizure_risk[patho_class]['confidence_interval'][1]:.1%})",
                (
                    f"{seizure_risk_diff[patho_class]['risk_difference']:.1%} (from {seizure_risk_diff[patho_class]['confidence_interval'][0]:.1%} to {seizure_risk_diff[patho_class]['confidence_interval'][1]:.1%})"
                    if patho_class != ref_group
                    else "REF"
                ),
            ]
        )
        ret_dict["病理分型粗"][patho_class + (Ref_indicator if patho_class == ref_group else "")] = {
            "Affected": {
                "n": n_affected[patho_class]["total"],
                "percent": n_affected[patho_class]["total"] / len(df_data),
                "t/v": f"{n_affected[patho_class]['train']}/{n_affected[patho_class]['test']}",
            },
            "seiuzre_risk": {
                "n": n_positive[patho_class],
                "percent": n_positive[patho_class] / n_affected[patho_class]["total"],
                "confidence_interval": seizure_risk[patho_class]["confidence_interval"],
            },
            "seizure_risk_difference": {
                "risk_difference": seizure_risk_diff[patho_class]["risk_difference"] if patho_class != ref_group else 0,
                "confidence_interval": (
                    seizure_risk_diff[patho_class]["confidence_interval"] if patho_class != ref_group else (0, 0)
                ),
            },
        }

    # comorbidity
    comorbidities = [
        "低钠血症",
        "低蛋白血症",
        "低钾血症",
        "高氯血症",
        "低氯血症",
        "中枢神经感染",
        "凝血功能异常",
    ]
    prefix = "合并症_"
    if comorbidity_type == 1:
        ref_group = "低钠血症"
        n_affected = {
            comorbidity: {
                "total": df_data[df_data[prefix + comorbidity] == 1].shape[0],
                "train": df_train[df_train[prefix + comorbidity] == 1].shape[0],
                "test": df_test[df_test[prefix + comorbidity] == 1].shape[0],
            }
            for comorbidity in comorbidities
        }
        n_positive = {
            comorbidity: df_data[df_data[prefix + comorbidity] == 1][DataPreprocessConfig.y_col].sum()
            for comorbidity in comorbidities
        }
        rows.append(["合并症", "", "", "", "", "", "", ""])
        ret_dict["合并症"] = {}
        for comorbidity in comorbidities:
            seizure_risk_confint = compute_confidence_interval(
                n_positive[comorbidity],
                n_affected[comorbidity]["total"],
                method=_CONFINT_METHOD,
            ).astuple()
            seizure_risk_diff_confint = compute_difference_confidence_interval(
                n_positive[comorbidity],
                n_affected[comorbidity]["total"],
                n_positive[ref_group],
                n_affected[ref_group]["total"],
                method=_CONFINT_METHOD,
            ).astuple()
            rows.append(
                [
                    "",
                    comorbidity,
                    f"{n_affected[comorbidity]['total']}",
                    f"{n_affected[comorbidity]['total'] / len(df_data):.1%}",
                    f"{n_affected[comorbidity]['train']}/{n_affected[comorbidity]['test']}",
                    f"{n_positive[comorbidity]}",
                    f"{n_positive[comorbidity] / n_affected[comorbidity]['total']:.1%} (from {seizure_risk_confint[0]:.1%} to {seizure_risk_confint[1]:.1%})",
                    (
                        f"{n_positive[comorbidity] / n_affected[comorbidity]['total'] - n_positive[ref_group] / n_affected[ref_group]['total']:.1%} (from {seizure_risk_diff_confint[0]:.1%} to {seizure_risk_diff_confint[1]:.1%})"
                        if comorbidity != ref_group
                        else "REF"
                    ),
                ]
            )
            ret_dict["合并症"][comorbidity + (Ref_indicator if comorbidity == ref_group else "")] = {
                "Affected": {
                    "n": n_affected[comorbidity]["total"],
                    "percent": n_affected[comorbidity]["total"] / len(df_data),
                    "t/v": f"{n_affected[comorbidity]['train']}/{n_affected[comorbidity]['test']}",
                },
                "seiuzre_risk": {
                    "n": n_positive[comorbidity],
                    "percent": n_positive[comorbidity] / n_affected[comorbidity]["total"],
                    "confidence_interval": seizure_risk_confint,
                },
                "seizure_risk_difference": {
                    "risk_difference": (
                        n_positive[comorbidity] / n_affected[comorbidity]["total"]
                        - n_positive[ref_group] / n_affected[ref_group]["total"]
                        if comorbidity != ref_group
                        else 0
                    ),
                    "confidence_interval": seizure_risk_diff_confint if comorbidity != ref_group else (0, 0),
                },
            }
    elif comorbidity_type == 0:
        ref_group = "Yes"
        for comorbidity in comorbidities:
            n_affected = {
                "Yes": {
                    "train": df_train[df_train[prefix + comorbidity] == 1].shape[0],
                    "test": df_test[df_test[prefix + comorbidity] == 1].shape[0],
                    "total": df_data[df_data[prefix + comorbidity] == 1].shape[0],
                },
                "No": {
                    "train": df_train[df_train[prefix + comorbidity] == 0].shape[0],
                    "test": df_test[df_test[prefix + comorbidity] == 0].shape[0],
                    "total": df_data[df_data[prefix + comorbidity] == 0].shape[0],
                },
            }
            n_positive = {
                "Yes": df_data[df_data[prefix + comorbidity] == 1][DataPreprocessConfig.y_col].sum(),
                "No": df_data[df_data[prefix + comorbidity] == 0][DataPreprocessConfig.y_col].sum(),
            }
            rows.append([comorbidity, "", "", "", "", "", "", ""])
            ret_dict[comorbidity] = {}
            for group in ["Yes", "No"]:
                seizure_risk_confint = compute_confidence_interval(
                    n_positive[group],
                    n_affected[group]["total"],
                    method=_CONFINT_METHOD,
                ).astuple()
                seizure_risk_diff_confint = compute_difference_confidence_interval(
                    n_positive[group],
                    n_affected[group]["total"],
                    n_positive[ref_group],
                    n_affected[ref_group]["total"],
                    method=_CONFINT_METHOD,
                ).astuple()
                rows.append(
                    [
                        "",
                        group,
                        f"{n_affected[group]['total']}",
                        f"{n_affected[group]['total'] / len(df_data):.1%}",
                        f"{n_affected[group]['train']}/{n_affected[group]['test']}",
                        f"{n_positive[group]}",
                        f"{n_positive[group] / n_affected[group]['total']:.1%} (from {seizure_risk_confint[0]:.1%} to {seizure_risk_confint[1]:.1%})",
                        (
                            f"{n_positive[group] / n_affected[group]['total'] - n_positive[ref_group] / n_affected[ref_group]['total']:.1%} (from {seizure_risk_diff_confint[0]:.1%} to {seizure_risk_diff_confint[1]:.1%})"
                            if group != ref_group
                            else "REF"
                        ),
                    ]
                )
                ret_dict[comorbidity][group + (Ref_indicator if group == ref_group else "")] = {
                    "Affected": {
                        "n": n_affected[group]["total"],
                        "percent": n_affected[group]["total"] / len(df_data),
                        "t/v": f"{n_affected[group]['train']}/{n_affected[group]['test']}",
                    },
                    "seiuzre_risk": {
                        "n": n_positive[group],
                        "percent": n_positive[group] / n_affected[group]["total"],
                        "confidence_interval": seizure_risk_confint,
                    },
                    "seizure_risk_difference": {
                        "risk_difference": (
                            n_positive[group] / n_affected[group]["total"]
                            - n_positive[ref_group] / n_affected[ref_group]["total"]
                            if group != ref_group
                            else 0
                        ),
                        "confidence_interval": seizure_risk_diff_confint if group != ref_group else (0, 0),
                    },
                }
    else:
        raise ValueError("comorbidity_type should be 0 or 1")

    # surgical resection method
    ref_group = "全切"  # "部分切除"
    inv_dict = {v: k for k, v in DataPreprocessConfig.x_col_mappings.手术切除方式.items()}

    def _simplify_surgical(x: str) -> str:
        return x if x == "全切" else "其它切除方式"

    df_data.loc[:, "手术切除方式"] = df_data.手术切除方式.map(inv_dict).apply(_simplify_surgical)
    df_train.loc[:, "手术切除方式"] = df_train.手术切除方式.map(inv_dict).apply(_simplify_surgical)
    df_test.loc[:, "手术切除方式"] = df_test.手术切除方式.map(inv_dict).apply(_simplify_surgical)
    n_affected = {
        resection_method: {
            "total": df_data[df_data.手术切除方式 == resection_method].shape[0],
            "train": df_train[df_train.手术切除方式 == resection_method].shape[0],
            "test": df_test[df_test.手术切除方式 == resection_method].shape[0],
        }
        for resection_method in df_data.手术切除方式.unique()
    }
    n_positive = {
        resection_method: df_data[df_data.手术切除方式 == resection_method][DataPreprocessConfig.y_col].sum()
        for resection_method in df_data.手术切除方式.unique()
    }
    seizure_risk = get_seizure_risk(df_data=df_data, col="手术切除方式", positive_class=1, negative_class=0)
    seizure_risk_diff = get_seizure_risk_difference(
        df_data=df_data,
        col="手术切除方式",
        ref_class=ref_group,
        positive_class=1,
        negative_class=0,
    )
    rows.append(["手术切除方式", "", "", "", "", "", "", ""])
    ret_dict["手术切除方式"] = {}
    # for resection_method in ["部分切除", "大部切除", "次全切", "近全切", "全切"]:
    for resection_method in ["全切", "其它切除方式"]:
        rows.append(
            [
                "",
                resection_method,
                f"{n_affected[resection_method]['total']}",
                f"{n_affected[resection_method]['total'] / len(df_data):.1%}",
                f"{n_affected[resection_method]['train']}/{n_affected[resection_method]['test']}",
                f"{n_positive[resection_method]}",
                f"{seizure_risk[resection_method]['risk']:.1%} (from {seizure_risk[resection_method]['confidence_interval'][0]:.1%} to {seizure_risk[resection_method]['confidence_interval'][1]:.1%})",
                (
                    f"{seizure_risk_diff[resection_method]['risk_difference']:.1%} (from {seizure_risk_diff[resection_method]['confidence_interval'][0]:.1%} to {seizure_risk_diff[resection_method]['confidence_interval'][1]:.1%})"
                    if resection_method != ref_group
                    else "REF"
                ),
            ]
        )
        ret_dict["手术切除方式"][resection_method + (Ref_indicator if resection_method == ref_group else "")] = {
            "Affected": {
                "n": n_affected[resection_method]["total"],
                "percent": n_affected[resection_method]["total"] / len(df_data),
                "t/v": f"{n_affected[resection_method]['train']}/{n_affected[resection_method]['test']}",
            },
            "seiuzre_risk": {
                "n": n_positive[resection_method],
                "percent": seizure_risk[resection_method]["risk"],
                "confidence_interval": seizure_risk[resection_method]["confidence_interval"],
            },
            "seizure_risk_difference": {
                "risk_difference": (
                    seizure_risk_diff[resection_method]["risk_difference"] if resection_method != ref_group else 0
                ),
                "confidence_interval": (
                    seizure_risk_diff[resection_method]["confidence_interval"] if resection_method != ref_group else (0, 0)
                ),
            },
        }

    # categorical tumor size
    ref_group = "<5"
    inv_dict = {v: k for k, v in DataPreprocessConfig.x_col_mappings.C肿瘤最大直径.items()}
    df_data.loc[:, "C肿瘤最大直径"] = df_data.C肿瘤最大直径.map(inv_dict)
    df_train.loc[:, "C肿瘤最大直径"] = df_train.C肿瘤最大直径.map(inv_dict)
    df_test.loc[:, "C肿瘤最大直径"] = df_test.C肿瘤最大直径.map(inv_dict)
    n_affected = {
        tumor_size: {
            "total": df_data[df_data.C肿瘤最大直径 == tumor_size].shape[0],
            "train": df_train[df_train.C肿瘤最大直径 == tumor_size].shape[0],
            "test": df_test[df_test.C肿瘤最大直径 == tumor_size].shape[0],
        }
        for tumor_size in df_data.C肿瘤最大直径.unique()
    }
    n_positive = {
        tumor_size: df_data[df_data.C肿瘤最大直径 == tumor_size][DataPreprocessConfig.y_col].sum()
        for tumor_size in df_data.C肿瘤最大直径.unique()
    }
    seizure_risk = get_seizure_risk(df_data=df_data, col="C肿瘤最大直径", positive_class=1, negative_class=0)
    seizure_risk_diff = get_seizure_risk_difference(
        df_data=df_data,
        col="C肿瘤最大直径",
        ref_class=ref_group,
        positive_class=1,
        negative_class=0,
    )
    rows.append(["C肿瘤最大直径", "", "", "", "", "", "", ""])
    ret_dict["C肿瘤最大直径"] = {}
    for tumor_size in ["<5", ">=5"]:
        rows.append(
            [
                "",
                tumor_size,
                f"{n_affected[tumor_size]['total']}",
                f"{n_affected[tumor_size]['total'] / len(df_data):.1%}",
                f"{n_affected[tumor_size]['train']}/{n_affected[tumor_size]['test']}",
                f"{n_positive[tumor_size]}",
                f"{seizure_risk[tumor_size]['risk']:.1%} (from {seizure_risk[tumor_size]['confidence_interval'][0]:.1%} to {seizure_risk[tumor_size]['confidence_interval'][1]:.1%})",
                (
                    f"{seizure_risk_diff[tumor_size]['risk_difference']:.1%} (from {seizure_risk_diff[tumor_size]['confidence_interval'][0]:.1%} to {seizure_risk_diff[tumor_size]['confidence_interval'][1]:.1%})"
                    if tumor_size != ref_group
                    else "REF"
                ),
            ]
        )
        ret_dict["C肿瘤最大直径"][str(tumor_size) + (Ref_indicator if tumor_size == ref_group else "")] = {
            "Affected": {
                "n": n_affected[tumor_size]["total"],
                "percent": n_affected[tumor_size]["total"] / len(df_data),
                "t/v": f"{n_affected[tumor_size]['train']}/{n_affected[tumor_size]['test']}",
            },
            "seiuzre_risk": {
                "n": n_positive[tumor_size],
                "percent": seizure_risk[tumor_size]["risk"],
                "confidence_interval": seizure_risk[tumor_size]["confidence_interval"],
            },
            "seizure_risk_difference": {
                "risk_difference": seizure_risk_diff[tumor_size]["risk_difference"] if tumor_size != ref_group else 0,
                "confidence_interval": (
                    seizure_risk_diff[tumor_size]["confidence_interval"] if tumor_size != ref_group else (0, 0)
                ),
            },
        }

    def _simplify_BIO(x: str) -> str:
        return x if x in ["-", DataPreprocessConfig.BIO_na_fillvalue] else "+"

    # BIO
    ref_group = "-"
    inv_dict = {v: k for k, v in DataPreprocessConfig.BIO_mapping.items()}
    # for col in DataPreprocessConfig.BIO_cate_var:
    for col in ["BIO_IDH1-R132"]:  # categorical biomarker variables
        df_data.loc[:, col] = df_data[col].map(inv_dict)
        df_train.loc[:, col] = df_train[col].map(inv_dict)
        df_test.loc[:, col] = df_test[col].map(inv_dict)
        if biomarker_type == 0:
            df_data.loc[:, col] = df_data[col].apply(_simplify_BIO)
            df_train.loc[:, col] = df_train[col].apply(_simplify_BIO)
            df_test.loc[:, col] = df_test[col].apply(_simplify_BIO)
            bio_levels = ["-", "+", DataPreprocessConfig.BIO_na_fillvalue]
        else:
            bio_levels = [
                "-",
                "±",
                "+",
                "++",
                "+++",
                DataPreprocessConfig.BIO_na_fillvalue,
            ]
        n_affected = {
            bio: {
                "total": df_data[df_data[col] == bio].shape[0],
                "train": df_train[df_train[col] == bio].shape[0],
                "test": df_test[df_test[col] == bio].shape[0],
            }
            for bio in bio_levels
        }
        n_positive = {bio: df_data[df_data[col] == bio][DataPreprocessConfig.y_col].sum() for bio in bio_levels}
        rows.append([col.replace("BIO_", ""), "", "", "", "", "", "", ""])
        ret_dict[col.replace("BIO_", "")] = {}
        for bio in bio_levels:
            if n_affected[bio]["total"] == 0:
                continue
            seizure_risk_confint = compute_confidence_interval(
                n_positive[bio], n_affected[bio]["total"], method=_CONFINT_METHOD
            ).astuple()
            seizure_risk_diff_confint = compute_difference_confidence_interval(
                n_positive[bio],
                n_affected[bio]["total"],
                n_positive[ref_group],
                n_affected[ref_group]["total"],
                method=_CONFINT_METHOD,
            ).astuple()
            rows.append(
                [
                    "",
                    bio.replace(DataPreprocessConfig.BIO_na_fillvalue, _BIO_na_fillvalue),
                    f"{n_affected[bio]['total']}",
                    f"{n_affected[bio]['total'] / len(df_data):.1%}",
                    f"{n_affected[bio]['train']}/{n_affected[bio]['test']}",
                    f"{n_positive[bio]}",
                    f"{n_positive[bio] / n_affected[bio]['total']:.1%} (from {seizure_risk_confint[0]:.1%} to {seizure_risk_confint[1]:.1%})",
                    (
                        f"{n_positive[bio] / n_affected[bio]['total'] - n_positive[ref_group] / n_affected[ref_group]['total']:.1%} (from {seizure_risk_diff_confint[0]:.1%} to {seizure_risk_diff_confint[1]:.1%})"
                        if bio != ref_group
                        else "REF"
                    ),
                ]
            )
            ret_dict[col.replace("BIO_", "")][
                bio.replace(DataPreprocessConfig.BIO_na_fillvalue, _BIO_na_fillvalue)
                + (Ref_indicator if bio == ref_group else "")
            ] = {
                "Affected": {
                    "n": n_affected[bio]["total"],
                    "percent": n_affected[bio]["total"] / len(df_data),
                    "t/v": f"{n_affected[bio]['train']}/{n_affected[bio]['test']}",
                },
                "seiuzre_risk": {
                    "n": n_positive[bio],
                    "percent": n_positive[bio] / n_affected[bio]["total"],
                    "confidence_interval": seizure_risk_confint,
                },
                "seizure_risk_difference": {
                    "risk_difference": (
                        (n_positive[bio] / n_affected[bio]["total"] - n_positive[ref_group] / n_affected[ref_group]["total"])
                        if bio != ref_group
                        else 0
                    ),
                    "confidence_interval": seizure_risk_diff_confint if bio != ref_group else (0, 0),
                },
            }

    split_values = [10, 30, 60]
    bio_levels = (
        [f"<= {split_values[0]}%"]
        + [f"{split_values[i-1]}% - {split_values[i]}%" for i in range(1, len(split_values))]
        + [f"> {split_values[-1]}%"]
        + [_BIO_na_fillvalue]
    )
    ref_group = bio_levels[0]

    def _BIO_categorizer(x: float) -> str:
        if np.isnan(x):
            return _BIO_na_fillvalue
        if x <= split_values[0]:
            return f"<= {split_values[0]}%"
        for i in range(len(split_values) - 1):
            if split_values[i] < x <= split_values[i + 1]:
                return f"{split_values[i]}% - {split_values[i+1]}%"
        return f"> {split_values[-1]}%"

    for col in ["BIO_Ki-67"]:  # continuous biomarker variables
        df_data.loc[:, col] = df_data[col].apply(_BIO_categorizer)
        df_train.loc[:, col] = df_train[col].apply(_BIO_categorizer)
        df_test.loc[:, col] = df_test[col].apply(_BIO_categorizer)
        n_affected = {
            bio: {
                "total": df_data[df_data[col] == bio].shape[0],
                "train": df_train[df_train[col] == bio].shape[0],
                "test": df_test[df_test[col] == bio].shape[0],
            }
            for bio in bio_levels
        }
        n_positive = {bio: df_data[df_data[col] == bio][DataPreprocessConfig.y_col].sum() for bio in bio_levels}
        rows.append([col.replace("BIO_", ""), "", "", "", "", "", "", ""])
        ret_dict[col.replace("BIO_", "")] = {}
        for bio in bio_levels:
            if n_affected[bio]["total"] == 0:
                continue
            seizure_risk_confint = compute_confidence_interval(
                n_positive[bio], n_affected[bio]["total"], method=_CONFINT_METHOD
            ).astuple()
            seizure_risk_diff_confint = compute_difference_confidence_interval(
                n_positive[bio],
                n_affected[bio]["total"],
                n_positive[ref_group],
                n_affected[ref_group]["total"],
                method=_CONFINT_METHOD,
            ).astuple()
            rows.append(
                [
                    "",
                    bio,
                    f"{n_affected[bio]['total']}",
                    f"{n_affected[bio]['total'] / len(df_data):.1%}",
                    f"{n_affected[bio]['train']}/{n_affected[bio]['test']}",
                    f"{n_positive[bio]}",
                    f"{n_positive[bio] / n_affected[bio]['total']:.1%} (from {seizure_risk_confint[0]:.1%} to {seizure_risk_confint[1]:.1%})",
                    (
                        f"{n_positive[bio] / n_affected[bio]['total'] - n_positive[ref_group] / n_affected[ref_group]['total']:.1%} (from {seizure_risk_diff_confint[0]:.1%} to {seizure_risk_diff_confint[1]:.1%})"
                        if bio != ref_group
                        else "REF"
                    ),
                ]
            )
            ret_dict[col.replace("BIO_", "")][bio + (Ref_indicator if bio == ref_group else "")] = {
                "Affected": {
                    "n": n_affected[bio]["total"],
                    "percent": n_affected[bio]["total"] / len(df_data),
                    "t/v": f"{n_affected[bio]['train']}/{n_affected[bio]['test']}",
                },
                "seiuzre_risk": {
                    "n": n_positive[bio],
                    "percent": n_positive[bio] / n_affected[bio]["total"],
                    "confidence_interval": seizure_risk_confint,
                },
                "seizure_risk_difference": {
                    "risk_difference": (
                        (n_positive[bio] / n_affected[bio]["total"] - n_positive[ref_group] / n_affected[ref_group]["total"])
                        if bio != ref_group
                        else 0
                    ),
                    "confidence_interval": seizure_risk_diff_confint if bio != ref_group else (0, 0),
                },
            }

    if zh2en:
        for row_idx, row in enumerate(rows):
            for item_idx, item in enumerate(row):
                if item in zh2en_mapping:
                    rows[row_idx][item_idx] = zh2en_mapping[item].replace("Of", "of").replace("Or", "or")
        _tmp_dict = deepcopy(ret_dict)
        ret_dict = {}
        for k, v in _tmp_dict.items():
            new_k = zh2en_mapping.get(k, k)
            ret_dict[new_k] = {}
            for key, val in v.items():
                if key.replace(Ref_indicator, "") in zh2en_mapping:
                    new_key = zh2en_mapping[key.replace(Ref_indicator, "")].replace("Of", "of").replace("Or", "or")
                    if Ref_indicator in key:
                        new_key = new_key + Ref_indicator
                    ret_dict[new_k][new_key] = val
                else:
                    ret_dict[new_k][key] = val
        del _tmp_dict

    df = pd.DataFrame(rows)

    if save_path is None:
        save_path = DEFAULTS.DATA_DIR / "seizure_risk_table.csv"
    if not save_path.is_file() or overwrite:
        df.to_csv(save_path, index=False, header=False)
        df.to_excel(save_path.with_suffix(".xlsx"), index=False, header=False)

    if return_type.lower() == "pd":
        return df
    elif return_type.lower() == "latex":
        rows = df.to_latex(header=False, index=False).splitlines()
        rows[0] = r"\begin{tabular}{@{\extracolsep{6pt}}llllllll@{}}"
        rows[2] = (
            r"\multicolumn{2}{l}{Feature} & \multicolumn{3}{l}{Affected} & \multicolumn{2}{l}{Seizure Risk ($95\%$ CI)} & Seizure Risk Difference  ($95\%$ CI) \\ \cline{1-2}\cline{3-5}\cline{6-7}\cline{8-8}"
        )
        ret_lines = "\n".join(rows)
        if not save_path.with_suffix(".tex").is_file() or overwrite:
            save_path.with_suffix(".tex").write_text(ret_lines)
        return ret_lines
    elif return_type.lower() in ["md", "markdown"]:
        return df.to_markdown(index=False)
    elif return_type.lower() == "html":
        return df.to_html(index=False)
    elif return_type.lower() == "dict":
        return ret_dict


if __name__ == "__main__":
    gen_seizure_risk_diff_TDSB_ext(return_type="pd", overwrite=True)
    gen_seizure_risk_diff_TDSB_ext(return_type="latex", overwrite=True)
