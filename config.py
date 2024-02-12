"""
"""

from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, MutableMapping, Optional

import numpy as np
from sklearn.model_selection import ParameterGrid

__all__ = [
    "CFG",
    "DEFAULTS",
    "DataPreprocessConfig",
    "FeatureConfig",
    "GridSearchConfig",
    "FeatureSelectionConfig",
    "ServingConfig",
    "clf_metric_aliases",
]


class CFG(dict):
    """This class is created in order to renew the :meth:`update` method,
    to fit the hierarchical structure of configurations.

    Examples
    --------
    >>> c = CFG(hehe={"a": 1, "b": 2})
    >>> c.update(hehe={"a": [-1]})
    >>> c
    {'hehe': {'a': [-1], 'b': 2}}
    >>> c.update(hehe={"c": -10})
    >>> c
    {'hehe': {'a': [-1], 'b': 2, 'c': -10}}
    >>> c.hehe.pop("a")
    [-1]
    >>> c
    {'hehe': {'b': 2, 'c': -10}}

    """

    __name__ = "CFG"

    def __init__(self, *args, **kwargs) -> None:
        if len(args) > 1:
            raise TypeError(f"expected at most 1 arguments, got {len(args)}")
        elif len(args) == 1:
            d = args[0]
            assert isinstance(d, MutableMapping)
        else:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            try:
                setattr(self, k, v)
            except Exception:
                dict.__setitem__(self, k, v)
        # Class attributes
        exclude_fields = ["update", "pop"]
        for k in self.__class__.__dict__:
            if not (k.startswith("__") and k.endswith("__")) and k not in exclude_fields:
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, new_cfg: Optional[MutableMapping] = None, **kwargs: Any) -> None:
        """The new hierarchical update method.

        Parameters
        ----------
        new_cfg : MutableMapping, optional
            The new configuration, by default None.
        **kwargs : dict, optional
            Key value pairs, by default None.

        """
        _new_cfg = new_cfg or CFG()
        if len(kwargs) > 0:  # avoid RecursionError
            _new_cfg.update(kwargs)
        for k in _new_cfg:
            if isinstance(_new_cfg[k], MutableMapping) and k in self:
                if isinstance(self[k], MutableMapping):
                    self[k].update(_new_cfg[k])
                else:  # for example, self[k] is `None`
                    self[k] = _new_cfg[k]  # deepcopy?
            else:
                try:
                    setattr(self, k, _new_cfg[k])
                except Exception:
                    dict.__setitem__(self, k, _new_cfg[k])

    def pop(self, key: str, default: Optional[Any] = None) -> Any:
        """The updated pop method.

        Parameters
        ----------
        key : str
            The key to pop.
        default : Any, optional
            The default value, by default None.

        """
        if key in self:
            delattr(self, key)
        return super().pop(key, default)

    def __repr__(self) -> str:
        return repr({k: v for k, v in self.items() if not callable(v)})

    def __str__(self) -> str:
        return str({k: v for k, v in self.items() if not callable(v)})


_PROJECT_DIR = Path(__file__).resolve().parent


DEFAULTS = CFG()

DEFAULTS.eps = 1e-7

DEFAULTS.SEED = 42
DEFAULTS.RNG = np.random.default_rng(seed=DEFAULTS.SEED)
DEFAULTS.RNG_sample = partial(DEFAULTS.RNG.choice, replace=False, shuffle=False)
DEFAULTS.RNG_randint = partial(DEFAULTS.RNG.integers, endpoint=True)

DEFAULTS.DATA_DIR = _PROJECT_DIR / "data"
DEFAULTS.FONT_DIR = _PROJECT_DIR / "fonts"
DEFAULTS.SAVE_DIR = _PROJECT_DIR / "save"
DEFAULTS.LOG_DIR = _PROJECT_DIR / "log"
DEFAULTS.DATA_DIR.mkdir(exist_ok=True)
DEFAULTS.SAVE_DIR.mkdir(exist_ok=True)
DEFAULTS.LOG_DIR.mkdir(exist_ok=True)


DataPreprocessConfig = CFG()

DataPreprocessConfig.zh2en = False
DataPreprocessConfig.zh2en_mapping = {
    "患者编号": "SubjectID",
    "整体有癫痫": "HasSeizure",
    "初步诊断": "PreliminaryDiagnosis",
    "病理诊断": "PathologicalDiagnosis",
    "出院诊断名称": "DischargeDiagnosis",
    "出院第一诊断名称": "DischargeDiagnosisFirst",
    "出院其他诊断名称": "DischargeDiagnosisRest",
    "年龄": "Age",
    "性别": "Gender",
    "男": "Male",
    "女": "Female",
    "年龄段": "AgeGroup",
    "并发症感染": "ComplicationInfection",
    "手术切除方式": "ExtentOfResection",
    "部分切除": "Partial",
    "大部切除": "Partial",
    "次全切": "SubTotal",
    "近全切": "SubTotal",
    "全切": "GrossTotal",
    "其它切除方式": "SubtotalOrPartial",
    "病理分级": "WHOGrading",
    "复发胶质瘤": "RecurrentGlioma",
    "去骨瓣减压术": "DecompressiveCraniectomy",
    "病理分型粗": "Pathology",
    "分型胶质母": "GlioblastomaMultiforme",
    "分型间变型星形": "AnaplasticAstrocytoma",
    "分型少突星形": "Oligoastrocytoma",
    "分型星形": "Asrocytoma",
    "分型毛细胞星形": "PilocyticAstrocytoma",
    "分型中枢神经": "Neurocytoma",
    "分型混合": "Mixed",
    "分型其它": "Others",
    "手术时长": "SurgeryDuration",
    "肿瘤大小": "MaximumDiameter(cm)",
    "出血量": "BleedingAmount",
    "合并症": "Comorbidity",
    "合并症_低钠血症": "Comorbidity_Hyponatremia",
    "合并症_低蛋白血症": "Comorbidity_Hypoproteinemia",
    "合并症_低钾血症": "Comorbidity_Hypokalemia",
    "合并症_高氯血症": "Comorbidity_Hyperchloremia",
    "合并症_低氯血症": "Comorbidity_Hypochloremia",
    "合并症_中枢神经感染": "Comorbidity_CentralNervousSystemInfection",
    "合并症_脑积水": "Comorbidity_Hydrocephalus",
    "合并症_凝血功能异常": "Comorbidity_CoagulationDisorders",
    "肿瘤分区": "RegionInvolved",
    "肿瘤分区_颞": "Region_Temporal",
    "肿瘤分区_顶": "Region_Parietal",
    "肿瘤分区_额": "Region_Frontal",
    "肿瘤分区_小脑": "Region_Cerebellum",
    "肿瘤分区_脑室": "Region_Ventricle",
    "肿瘤分区_岛": "Region_Insular",
    "肿瘤分区_枕": "Region_Occipital",
    "肿瘤分区_丘脑": "Region_Thalamus",
    "肿瘤分区_胼胝体": "Region_CorpusCallosum",
    "肿瘤分区_其它": "Region_Others",
    "肿瘤分区_基底节": "Region_BasalGanglia",
    "肿瘤分区_鞍区": "Region_SellarRegion",
    "肿瘤分区_脑干": "Region_Brainstem",
    "肿瘤分区_额或颞": "Region_FrontalOrTemporal",  # NOT used
    "C年龄": "CateAge",
    "C手术时长小时": "CateSurgeryDuration",
    "C肿瘤最大直径": "CateMaximumDiameter (cm)",
    "C出血量": "CateBleedingAmount",
    "C病理分级": "CatePathologicalGrading",
    "C肿瘤分区1": "CateRegion1",
    "C肿瘤分区2": "CateRegion2",
    "术后是否用药": "IsDrugAfterSurgery",
    "用药改变": "MedicationChange",
}
_tmp_dict = {f"WHO{idx}级": f"WHO{idx}" for idx in ["I", "II", "III", "IV"]}
DataPreprocessConfig.zh2en_mapping.update(deepcopy(_tmp_dict))
_tmp_dict = {}
for k, v in DataPreprocessConfig.zh2en_mapping.items():
    if "_" in k:
        _tmp_dict[k.split("_")[1]] = v.split("_")[1]
DataPreprocessConfig.zh2en_mapping.update(deepcopy(_tmp_dict))
del _tmp_dict

DataPreprocessConfig.y_col = "整体有癫痫"
DataPreprocessConfig.positive_class = "是"
DataPreprocessConfig.negative_class = "否"

DataPreprocessConfig.BIO_cont_var = ["BIO_Ki-67"]
DataPreprocessConfig.BIO_cate_var = [
    "BIO_PD-L1",
    "BIO_Olig-2",
    "BIO_P53",
    "BIO_IDH1-R132",
    "BIO_IDH2-R172",
    "BIO_1p36-lose",
    "BIO_19q13-lose",
    "BIO_TERT-C228T-chg",
    "BIO_TERT-C250T-chg",
    "BIO_MGMT",
    "BIO_ATRX",
    "BIO_EGFR",
    "BIO_GFAP",
    "BIO_VEGF",
    "BIO_MMP-9",
    "BIO_H3K27M",
]
DataPreprocessConfig.BIO_var = DataPreprocessConfig.BIO_cont_var + DataPreprocessConfig.BIO_cate_var
DataPreprocessConfig.BIO_na_fillvalue = " "

DataPreprocessConfig.demographic_cont_var = [
    "年龄",
]
DataPreprocessConfig.demographic_cate_var = [
    "性别",
    # "年龄段",
]
DataPreprocessConfig.demographic_var = DataPreprocessConfig.demographic_cont_var + DataPreprocessConfig.demographic_cate_var

DataPreprocessConfig.tumor_cate_var = [
    "肿瘤分区_颞",
    "肿瘤分区_顶",
    "肿瘤分区_额",
    "肿瘤分区_小脑",
    "肿瘤分区_脑室",
    "肿瘤分区_岛",
    "肿瘤分区_枕",
    # "肿瘤分区_额或颞",
    "肿瘤分区_丘脑",
    "肿瘤分区_胼胝体",
    "肿瘤分区_其它",
    "肿瘤分区_基底节",
    "肿瘤分区_鞍区",
    "肿瘤分区_脑干",
    # "C肿瘤最大直径",
    "病理分型粗",
    "病理分级",
]
DataPreprocessConfig.tumor_cont_var = [
    # "肿瘤大小",
]
DataPreprocessConfig.tumor_var = DataPreprocessConfig.tumor_cate_var + DataPreprocessConfig.tumor_cont_var

DataPreprocessConfig.comorbidity_var = [
    "合并症_低钠血症",
    "合并症_低蛋白血症",
    "合并症_低钾血症",
    "合并症_高氯血症",
    "合并症_低氯血症",
    "合并症_中枢神经感染",
    "合并症_脑积水",
    "合并症_凝血功能异常",
]
DataPreprocessConfig.surgery_cate_var = [
    "并发症感染",
    "手术切除方式",
    "C肿瘤最大直径",
    # "C手术时长小时",
] + DataPreprocessConfig.comorbidity_var
DataPreprocessConfig.surgery_cont_var = [
    # "手术时长",
    # "肿瘤大小",
    # "出血量",
]
DataPreprocessConfig.surgery_var = DataPreprocessConfig.surgery_cate_var + DataPreprocessConfig.surgery_cont_var

DataPreprocessConfig.medication_var = [
    "术后是否用药",
    "用药改变",
    "乙N丙N",
]

# DataPreprocessConfig.diagnosis_var = ["初步诊断", "病理诊断", "出院诊断名称", "出院第一诊断名称", "出院其他诊断名称"]

DataPreprocessConfig.categorical_var = (
    [
        "复发胶质瘤",
        "去骨瓣减压术",
        # "肿瘤分区",
        # "C年龄",
    ]
    + DataPreprocessConfig.BIO_cate_var
    + DataPreprocessConfig.demographic_cate_var
    + DataPreprocessConfig.tumor_cate_var
    + DataPreprocessConfig.surgery_cate_var
    + DataPreprocessConfig.medication_var
    # + DataPreprocessConfig.diagnosis_var
)
DataPreprocessConfig.continuous_var = (
    DataPreprocessConfig.BIO_cont_var
    + DataPreprocessConfig.demographic_cont_var
    + DataPreprocessConfig.tumor_cont_var
    + DataPreprocessConfig.surgery_cont_var
)

DataPreprocessConfig.x_cols = DataPreprocessConfig.categorical_var + DataPreprocessConfig.continuous_var
for item in [
    "年龄段",
    "并发症感染",
    "复发胶质瘤",
    "去骨瓣减压术",
    "C年龄",
    "C手术时长小时",
    "C肿瘤最大直径",
    "C出血量",
    "C病理分级",
    "C肿瘤分区1",
    "C肿瘤分区2",
]:
    if item not in DataPreprocessConfig.x_cols:
        DataPreprocessConfig.x_cols.append(item)

DataPreprocessConfig.dropped_cols = [
    "总用药天数",
    "手术麻醉时间",
    "入院是否有癫痫",
    "初步诊断",
    "手术名称",
    "手术时间",
    "住院号",
    "住院天数",
    "民族",
    "病理诊断",
    "出院诊断名称",
    "出院第一诊断名称",
    "出院其他诊断名称",
    "复诊周期月",
    "复诊次数",
    "随访天数",
    "出院发生癫痫次数",
    "最后开药时间",
]

DataPreprocessConfig.y_col_mapping = {"是": 1, "否": 0}
DataPreprocessConfig.x_col_mappings = CFG()
DataPreprocessConfig.x_col_mappings.性别 = {"女": 0, "男": 1}
DataPreprocessConfig.x_col_mappings.年龄段 = {
    ">60": 3,
    "40~60": 2,
    "20~40": 1,
    "<=20": 0,
}
DataPreprocessConfig.x_col_mappings.手术切除方式 = {
    "全切": 4,
    "近全切": 3,
    "次全切": 2,
    "大部切除": 1,
    "部分切除": 0,
}
DataPreprocessConfig.x_col_mappings.病理分级 = {
    "WHOIV级": 3,
    "WHOIII级": 2,
    "WHOII级": 1,
    "WHOI级": 0,
}
DataPreprocessConfig.x_col_mappings.病理分型粗 = {
    # 病理分型：间变性星形细胞瘤、少突星形细胞瘤与胶质母细胞瘤相比，癫痫风险增加。毛细胞星形细胞瘤、髓母细胞瘤、室管膜瘤癫痫风险较低。
    "分型间变型星形": 7,
    "分型少突星形": 6,
    "分型胶质母": 5,
    "分型星形": 4,
    "分型毛细胞星形": 3,
    "分型中枢神经": 2,
    "分型混合": 1,
    "分型其它": 0,
}
DataPreprocessConfig.x_col_mappings.术后是否用药 = {"是": 1, "否": 0}
DataPreprocessConfig.x_col_mappings.用药改变 = {"术后改变用药": 2, "单一用药": 1, "不用药": 0}
DataPreprocessConfig.x_col_mappings.乙N丙N = {"左0丙0": 0, "左1丙0": 1, "左0丙1": 2, "左1丙1": 3}
DataPreprocessConfig.x_col_mappings.C肿瘤最大直径 = {"<5": 1, ">=5": 0}

DataPreprocessConfig.BIO_mapping = {
    "+++": 5,
    "++": 4,
    "+": 3,
    "±": 2,
    "-": 1,
    DataPreprocessConfig.BIO_na_fillvalue: 0,
}

DataPreprocessConfig.split_file_path = DEFAULTS.DATA_DIR / "train-test-split-8-2.json"

DataPreprocessConfig.age_scaler_path = DEFAULTS.SAVE_DIR / "age_scaler.pkl"


FeatureConfig = CFG()
FeatureConfig.set_name = None
FeatureConfig.feature_list = None
FeatureConfig.sets = CFG()
FeatureConfig.sets.TD = DataPreprocessConfig.tumor_var + DataPreprocessConfig.demographic_var
FeatureConfig.sets.TDS = FeatureConfig.sets.TD + DataPreprocessConfig.surgery_var
# FeatureConfig.sets.TDM = FeatureConfig.sets.TD + DataPreprocessConfig.medication_var
# FeatureConfig.sets.TDSM = FeatureConfig.sets.TDS + DataPreprocessConfig.medication_var
FeatureConfig.sets.TDB = FeatureConfig.sets.TD + ["BIO_IDH1-R132"]
FeatureConfig.sets.TDSB = FeatureConfig.sets.TDS + ["BIO_IDH1-R132"]
FeatureConfig.sets.TDSB_ext = FeatureConfig.sets.TDS + DataPreprocessConfig.BIO_var
# FeatureConfig.sets.TDMB = FeatureConfig.sets.TDM + ["BIO_IDH1-R132"]
# FeatureConfig.sets.TDSMB = FeatureConfig.sets.TDSM + ["BIO_IDH1-R132"]
FeatureConfig.BIO_na_strategy = "drop"  # "knn", "random_uniform", "random_distributional", "keep"
FeatureConfig.set_list = list(FeatureConfig.sets)

FeatureConfig.binarize_variables = False

FeatureConfig.over_sampler = "SMOTENC"  # "RandomOverSampler", None


GridSearchConfig = CFG()
GridSearchConfig.rf = ParameterGrid(
    {
        "n_estimators": [10, 15, 20, 50, 100],
        "criterion": ["gini", "entropy"],
        # 'max_depth': None,
        "min_samples_split": [2, 3, 4],
        # 'min_samples_leaf': 1,
        # 'min_weight_fraction_leaf': 0.0,
        "max_features": ["auto", "sqrt", "log2"],
        # 'max_leaf_nodes': None,
        # 'min_impurity_decrease': 0.0,
        "bootstrap": [True, False],
        "oob_score": [True, False],
        # 'n_jobs': None,
        # 'random_state': None,
        # 'verbose': 0,
        "warm_start": [True, False],
        "class_weight": ["balanced", "balanced_subsample", None],
        # 'ccp_alpha': 0.0,
        # 'max_samples': None,
    }
)
GridSearchConfig.xgb = ParameterGrid(
    {
        "n_estimators": [10, 15, 20, 50],
        "learning_rate": [0.01, 0.05, 0.1],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.0, 0.1, 0.5, 1.0],
        "max_depth": [3, 5, 8],
        "verbosity": [0],
    }
)
GridSearchConfig.gdbt = ParameterGrid(
    {
        "n_estimators": [10, 15, 20, 50, 100],
        "loss": ["deviance", "exponential"],
        "learning_rate": [0.01, 0.05, 0.1],
        "criterion": ["friedman_mse", "mse"],
        "min_samples_split": [2, 3, 4],
        "max_features": ["auto", "sqrt", "log2"],
        "warm_start": [True, False],
        "ccp_alpha": [0.0, 0.1, 0.5, 1.0],
    }
)
GridSearchConfig.svc = ParameterGrid(
    {
        "C": [0.1, 0.5, 1, 10],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [2, 3, 5],  # for "poly" kernel
        "gamma": [
            "scale",
            "auto",
        ],  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        "coef0": [0.0, 0.2, 0.5, 1.0],  # for 'poly' and 'sigmoid'
        "class_weight": ["balanced", None],
        # https://stackoverflow.com/questions/62341001/svm-stuck-in-fitting-the-model
        "probability": [True],
        "shrinking": [True, False],
    }
)
GridSearchConfig.lr = ParameterGrid(
    {
        "penalty": ["l1", "l2", "elasticnet"],
        "C": [0.1, 0.5, 1, 10],
        "class_weight": ["balanced", None],
        "warm_start": [True, False],
        "l1_ratio": [0.1, 0.5, 0.9],
    }
)
GridSearchConfig.bagging = ParameterGrid(
    {
        "n_estimators": [10, 15, 20, 50, 100],
        "max_features": [0.1, 0.2, 0.5, 0.9, 1.0],
        "bootstrap": [True, False],
        "bootstrap_features": [True, False],
        "oob_score": [True, False],
        "warm_start": [True, False],
    }
)
# 'RidgeClassifier' object has no attribute 'predict_proba'
# GridSearchConfig.ridge = ParameterGrid(
#     {
#         "alpha": [0.1, 1, 10],
#         "class_weight": ["balanced", None],
#     }
# )
GridSearchConfig.sk_mlp = ParameterGrid(
    {
        "hidden_layer_sizes": [(50,), (100,), (50, 100), (50, 100, 50)],
        "activation": ["logistic", "tanh", "relu"],
        "solver": ["lbfgs", "sgd", "adam"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["constant", "invscaling", "adaptive"],
        "learning_rate_init": [
            0.001,
            0.01,
        ],
        "warm_start": [True, False],
    }
)


FeatureSelectionConfig = CFG()
FeatureSelectionConfig.rfe = CFG(
    model="rf",
    model_params=CFG(n_estimators=20, criterion="entropy", max_features="sqrt"),
    n_features_to_select=np.arange(0.3, 1.0, 0.1).tolist(),
    n_features_to_select_key="n_features_to_select",
)
FeatureSelectionConfig.kbest = CFG(
    score_func="chi2",
    n_features_to_select=np.arange(0.3, 1.0, 0.1).tolist(),
    n_features_to_select_key="k",
)
FeatureSelectionConfig.vt = CFG(
    threshold=(0.9 * (1 - 0.9)),
)
FeatureSelectionConfig.sfm = CFG(
    model="rf",
    model_params=CFG(n_estimators=20, criterion="entropy", max_features="sqrt"),
    n_features_to_select=np.arange(0.3, 1.0, 0.1).tolist(),
    n_features_to_select_key="max_features",
)
FeatureSelectionConfig.sequential = CFG(
    model="rf",
    model_params=CFG(n_estimators=20, criterion="entropy", max_features="sqrt"),
    n_features_to_select=np.arange(0.3, 1.0, 0.1).tolist(),
    n_features_to_select_key="n_features_to_select",
)


ServingConfig = CFG()
ServingConfig.model_path = DEFAULTS.SAVE_DIR / "spm_rf_TDSB_drop.pkl"
ServingConfig.public_domain = None
ServingConfig.public_ip = "43.140.244.112"
ServingConfig.internal_ip = "0.0.0.0"
ServingConfig.port = 11111
ServingConfig.name = "seizure_prediction"


clf_metric_aliases = CFG(
    # https://en.wikipedia.org/wiki/Precision_and_recall
    {
        "sens": "sensitivity, recall, hit rate, true positive rate".split(", "),
        "spec": "specificity, selectivity, true negative rate".split(", "),
        "prec": "precision, positive predictive value".split(", "),
        "npv": "negative predictive value".split(", "),
        "jac": "jaccard index, threat score, critical success index".split(", "),
        "acc": "accuracy".split(", "),
        "phi": "phi coefficient, matthews correlation coefficient".split(", "),
        "fnr": "false negative rate, miss rate".split(", "),
        "fpr": "false positive rate, fall-out".split(", "),
        "fdr": "false discovery rate".split(", "),
        "for": "false omission rate".split(", "),
        "plr": "positive likelihood ratio".split(", "),
        "nlr": "negative likelihood ratio".split(", "),
        "pt": "prevalence threshold".split(", "),
        "ba": "balanced accuracy".split(", "),
        "f1": "f1-measure".split(", "),
        "fm": "fowlkes-mallows index".split(", "),
        "bm": "bookmaker informedness".split(", "),
        "mk": "markedness".split(", "),
        "dor": "diagnostic odds ratio".split(", "),
    }
)
