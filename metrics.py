"""
utilities for computing metrics.
Adapted from: [torch_ecg](https://github.com/DeepPSP/torch_ecg/blob/master/torch_ecg/components/metrics.py)
"""

import warnings
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import roc_auc_score

from utils import add_docstring

__all__ = [
    "confusion_matrix",
    "ovr_confusion_matrix",
    "metrics_from_confusion_matrix",
    "ClassificationMetrics",
    "SeizureClassificationMetrics",
]


def confusion_matrix(
    labels: np.ndarray,
    outputs: np.ndarray,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Compute a binary confusion matrix, where the columns are expert labels and rows are classifier labels.

    Parameters
    ----------
    labels : np.ndarray or Tensor
        Binary labels, of shape: ``(n_samples, n_classes)``,
        or indices of each label class, of shape: ``(n_samples,)``.
    outputs : np.ndarray or Tensor
        Binary outputs, of shape: ``(n_samples, n_classes)``,
        or indices of each class predicted, of shape: ``(n_samples,)``.
    num_classes : int, optional
        Number of classes.
        If `labels` and `outputs` are both of shape ``(n_samples,)``,
        then `num_classes` must be specified.

    Returns
    -------
    cm : np.ndarray
        Confusion matrix, of shape: ``(n_classes, n_classes)``.

    """
    labels, outputs = cls_to_bin(labels, outputs, num_classes)
    assert np.shape(labels) == np.shape(outputs)
    assert all(value in (0, 1) for value in np.unique(labels))
    assert all(value in (0, 1) for value in np.unique(outputs))

    num_samples, num_classes = np.shape(labels)

    cm = np.zeros((num_classes, num_classes))
    for k in range(num_samples):
        i = np.argmax(outputs[k, :])
        j = np.argmax(labels[k, :])
        cm[i, j] += 1

    return cm


def one_vs_rest_confusion_matrix(
    labels: np.ndarray,
    outputs: np.ndarray,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Compute binary one-vs-rest confusion matrices,
    where the columns are expert labels and rows are classifier labels.

    Parameters
    ----------
    labels : np.ndarray
        Binary labels, of shape: ``(n_samples, n_classes)``
        or indices of each label class, of shape: ``(n_samples,)``.
    outputs : np.ndarray
        Binary outputs, of shape: ``(n_samples, n_classes)``
        or indices of each class predicted, of shape: ``(n_samples,)``.
    num_classes : int, optional,
        Number of classes,
        If `labels` and `outputs` are both of shape ``(n_samples,)``,
        then `num_classes` must be specified.

    Returns
    -------
    ovr_cm : np.ndarray
        One-vs-rest confusion matrix, of shape: ``(n_classes, 2, 2)``.

    """
    labels, outputs = cls_to_bin(labels, outputs, num_classes)
    assert np.shape(labels) == np.shape(outputs)
    assert all(value in (0, 1) for value in np.unique(labels))
    assert all(value in (0, 1) for value in np.unique(outputs))

    num_samples, num_classes = np.shape(labels)

    ovr_cm = np.zeros((num_classes, 2, 2))
    for i in range(num_samples):
        for j in range(num_classes):
            if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                ovr_cm[j, 0, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                ovr_cm[j, 0, 1] += 1
            elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                ovr_cm[j, 1, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                ovr_cm[j, 1, 1] += 1

    return ovr_cm


# alias
ovr_confusion_matrix = one_vs_rest_confusion_matrix


_METRICS_FROM_CONFUSION_MATRIX_PARAMS = """
    Compute macro {metric}, and {metrics} for each class.

    Parameters
    ----------
    labels : np.ndarray
        Binary labels, of shape: ``(n_samples, n_classes)``
        or indices of each label class, of shape: ``(n_samples,)``
    outputs : np.ndarray
        Binary outputs, of shape: ``(n_samples, n_classes)``
        or indices of each class predicted, of shape: ``(n_samples,)``
    num_classes : int, optional
        Number of classes,
        If `labels` and `outputs` are both of shape ``(n_samples,)``,
        then `num_classes` must be specified.
    weights : np.ndarray, optional
        Weights for each class, of shape: ``(n_classes,)``,
        used to compute macro {metric},
"""


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(metric="metrics", metrics="metrics"),
    "prepend",
)
def metrics_from_confusion_matrix(
    labels: np.ndarray,
    outputs: np.ndarray,
    num_classes: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Returns
    -------
    metrics : dict
        Metrics computed from the one-vs-rest confusion matrix.

    Examples
    --------
    >>> from torch_ecg.cfg import DEFAULTS
    >>> labels, outputs = DEFAULTS.RNG_randint(0,2,(100,10)), DEFAULTS.RNG_randint(0,2,(100,10))
    >>> metrics = metrics_from_confusion_matrix(labels, outputs)

    References
    ----------
    1. https://en.wikipedia.org/wiki/Precision_and_recall

    """
    labels, outputs = cls_to_bin(labels, outputs, num_classes)
    num_samples, num_classes = np.shape(labels)

    ovr_cm = ovr_confusion_matrix(labels, outputs)

    # sens: sensitivity, recall, hit rate, or true positive rate
    # spec: specificity, selectivity or true negative rate
    # prec: precision or positive predictive value
    # npv: negative predictive value
    # jac: jaccard index, threat score, or critical success index
    # acc: accuracy
    # phi: phi coefficient, or matthews correlation coefficient
    # NOTE: never use repeat here, because it will cause bugs
    # sens, spec, prec, npv, jac, acc, phi = list(repeat(np.zeros(num_classes), 7))
    sens, spec, prec, npv, jac, acc, phi = [np.zeros(num_classes) for _ in range(7)]
    auroc = np.zeros(num_classes)  # area under the receiver-operater characteristic curve (ROC AUC)
    auprc = np.zeros(num_classes)  # area under the precision-recall curve
    for k in range(num_classes):
        tp, fp, fn, tn = (
            ovr_cm[k, 0, 0],
            ovr_cm[k, 0, 1],
            ovr_cm[k, 1, 0],
            ovr_cm[k, 1, 1],
        )
        if tp + fn > 0:
            sens[k] = tp / (tp + fn)
        else:
            sens[k] = float("nan")
        if tp + fp > 0:
            prec[k] = tp / (tp + fp)
        else:
            prec[k] = float("nan")
        if tn + fp > 0:
            spec[k] = tn / (tn + fp)
        else:
            spec[k] = float("nan")
        if tn + fn > 0:
            npv[k] = tn / (tn + fn)
        else:
            npv[k] = float("nan")
        if tp + fn + fp > 0:
            jac[k] = tp / (tp + fn + fp)
        else:
            jac[k] = float("nan")
        acc[k] = (tp + tn) / num_samples
        phi[k] = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k] == 1)
        tn[0] = np.sum(labels[:, k] == 0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_samples and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr_ = np.zeros(num_thresholds)
        tnr_ = np.zeros(num_thresholds)
        ppv_ = np.zeros(num_thresholds)
        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr_[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr_[j] = float("nan")
            if fp[j] + tn[j]:
                tnr_[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr_[j] = float("nan")
            if tp[j] + fp[j]:
                ppv_[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv_[j] = float("nan")

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds - 1):
            auroc[k] += 0.5 * (tpr_[j + 1] - tpr_[j]) * (tnr_[j + 1] + tnr_[j])
            auprc[k] += (tpr_[j + 1] - tpr_[j]) * ppv_[j + 1]

    fnr = 1 - sens  # false negative rate, miss rate
    fpr = 1 - spec  # false positive rate, fall-out
    fdr = 1 - prec  # false discovery rate
    for_ = 1 - npv  # false omission rate
    plr = sens / fpr  # positive likelihood ratio
    nlr = fnr / spec  # negative likelihood ratio
    pt = np.sqrt(fpr) / (np.sqrt(sens) + np.sqrt(fpr))  # prevalence threshold
    ba = (sens + spec) / 2  # balanced accuracy
    f1 = 2 * sens * prec / (sens + prec)  # f1-measure
    fm = np.sqrt(prec * sens)  # fowlkes-mallows index
    bm = sens + spec - 1  # informedness, bookmaker informedness
    mk = prec + npv - 1  # markedness
    dor = plr / nlr  # diagnostic odds ratio

    if weights is None:
        _weights = np.ones(num_classes)
    else:
        _weights = weights / np.mean(weights)
    metrics = {}
    for m in [
        "sens",  # sensitivity, recall, hit rate, or true positive rate
        "spec",  # specificity, selectivity or true negative rate
        "prec",  # precision or positive predictive value
        "npv",  # negative predictive value
        "jac",  # jaccard index, threat score, or critical success index
        "acc",  # accuracy
        "phi",  # phi coefficient, or matthews correlation coefficient
        "fnr",  # false negative rate, miss rate
        "fpr",  # false positive rate, fall-out
        "fdr",  # false discovery rate
        "for_",  # false omission rate
        "plr",  # positive likelihood ratio
        "nlr",  # negative likelihood ratio
        "pt",  # prevalence threshold
        "ba",  # balanced accuracy
        "f1",  # f1-measure
        "fm",  # fowlkes-mallows index
        "bm",  # bookmaker informedness
        "mk",  # markedness
        "dor",  # diagnostic odds ratio
        "auroc",  # area under the receiver-operater characteristic curve (ROC AUC)
        "auprc",  # area under the precision-recall curve
    ]:
        metrics[m.strip("_")] = eval(m)
        metrics[f"macro_{m}".strip("_")] = np.nanmean(eval(m) * _weights) if np.any(np.isfinite(eval(m))) else float("nan")
    return metrics


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(metric="F1-measure", metrics="F1-measures"),
    "prepend",
)
def f_measure(
    labels: np.ndarray,
    outputs: np.ndarray,
    num_classes: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
    Returns
    -------
    macro_f1 : float
        Macro F1-measure.
    f1 : np.ndarray
        F1-measures for each class, of shape: ``(n_classes,)``.

    """
    m = metrics_from_confusion_matrix(labels, outputs, num_classes, weights)

    return m["macro_f1"], m["f1"]  # type: ignore


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(metric="sensitivity", metrics="sensitivities"),
    "prepend",
)
def sensitivity(
    labels: np.ndarray,
    outputs: np.ndarray,
    num_classes: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
    Returns
    -------
    macro_sens : float
        Macro sensitivity.
    sens : np.ndarray
        Sensitivities for each class, of shape: ``(n_classes,)``.

    """
    m = metrics_from_confusion_matrix(labels, outputs, num_classes, weights)

    return m["macro_sens"], m["sens"]  # type: ignore


# aliases
recall = sensitivity
true_positive_rate = sensitivity
hit_rate = sensitivity


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(metric="precision", metrics="precisions"),
    "prepend",
)
def precision(
    labels: np.ndarray,
    outputs: np.ndarray,
    num_classes: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
    Returns
    -------
    macro_prec : float
        Macro precision.
    prec : np.ndarray
        Precisions for each class, of shape: ``(n_classes,)``.

    """
    m = metrics_from_confusion_matrix(labels, outputs, num_classes, weights)

    return m["macro_prec"], m["prec"]  # type: ignore


# aliases
positive_predictive_value = precision


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(metric="specificity", metrics="specificities"),
    "prepend",
)
def specificity(
    labels: np.ndarray,
    outputs: np.ndarray,
    num_classes: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
    Returns
    -------
    macro_spec : float
        Macro specificity.
    spec : np.ndarray
        Specificities for each class, of shape: ``(n_classes,)``.

    """
    m = metrics_from_confusion_matrix(labels, outputs, num_classes, weights)

    return m["macro_spec"], m["spec"]  # type: ignore


# aliases
selectivity = specificity
true_negative_rate = specificity


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(metric="AUROC and macro AUPRC", metrics="AUPRCs, AUPRCs"),
    "prepend",
)
def auc(
    labels: np.ndarray,
    outputs: np.ndarray,
    num_classes: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    macro_auroc : float
        Macro AUROC.
    macro_auprc : float
        Macro AUPRC.
    auprc : np.ndarray
        AUPRCs for each class, of shape: ``(n_classes,)``.
    auprc : np.ndarray
        AUPRCs for each class, of shape: ``(n_classes,)``.

    """
    m = metrics_from_confusion_matrix(labels, outputs, num_classes, weights)

    return m["macro_auroc"], m["macro_auprc"], m["auroc"], m["auprc"]  # type: ignore


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(metric="accuracy", metrics="accuracies"),
    "prepend",
)
def accuracy(
    labels: np.ndarray,
    outputs: np.ndarray,
    num_classes: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Returns
    -------
    macro_acc : float
        The macro accuracy.
    acc : np.ndarray
        Accuracies for each class, of shape: ``(n_classes,)``.

    """
    m = metrics_from_confusion_matrix(labels, outputs, num_classes, weights)

    return m["macro_acc"], m["acc"]  # type: ignore


def cls_to_bin(
    labels: np.ndarray,
    outputs: np.ndarray,
    num_classes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """ """
    if labels.ndim == outputs.ndim == 1:
        assert num_classes is not None
        shape = (labels.shape[0], num_classes)
        labels = _cls_to_bin(labels, shape)  # type: ignore
        outputs = _cls_to_bin(outputs, shape)  # type: ignore
    elif labels.ndim == 1:
        shape = outputs.shape
        labels = _cls_to_bin(labels, shape)
    elif outputs.ndim == 1:
        shape = labels.shape
        outputs = _cls_to_bin(outputs, shape)
    return labels, outputs


def _cls_to_bin(cls: np.ndarray, shape: Tuple[int]) -> np.ndarray:
    """ """
    bin_ = np.zeros(shape)
    for i in range(shape[0]):
        bin_[i, cls[i]] = 1
    return bin_


class ClassificationMetrics:
    """
    Classification metrics for multi-label and multi-class classification.

    Parameters
    ----------
    multi_label: bool,
        whether is multi-label classification
    macro: bool,
        whether to use macro-averaged metrics
    extra_metrics: Callable,
        extra metrics to compute,
        has to be a function with signature:
        `def extra_metrics(
            labels: np.ndarray,
            outputs: np.ndarray,
            num_classes: Optional[int]=None,
            weights: Optional[np.ndarray]=None
        ) -> dict`

    """

    __name__ = "ClassificationMetrics"

    def __init__(
        self,
        multi_label: bool = True,
        macro: bool = True,
        extra_metrics: Optional[Callable] = None,
    ) -> None:
        self.multi_label = multi_label
        self.set_macro(macro)
        self._extra_metrics = extra_metrics
        self._em = {}
        self._metrics = {
            k: np.nan
            for k in [
                "sens",  # sensitivity, recall, hit rate, or true positive rate
                "spec",  # specificity, selectivity or true negative rate
                "prec",  # precision or positive predictive value
                "npv",  # negative predictive value
                "jac",  # jaccard index, threat score, or critical success index
                "acc",  # accuracy
                "phi",  # phi coefficient, or matthews correlation coefficient
                "fnr",  # false negative rate, miss rate
                "fpr",  # false positive rate, fall-out
                "fdr",  # false discovery rate
                "for",  # false omission rate
                "plr",  # positive likelihood ratio
                "nlr",  # negative likelihood ratio
                "pt",  # prevalence threshold
                "ba",  # balanced accuracy
                "f1",  # f1-measure
                "fm",  # fowlkes-mallows index
                "bm",  # bookmaker informedness
                "mk",  # markedness
                "dor",  # diagnostic odds ratio
                "auroc",  # area under the receiver-operater characteristic curve (ROC AUC)
                "auprc",  # area under the precision-recall curve
            ]
        }
        self._metrics.update({f"macro_{k}": np.nan for k in self._metrics})
        self._cm = None
        self._cm_ovr = None

    def set_macro(self, macro: bool) -> None:
        """
        Set whether to use macro-averaged metrics.

        Parameters
        ----------
        macro : bool
            Whether to use macro-averaged metrics.

        """
        self.__prefix = ""
        self.macro = macro
        if macro:
            self.__prefix = "macro_"

    def compute(
        self,
        labels: np.ndarray,
        outputs: np.ndarray,
        num_classes: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
    ) -> "ClassificationMetrics":
        """
        Compute classification metrics.

        Parameters
        ----------
        labels : np.ndarray
            Binary labels, of shape: ``(n_samples, n_classes)``
            or indices of each label class, of shape: ``(n_samples,)``
        outputs : np.ndarray
            Binary outputs, of shape: ``(n_samples, n_classes)``
            or indices of each class predicted, of shape: ``(n_samples,)``
        num_classes : int, optional
            Number of classes,
            If `labels` and `outputs` are both of shape ``(n_samples,)``,
            then `num_classes` must be specified.
        weights : np.ndarray, optional
            Weights for each class, of shape: ``(n_classes,)``,
            used to compute macro metrics.

        Returns
        -------
        self : ClassificationMetrics
            The ClassificationMetrics object itself.

        """
        self._cm = confusion_matrix(labels, outputs, num_classes)
        self._cm_ovr = ovr_confusion_matrix(labels, outputs, num_classes)
        self._metrics = metrics_from_confusion_matrix(labels, outputs, num_classes, weights)
        if self._extra_metrics is not None:
            self._em = self._extra_metrics(labels, outputs, num_classes, weights)
            self._metrics.update(self._em)

        return self

    compute.__doc__ = metrics_from_confusion_matrix.__doc__.replace("metrics: dict,", f"{__name__},").replace(  # type: ignore
        "metrics = metrics_from_confusion_matrix(labels, outputs)",
        """metrics = ClassificationMetrics()
    >>> metrics = metrics.compute(labels, outputs)
    >>> metrics.fl_measure
    0.5062821146226457
    >>> metrics.set_macro(False)
    >>> metrics.fl_measure
    array([0.46938776, 0.4742268 , 0.4375    , 0.52941176, 0.58      ,
       0.57692308, 0.55769231, 0.48351648, 0.55855856, 0.3956044 ])""",
    )

    @add_docstring(compute.__doc__)
    def __call__(
        self,
        labels: np.ndarray,
        outputs: np.ndarray,
        num_classes: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
    ) -> "ClassificationMetrics":
        return self.compute(labels, outputs, num_classes, weights)

    @property
    def sensitivity(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}sens"]

    @property
    def recall(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}sens"]

    @property
    def hit_rate(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}sens"]

    @property
    def true_positive_rate(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}sens"]

    @property
    def specificity(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}spec"]

    @property
    def selectivity(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}spec"]

    @property
    def true_negative_rate(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}spec"]

    @property
    def precision(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}prec"]

    @property
    def positive_predictive_value(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}prec"]

    @property
    def negative_predictive_value(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}npv"]

    @property
    def jaccard_index(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}jac"]

    @property
    def threat_score(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}jac"]

    @property
    def critical_success_index(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}jac"]

    @property
    def accuracy(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}acc"]

    @property
    def phi_coefficient(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}phi"]

    @property
    def matthews_correlation_coefficient(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}phi"]

    @property
    def false_negative_rate(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}fnr"]

    @property
    def miss_rate(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}fnr"]

    @property
    def false_positive_rate(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}fpr"]

    @property
    def fall_out(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}fpr"]

    @property
    def false_discovery_rate(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}fdr"]

    @property
    def false_omission_rate(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}for"]

    @property
    def positive_likelihood_ratio(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}plr"]

    @property
    def negative_likelihood_ratio(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}nlr"]

    @property
    def prevalence_threshold(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}pt"]

    @property
    def balanced_accuracy(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}ba"]

    @property
    def f1_measure(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}f1"]

    @property
    def fowlkes_mallows_index(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}fm"]

    @property
    def bookmaker_informedness(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}bm"]

    @property
    def markedness(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}mk"]

    @property
    def diagnostic_odds_ratio(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}dor"]

    @property
    def area_under_the_receiver_operater_characteristic_curve(
        self,
    ) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}auroc"]

    @property
    def auroc(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}auroc"]

    @property
    def area_under_the_precision_recall_curve(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}auprc"]

    @property
    def auprc(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}auprc"]

    @property
    def extra_metrics(self) -> dict:
        return self._em


class SeizureClassificationMetrics(ClassificationMetrics):
    """Binary classification metrics for seizure detection.

    Parameters
    ----------
    positive_class : int
        The positive class index, default is 1.
    subset : Optional[Sequence[str]]
        The subset of metrics to return. Default is
        ["sens", "spec", "prec", "acc", "npv", "auc", "f1"].

    """

    __name__ = "SeizureClassificationMetrics"

    def __init__(
        self,
        positive_class: int = 1,
        subset: Optional[Sequence[str]] = [
            "sens",
            "spec",
            "prec",
            "acc",
            "npv",
            "auc",
            "f1",
        ],
    ) -> None:
        self.positive_class = positive_class
        self.subset = subset
        super().__init__(multi_label=False, macro=False)

    def __call__(self, y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:  # type: ignore
        """
        Compute seizure classification metrics.

        Parameters
        ----------
        y_true : np.ndarray
            True labels, of shape: ``(n_samples,)``.
        y_prob : np.ndarray
            Predicted probabilities, of shape: ``(n_samples,)`` or ``(n_samples, 2)``.
        thr : float
            Threshold to convert probabilities to binary predictions.

        Returns
        -------
        metrics : Dict[str, float]
            Dictionary of computed metrics.

        """
        assert y_true.ndim == 1, "y_true must be 1D"
        if y_prob.ndim == 1:
            assert y_prob.shape[0] == y_true.shape[0], "labels and probabilities must have the same length"
            y_pred = (y_prob > thr).astype(int)
        elif y_prob.ndim == 2:
            assert y_prob.shape == (
                y_true.shape[0],
                2,
            ), "`y_prob` must have shape (n_samples, 2)"
            y_prob = y_prob[:, self.positive_class]
            if self.positive_class == 1:
                y_pred = (y_prob > thr).astype(int)
            else:
                y_pred = (y_prob < thr).astype(int)
        else:
            raise ValueError("`y_prob` must be 1D or 2D")

        super().__call__(y_true, y_pred, num_classes=2)

        metrics = {k: v[self.positive_class] for k, v in self._metrics.items() if not k.startswith("macro_")}  # type: ignore
        # use the AUC score computed from roc_auc_score from sklearn
        metrics.pop("auroc")
        metrics.pop("auprc")
        metrics["auc"] = roc_auc_score(y_true, y_prob)

        if self.subset is not None:
            metrics = {k: v for k, v in metrics.items() if k in self.subset}
            if set(self.subset) - set(metrics.keys()):
                warnings.warn(f"Metrics {set(self.subset) - set(metrics.keys())} not available")

        # convert all numpy floats to python floats
        for k, v in metrics.items():
            if isinstance(v, np.generic):
                metrics[k] = v.item()

        return metrics
