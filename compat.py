"""
compat_metrics.py
=================
A compatibility shim for removed sklearn plotting functions.

Supports:
    - plot_roc_curve
    - plot_confusion_matrix
    - plot_precision_recall_curve
    - plot_partial_dependence

Works with both old and new scikit-learn versions.
"""

import warnings

import sklearn
from packaging import version

SKLEARN_VERSION = version.parse(sklearn.__version__)

# Try to import modern display classes
try:
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
except ImportError:
    RocCurveDisplay = None
    ConfusionMatrixDisplay = None
    PrecisionRecallDisplay = None

try:
    from sklearn.inspection import PartialDependenceDisplay
except ImportError:
    PartialDependenceDisplay = None


def _warn_and_call_new(display_cls, method_name, estimator, *args, **kwargs):
    """Helper for displaying deprecation warning and calling new display API."""
    if display_cls is None:
        raise ImportError(f"Cannot import {method_name} display class from sklearn.")
    if kwargs.get("raise_warnining", False):
        warnings.warn(
            f"{method_name} is removed in scikit-learn>=1.2; " f"using {display_cls.__name__}.from_estimator() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    return display_cls.from_estimator(estimator, *args, **kwargs)


def plot_roc_curve(estimator, X, y, ax=None, **kwargs):
    """Backward-compatible plot_roc_curve."""
    if SKLEARN_VERSION < version.parse("1.2"):
        from sklearn.metrics import plot_roc_curve as _plot_roc_curve

        return _plot_roc_curve(estimator, X, y, ax=ax, **kwargs)
    return _warn_and_call_new(RocCurveDisplay, "plot_roc_curve", estimator, X, y, ax=ax, **kwargs)


def plot_confusion_matrix(estimator, X, y_true, ax=None, **kwargs):
    """Backward-compatible plot_confusion_matrix."""
    if SKLEARN_VERSION < version.parse("1.2"):
        from sklearn.metrics import plot_confusion_matrix as _plot_confusion_matrix

        return _plot_confusion_matrix(estimator, X, y_true, ax=ax, **kwargs)
    return _warn_and_call_new(ConfusionMatrixDisplay, "plot_confusion_matrix", estimator, X, y_true, ax=ax, **kwargs)


def plot_precision_recall_curve(estimator, X, y, ax=None, **kwargs):
    """Backward-compatible plot_precision_recall_curve."""
    if SKLEARN_VERSION < version.parse("1.2"):
        from sklearn.metrics import plot_precision_recall_curve as _plot_precision_recall_curve

        return _plot_precision_recall_curve(estimator, X, y, ax=ax, **kwargs)
    return _warn_and_call_new(PrecisionRecallDisplay, "plot_precision_recall_curve", estimator, X, y, ax=ax, **kwargs)


def plot_partial_dependence(estimator, X, features, ax=None, **kwargs):
    """Backward-compatible plot_partial_dependence."""
    if SKLEARN_VERSION < version.parse("1.2"):
        from sklearn.inspection import plot_partial_dependence as _plot_partial_dependence

        return _plot_partial_dependence(estimator, X, features, ax=ax, **kwargs)
    return _warn_and_call_new(
        PartialDependenceDisplay, "plot_partial_dependence", estimator, X, features=features, ax=ax, **kwargs
    )
