from typing import Dict, List

from .base import MetricFn

# ---------------------------------------------------------------------------
# Standard classification metrics (sklearn-backed)
# ---------------------------------------------------------------------------

def _accuracy(y_true: List[int], y_pred: List[int], **kwargs) -> float:
    n = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return round(correct / n, 4) if n else 0.0


def _auc(y_true: List[int], y_score: List[float], **kwargs) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return round(float(roc_auc_score(y_true, y_score)), 4)
    except ValueError:
        return float("nan")


def _f1(y_true: List[int], y_pred: List[int], **kwargs) -> float:
    from sklearn.metrics import f1_score
    yp = [p if p in (0, 1) else (1 - t) for t, p in zip(y_true, y_pred)]
    return round(float(f1_score(y_true, yp, zero_division=0)), 4)


def _avg_precision(y_true: List[int], y_score: List[float], **kwargs) -> float:
    from sklearn.metrics import average_precision_score
    try:
        return round(float(average_precision_score(y_true, y_score)), 4)
    except ValueError:
        return float("nan")


accuracy = MetricFn(name="Accuracy", fn=_accuracy)
auc = MetricFn(name="AUC", fn=_auc)
f1 = MetricFn(name="F1", fn=_f1)
avg_precision = MetricFn(name="Avg Precision", fn=_avg_precision)
