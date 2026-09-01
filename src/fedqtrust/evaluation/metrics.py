"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["auc_roc"] = float("nan")
    else:
        metrics["auc_roc"] = float("nan")
    return metrics

