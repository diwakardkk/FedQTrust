"""Byzantine detection metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def detection_metrics(is_malicious: np.ndarray, predicted_malicious: np.ndarray, score: np.ndarray | None = None) -> dict[str, float]:
    is_malicious = np.asarray(is_malicious).astype(bool)
    predicted = np.asarray(predicted_malicious).astype(bool)
    tp = int(np.sum(is_malicious & predicted))
    tn = int(np.sum(~is_malicious & ~predicted))
    fp = int(np.sum(~is_malicious & predicted))
    fn = int(np.sum(is_malicious & ~predicted))
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = float("nan")
    if score is not None and len(np.unique(is_malicious)) == 2:
        auc = float(roc_auc_score(is_malicious.astype(int), score))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "TPR": tpr, "FPR": fpr, "precision": precision, "recall": recall, "F1": f1, "ROC_AUC": auc}

