"""
Evaluation metrics for drone audio binary classification.

Primary target: recall >= 97% (missed drone = security failure).
Secondary:      precision >= 93% (false alarms erode trust).

All functions accept probabilities (0-1) and binary labels (0/1).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute all evaluation metrics at a given decision threshold.

    Args:
        probs:     (N,) float tensor of predicted probabilities [0, 1].
        labels:    (N,) float tensor of ground-truth labels {0, 1}.
        threshold: Decision threshold for binary prediction.

    Returns:
        Dict with keys: accuracy, precision, recall, f1, roc_auc, pr_auc,
                        tp, fp, tn, fn.
    """
    p = probs.numpy() if isinstance(probs, torch.Tensor) else np.array(probs)
    y = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    preds = (p >= threshold).astype(int)

    cm = confusion_matrix(y, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)

    # ROC-AUC and PR-AUC require probability scores
    try:
        roc_auc = roc_auc_score(y, p)
        pr_auc = average_precision_score(y, p)
    except ValueError:
        roc_auc = 0.0
        pr_auc = 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def find_threshold_for_recall(
    probs: torch.Tensor,
    labels: torch.Tensor,
    min_recall: float = 0.97,
) -> tuple[float, dict[str, float]]:
    """
    Find the lowest decision threshold that achieves at least min_recall,
    while maximising precision subject to that constraint.

    Returns:
        (optimal_threshold, metrics_at_that_threshold)
    """
    p = probs.numpy() if isinstance(probs, torch.Tensor) else np.array(probs)
    y = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)

    precisions, recalls, thresholds = precision_recall_curve(y, p)

    # thresholds has length N-1; precisions/recalls have length N
    best_thresh = 0.5
    best_precision = 0.0
    for prec, rec, thr in zip(precisions[:-1], recalls[:-1], thresholds):
        if rec >= min_recall and prec > best_precision:
            best_precision = prec
            best_thresh = float(thr)

    metrics = compute_metrics(probs, labels, threshold=best_thresh)
    return best_thresh, metrics


def print_metrics(metrics: dict[str, float], header: str = "") -> None:
    if header:
        print(f"\n{'─'*50}")
        print(f"  {header}")
        print(f"{'─'*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}  ← primary target (>0.97)")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC    : {metrics['pr_auc']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")
