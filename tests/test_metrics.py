"""Tests for evaluation metrics."""
import torch
import pytest

from dronedetection.evaluation.metrics import compute_metrics, find_threshold_for_recall


def test_perfect_classifier():
    probs = torch.tensor([0.9, 0.8, 0.1, 0.05])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    m = compute_metrics(probs, labels, threshold=0.5)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["tn"] == 2
    assert m["fn"] == 0


def test_all_wrong():
    probs = torch.tensor([0.1, 0.05, 0.9, 0.8])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    m = compute_metrics(probs, labels, threshold=0.5)
    assert m["tp"] == 0
    assert m["fn"] == 2
    assert m["fp"] == 2
    assert m["tn"] == 0


def test_recall_prioritised_threshold():
    # 10 drones, 10 non-drones
    probs = torch.cat([torch.linspace(0.4, 0.9, 10), torch.linspace(0.1, 0.5, 10)])
    labels = torch.cat([torch.ones(10), torch.zeros(10)])
    best_thresh, metrics = find_threshold_for_recall(probs, labels, min_recall=0.9)
    assert metrics["recall"] >= 0.9, f"Recall {metrics['recall']} < 0.9"
    assert 0.0 <= best_thresh <= 1.0


def test_metrics_keys():
    probs = torch.rand(20)
    labels = (torch.rand(20) > 0.5).float()
    m = compute_metrics(probs, labels)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "tp", "fp", "tn", "fn"):
        assert key in m


def test_roc_auc_range():
    probs = torch.rand(50)
    labels = (torch.rand(50) > 0.5).float()
    m = compute_metrics(probs, labels)
    assert 0.0 <= m["roc_auc"] <= 1.0
