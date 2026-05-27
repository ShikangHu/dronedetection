"""Tests for inference pipeline."""
import tempfile
from pathlib import Path

import soundfile as sf
import numpy as np
import torch
import pytest
from omegaconf import OmegaConf

from dronedetection.inference.predictor import FilePredictor
from dronedetection.models.lightweight import LightweightCNN


@pytest.fixture
def cfg():
    return OmegaConf.load("configs/default.yaml")


@pytest.fixture
def dummy_wav(cfg, tmp_path):
    """Write a 5-second silent WAV file."""
    sr = cfg.data.sample_rate
    audio = np.zeros(sr * 5, dtype=np.float32)
    path = tmp_path / "test.wav"
    sf.write(str(path), audio, sr)
    return path


def test_file_predictor_returns_result(cfg, dummy_wav):
    model = LightweightCNN(dropout=0.0)
    predictor = FilePredictor(model, cfg, stats_dir=None)
    result = predictor.predict(dummy_wav)
    assert "drone_detected" in result
    assert "confidence" in result
    assert "segment_probs" in result
    assert isinstance(result["drone_detected"], bool)
    assert 0.0 <= result["confidence"] <= 1.0


def test_file_predictor_produces_segments(cfg, dummy_wav):
    model = LightweightCNN(dropout=0.0)
    predictor = FilePredictor(model, cfg, stats_dir=None)
    result = predictor.predict(dummy_wav)
    # 5-second clip / 1.5s hop = at least 2 segments
    assert len(result["segment_probs"]) >= 2


def test_file_predictor_short_clip(cfg, tmp_path):
    """A clip shorter than one segment should still return a result."""
    sr = cfg.data.sample_rate
    audio = np.zeros(sr * 2, dtype=np.float32)  # 2s < 3s segment
    path = tmp_path / "short.wav"
    sf.write(str(path), audio, sr)
    model = LightweightCNN(dropout=0.0)
    predictor = FilePredictor(model, cfg, stats_dir=None)
    # Should not raise even though the clip is shorter than one segment
    # (no segments produced → confidence = 0.0)
    result = predictor.predict(path)
    assert isinstance(result, dict)
