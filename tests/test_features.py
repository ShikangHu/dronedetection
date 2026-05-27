"""Tests for feature extraction pipeline."""
import torch
import pytest
from omegaconf import OmegaConf

from dronedetection.data.features import (
    MelSpectrogramExtractor,
    peak_normalize,
    compute_dataset_stats,
    standardize,
)


@pytest.fixture
def cfg():
    return OmegaConf.load("configs/default.yaml")


def test_peak_normalize_unit_peak():
    waveform = torch.tensor([[0.5, -1.0, 0.3]])
    normed = peak_normalize(waveform)
    assert normed.abs().max().item() == pytest.approx(1.0, abs=1e-5)


def test_peak_normalize_silence():
    waveform = torch.zeros(1, 100)
    normed = peak_normalize(waveform)
    assert normed.abs().max().item() == pytest.approx(0.0, abs=1e-5)


def test_mel_extractor_output_shape(cfg):
    sr = cfg.data.sample_rate
    seg_len = int(sr * cfg.data.segment_duration)
    waveform = torch.randn(1, seg_len)
    extractor = MelSpectrogramExtractor(cfg.data)
    spec = extractor(waveform)
    # Expected: (1, n_mels, T)
    assert spec.shape[0] == 1
    assert spec.shape[1] == cfg.data.n_mels
    assert spec.shape[2] > 0


def test_mel_extractor_no_nan(cfg):
    sr = cfg.data.sample_rate
    seg_len = int(sr * cfg.data.segment_duration)
    waveform = torch.randn(1, seg_len)
    extractor = MelSpectrogramExtractor(cfg.data)
    spec = extractor(waveform)
    assert not torch.isnan(spec).any()


def test_standardize():
    spec = torch.randn(1, 128, 129)
    mean, std = 0.5, 2.0
    out = standardize(spec, mean, std)
    assert out.shape == spec.shape
    expected = (spec - mean) / (std + 1e-9)
    assert torch.allclose(out, expected)


def test_compute_dataset_stats():
    specs = [torch.ones(1, 128, 129) * i for i in range(5)]
    mean, std = compute_dataset_stats(specs)
    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert std >= 0
