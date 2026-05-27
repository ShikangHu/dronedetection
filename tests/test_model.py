"""Tests for model architectures."""
import torch
import pytest
from omegaconf import OmegaConf

from dronedetection.models.efficientnet import EfficientNetB0Classifier
from dronedetection.models.lightweight import LightweightCNN
from dronedetection.models.factory import build_model


@pytest.fixture
def cfg():
    return OmegaConf.load("configs/default.yaml")


@pytest.fixture
def dummy_input(cfg):
    """(batch=2, 1, n_mels, T) log-mel spectrogram."""
    sr = cfg.data.sample_rate
    seg = cfg.data.segment_duration
    hop = cfg.data.hop_length
    T = int(sr * seg) // hop + 1
    return torch.randn(2, 1, cfg.data.n_mels, T)


class TestEfficientNetB0:
    def test_output_shape(self, dummy_input):
        model = EfficientNetB0Classifier(pretrained=False)
        out = model(dummy_input)
        assert out.shape == (2, 1)

    def test_output_is_finite(self, dummy_input):
        model = EfficientNetB0Classifier(pretrained=False)
        out = model(dummy_input)
        assert torch.isfinite(out).all()

    def test_freeze_backbone(self, dummy_input):
        model = EfficientNetB0Classifier(pretrained=False, freeze_backbone=True)
        for p in model.features.parameters():
            assert not p.requires_grad

    def test_unfreeze_backbone(self):
        model = EfficientNetB0Classifier(pretrained=False, freeze_backbone=True)
        model.unfreeze_backbone()
        for p in model.features.parameters():
            assert p.requires_grad

    def test_param_groups(self):
        model = EfficientNetB0Classifier(pretrained=False)
        groups = model.get_param_groups(lr_backbone=1e-4, lr_head=1e-3)
        assert len(groups) == 2
        assert groups[0]["lr"] == 1e-4
        assert groups[1]["lr"] == 1e-3


class TestLightweightCNN:
    def test_output_shape(self, dummy_input):
        model = LightweightCNN()
        out = model(dummy_input)
        assert out.shape == (2, 1)

    def test_output_is_finite(self, dummy_input):
        model = LightweightCNN()
        out = model(dummy_input)
        assert torch.isfinite(out).all()

    def test_param_count(self):
        model = LightweightCNN()
        n = sum(p.numel() for p in model.parameters())
        assert n < 1_000_000, f"Expected < 1M params, got {n:,}"


class TestModelFactory:
    def test_build_efficientnet(self, cfg):
        model = build_model(cfg)
        assert isinstance(model, EfficientNetB0Classifier)

    def test_build_lightweight(self, cfg):
        from omegaconf import OmegaConf
        cfg2 = OmegaConf.merge(cfg, OmegaConf.create({"model": {"name": "lightweight_cnn"}}))
        model = build_model(cfg2)
        assert isinstance(model, LightweightCNN)

    def test_unknown_model_raises(self, cfg):
        from omegaconf import OmegaConf
        cfg2 = OmegaConf.merge(cfg, OmegaConf.create({"model": {"name": "nonexistent"}}))
        with pytest.raises(ValueError):
            build_model(cfg2)
