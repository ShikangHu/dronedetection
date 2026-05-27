"""Tests for audio augmentation pipeline."""
import torch
import pytest
from omegaconf import OmegaConf

from dronedetection.data.augmentation import (
    add_gaussian_noise,
    mix_background,
    spec_augment,
    time_shift,
)


SR = 22050
SEG_LEN = SR * 3  # 3-second segment


def test_time_shift_preserves_shape():
    w = torch.randn(1, SEG_LEN)
    out = time_shift(w, SR, max_ms=500)
    assert out.shape == w.shape


def test_add_gaussian_noise_preserves_shape():
    w = torch.randn(1, SEG_LEN)
    out = add_gaussian_noise(w, snr_min_db=20, snr_max_db=40)
    assert out.shape == w.shape


def test_add_gaussian_noise_is_different():
    w = torch.zeros(1, SEG_LEN)
    out = add_gaussian_noise(w, snr_min_db=20, snr_max_db=20)
    assert not torch.allclose(w, out)


def test_mix_background_preserves_shape():
    w = torch.randn(1, SEG_LEN)
    bg = torch.randn(1, SEG_LEN // 2)  # shorter background
    out = mix_background(w, bg)
    assert out.shape == w.shape


def test_spec_augment_preserves_shape():
    spec = torch.randn(1, 128, 129)
    out = spec_augment(spec, num_freq_masks=2, freq_mask_width=10,
                       num_time_masks=2, time_mask_width=15)
    assert out.shape == spec.shape


def test_spec_augment_zeros_regions():
    spec = torch.ones(1, 128, 129)
    out = spec_augment(spec, num_freq_masks=1, freq_mask_width=10,
                       num_time_masks=1, time_mask_width=10)
    # Some values should be zeroed
    assert (out == 0.0).any()
