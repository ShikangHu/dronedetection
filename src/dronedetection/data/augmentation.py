"""
Audio augmentation pipeline.

Two stages:
  1. Waveform-level augmentations (applied before feature extraction)
  2. Spectrogram-level augmentations / SpecAugment (applied on the log-mel tensor)

All augmentations are probabilistic and only applied during training.
"""
from __future__ import annotations

import random

import torch
import torchaudio.functional as F
from omegaconf import DictConfig


# ── Waveform augmentations ────────────────────────────────────────────────────

def time_shift(waveform: torch.Tensor, sample_rate: int, max_ms: float) -> torch.Tensor:
    """Randomly shift the waveform left or right by up to max_ms milliseconds."""
    max_samples = int(sample_rate * max_ms / 1000)
    shift = random.randint(-max_samples, max_samples)
    return torch.roll(waveform, shift, dims=-1)


def add_gaussian_noise(
    waveform: torch.Tensor,
    snr_min_db: float = 20.0,
    snr_max_db: float = 40.0,
) -> torch.Tensor:
    """Add Gaussian noise at a random SNR (dB) within [snr_min_db, snr_max_db]."""
    snr_db = random.uniform(snr_min_db, snr_max_db)
    snr_linear = 10 ** (snr_db / 20.0)
    signal_power = waveform.norm(p=2)
    noise = torch.randn_like(waveform)
    noise_power = noise.norm(p=2)
    if noise_power > 0:
        noise = noise * (signal_power / (snr_linear * noise_power + 1e-9))
    return waveform + noise


def mix_background(
    waveform: torch.Tensor,
    background: torch.Tensor,
    snr_min_db: float = 5.0,
    snr_max_db: float = 20.0,
) -> torch.Tensor:
    """
    Mix a background clip into the waveform at a random SNR.
    The background is repeated/trimmed to match waveform length.
    """
    target_len = waveform.shape[-1]
    bg = background
    # Tile or trim background to match length
    if bg.shape[-1] < target_len:
        repeats = (target_len // bg.shape[-1]) + 1
        bg = bg.repeat(1, repeats)[..., :target_len]
    else:
        start = random.randint(0, bg.shape[-1] - target_len)
        bg = bg[..., start:start + target_len]

    snr_db = random.uniform(snr_min_db, snr_max_db)
    snr_linear = 10 ** (snr_db / 20.0)
    signal_power = waveform.norm(p=2)
    bg_power = bg.norm(p=2)
    if bg_power > 0:
        bg = bg * (signal_power / (snr_linear * bg_power + 1e-9))
    return waveform + bg


def pitch_shift(
    waveform: torch.Tensor,
    sample_rate: int,
    semitones: float,
) -> torch.Tensor:
    """Shift pitch by a random number of semitones in [-semitones, +semitones]."""
    n = random.uniform(-semitones, semitones)
    return F.pitch_shift(waveform, sample_rate, n_steps=n)


def time_stretch(
    waveform: torch.Tensor,
    sample_rate: int,
    rate_min: float = 0.9,
    rate_max: float = 1.1,
    target_length: int | None = None,
) -> torch.Tensor:
    """Stretch time by a random rate, then pad/trim back to original length."""
    rate = random.uniform(rate_min, rate_max)
    stretched = F.resample(waveform, int(sample_rate * rate), sample_rate)
    orig_len = target_length or waveform.shape[-1]
    if stretched.shape[-1] >= orig_len:
        return stretched[..., :orig_len]
    # Pad with zeros
    pad = torch.zeros(*stretched.shape[:-1], orig_len - stretched.shape[-1])
    return torch.cat([stretched, pad], dim=-1)


# ── SpecAugment ───────────────────────────────────────────────────────────────

def spec_augment(
    spec: torch.Tensor,
    num_freq_masks: int = 2,
    freq_mask_width: int = 20,
    num_time_masks: int = 2,
    time_mask_width: int = 30,
) -> torch.Tensor:
    """
    Apply SpecAugment (frequency and time masking) to a log-mel spectrogram.

    Args:
        spec: (1, n_mels, T) tensor.

    Returns:
        Augmented spectrogram of the same shape.
    """
    spec = spec.clone()
    _, n_mels, T = spec.shape

    for _ in range(num_freq_masks):
        w = random.randint(0, freq_mask_width)
        f0 = random.randint(0, max(0, n_mels - w))
        spec[:, f0:f0 + w, :] = 0.0

    for _ in range(num_time_masks):
        w = random.randint(0, time_mask_width)
        t0 = random.randint(0, max(0, T - w))
        spec[:, :, t0:t0 + w] = 0.0

    return spec


# ── Composed pipeline ─────────────────────────────────────────────────────────

class WaveformAugmenter:
    """
    Applies a stochastic sequence of waveform-level augmentations.

    background_pool: list of (1, samples) tensors used for background mixing.
                     Can be empty (background mixing will be skipped).
    """

    def __init__(self, cfg: DictConfig, background_pool: list[torch.Tensor] | None = None):
        self.cfg = cfg
        self.bg_pool = background_pool or []
        self.sample_rate = cfg.data.sample_rate

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        aug = self.cfg.augmentation
        if not aug.enabled:
            return waveform

        if random.random() < aug.prob_time_shift:
            waveform = time_shift(waveform, self.sample_rate, aug.time_shift_ms)

        if random.random() < aug.prob_noise:
            waveform = add_gaussian_noise(waveform, aug.gaussian_snr_min, aug.gaussian_snr_max)

        if self.bg_pool and random.random() < aug.prob_background:
            bg = random.choice(self.bg_pool)
            waveform = mix_background(waveform, bg, aug.background_snr_min, aug.background_snr_max)

        if random.random() < aug.prob_time_stretch:
            waveform = time_stretch(waveform, self.sample_rate,
                                    aug.time_stretch_min, aug.time_stretch_max,
                                    target_length=waveform.shape[-1])

        if random.random() < aug.prob_pitch_shift:
            waveform = pitch_shift(waveform, self.sample_rate, aug.pitch_shift_semitones)

        return waveform


class SpectrogramAugmenter:
    """Applies SpecAugment to a log-mel spectrogram tensor."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        aug = self.cfg.augmentation
        if not aug.enabled:
            return spec
        if random.random() < aug.prob_spec_augment:
            spec = spec_augment(
                spec,
                num_freq_masks=aug.spec_freq_num_masks,
                freq_mask_width=aug.spec_freq_mask_width,
                num_time_masks=aug.spec_time_num_masks,
                time_mask_width=aug.spec_time_mask_width,
            )
        return spec
