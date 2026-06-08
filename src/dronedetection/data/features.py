"""
Mel spectrogram feature extraction and normalization.

All transforms use torchaudio for GPU-accelerated computation.

Output shape per sample: (1, n_mels, time_frames)
  e.g. (1, 128, 129) for a 3-second clip at 22,050 Hz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from omegaconf import DictConfig

from dronedetection.utils.logging import get_logger

log = get_logger(__name__)


class MelSpectrogramExtractor:
    """
    Converts a raw audio waveform tensor to a log-mel spectrogram.

    Args:
        cfg: The data section of the Hydra config (cfg.data).
        device: Torch device for the transforms.
    """

    def __init__(self, cfg: DictConfig, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self._mel = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=cfg.power,
        ).to(self.device)

    @torch.no_grad()
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (1, num_samples) float32 tensor, mono, peak-normalised.

        Returns:
            (1, n_mels, time_frames) log-mel spectrogram.
        """
        waveform = waveform.to(self.device)
        mel = self._mel(waveform)                          # (1, n_mels, T)
        log_mel = torch.log(mel + self.cfg.log_offset)     # log compression
        return log_mel.cpu()


# ── Peak normalisation ────────────────────────────────────────────────────────

def peak_normalize(waveform: torch.Tensor) -> torch.Tensor:
    """Normalize waveform to [-1, 1] by its peak absolute value."""
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform


# ── Dataset-level statistics (fit on training split) ─────────────────────────

def compute_dataset_stats(
    spec_tensors: list[torch.Tensor],
) -> tuple[float, float]:
    """
    Compute global mean and std over all log-mel spectrogram tensors.
    Used for z-score standardisation at inference time.

    Args:
        spec_tensors: List of (1, n_mels, T) tensors.

    Returns:
        (mean, std) as Python floats.
    """
    all_vals = torch.cat([t.flatten() for t in spec_tensors])
    return float(all_vals.mean()), float(all_vals.std())


def save_stats(mean: float, std: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / "mean.npy"), np.array(mean, dtype=np.float32))
    np.save(str(out_dir / "std.npy"), np.array(std, dtype=np.float32))
    log.info("Saved normalisation stats → %s  (mean=%.4f, std=%.4f)", out_dir, mean, std)


def load_stats(stats_dir: Path) -> tuple[float, float]:
    mean = float(np.load(str(stats_dir / "mean.npy")))
    std = float(np.load(str(stats_dir / "std.npy")))
    return mean, std


def standardize(spec: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Apply z-score standardisation: (x - mean) / std."""
    return (spec - mean) / (std + 1e-9)
