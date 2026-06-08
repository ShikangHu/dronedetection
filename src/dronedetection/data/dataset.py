"""
DroneAudioDataset — PyTorch Dataset for drone audio binary classification.

Pipeline per sample
───────────────────
  1. Load raw audio file, resample to cfg.data.sample_rate
  2. Convert to mono, peak-normalize
  3. Segment into fixed-length chunks with overlap (done at build time)
  4. (Training) Apply waveform augmentations
  5. Extract log-mel spectrogram
  6. (Training) Apply SpecAugment
  7. Standardize with pre-computed mean/std

Label encoding
──────────────
  1 = drone, 0 = no-drone

CSV manifest format (data/splits/{train,val,test}.csv)
──────────────────────────────────────────────────────
  path, label, source, split
  data/raw/droneaudioset/drone_000001.wav, 1, droneaudioset, train
  ...
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import soundfile as sf
import pandas as pd
import torch
import torchaudio
from omegaconf import DictConfig
from torch.utils.data import Dataset, WeightedRandomSampler

from dronedetection.data.augmentation import SpectrogramAugmenter, WaveformAugmenter
from dronedetection.data.features import MelSpectrogramExtractor, peak_normalize, standardize
from dronedetection.utils.logging import get_logger

log = get_logger(__name__)


# ── Segmentation ──────────────────────────────────────────────────────────────

def segment_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    segment_duration: float,
    overlap: float,
) -> list[torch.Tensor]:
    """
    Split a waveform into fixed-length overlapping segments.

    Args:
        waveform: (1, num_samples)
        sample_rate: target sample rate
        segment_duration: seconds per segment
        overlap: fractional overlap [0, 1)

    Returns:
        List of (1, segment_samples) tensors.
    """
    seg_len = int(sample_rate * segment_duration)
    hop = int(seg_len * (1 - overlap))
    num_samples = waveform.shape[-1]
    segments = []
    start = 0
    while start + seg_len <= num_samples:
        segments.append(waveform[:, start:start + seg_len])
        start += hop
    return segments


# ── Manifest builder ──────────────────────────────────────────────────────────

def _is_silent(waveform: torch.Tensor, min_rms_db: float = -60.0) -> bool:
    rms = waveform.norm(p=2) / (waveform.shape[-1] ** 0.5 + 1e-9)
    rms_db = 20 * torch.log10(rms + 1e-9)
    return rms_db.item() < min_rms_db


def build_manifest(
    raw_dir: Path,
    splits_dir: Path,
    cfg: DictConfig,
    drone_dirs: list[str] | None = None,
    no_drone_dirs: list[str] | None = None,
) -> None:
    """
    Scan raw_dir for audio files, quality-filter, and write
    train.csv / val.csv / test.csv to splits_dir.

    Splits are performed at the recording level (not segment level)
    to prevent data leakage.

    Args:
        raw_dir: Root of downloaded datasets.
        splits_dir: Directory where CSVs will be written.
        cfg: Full hydra config.
        drone_dirs: Subdirectory names in raw_dir that contain drone audio.
        no_drone_dirs: Subdirectory names in raw_dir that contain non-drone audio.
    """
    import random as _random
    _random.seed(cfg.data.random_seed)

    drone_dirs = drone_dirs or ["droneaudioset", "dads", "alemadi"]
    no_drone_dirs = no_drone_dirs or ["esc50", "urbansound8k", "fsd50k"]

    AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
    sr = cfg.data.sample_rate
    seg_dur = cfg.data.segment_duration
    overlap = cfg.data.overlap
    train_ratio = cfg.data.train_ratio
    val_ratio = cfg.data.val_ratio

    records: list[dict] = []

    def _collect(dirs: list[str], label: int) -> None:
        for d in dirs:
            src_dir = raw_dir / d
            if not src_dir.exists():
                log.warning("Directory not found, skipping: %s", src_dir)
                continue
            files = [f for f in src_dir.rglob("*") if f.suffix.lower() in AUDIO_EXTS]
            log.info("Found %d audio files in %s (label=%d)", len(files), d, label)

            # Split at the recording level
            _random.shuffle(files)
            n = len(files)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            for i, fpath in enumerate(files):
                split = "train" if i < train_end else ("val" if i < val_end else "test")
                try:
                    import soundfile as _sf
                    info = _sf.info(str(fpath))
                except Exception as e:
                    log.warning("Skipping corrupt file %s: %s", fpath, e)
                    continue
                dur = info.frames / info.samplerate
                if dur < 1.0:
                    continue  # too short
                records.append({
                    "path": str(fpath),
                    "label": label,
                    "source": d,
                    "split": split,
                })

    _collect(drone_dirs, label=1)
    _collect(no_drone_dirs, label=0)

    splits_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in records:
        splits[r["split"]].append(r)

    for split_name, rows in splits.items():
        out = splits_dir / f"{split_name}.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "split"])
            writer.writeheader()
            writer.writerows(rows)
        log.info("Wrote %d records → %s", len(rows), out)


# ── Dataset ───────────────────────────────────────────────────────────────────

class DroneAudioDataset(Dataset):
    """
    PyTorch Dataset for drone audio binary classification.

    Each item is a (spec, label) pair where:
      spec:  (1, n_mels, T) standardised log-mel spectrogram
      label: scalar float tensor (1.0 or 0.0)
    """

    def __init__(
        self,
        manifest_csv: Path,
        cfg: DictConfig,
        split: str = "train",
        stats: tuple[float, float] | None = None,
        waveform_augmenter: WaveformAugmenter | None = None,
        spec_augmenter: SpectrogramAugmenter | None = None,
    ):
        self.cfg = cfg
        self.split = split
        self.stats = stats  # (mean, std) from training split
        self.waveform_augmenter = waveform_augmenter
        self.spec_augmenter = spec_augmenter
        self.extractor = MelSpectrogramExtractor(cfg.data)

        df = pd.read_csv(manifest_csv)
        self.records = df.to_dict("records")
        log.info("Loaded %d records from %s (split=%s)", len(self.records), manifest_csv, split)

        # Pre-segment: build a flat list of (path, label, seg_idx)
        self.segments: list[tuple[str, int, int]] = []
        sr = cfg.data.sample_rate
        seg_dur = cfg.data.segment_duration
        overlap = cfg.data.overlap
        seg_len = int(sr * seg_dur)
        hop = int(seg_len * (1 - overlap))

        for rec in self.records:
            try:
                info = sf.info(str(rec["path"]))
            except Exception:
                continue
            orig_len = info.frames
            orig_sr = info.samplerate
            resampled_len = int(orig_len * sr / orig_sr)
            n_segs = max(1, (resampled_len - seg_len) // hop + 1)
            for i in range(n_segs):
                self.segments.append((rec["path"], rec["label"], i, orig_sr))

        log.info("Total segments for split=%s: %d", split, len(self.segments))

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label, seg_idx, orig_sr = self.segments[idx]
        cfg_d = self.cfg.data

        seg_len = int(cfg_d.sample_rate * cfg_d.segment_duration)
        hop = int(seg_len * (1 - cfg_d.overlap))
        start_sample = seg_idx * hop

        # Load only the required slice (efficient for large files)
        frame_offset = max(0, int(start_sample * (orig_sr / cfg_d.sample_rate)))
        num_frames = int(seg_len * (orig_sr / cfg_d.sample_rate) + 1)

        audio, orig_sr = sf.read(path, start=frame_offset, stop=frame_offset + num_frames, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.T)  # (channels, frames)

        # Resample
        if orig_sr != cfg_d.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_sr, cfg_d.sample_rate)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Tile or trim to exact segment length (tiling avoids silence-padding artifacts
        # for short clips like the 0.7s Kaggle dataset files)
        if waveform.shape[-1] < seg_len:
            repeats = -(-seg_len // waveform.shape[-1])  # ceiling division
            waveform = waveform.repeat(1, repeats)[:, :seg_len]
        else:
            waveform = waveform[:, :seg_len]

        # Peak normalize
        waveform = peak_normalize(waveform)

        # Waveform augmentation (train only)
        if self.waveform_augmenter is not None and self.split == "train":
            waveform = self.waveform_augmenter(waveform)

        # Extract log-mel spectrogram
        spec = self.extractor(waveform)

        # Spectrogram augmentation (train only)
        if self.spec_augmenter is not None and self.split == "train":
            spec = self.spec_augmenter(spec)

        # Standardize
        if self.stats is not None:
            spec = standardize(spec, *self.stats)

        return spec, torch.tensor(float(label))


# ── Sampler ───────────────────────────────────────────────────────────────────

def make_weighted_sampler(dataset: DroneAudioDataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler for balanced class sampling."""
    labels = [dataset.segments[i][1] for i in range(len(dataset))]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    w_pos = 1.0 / (n_pos + 1e-9)
    w_neg = 1.0 / (n_neg + 1e-9)
    weights = [w_pos if l == 1 else w_neg for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
