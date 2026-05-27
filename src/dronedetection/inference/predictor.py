"""
Inference module — single-file prediction and real-time streaming.

Two modes
─────────
1. FilePredictor:   classify a WAV/FLAC/MP3 file (sliding window + vote)
2. StreamPredictor: classify a live microphone stream in near-real-time

Both use the same core pipeline:
  load/capture → resample → segment → extract features → model → smooth → threshold
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import torch
import torchaudio

from dronedetection.data.features import (
    MelSpectrogramExtractor,
    load_stats,
    peak_normalize,
    standardize,
)
from dronedetection.utils.logging import get_logger

log = get_logger(__name__)


class _BasePredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg,
        stats_dir: Path | None = None,
        device: str | torch.device | None = None,
    ):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.extractor = MelSpectrogramExtractor(cfg.data)
        self.threshold = cfg.evaluation.threshold

        self.mean: float | None = None
        self.std: float | None = None
        if stats_dir is not None:
            self.mean, self.std = load_stats(Path(stats_dir))

        self._seg_len = int(cfg.data.sample_rate * cfg.data.segment_duration)
        self._smoothing = cfg.inference.smoothing_window
        self._recent: deque[float] = deque(maxlen=self._smoothing)

    @torch.no_grad()
    def _predict_segment(self, waveform: torch.Tensor) -> float:
        """Return probability of 'drone' for one segment."""
        waveform = peak_normalize(waveform)
        spec = self.extractor(waveform)
        if self.mean is not None:
            spec = standardize(spec, self.mean, self.std)
        spec = spec.unsqueeze(0).to(self.device)  # (1, 1, n_mels, T)
        with torch.cuda.amp.autocast(enabled=self.cfg.training.use_amp):
            logit = self.model(spec)
        prob = torch.sigmoid(logit).item()
        self._recent.append(prob)
        return prob

    def _smooth_prediction(self) -> float:
        """Return median of the recent prediction window."""
        if not self._recent:
            return 0.0
        sorted_r = sorted(self._recent)
        n = len(sorted_r)
        if n % 2 == 0:
            return (sorted_r[n // 2 - 1] + sorted_r[n // 2]) / 2
        return sorted_r[n // 2]


class FilePredictor(_BasePredictor):
    """
    Classify an audio file using a sliding window.

    Returns:
        {
            "drone_detected": bool,
            "confidence": float,          # smoothed probability
            "segment_probs": list[float], # per-segment probabilities
        }
    """

    def predict(self, audio_path: str | Path) -> dict:
        waveform, sr = torchaudio.load(str(audio_path))
        if sr != self.cfg.data.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.cfg.data.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        hop = int(self.cfg.inference.step_duration * self.cfg.data.sample_rate)
        seg_len = self._seg_len
        probs: list[float] = []
        start = 0

        while start + seg_len <= waveform.shape[-1]:
            seg = waveform[:, start:start + seg_len]
            p = self._predict_segment(seg)
            probs.append(p)
            start += hop

        smoothed = self._smooth_prediction()
        return {
            "drone_detected": smoothed >= self.threshold,
            "confidence": smoothed,
            "segment_probs": probs,
        }


class StreamPredictor(_BasePredictor):
    """
    Real-time streaming predictor using sounddevice.

    Usage
    ─────
        predictor = StreamPredictor(model, cfg, stats_dir="checkpoints")
        predictor.start()  # blocks; Ctrl-C to stop
    """

    def start(self, on_detection: callable | None = None) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            raise RuntimeError("Install sounddevice: pip install sounddevice")

        sr = self.cfg.data.sample_rate
        seg_samples = self._seg_len
        hop_samples = int(self.cfg.inference.step_duration * sr)
        buffer = torch.zeros(1, seg_samples)

        log.info("Starting real-time stream at %d Hz. Ctrl-C to stop.", sr)

        def callback(indata, frames, time, status):
            nonlocal buffer
            chunk = torch.from_numpy(indata[:, 0]).unsqueeze(0)  # (1, frames)
            buffer = torch.roll(buffer, -frames, dims=-1)
            buffer[:, -frames:] = chunk
            prob = self._predict_segment(buffer.clone())
            smoothed = self._smooth_prediction()
            detected = smoothed >= self.threshold
            label = "[DRONE]" if detected else "[clear]"
            print(f"\r{label}  prob={smoothed:.3f}  raw={prob:.3f}  ", end="", flush=True)
            if detected and on_detection:
                on_detection(smoothed)

        try:
            with sd.InputStream(
                samplerate=sr,
                channels=1,
                blocksize=hop_samples,
                callback=callback,
                dtype="float32",
            ):
                while True:
                    sd.sleep(1000)
        except KeyboardInterrupt:
            log.info("Stream stopped.")
