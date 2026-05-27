"""
Run inference on a single audio file using a trained checkpoint.

Segments the file into overlapping windows (matching training config),
runs each segment through the model, and reports per-segment predictions
plus an overall verdict.

Usage
─────
  python3.10 scripts/infer_file.py --audio /path/to/file.wav
  python3.10 scripts/infer_file.py --audio /path/to/file.wav --checkpoint checkpoints/best_model_v3.pt
  python3.10 scripts/infer_file.py --audio /path/to/file.wav --threshold 0.988
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import soundfile as sf
import torch
import torchaudio
from omegaconf import OmegaConf

from dronedetection.data.features import MelSpectrogramExtractor, load_stats, peak_normalize, standardize
from dronedetection.models.factory import build_model

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model_v3.pt")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (default: from config)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    threshold = args.threshold if args.threshold is not None else cfg.evaluation.threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load normalization stats
    stats_dir = Path(cfg.paths.checkpoints)
    mean, std = load_stats(stats_dir)

    # Load model
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model = model.to(device).eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load audio
    audio_path = Path(args.audio)
    info = sf.info(str(audio_path))
    orig_sr = info.samplerate
    duration = info.frames / orig_sr
    print(f"\nFile: {audio_path.name}")
    print(f"  Duration:    {duration:.2f}s")
    print(f"  Sample rate: {orig_sr} Hz")
    print(f"  Channels:    {info.channels}")

    audio, orig_sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T)  # (channels, frames)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    target_sr = cfg.data.sample_rate
    if orig_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, target_sr)
        print(f"  Resampled to {target_sr} Hz")

    # Segment
    seg_len = int(target_sr * cfg.data.segment_duration)
    hop = int(seg_len * (1 - cfg.data.overlap))
    total_samples = waveform.shape[-1]
    n_segs = max(1, (total_samples - seg_len) // hop + 1)
    print(f"\nSegmenting: {n_segs} segments ({cfg.data.segment_duration}s each, {cfg.data.overlap*100:.0f}% overlap)")
    print(f"Threshold: {threshold}")

    extractor = MelSpectrogramExtractor(cfg.data)

    seg_results: list[dict] = []
    batch_specs: list[torch.Tensor] = []
    batch_info: list[dict] = []

    def run_batch(specs_list, infos_list):
        batch = torch.stack(specs_list).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg.training.use_amp and device.type == "cuda"):
                logits = model(batch)
        probs = torch.sigmoid(logits).cpu().squeeze(1).tolist()
        for info_d, prob in zip(infos_list, probs):
            info_d["prob"] = prob
            info_d["pred"] = int(prob >= threshold)
            seg_results.append(info_d)

    BATCH_SIZE = 64
    for i in range(n_segs):
        start = i * hop
        chunk = waveform[:, start:start + seg_len]

        # Tile if short (matches training behaviour)
        if chunk.shape[-1] < seg_len:
            repeats = -(-seg_len // chunk.shape[-1])
            chunk = chunk.repeat(1, repeats)[:, :seg_len]

        chunk = peak_normalize(chunk)
        spec = extractor(chunk)
        spec = standardize(spec, mean, std)

        t_start = start / target_sr
        t_end = min(t_start + cfg.data.segment_duration, duration)
        batch_specs.append(spec)
        batch_info.append({"seg": i, "t_start": t_start, "t_end": t_end})

        if len(batch_specs) == BATCH_SIZE:
            run_batch(batch_specs, batch_info)
            batch_specs, batch_info = [], []

    if batch_specs:
        run_batch(batch_specs, batch_info)

    # ── Results ──────────────────────────────────────────────────────────────
    probs = [r["prob"] for r in seg_results]
    preds = [r["pred"] for r in seg_results]
    n_drone = sum(preds)
    n_total = len(preds)
    drone_frac = n_drone / n_total if n_total else 0

    print(f"\n{'─'*60}")
    print(f"RESULTS: {n_drone}/{n_total} segments classified as DRONE ({drone_frac*100:.1f}%)")
    print(f"  Mean confidence:   {np.mean(probs):.4f}")
    print(f"  Median confidence: {np.median(probs):.4f}")
    print(f"  Max confidence:    {np.max(probs):.4f}")
    print(f"  Min confidence:    {np.min(probs):.4f}")

    if drone_frac >= 0.5:
        print(f"\n  VERDICT: DRONE DETECTED")
    else:
        print(f"\n  VERDICT: NO DRONE DETECTED")

    # Per-segment detail (show drone-flagged segments and some no-drone)
    drone_segs = [r for r in seg_results if r["pred"] == 1]
    no_drone_segs = [r for r in seg_results if r["pred"] == 0]

    if drone_segs:
        print(f"\n── Drone-flagged segments ({len(drone_segs)}) ──")
        for r in drone_segs[:30]:
            print(f"  [{r['t_start']:7.2f}s – {r['t_end']:7.2f}s]  conf={r['prob']:.4f}  DRONE")
        if len(drone_segs) > 30:
            print(f"  ... and {len(drone_segs)-30} more")

    # Timeline: show confidence every 10 seconds
    print(f"\n── Confidence timeline (10s bins) ──")
    bin_size = 10.0
    n_bins = int(duration / bin_size) + 1
    for b in range(n_bins):
        t0, t1 = b * bin_size, (b + 1) * bin_size
        bin_segs = [r for r in seg_results if r["t_start"] >= t0 and r["t_start"] < t1]
        if not bin_segs:
            continue
        bin_probs = [r["prob"] for r in bin_segs]
        bin_drone = sum(r["pred"] for r in bin_segs)
        bar = "█" * int(np.mean(bin_probs) * 20)
        status = "DRONE" if bin_drone > 0 else "     "
        print(f"  {t0:6.0f}s-{t1:6.0f}s  {status}  {bar:<20}  mean_conf={np.mean(bin_probs):.3f}  ({bin_drone}/{len(bin_segs)} segs)")


if __name__ == "__main__":
    main()
