"""Detect the 4 sync claps in two play videos and compute the camera-to-camera offset.

Workflow:
    1. Extract the first ~60s of audio from each video (mono, 48 kHz) via ffmpeg.
    2. Compute short-time energy and find the 4 highest peaks in the 15–40 s window.
    3. Refine each peak time to sub-sample precision via parabolic interpolation.
    4. Match claps 1:1 across A and B (sorted by time), compute mean offset + dispersion.
    5. Use nominal fps ratio (e.g. 60.00 / 59.94 = 1.001001) for the drift slope —
       all 4 claps are clustered in the first 40 s so empirical drift cannot be measured.
    6. Save sync.json + a diagnostic PNG.

Usage:
    uv run python scripts/detect_claps.py \
        --video-a /path/to/A_play.MOV --fps-a 59.94005994 \
        --video-b /path/to/B_play.MOV --fps-b 60.0 \
        --window-start 15 --window-end 45 \
        --output data/sessions/2026-06-02_1on1/sync.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, find_peaks, sosfiltfilt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_audio(video: Path, out_wav: Path, duration_s: float) -> None:
    """Extract first `duration_s` seconds of audio to mono 48 kHz wav via ffmpeg."""
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-t", str(duration_s),
        "-i", str(video),
        "-ac", "1", "-ar", "48000", "-vn",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)


def short_time_energy(audio: np.ndarray, sr: int, win_ms: float = 20.0) -> np.ndarray:
    """Convolve squared-audio with a uniform window to get short-time energy."""
    win = max(1, int(round(win_ms * 1e-3 * sr)))
    return np.convolve(audio.astype(np.float64) ** 2, np.ones(win) / win, mode="same")


def parabolic_refine(y: np.ndarray, peak_idx: int) -> float:
    """Return a sub-sample-refined peak position via 3-point parabolic interpolation."""
    if peak_idx <= 0 or peak_idx >= len(y) - 1:
        return float(peak_idx)
    y0, y1, y2 = y[peak_idx - 1], y[peak_idx], y[peak_idx + 1]
    denom = (y0 - 2 * y1 + y2)
    if denom == 0:
        return float(peak_idx)
    delta = 0.5 * (y0 - y2) / denom
    return float(peak_idx + delta)


def detect_claps(
    wav_path: Path, window_start_s: float, window_end_s: float, expected_n: int = 4
) -> tuple[list[float], np.ndarray, int]:
    """Return (clap_times_s, energy, sr) for the top-N peaks in [window_start, window_end]."""
    sr, audio = wavfile.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32) / max(1.0, float(np.iinfo(audio.dtype).max if np.issubdtype(audio.dtype, np.integer) else 1.0))

    # High-pass 500 Hz to suppress HVAC/ambient rumble; clap energy is broadband but sharp transient lives ≥ 1 kHz.
    sos = butter(N=4, Wn=500.0, btype="highpass", fs=sr, output="sos")
    audio_hp = sosfiltfilt(sos, audio)

    energy = short_time_energy(audio_hp, sr, win_ms=20.0)

    # Peak search: well-separated (≥ 0.25 s apart), above a relative threshold.
    rel_thr = 0.05 * float(np.percentile(energy, 99.5))
    peaks, props = find_peaks(energy, height=rel_thr, distance=int(0.25 * sr))

    # Restrict to the requested window and pick the top `expected_n` by height.
    s_start, s_end = int(window_start_s * sr), int(window_end_s * sr)
    mask = (peaks >= s_start) & (peaks <= s_end)
    cand_peaks = peaks[mask]
    cand_heights = props["peak_heights"][mask]
    if len(cand_peaks) < expected_n:
        raise RuntimeError(
            f"Only {len(cand_peaks)} peaks found in [{window_start_s},{window_end_s}] s — expected {expected_n}. "
            f"Try widening the window, lowering rel_thr, or inspect the diagnostic plot."
        )
    top_order = np.argsort(cand_heights)[-expected_n:]
    top_peaks = sorted(cand_peaks[top_order].tolist())  # chronological

    clap_times = [parabolic_refine(energy, int(pk)) / sr for pk in top_peaks]
    return clap_times, energy, sr


def diag_plot(
    out_png: Path,
    energy_a: np.ndarray, sr_a: int, claps_a: list[float],
    energy_b: np.ndarray, sr_b: int, claps_b: list[float],
    window_start_s: float, window_end_s: float,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, name, energy, sr, claps in [
        (axes[0], "A_play", energy_a, sr_a, claps_a),
        (axes[1], "B_play", energy_b, sr_b, claps_b),
    ]:
        t = np.arange(len(energy)) / sr
        ax.plot(t, energy, lw=0.6, color="steelblue")
        for ci, ct in enumerate(claps):
            ax.axvline(ct, color="red", lw=0.8, alpha=0.7)
            ax.text(ct, ax.get_ylim()[1] * 0.95, f"#{ci+1}\n{ct:.3f}s",
                    color="red", fontsize=8, ha="center", va="top")
        ax.axvspan(window_start_s, window_end_s, alpha=0.1, color="green")
        ax.set_ylabel(f"{name} energy")
    axes[1].set_xlabel("time (s)")
    axes[0].set_title("Detected sync claps (red lines) — search window in green")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video-a", required=True, type=Path)
    p.add_argument("--video-b", required=True, type=Path)
    p.add_argument("--fps-a", type=float, default=60000 / 1001)
    p.add_argument("--fps-b", type=float, default=60.0)
    p.add_argument("--window-start", type=float, default=10.0, help="search window start (s)")
    p.add_argument("--window-end", type=float, default=45.0, help="search window end (s)")
    p.add_argument("--n-claps", type=int, default=4)
    p.add_argument("--audio-duration", type=float, default=60.0,
                   help="duration to extract from each video for analysis (s)")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Extracting audio from {args.video_a.name} and {args.video_b.name}...")
    with tempfile.TemporaryDirectory() as td:
        wav_a = Path(td) / "a.wav"
        wav_b = Path(td) / "b.wav"
        extract_audio(args.video_a, wav_a, args.audio_duration)
        extract_audio(args.video_b, wav_b, args.audio_duration)

        print(f"Detecting {args.n_claps} claps in [{args.window_start}, {args.window_end}] s on each...")
        claps_a, energy_a, sr_a = detect_claps(wav_a, args.window_start, args.window_end, args.n_claps)
        claps_b, energy_b, sr_b = detect_claps(wav_b, args.window_start, args.window_end, args.n_claps)
        print(f"  A claps (s): {[round(t,4) for t in claps_a]}")
        print(f"  B claps (s): {[round(t,4) for t in claps_b]}")

        diag_png = args.output.with_suffix(".diag.png")
        diag_plot(diag_png, energy_a, sr_a, claps_a, energy_b, sr_b, claps_b,
                  args.window_start, args.window_end)
        print(f"  ✓ Diagnostic plot → {diag_png}")

    # Pairwise offsets — sorted, so claps_a[i] should correspond to claps_b[i].
    diffs = np.array(claps_b) - np.array(claps_a)
    offset_mean = float(np.mean(diffs))
    offset_std = float(np.std(diffs))
    offset_max_disagreement = float(np.max(diffs) - np.min(diffs))
    print(f"  B_minus_A offsets per clap (s): {[round(d,4) for d in diffs.tolist()]}")
    print(f"  Mean offset: {offset_mean:.4f} s  (std {offset_std:.4f} s, range {offset_max_disagreement:.4f} s)")

    # Drift slope = ratio of B's seconds to A's seconds.
    drift_slope_nominal = args.fps_a / args.fps_b
    # i.e. for every 1.0 s on A's clock, B's clock advances drift_slope * 1.0 s
    # If fps_a < fps_b then B's clock advances faster -> drift_slope > 1.

    sync = {
        "anchor_camera": "A",
        "video_a": str(args.video_a),
        "video_b": str(args.video_b),
        "fps_a": args.fps_a,
        "fps_b": args.fps_b,
        "search_window_s": [args.window_start, args.window_end],
        "claps_a_s": [round(t, 6) for t in claps_a],
        "claps_b_s": [round(t, 6) for t in claps_b],
        "per_clap_offset_b_minus_a_s": [round(d, 6) for d in diffs.tolist()],
        "offset_b_minus_a_s_mean": round(offset_mean, 6),
        "offset_b_minus_a_s_std": round(offset_std, 6),
        "offset_max_disagreement_s": round(offset_max_disagreement, 6),
        "drift_slope_b_per_a": float(args.fps_b / args.fps_a),  # B-seconds per A-second
        "drift_source": "nominal_fps_ratio",  # all claps clustered early; can't measure empirically
        "drift_slope_warning": (
            "All 4 claps are in the first ~40 s, so drift slope is computed from the nominal fps ratio, "
            "not empirically. Over ~700 s of footage this accrues ~42 frames between the two clocks."
        ),
        "frame_correspondence_formula": (
            "B_frame(A_frame) = round( ((A_frame / fps_a) * drift_slope_b_per_a + offset_b_minus_a_s_mean) * fps_b )"
        ),
    }
    args.output.write_text(json.dumps(sync, indent=2))
    print(f"  ✓ sync.json → {args.output}")

    # Print quality assessment
    if offset_max_disagreement < 0.020:
        print(f"  ✓ Per-clap offset disagreement {offset_max_disagreement*1000:.1f} ms — excellent sync precision.")
    elif offset_max_disagreement < 0.100:
        print(f"  ⚠ Per-clap offset disagreement {offset_max_disagreement*1000:.1f} ms — acceptable but inspect the diag plot.")
    else:
        print(f"  ⚠⚠ Per-clap offset disagreement {offset_max_disagreement*1000:.1f} ms — likely a detection error (some peaks aren't the same clap). Inspect the diag plot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
