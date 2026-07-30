#!/usr/bin/env python3
"""
Measure MuseTalk's Real-Time Factor (RTF): given N seconds of driving audio,
how many seconds of wall-clock compute does the persistent worker need to
turn that into an animated clip?

This is the one number that determines whether a continuous streaming
architecture (encoder keeps producing frames while playback consumes them,
connected by a small ring buffer) is realistic, or whether the pipeline needs
to fall back to generate-then-play chunking with real buffering:

  - throughput factor = audio_seconds / compute_seconds
    (this is the "3 seconds of animation in ~1 second" number: >1 means
    MuseTalk outruns real time; the bigger the margin, the more comfortably
    the encoder can stay ahead of playback.)

  - RTF = compute_seconds / audio_seconds
    (the conventional speech/video-processing definition; <1 means faster
    than real time. Reported alongside the throughput factor since both
    conventions show up in the literature.)

The script spawns the same persistent worker used in production
(backend/musetalk_worker.py, driven exactly like
app/services/animator.py::AvatarAnimator._worker_infer), feeds it silent
audio clips of several durations, and times each job end-to-end: whisper
feature extraction, UNet/VAE inference, PNG frame writes, and the two ffmpeg
encode/mux passes — i.e. everything the real pipeline pays per turn, not just
the GPU forward pass.

Usage:
    conda_envs/ai-avatar/bin/python scripts/benchmark_musetalk_rtf.py
    conda_envs/ai-avatar/bin/python scripts/benchmark_musetalk_rtf.py --durations 2 5 10 20 --repeats 5
    conda_envs/ai-avatar/bin/python scripts/benchmark_musetalk_rtf.py --source-image path/to/photo.jpg --json out.json

Run against an otherwise-idle GPU (no backend server already up) for clean
numbers — a resident backend process competes for the same CUDA context.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

os.chdir(BACKEND_DIR)  # app.config resolves .env relative to cwd/its own file either way

from app.config import settings  # noqa: E402

DEFAULT_DURATIONS = [2.0, 5.0, 10.0, 20.0]
# audio_seconds / compute_seconds this far above 1.0 before we call
# "continuous streaming" comfortable rather than merely feasible.
COMFORTABLE_MARGIN = 1.5


# ── worker lifecycle (mirrors app/services/animator.py) ─────────────────────


def resolve_worker_script() -> tuple[Path, Path]:
    """Returns (musetalk_dir, worker_script)."""
    musetalk_dir = (BACKEND_DIR / settings.MUSETALK_PATH).resolve()
    worker_script = musetalk_dir / "scripts" / "musetalk_worker.py"
    if not worker_script.exists():
        tracked = BACKEND_DIR / "models" / "MuseTalk" / "scripts" / "musetalk_worker.py"
        if not tracked.exists():
            raise FileNotFoundError(
                f"MuseTalk worker script not found: {worker_script} or {tracked}"
            )
        worker_script = tracked
    return musetalk_dir, worker_script


def spawn_worker(ready_timeout: float = 300.0) -> subprocess.Popen:
    musetalk_dir, worker_script = resolve_worker_script()

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(musetalk_dir) + (":" + existing if existing else "")
    env["MUSETALK_TAIL_HOLD_FRAMES"] = str(settings.MUSETALK_TAIL_HOLD_FRAMES)

    import torch

    use_float16 = torch.cuda.is_available()

    proc = subprocess.Popen(
        [sys.executable, str(worker_script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(musetalk_dir),
        env=env,
        text=True,
    )

    init_msg = (
        json.dumps(
            {
                "unet_model_path": str(musetalk_dir / "models" / "musetalkV15" / "unet.pth"),
                "unet_config": str(musetalk_dir / "models" / "musetalkV15" / "musetalk.json"),
                "whisper_dir": str(musetalk_dir / "models" / "whisper"),
                "vae_type": str(musetalk_dir / "models" / "sd-vae"),
                "use_float16": use_float16,
            }
        )
        + "\n"
    )
    proc.stdin.write(init_msg)
    proc.stdin.flush()

    print("Loading MuseTalk worker (models load once, then every job below reuses them)...")
    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            stderr = proc.stderr.read()
            raise RuntimeError(f"MuseTalk worker exited before READY:\n{stderr[-4000:]}")
        if line.strip().startswith("READY"):
            print("Worker ready.\n")
            return proc
    proc.kill()
    raise TimeoutError("MuseTalk worker timed out waiting for READY")


def send_job(
    proc: subprocess.Popen, image: Path, audio: Path, output: Path, coord_cache: Optional[Path],
    timeout: float,
) -> float:
    """Send one inference job, block for the result, return wall-clock seconds."""
    job = (
        json.dumps(
            {
                "image": str(image.resolve()),
                "audio": str(audio.resolve()),
                "output": str(output.resolve()),
                "coord_cache": str(coord_cache.resolve()) if coord_cache else None,
            }
        )
        + "\n"
    )

    t0 = time.time()
    try:
        proc.stdin.write(job)
        proc.stdin.flush()
    except (BrokenPipeError, OSError) as e:
        stderr = proc.stderr.read()
        raise RuntimeError(f"Worker pipe is dead: {e}\nstderr:\n{stderr[-4000:]}") from e

    # readline() blocks the main thread; the worker only ever emits protocol
    # JSON on stdout (see musetalk_worker.py's fd-1 redirect), so a plain
    # blocking read with a wall-clock deadline check is enough here — this
    # script issues jobs serially, unlike animator.py's asyncio timeout.
    import select

    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        raise TimeoutError(f"MuseTalk job timed out after {timeout}s")
    line = proc.stdout.readline()
    elapsed = time.time() - t0

    if not line:
        stderr = proc.stderr.read()
        raise RuntimeError(f"Worker exited mid-job. stderr:\n{stderr[-4000:]}")

    result = json.loads(line.strip())
    if result.get("status") != "ok":
        raise RuntimeError(f"Worker reported error: {result.get('msg')}")

    return elapsed


# ── audio + media helpers ────────────────────────────────────────────────────


def make_silence(path: Path, seconds: float, sample_rate: int = 16000) -> None:
    """Silent WAV of a given duration — content doesn't matter for RTF since
    MuseTalk's compute cost scales with frame count (audio duration * fps),
    not with what's actually being said; this keeps the benchmark
    reproducible without needing a TTS pass first."""
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=mono",
            "-t", str(seconds),
            "-acodec", "pcm_s16le",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Could not generate silence: {proc.stderr}")


def media_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return float(out)


def find_default_source_image() -> Optional[Path]:
    avatars_dir = BACKEND_DIR / "uploads" / "avatars"
    if not avatars_dir.exists():
        return None
    for candidate in sorted(avatars_dir.glob("*/image.jpg")):
        return candidate
    return None


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-image", type=Path, default=None, help="Avatar photo (or video) to drive")
    parser.add_argument(
        "--durations", type=float, nargs="+", default=DEFAULT_DURATIONS,
        help=f"Audio-clip durations in seconds to test (default: {DEFAULT_DURATIONS})",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timed runs per duration (default: 3)")
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="Per-job timeout in seconds (default: 120)",
    )
    parser.add_argument("--json", type=Path, default=None, help="Also write raw results as JSON")
    parser.add_argument("--keep-outputs", action="store_true", help="Don't delete generated clips")
    args = parser.parse_args()

    source_image = args.source_image or find_default_source_image()
    if source_image is None:
        parser.error(
            "No avatar image found under backend/uploads/avatars/*/image.jpg — pass --source-image"
        )
    if not Path(source_image).exists():
        parser.error(f"--source-image not found: {source_image}")

    work_dir = Path(tempfile.mkdtemp(prefix="musetalk-rtf-"))
    print(f"Avatar: {source_image}")
    print(f"Scratch dir: {work_dir}\n")

    proc = spawn_worker()
    coord_cache = work_dir / "coords.pkl"

    rows: list[dict] = []
    try:
        # Warm-up job: pays face-detection (populates coord_cache) and first-batch
        # CUDA/cuDNN kernel autotuning, neither of which a real streaming session
        # pays more than once per avatar. Not counted in the timed results below.
        print("Warm-up job (face detection + kernel warmup, not timed)...")
        warmup_audio = work_dir / "warmup.wav"
        make_silence(warmup_audio, 2.0)
        send_job(
            proc, Path(source_image), warmup_audio, work_dir / "warmup.mp4", coord_cache,
            timeout=max(args.timeout, 180.0),
        )
        print("Warm-up done.\n")

        print(f"{'duration':>10}  {'run':>4}  {'compute_s':>10}  {'RTF':>6}  {'throughput':>10}")
        for duration in args.durations:
            audio_path = work_dir / f"audio_{duration:g}s.wav"
            make_silence(audio_path, duration)
            actual_audio_s = media_duration(audio_path)

            for run in range(1, args.repeats + 1):
                output_path = work_dir / f"out_{duration:g}s_run{run}.mp4"
                compute_s = send_job(
                    proc, Path(source_image), audio_path, output_path, coord_cache,
                    timeout=args.timeout,
                )
                video_s = media_duration(output_path)
                rtf = compute_s / actual_audio_s
                throughput = actual_audio_s / compute_s
                rows.append(
                    {
                        "requested_duration_s": duration,
                        "audio_duration_s": round(actual_audio_s, 3),
                        "video_duration_s": round(video_s, 3),
                        "run": run,
                        "compute_s": round(compute_s, 3),
                        "rtf": round(rtf, 3),
                        "throughput_factor": round(throughput, 3),
                    }
                )
                print(
                    f"{duration:>9.1f}s  {run:>4}  {compute_s:>9.2f}s  {rtf:>6.2f}  {throughput:>9.2f}x"
                )

                if not args.keep_outputs:
                    output_path.unlink(missing_ok=True)

    finally:
        proc.kill()
        if not args.keep_outputs:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)

    # ── summary ──────────────────────────────────────────────────────────────
    throughputs = [r["throughput_factor"] for r in rows]
    rtfs = [r["rtf"] for r in rows]
    mean_throughput = statistics.mean(throughputs)
    min_throughput = min(throughputs)
    mean_rtf = statistics.mean(rtfs)

    print("\nPer-duration averages:")
    print(f"{'duration':>10}  {'mean compute_s':>15}  {'mean RTF':>9}  {'mean throughput':>16}")
    for duration in args.durations:
        subset = [r for r in rows if r["requested_duration_s"] == duration]
        mc = statistics.mean(r["compute_s"] for r in subset)
        mr = statistics.mean(r["rtf"] for r in subset)
        mt = statistics.mean(r["throughput_factor"] for r in subset)
        print(f"{duration:>9.1f}s  {mc:>14.2f}s  {mr:>9.2f}  {mt:>15.2f}x")

    print(
        f"\nOverall: mean RTF={mean_rtf:.2f} (compute_s / audio_s), "
        f"mean throughput={mean_throughput:.2f}x, worst-case throughput={min_throughput:.2f}x"
    )

    print("\nVerdict:")
    if min_throughput >= COMFORTABLE_MARGIN:
        print(
            f"  MuseTalk consistently generates audio faster than it plays back "
            f"(worst case {min_throughput:.2f}x, i.e. every real-time second of audio "
            f"costs <= {1 / min_throughput:.2f}s of compute). A continuous streaming "
            f"architecture is realistic here: the encoder can stay comfortably ahead "
            f"of playback with just a small ring buffer to absorb jitter."
        )
    elif min_throughput >= 1.0:
        print(
            f"  MuseTalk is faster than real time but not by a wide margin "
            f"(worst case {min_throughput:.2f}x). Continuous streaming can work, but "
            f"size the buffer for the worst-case duration/run above, not the average — "
            f"a few slow jobs in a row could let playback catch up to the encoder."
        )
    else:
        print(
            f"  MuseTalk is barely real time or slower in at least one case "
            f"(worst case {min_throughput:.2f}x, i.e. {1 / min_throughput:.2f}s of compute "
            f"per second of audio). Continuous low-latency streaming is not realistic as-is — "
            f"plan for a generate-then-play (chunked) pipeline with real buffering regardless "
            f"of transport, or invest in speeding up inference (smaller batch latency, "
            f"lower resolution, TensorRT/fp8, etc.) before redesigning the media pipeline."
        )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "rows": rows,
                    "summary": {
                        "mean_rtf": round(mean_rtf, 3),
                        "mean_throughput_factor": round(mean_throughput, 3),
                        "worst_case_throughput_factor": round(min_throughput, 3),
                    },
                },
                indent=2,
            )
        )
        print(f"\nWrote raw results to {args.json}")


if __name__ == "__main__":
    main()
