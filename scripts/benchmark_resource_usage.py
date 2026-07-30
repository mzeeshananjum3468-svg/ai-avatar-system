#!/usr/bin/env python3
"""
Measure VRAM, RAM, and load time for each stage of the avatar pipeline:
Startup -> Whisper -> Chatterbox -> MuseTalk -> LivePortrait.

Each stage is loaded on top of the previous one (nothing is torn down in
between) so the reported VRAM/RAM is the *cumulative* footprint of having
that many stages resident at once, same as the real backend process ends up
after `main.py`'s startup warmup + a LivePortrait avatar-upload run.

Whisper and Chatterbox load in-process, exactly like
app/services/stt.py::STTService._build_model and
app/services/tts.py::TTSService.initialize. MuseTalk and LivePortrait load
in subprocesses (a persistent worker and a one-shot script, respectively),
mirroring app/services/animator.py and app/services/liveportrait.py — so
their VRAM/RAM is measured across the whole process tree, not just this
script's own process.

LivePortrait has no separate "load, then sit idle" mode: its inference.py
loads weights and renders one video in a single script run. The number
reported for it is therefore load + one short inference pass, not a pure
model-load cost — called out again in the printed report.

Usage:
    conda_envs/ai-avatar/bin/python scripts/benchmark_resource_usage.py
    conda_envs/ai-avatar/bin/python scripts/benchmark_resource_usage.py --skip-liveportrait
    conda_envs/ai-avatar/bin/python scripts/benchmark_resource_usage.py --json out.json

Run this against an otherwise-idle GPU (no backend server already up) —
the script reports absolute nvidia-smi/RSS numbers, so anything else
resident on the GPU/host inflates every row.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import psutil  # noqa: E402

os.chdir(BACKEND_DIR)  # app.config resolves .env relative to cwd/its own file either way

from app.config import settings  # noqa: E402

ZEESHAN_ROOT = BACKEND_DIR.parent.parent


# ── resource sampling ────────────────────────────────────────────────────────


def gpu_used_mb(gpu_index: int = 0) -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        return float(lines[gpu_index])
    except Exception:
        return 0.0


def proc_tree_rss_mb(pid: int) -> float:
    """RSS of `pid` plus every descendant it has spawned (subprocess workers)."""
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0.0
    procs = [root] + root.children(recursive=True)
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / 1024**2


class ResourceSampler(threading.Thread):
    """Polls VRAM/RAM in the background and tracks the peak seen per stage."""

    def __init__(self, main_pid: int, gpu_index: int = 0, interval: float = 0.2):
        super().__init__(daemon=True)
        self.main_pid = main_pid
        self.gpu_index = gpu_index
        self.interval = interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._stage: Optional[str] = None
        self.peaks: dict[str, tuple[float, float]] = {}

    def run(self) -> None:
        while not self._stop.is_set():
            rss = proc_tree_rss_mb(self.main_pid)
            vram = gpu_used_mb(self.gpu_index)
            with self._lock:
                if self._stage is not None:
                    pr, pv = self.peaks.get(self._stage, (0.0, 0.0))
                    self.peaks[self._stage] = (max(pr, rss), max(pv, vram))
            self._stop.wait(self.interval)

    def start_stage(self, name: str) -> None:
        with self._lock:
            self._stage = name
            self.peaks.setdefault(name, (0.0, 0.0))

    def peak_for(self, name: str) -> tuple[float, float]:
        with self._lock:
            return self.peaks.get(name, (0.0, 0.0))

    def stop(self) -> None:
        self._stop.set()


# ── stage implementations ───────────────────────────────────────────────────


def load_startup() -> None:
    """Baseline: import torch and force CUDA context init, same cost every
    stage below pays implicitly the first time it touches the GPU."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.init()
        torch.zeros(1, device="cuda")


def load_whisper():
    """Mirrors app/services/stt.py::STTService._build_model."""
    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(settings.WHISPER_MODEL, device=device, compute_type=compute_type)


def load_chatterbox():
    """Mirrors app/services/tts.py::TTSService.initialize."""
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ChatterboxMultilingualTTS.from_pretrained(device=device)


def spawn_musetalk_worker(ready_timeout: float = 300.0):
    """Mirrors app/services/animator.py::AvatarAnimator._ensure_worker —
    spawns the persistent worker and waits for READY, no inference job."""
    musetalk_dir = (BACKEND_DIR / settings.MUSETALK_PATH).resolve()
    worker_script = musetalk_dir / "scripts" / "musetalk_worker.py"
    if not worker_script.exists():
        # MUSETALK_PATH may point at an external checkout that predates our
        # worker script (or is a different MuseTalk version altogether) —
        # fall back to the tracked copy shipped with this repo, same as
        # app/services/animator.py::AvatarAnimator._resolve_worker_script.
        tracked = BACKEND_DIR / "models" / "MuseTalk" / "scripts" / "musetalk_worker.py"
        if not tracked.exists():
            raise FileNotFoundError(
                f"MuseTalk worker script not found: {worker_script} or {tracked}"
            )
        worker_script = tracked

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

    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            stderr = proc.stderr.read()
            raise RuntimeError(f"MuseTalk worker exited before READY:\n{stderr[-4000:]}")
        if line.strip().startswith("READY"):
            return proc
    proc.kill()
    raise TimeoutError("MuseTalk worker timed out waiting for READY")


def discover_liveportrait() -> dict:
    """Mirrors app/services/liveportrait.py's discovery logic."""
    checkout = ZEESHAN_ROOT / "LivePortrait"
    python = ZEESHAN_ROOT / "conda_envs" / "LivePortrait" / "bin" / "python"
    driving = BACKEND_DIR / "models" / "MuseTalk" / "data" / "video" / "yongen.mp4"

    missing = [
        label
        for label, path in [("checkout", checkout / "inference.py"), ("python", python), ("driving video", driving)]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"LivePortrait not fully available, missing: {', '.join(missing)}")
    return {"checkout": checkout, "python": python, "driving": driving}


def run_liveportrait_once(source_image: Path, timeout: float = 600.0) -> None:
    lp = discover_liveportrait()
    with tempfile.TemporaryDirectory(prefix="lp-benchmark-") as out_dir:
        cmd = [
            str(lp["python"]),
            "inference.py",
            "--source",
            str(source_image),
            "--driving",
            str(lp["driving"]),
            "--output_dir",
            out_dir,
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(lp["checkout"]),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"LivePortrait inference failed:\n{proc.stderr[-4000:]}")


# ── main ─────────────────────────────────────────────────────────────────────


def find_default_source_image() -> Optional[Path]:
    avatars_dir = BACKEND_DIR / "uploads" / "avatars"
    if not avatars_dir.exists():
        return None
    for candidate in sorted(avatars_dir.glob("*/image.jpg")):
        return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument("--skip-chatterbox", action="store_true")
    parser.add_argument("--skip-musetalk", action="store_true")
    parser.add_argument("--skip-liveportrait", action="store_true")
    parser.add_argument("--source-image", type=Path, default=None, help="Avatar photo for the LivePortrait pass")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None, help="Also write raw results as JSON")
    args = parser.parse_args()

    source_image = args.source_image or find_default_source_image()
    if not args.skip_liveportrait and source_image is None:
        parser.error(
            "No avatar image found under backend/uploads/avatars/*/image.jpg — "
            "pass --source-image or use --skip-liveportrait"
        )

    baseline_vram = gpu_used_mb(args.gpu_index)
    if baseline_vram > 200:
        print(
            f"WARNING: GPU already has {baseline_vram:.0f} MB used before this script started — "
            f"stop any running backend/model server for clean numbers. Every row below includes that.\n"
        )

    sampler = ResourceSampler(os.getpid(), gpu_index=args.gpu_index)
    sampler.start()

    rows: list[dict] = []
    # Keep every loaded model/process referenced so its memory stays resident
    # for the rest of the run — later stages measure cumulative footprint.
    keep_alive: list = []
    musetalk_proc = None

    def run_stage(name: str, fn) -> None:
        sampler.start_stage(name)
        t0 = time.time()
        result = fn()
        elapsed = time.time() - t0
        keep_alive.append(result)
        time.sleep(0.5)  # let the sampler catch the post-load steady state
        rss_mb, vram_mb = sampler.peak_for(name)
        rows.append(
            {
                "stage": name,
                "vram_gb": round(vram_mb / 1024, 2),
                "ram_gb": round(rss_mb / 1024, 2),
                "load_time_s": round(elapsed, 1),
            }
        )
        print(f"{name:<12} vram={vram_mb / 1024:6.1f} GB  ram={rss_mb / 1024:6.1f} GB  load={elapsed:6.1f}s")

    try:
        run_stage("Startup", load_startup)

        if not args.skip_whisper:
            run_stage("Whisper", load_whisper)

        if not args.skip_chatterbox:
            run_stage("Chatterbox", load_chatterbox)

        if not args.skip_musetalk:
            def _start_musetalk():
                nonlocal musetalk_proc
                musetalk_proc = spawn_musetalk_worker()
                return musetalk_proc

            run_stage("MuseTalk", _start_musetalk)

        if not args.skip_liveportrait:
            run_stage("LivePortrait", lambda: run_liveportrait_once(source_image))

    finally:
        sampler.stop()
        if musetalk_proc is not None:
            musetalk_proc.kill()

    print("\nStage        VRAM        RAM         Load Time")
    for row in rows:
        print(f"{row['stage']:<12} {row['vram_gb']:>6.1f} GB   {row['ram_gb']:>6.1f} GB   {row['load_time_s']:>6.1f}s")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"\nWrote raw results to {args.json}")


if __name__ == "__main__":
    main()
