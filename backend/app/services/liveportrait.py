"""
Idle/thinking loop video generation via LivePortrait.

LivePortrait lives in its own conda env (its own torch/Python build,
separate from the backend's), so — same reasoning as the MuseTalk
worker in animator.py — it can't be imported in-process. Unlike the
MuseTalk worker it's invoked as a plain one-shot subprocess per video
rather than a persistent process, since this only runs once per
avatar upload, not once per chat turn.

Given an avatar's processed source image, two videos are produced:
  - idle:     source image driven by a neutral "resting" driving video
  - thinking: source image driven by a "thinking" driving video
Both are muted after generation — LivePortrait dubs the driving
video's own audio onto the output since a still image has none, but
these loops play silently under the app's own TTS audio.
"""

import asyncio
import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

TMPDIR = Path(tempfile.gettempdir())

# backend/app/services/liveportrait.py -> services -> app -> backend ->
# ai-avatar-system -> Zeeshan (the shared parent of every sibling project).
_ZEESHAN_ROOT = Path(__file__).resolve().parents[4]


class LivePortraitService:
    def __init__(self):
        self._checkout: Optional[Path] = None
        self._python: Optional[str] = None
        self._idle_driving: Optional[Path] = None
        self._thinking_driving: Optional[Path] = None
        self._initialised = False
        self._available = False

    # ── discovery ─────────────────────────────────────────────────────────────

    def _discover_checkout(self) -> Optional[Path]:
        candidates = [
            Path(settings.LIVEPORTRAIT_PATH) if settings.LIVEPORTRAIT_PATH else None,
            _ZEESHAN_ROOT / "LivePortrait",
        ]
        for candidate in candidates:
            if candidate and (candidate / "inference.py").exists():
                return candidate.resolve()
        return None

    def _discover_python(self) -> Optional[str]:
        if settings.LIVEPORTRAIT_PYTHON:
            configured = Path(settings.LIVEPORTRAIT_PYTHON)
            if configured.exists():
                return str(configured)
            logger.warning(f"LIVEPORTRAIT_PYTHON={configured} does not exist")

        guess = _ZEESHAN_ROOT / "conda_envs" / "LivePortrait" / "bin" / "python"
        if guess.exists():
            return str(guess)
        return None

    def _discover_driving_video(self, configured: str, fallbacks: list) -> Optional[Path]:
        if configured:
            path = Path(configured)
            if path.exists():
                return path.resolve()
            logger.warning(f"Configured driving video does not exist: {path}")

        for candidate in fallbacks:
            if candidate.exists():
                return candidate.resolve()
        return None

    def initialize(self) -> None:
        if self._initialised:
            return
        self._initialised = True

        self._checkout = self._discover_checkout()
        self._python = self._discover_python()

        ai_avatar_root = _ZEESHAN_ROOT / "ai-avatar-system"
        self._idle_driving = self._discover_driving_video(
            settings.IDLE_DRIVING_VIDEO,
            [
                ai_avatar_root / "backend" / "models" / "MuseTalk" / "data" / "video" / "yongen.mp4",
                _ZEESHAN_ROOT / "MuseTalk" / "data" / "video" / "yongen.mp4",
            ],
        )
        self._thinking_driving = self._discover_driving_video(
            settings.THINKING_DRIVING_VIDEO,
            [ai_avatar_root / "source.mp4"],
        )

        self._available = bool(
            self._checkout and self._python and self._idle_driving and self._thinking_driving
        )
        if self._available:
            logger.info(
                "LivePortrait ready: checkout=%s python=%s idle_driving=%s thinking_driving=%s",
                self._checkout,
                self._python,
                self._idle_driving,
                self._thinking_driving,
            )
        else:
            logger.warning(
                "LivePortrait unavailable (checkout=%s python=%s idle_driving=%s "
                "thinking_driving=%s) — idle/thinking loop videos will not be "
                "generated for new avatars.",
                self._checkout,
                self._python,
                self._idle_driving,
                self._thinking_driving,
            )

    @property
    def available(self) -> bool:
        if not self._initialised:
            self.initialize()
        return self._available

    # ── inference ─────────────────────────────────────────────────────────────

    async def _run_inference(self, source_image: Path, driving_video: Path, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self._python,
            "inference.py",
            "--source",
            str(source_image),
            "--driving",
            str(driving_video),
            "--output_dir",
            str(output_dir),
        ]
        logger.info(f"Running LivePortrait: {' '.join(cmd)} (cwd={self._checkout})")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self._checkout),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"LivePortrait inference failed (exit {proc.returncode}) for "
                f"source={source_image} driving={driving_video}:\n"
                f"{stderr.decode(errors='replace')[-4000:]}"
            )

        expected = output_dir / f"{source_image.stem}--{driving_video.stem}.mp4"
        if expected.exists():
            return expected

        # basename() convention didn't match (unexpected LivePortrait version) —
        # fall back to the newest non-concat mp4 actually produced.
        candidates = sorted(
            (p for p in output_dir.glob("*.mp4") if "_concat" not in p.name),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise RuntimeError(f"LivePortrait produced no output video in {output_dir}")
        return candidates[-1]

    async def _mute(self, src: Path, dest: Path) -> None:
        cmd = ["ffmpeg", "-y", "-i", str(src), "-c:v", "copy", "-an", str(dest)]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg mute failed for {src}:\n{stderr.decode(errors='replace')[-2000:]}")

    async def _reverse(self, src: Path, dest: Path) -> None:
        """Frame-reverse `src` into `dest`. Unlike _mute this can't stream-copy
        (reversing requires decoding the whole clip), but these are ~10s loop
        videos generated once at upload time, not a per-request cost.

        The client plays this forward-only, alternating it with the original
        clip on each 'ended' — real reverse video playback isn't supported by
        browsers (negative playbackRate is a no-op), and scrubbing `currentTime`
        backwards on compressed video is slow/unreliable (each step re-decodes
        from the nearest prior keyframe), which is what this file exists to
        avoid on the client.
        """
        cmd = ["ffmpeg", "-y", "-i", str(src), "-vf", "reverse", "-an", str(dest)]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg reverse failed for {src}:\n{stderr.decode(errors='replace')[-2000:]}")

    async def generate_idle_and_thinking(
        self, source_image_path: str, avatar_id: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Generate silent idle-loop and thinking-loop videos for an avatar,
        plus a frame-reversed companion of each.

        Returns (idle_video_path, idle_reversed_path, thinking_video_path,
        thinking_reversed_path) — any of these may be None if LivePortrait
        isn't installed/configured or inference fails; callers must treat
        every value as optional and leave the avatar usable without them (the
        reversed pair especially — the client falls back to a plain forward
        loop when they're missing).
        """
        if not self.available:
            return None, None, None, None

        work_dir = TMPDIR / f"liveportrait-{avatar_id}-{uuid.uuid4().hex[:8]}"
        try:
            # Sequential, not gather()'d — two concurrent LivePortrait
            # processes would each load the model onto the GPU at once and
            # risk OOMing a card that's sized for a single run.
            idle_raw = await self._run_inference(
                Path(source_image_path), self._idle_driving, work_dir / "idle"
            )
            thinking_raw = await self._run_inference(
                Path(source_image_path), self._thinking_driving, work_dir / "thinking"
            )

            idle_out = TMPDIR / f"{avatar_id}_idle.mp4"
            thinking_out = TMPDIR / f"{avatar_id}_thinking.mp4"
            await self._mute(idle_raw, idle_out)
            await self._mute(thinking_raw, thinking_out)

            idle_reversed_out = TMPDIR / f"{avatar_id}_idle_reversed.mp4"
            thinking_reversed_out = TMPDIR / f"{avatar_id}_thinking_reversed.mp4"
            try:
                await self._reverse(idle_out, idle_reversed_out)
            except Exception:
                logger.exception(f"Reversed idle clip failed for avatar {avatar_id} — continuing without it")
                idle_reversed_out = None
            try:
                await self._reverse(thinking_out, thinking_reversed_out)
            except Exception:
                logger.exception(f"Reversed thinking clip failed for avatar {avatar_id} — continuing without it")
                thinking_reversed_out = None

            return (
                str(idle_out),
                str(idle_reversed_out) if idle_reversed_out else None,
                str(thinking_out),
                str(thinking_reversed_out) if thinking_reversed_out else None,
            )
        except Exception:
            logger.exception(f"LivePortrait idle/thinking generation failed for avatar {avatar_id}")
            return None, None, None, None
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


# Global instance
liveportrait_service = LivePortraitService()
