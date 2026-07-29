import asyncio
import hashlib
import json
import logging
import os
import sys
import tempfile
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import torch
from PIL import Image

from app.config import settings

TMPDIR = Path(tempfile.gettempdir())

logger = logging.getLogger(__name__)


class AvatarAnimator:
    """
    Avatar Animation Service.
    Supported engines (set AVATAR_ENGINE in .env):
      - musetalk : MuseTalk V1.5 — persistent worker (models loaded once)
      - simple   : ffmpeg static image + audio, no lip-sync
    """

    def __init__(self):
        self.engine = settings.AVATAR_ENGINE
        self.resolution = settings.AVATAR_RESOLUTION
        self.fps = settings.AVATAR_FPS
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_float16 = self.device == "cuda"  # float16 on GPU = ~2× faster via Tensor Cores
        self._initialised = False
        self._musetalk_dir: Optional[Path] = None

        # Persistent worker handles
        self._worker_proc: Optional[asyncio.subprocess.Process] = None
        self._worker_lock = asyncio.Lock()
        self._worker_env: dict = {}
        # Rolling tail of the worker's stderr (its print()s land here — see
        # sys.stdout redirect at the top of musetalk_worker.py), kept for
        # error messages. Populated by _pump_worker_stderr, the only reader
        # of proc.stderr — a StreamReader can't be read from two places at
        # once, so failure paths read this deque instead of the pipe directly.
        self._worker_stderr_tail: deque = deque(maxlen=500)
        self._stderr_pump_task: Optional[asyncio.Task] = None

        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(
                f"AvatarAnimator: engine={self.engine}, device=cuda "
                f"({gpu_name}, {vram_gb:.1f} GB VRAM), float16={self.use_float16}"
            )
        else:
            logger.info(
                f"AvatarAnimator: engine={self.engine}, device=cpu "
                f"(no GPU — consider AWS g5/g6 instance for real-time performance)"
            )

    # ── initialisation ────────────────────────────────────────────────────────

    async def initialize(self):
        if self._initialised:
            return

        if self.engine == "musetalk":
            self._musetalk_dir = self._find_dir(settings.MUSETALK_PATH, "scripts/inference.py")
            if self._musetalk_dir is None:
                logger.warning(
                    "MuseTalk not found at '%s'. "
                    "Run scripts/setup_musetalk.sh to install it. "
                    "Falling back to simple animation.",
                    settings.MUSETALK_PATH,
                )
                self.engine = "simple"
            else:
                logger.info(f"MuseTalk found at: {self._musetalk_dir}")
                # Build env once
                existing = os.environ.get("PYTHONPATH", "")
                self._worker_env = os.environ.copy()
                self._worker_env["PYTHONPATH"] = str(self._musetalk_dir) + (
                    ":" + existing if existing else ""
                )
                # pydantic-settings' env_file only populates the `settings`
                # object, it never injects values into os.environ — so a
                # tunable set only in .env (not a genuinely exported shell/
                # docker env var) wouldn't otherwise reach the worker
                # subprocess, which reads its config straight from os.environ.
                self._worker_env["MUSETALK_TAIL_HOLD_FRAMES"] = str(
                    settings.MUSETALK_TAIL_HOLD_FRAMES
                )

        elif self.engine not in ("simple",):
            logger.warning(f"Unknown engine '{self.engine}', using simple animation.")
            self.engine = "simple"

        self._initialised = True

    def _find_dir(self, config_path: str, marker_file: str) -> Optional[Path]:
        candidates = [
            Path(config_path),
            Path(__file__).resolve().parent.parent.parent / config_path,
        ]
        for p in candidates:
            if (p / marker_file).exists():
                return p.resolve()
        return None

    def _resolve_worker_script(self, musetalk_dir: Path) -> Path:
        """Resolve the persistent MuseTalk worker script.

        Prefer the script inside the active MuseTalk checkout, but fall back to
        the tracked copy shipped with this repo so an external checkout can be
        used without needing a duplicate worker file in that tree.
        """
        in_checkout = musetalk_dir / "scripts" / "musetalk_worker.py"
        if in_checkout.exists():
            return in_checkout

        tracked = Path(__file__).resolve().parent.parent.parent / "models" / "MuseTalk" / "scripts" / "musetalk_worker.py"
        if tracked.exists():
            logger.info(f"Using tracked MuseTalk worker at {tracked}")
            return tracked

        raise FileNotFoundError(
            f"musetalk_worker.py not found in {in_checkout} or {tracked}. "
            "Run scripts/setup_musetalk.sh to install it."
        )

    # ── persistent worker management ─────────────────────────────────────────

    async def _ensure_worker(self) -> asyncio.subprocess.Process:
        """Start the persistent worker if not already running."""
        if self._worker_proc is not None and self._worker_proc.returncode is None:
            return self._worker_proc

        musetalk_dir: Path = self._musetalk_dir  # type: ignore[assignment]
        worker_script = self._resolve_worker_script(musetalk_dir)

        logger.info("Starting persistent MuseTalk worker (loading models once)…")
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(worker_script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(musetalk_dir),
            env=self._worker_env,
        )

        self._worker_stderr_tail.clear()
        self._stderr_pump_task = asyncio.create_task(self._pump_worker_stderr(proc))

        # Send init config — include float16 flag so worker can optimise for GPU
        init_msg = (
            json.dumps(
                {
                    "unet_model_path": str(musetalk_dir / "models" / "musetalkV15" / "unet.pth"),
                    "unet_config": str(musetalk_dir / "models" / "musetalkV15" / "musetalk.json"),
                    "whisper_dir": str(musetalk_dir / "models" / "whisper"),
                    "vae_type": str(musetalk_dir / "models" / "sd-vae"),
                    "use_float16": self.use_float16,
                }
            )
            + "\n"
        )
        proc.stdin.write(init_msg.encode())
        await proc.stdin.drain()

        # Wait for READY — GPU loads much faster (~60s) vs CPU (~5-10 min first time)
        model_load_timeout = 300 if self.device == "cuda" else 600
        logger.info(f"Waiting for worker to finish loading models (timeout={model_load_timeout}s)…")
        try:
            ready_line = await asyncio.wait_for(proc.stdout.readline(), timeout=model_load_timeout)
        except asyncio.TimeoutError:
            self._safe_kill(proc)
            raise RuntimeError("MuseTalk worker timed out while loading models")

        if not ready_line.decode().strip().startswith("READY"):
            self._safe_kill(proc)
            raise RuntimeError(
                f"Worker failed to start. stderr:\n{''.join(self._worker_stderr_tail)}"
            )

        logger.info("MuseTalk worker ready — models loaded")
        self._worker_proc = proc
        return proc

    async def _pump_worker_stderr(self, proc: asyncio.subprocess.Process) -> None:
        """Continuously drain the worker's stderr into our own logger.

        proc.stderr is a pipe — nothing reads it during normal operation, so
        without this every print() in musetalk_worker.py (its stdout is
        redirected to stderr, see that file's header) just sits unread in the
        OS pipe buffer: invisible now, and a potential deadlock later if the
        buffer fills while the worker blocks on a write. This is the only
        reader of proc.stderr; failure paths consult _worker_stderr_tail
        instead of reading the pipe themselves.
        """
        try:
            async for raw_line in proc.stderr:
                line = raw_line.decode(errors="replace").rstrip()
                self._worker_stderr_tail.append(line + "\n")
                logger.info(f"[musetalk_worker] {line}")
        except (asyncio.CancelledError, ValueError):
            pass

    async def _worker_infer(
        self, image_path: str, audio_path: str, output_path: str, coord_cache: Optional[str]
    ) -> str:
        """Send one job to the persistent worker and await its result."""
        async with self._worker_lock:
            proc = await self._ensure_worker()

            job = (
                json.dumps(
                    {
                        "image": str(Path(image_path).resolve()),
                        "audio": str(Path(audio_path).resolve()),
                        "output": str(Path(output_path).resolve()),
                        "coord_cache": coord_cache,
                    }
                )
                + "\n"
            )

            # If the worker died (OOM/segfault) its stdin is closed; writing
            # raises BrokenPipeError. Reset the handle so the NEXT job respawns
            # a fresh worker instead of repeatedly failing against a dead pipe.
            try:
                proc.stdin.write(job.encode())
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                self._safe_kill(proc)
                self._worker_proc = None
                raise RuntimeError(f"MuseTalk worker pipe is dead: {e}") from e

            # GPU: expect ~5-15s per sentence; CPU: up to 5 min
            infer_timeout = 60 if self.device == "cuda" else 300

            try:
                result_line = await asyncio.wait_for(proc.stdout.readline(), timeout=infer_timeout)
            except asyncio.TimeoutError:
                self._safe_kill(proc)
                self._worker_proc = None
                raise RuntimeError(f"MuseTalk inference timed out after {infer_timeout}s")
            except asyncio.CancelledError:
                # Caller (e.g. the websocket handler) is abandoning this request, most
                # likely a client disconnect. The worker subprocess is still mid-job in
                # the background and will eventually emit a result line nobody is
                # waiting on. If left alive, the NEXT unrelated job reuses this worker
                # and consumes that stale line as its own result — wrong-file errors on
                # reconnect. Kill and reset so the next call spawns a clean worker.
                self._safe_kill(proc)
                self._worker_proc = None
                raise

            # Empty read == worker exited mid-job (EOF on stdout). Reset so the
            # next call respawns instead of erroring on a half-dead process.
            if not result_line:
                # Give the stderr pump a beat to drain whatever the worker
                # printed on its way out (traceback, etc.) before we read it.
                await asyncio.sleep(0.2)
                self._safe_kill(proc)
                self._worker_proc = None
                raise RuntimeError(
                    f"MuseTalk worker exited before returning a result. stderr:\n"
                    f"{''.join(self._worker_stderr_tail)}"
                )

            result = json.loads(result_line.decode().strip())
            if result["status"] != "ok":
                raise RuntimeError(result.get("msg", "Unknown worker error"))

            return output_path

    # ── warmup ────────────────────────────────────────────────────────────────

    async def warmup_models(self) -> None:
        """
        Load MuseTalk's model weights onto the GPU ahead of any session, so
        the first real user of the process isn't the one paying the
        60s-10min model-load cost. No avatar image is needed for this —
        it's just spinning up the persistent worker and waiting for READY.
        Safe to call at server startup; best-effort (logs and returns on
        failure so a missing/broken MuseTalk checkout can't block startup).
        """
        if not self._initialised:
            await self.initialize()
        if self.engine != "musetalk":
            return
        try:
            async with self._worker_lock:
                await self._ensure_worker()
        except Exception:
            logger.exception("Model warmup failed — worker will load lazily on first real request")

    async def _write_silence(self, path: Path, seconds: float) -> None:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-t", str(seconds),
            "-acodec", "pcm_s16le",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Could not generate warmup silence: {stderr.decode(errors='replace')}")

    async def warmup_avatar(self, avatar_image_path: str) -> None:
        """
        Run one throwaway inference for this specific avatar: loads the
        worker if it isn't already up, computes/caches this avatar's face
        landmarks + latents (the per-avatar cost musetalk_worker.py would
        otherwise pay lazily on the first real turn), and warms the CUDA
        kernels for this crop shape. Called from the WS connect path so a
        session's first REAL message doesn't eat this latency.

        Best-effort — swallows all errors so a broken/missing avatar image
        can't block a session from ever starting; the real `animate()` call
        already falls back to the simple ffmpeg engine on failure anyway.
        """
        if not self._initialised:
            await self.initialize()
        if self.engine != "musetalk":
            return

        dummy_audio = TMPDIR / f"warmup-{uuid.uuid4().hex}.wav"
        dummy_output = TMPDIR / f"warmup-{uuid.uuid4().hex}.mp4"
        try:
            await self._write_silence(dummy_audio, seconds=1.0)
            await self._animate_musetalk(avatar_image_path, str(dummy_audio), str(dummy_output))
            logger.info(f"Avatar warmup complete: {avatar_image_path}")
        except Exception:
            logger.exception(f"Avatar warmup failed for {avatar_image_path} — continuing anyway")
        finally:
            dummy_audio.unlink(missing_ok=True)
            dummy_output.unlink(missing_ok=True)

    # ── public API ────────────────────────────────────────────────────────────

    async def animate(
        self,
        avatar_image_path: str,
        audio_path: str,
        output_path: str,
        cache_key: Optional[str] = None,
    ) -> str:
        """
        Animate avatar with audio. Returns path to the generated video.
        Falls back to simple (static image + audio) on any engine failure.
        """
        if not self._initialised:
            await self.initialize()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Animating [{self.engine}] source={avatar_image_path} audio={audio_path}")

        try:
            if self.engine == "musetalk":
                # Pass the source through untouched — if it's a video, the
                # MuseTalk worker detects that itself and animates every
                # frame (see musetalk_worker.py). Extracting a single frame
                # here would silently downgrade every video avatar to a
                # still-image lip-sync, which defeats the point.
                return await self._animate_musetalk(avatar_image_path, audio_path, output_path)
            else:
                return await self._animate_simple_from_source(avatar_image_path, audio_path, output_path)
        except Exception:
            logger.exception(f"Animation failed ({self.engine}): Falling back to simple.")
            return await self._animate_simple_from_source(avatar_image_path, audio_path, output_path)

    async def _animate_simple_from_source(
        self, avatar_path: str, audio_path: str, output_path: str
    ) -> str:
        """The ffmpeg 'simple' path needs a still image, so extract one here
        (only for this path — musetalk gets the original source untouched)."""
        frame_path = self._prepare_avatar_source(avatar_path)
        try:
            return await self._animate_simple(frame_path, audio_path, output_path)
        finally:
            if frame_path != avatar_path:
                Path(frame_path).unlink(missing_ok=True)

    def _prepare_avatar_source(self, avatar_path: str) -> str:
        """Return a still-image path for the simple ffmpeg fallback.

        If the supplied path points to a video, extract the first frame and use
        that as the avatar image. Only used by the 'simple' animation path,
        which genuinely requires a still image; musetalk receives the original
        source (image or video) untouched.
        """
        candidate = Path(avatar_path)
        if not candidate.exists():
            return avatar_path

        if candidate.suffix.lower() not in {
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".webm",
            ".m4v",
            ".mpg",
            ".mpeg",
            ".wmv",
        }:
            return avatar_path

        frame_path = Path(tempfile.gettempdir()) / f"avatar-frame-{uuid.uuid4().hex}.jpg"
        capture = cv2.VideoCapture(str(candidate))
        try:
            success, frame = capture.read()
            if not success or frame is None:
                return avatar_path

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(frame_path, quality=95)
            return str(frame_path)
        finally:
            capture.release()

    # ── MuseTalk ──────────────────────────────────────────────────────────────

    async def _animate_musetalk(
        self,
        avatar_path: str,
        audio_path: str,
        output_path: str,
    ) -> str:
        """Run MuseTalk via persistent worker (models stay loaded between calls)."""
        musetalk_dir: Path = self._musetalk_dir  # type: ignore[assignment]

        # Per-avatar face-coordinate cache (saves face-detection on repeat calls)
        source_path = Path(avatar_path)
        source_bytes = source_path.read_bytes() if source_path.exists() else b""
        avatar_id = hashlib.md5(source_bytes).hexdigest()
        coord_cache = str(musetalk_dir / "results" / "coords" / f"{avatar_id}.pkl")
        os.makedirs(os.path.dirname(coord_cache), exist_ok=True)

        # If the path changed but the content is different, don't reuse a stale
        # cache generated from an older still frame / video frame.
        if source_path.exists() and source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            legacy_cache = musetalk_dir / "results" / "coords" / f"{hashlib.md5(str(source_path.resolve()).encode()).hexdigest()}.pkl"
            if legacy_cache.exists() and not Path(coord_cache).exists():
                legacy_cache.unlink(missing_ok=True)

        await self._worker_infer(avatar_path, audio_path, output_path, coord_cache)

        logger.info(f"MuseTalk animation done: {output_path}")
        return output_path

    # ── Simple ffmpeg fallback ────────────────────────────────────────────────

    async def _animate_simple(
        self,
        avatar_path: str,
        audio_path: str,
        output_path: str,
    ) -> str:
        """Combine static image + audio with FFmpeg. No lip-sync."""
        logger.info("Using simple animation (static image + audio, no lip-sync)")

        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(avatar_path),
            "-i",
            str(audio_path),
            "-c:v",
            "libx264",
            "-tune",
            "stillimage",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            "-vf",
            (
                f"fps={self.fps},"
                f"scale={self.resolution}:{self.resolution}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={self.resolution}:{self.resolution}:(ow-iw)/2:(oh-ih)/2"
            ),
            output_path,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = stderr.decode(errors="replace")
            logger.error(f"FFmpeg error:\n{err}")
            raise RuntimeError("Simple animation (ffmpeg) failed")

        logger.info(f"Simple animation done: {output_path}")
        return output_path

    # ── helpers ───────────────────────────────────────────────────────────────

    def generate_cache_key(self, text: str, avatar_id: str) -> str:
        return hashlib.md5(f"{avatar_id}:{text}".encode()).hexdigest()

    @staticmethod
    def _safe_kill(proc: asyncio.subprocess.Process) -> None:
        """Kill a subprocess, tolerating the case where it already exited."""
        try:
            proc.kill()
        except ProcessLookupError:
            pass


# Global instance
avatar_animator = AvatarAnimator()