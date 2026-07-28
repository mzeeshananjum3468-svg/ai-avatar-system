"""
musetalk_worker.py — persistent MuseTalk V1.5 inference worker.

Protocol (spoken over stdin/stdout, one JSON object per line):

  Startup:
    stdin  <- {"unet_model_path": "...", "unet_config": "...",
               "whisper_dir": "...", "vae_type": "...", "use_float16": true}
    stdout -> "READY\n"                       (once models are loaded)

  Per job:
    stdin  <- {"image": "<avatar photo path>", "audio": "<wav/mp3 path>",
               "output": "<mp4 path to write>", "coord_cache": "<pkl path or null>"}
    stdout -> {"status": "ok"}                 on success
    stdout -> {"status": "error", "msg": "..."} on failure (worker keeps running)

This file is meant to live at <MUSETALK_PATH>/scripts/musetalk_worker.py, i.e.
inside the TMElyralab/MuseTalk checkout that scripts/setup_musetalk.sh creates.
It is spawned with cwd=<MUSETALK_PATH> and PYTHONPATH including <MUSETALK_PATH>,
so `import musetalk...` resolves to that checkout's package.

Avatar source ("image" field) may be a still image OR a video/gif. Stills are
animated the classic way (one static crop driven by audio). Videos are
animated frame-by-frame: every source frame gets its own face crop, latent,
and mask, and those are cycled forward-then-backward (ping-pong) to match
however many frames the audio produces, so the output actually shows the
source video moving instead of freezing on frame 0.

IMPORTANT: nothing except the two protocol messages above may ever reach the
real stdout file descriptor, or the parent (animator.py) will hang / crash
trying to json.loads() a log line. MuseTalk's own modules call print() during
model loading and preprocessing, and shell out to ffmpeg via os.system(), both
of which would otherwise land on stdout. This file redirects Python-level
stdout to stderr before importing anything, and runs ffmpeg via subprocess
with its stdout explicitly discarded, keeping fd 1 clean for protocol JSON.
"""

import sys

# Grab the real stdout for protocol messages BEFORE anything else can print
# to it, then redirect Python's sys.stdout so every print() from MuseTalk's
# own modules (imported below) goes to stderr instead.
_PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr


def _send(obj) -> None:
    """Write one protocol line to the real stdout and flush immediately."""
    import json
    _PROTOCOL_OUT.write(json.dumps(obj) + "\n")
    _PROTOCOL_OUT.flush()


def _send_ready() -> None:
    _PROTOCOL_OUT.write("READY\n")
    _PROTOCOL_OUT.flush()


import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# PyTorch >=2.6 flipped torch.load's default to weights_only=True, which
# refuses to load legacy-format .tar/.pth checkpoints (e.g. the face-parsing
# ResNet18 weights MuseTalk ships). These are local files from the model
# download, not untrusted network input, so it's safe to restore the old
# default here rather than patch MuseTalk's vendored source.
_orig_torch_load = torch.load


def _torch_load_legacy_ok(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_legacy_ok

# ── MuseTalk internals (this file must live under <MUSETALK_PATH>/scripts/) ──
from musetalk.utils.face_parsing import FaceParsing  # noqa: E402
from musetalk.utils.utils import datagen, load_all_model  # noqa: E402
from musetalk.utils.preprocessing import get_landmark_and_bbox  # noqa: E402
from musetalk.utils.blending import get_image_prepare_material, get_image_blending  # noqa: E402
from musetalk.utils.audio_processor import AudioProcessor  # noqa: E402
from transformers import WhisperModel  # noqa: E402

try:
    from musetalk.utils.preprocessing import coord_placeholder
except ImportError:
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)

# ── tunables that animator.py doesn't send explicitly; read from env instead,
#    since the worker inherits the backend process's environment. ────────────
FPS = int(os.environ.get("AVATAR_FPS", "25"))
BATCH_SIZE = int(os.environ.get("MUSETALK_BATCH_SIZE", "8"))
EXTRA_MARGIN = int(os.environ.get("MUSETALK_EXTRA_MARGIN", "10"))
PARSING_MODE = os.environ.get("MUSETALK_PARSING_MODE", "jaw")
LEFT_CHEEK_WIDTH = int(os.environ.get("MUSETALK_LEFT_CHEEK_WIDTH", "90"))
RIGHT_CHEEK_WIDTH = int(os.environ.get("MUSETALK_RIGHT_CHEEK_WIDTH", "90"))
AUDIO_PAD_LEFT = int(os.environ.get("MUSETALK_AUDIO_PAD_LEFT", "2"))
AUDIO_PAD_RIGHT = int(os.environ.get("MUSETALK_AUDIO_PAD_RIGHT", "2"))
# How many trailing frames per chunk get their whisper context window frozen
# on the last fully-real chunk instead of predicted individually — see the
# comment at the call site in Worker.infer(). Defaults to AUDIO_PAD_RIGHT
# since that's exactly how many trailing frames' windows overlap the
# fabricated end-padding (see get_whisper_chunk).
TAIL_HOLD_FRAMES = int(os.environ.get("MUSETALK_TAIL_HOLD_FRAMES", str(AUDIO_PAD_RIGHT)))
print("Using MuseTalk tail-hold frames:", TAIL_HOLD_FRAMES, file=sys.stderr)
# Cap on how many source frames we'll extract/encode for a *video* avatar.
# Long videos get evenly re-sampled down to this many frames before the
# ping-pong cycle is built, so prep time and cache size stay bounded.
MAX_AVATAR_FRAMES = int(os.environ.get("MUSETALK_MAX_AVATAR_FRAMES", "250"))
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".gif"}


def _run_silent(cmd: list) -> None:
    """Run a subprocess with stdout/stderr fully captured (never inherited),
    so ffmpeg output can't corrupt the protocol stream on fd 1."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(cmd)}): {proc.stdout.decode(errors='replace')[-4000:]}"
        )


class Worker:
    def __init__(self, cfg: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_float16 = bool(cfg.get("use_float16")) and self.device.type == "cuda"

        print(f"[musetalk_worker] loading models on {self.device} (float16={self.use_float16})")

        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=cfg["unet_model_path"],
            vae_type=cfg.get("vae_type", "sd-vae"),
            unet_config=cfg["unet_config"],
            device=self.device,
        )
        self.timesteps = torch.tensor([0], device=self.device)

        if self.use_float16:
            self.pe = self.pe.half().to(self.device)
            self.vae.vae = self.vae.vae.half().to(self.device)
            self.unet.model = self.unet.model.half().to(self.device)

        self.weight_dtype = self.unet.model.dtype

        self.audio_processor = AudioProcessor(feature_extractor_path=cfg["whisper_dir"])
        self.whisper = WhisperModel.from_pretrained(cfg["whisper_dir"])
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)

        self.fp = FaceParsing(
            left_cheek_width=LEFT_CHEEK_WIDTH,
            right_cheek_width=RIGHT_CHEEK_WIDTH,
        )

        print("[musetalk_worker] models loaded")

    # ── avatar material (face crop, latent, mask), per source frame ────────
    # material["frames"] is always a list of dicts, each with keys:
    #   frame, bbox, latent, mask, mask_crop_box
    # For a still image this list has exactly one entry. For a video it has
    # one entry per (possibly downsampled) source frame, in playback order.

    def _build_frame_material(self, frame: np.ndarray, bbox) -> dict:
        x1, y1, x2, y2 = bbox
        y2 = min(y2 + EXTRA_MARGIN, frame.shape[0])
        bbox = [x1, y1, x2, y2]

        crop_frame = frame[y1:y2, x1:x2]
        resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latent = self.vae.get_latents_for_unet(resized_crop_frame)

        mask, mask_crop_box = get_image_prepare_material(
            frame, bbox, fp=self.fp, mode=PARSING_MODE
        )

        return {
            "frame": frame,
            "bbox": bbox,
            "latent": latent,
            "mask": mask,
            "mask_crop_box": mask_crop_box,
        }

    def _prepare_avatar_image(self, image_path: str) -> dict:
        frame = cv2.imread(image_path)
        if frame is None:
            raise RuntimeError(f"Could not read avatar image: {image_path}")

        coord_list, _ = get_landmark_and_bbox([image_path], 0)
        bbox = coord_list[0]
        if bbox == coord_placeholder:
            raise RuntimeError("No face detected in avatar image")

        return {"mode": "image", "frames": [self._build_frame_material(frame, bbox)]}

    def _prepare_avatar_video(self, video_path: str) -> dict:
        capture = cv2.VideoCapture(video_path)
        try:
            raw_frames = []
            while True:
                success, frame = capture.read()
                if not success or frame is None:
                    break
                raw_frames.append(frame)
        finally:
            capture.release()

        if not raw_frames:
            raise RuntimeError(f"Could not read any frames from avatar video: {video_path}")

        # Evenly downsample long videos so prep/cache stays bounded.
        if len(raw_frames) > MAX_AVATAR_FRAMES:
            idxs = np.linspace(0, len(raw_frames) - 1, MAX_AVATAR_FRAMES).round().astype(int)
            raw_frames = [raw_frames[i] for i in sorted(set(idxs.tolist()))]

        tmp_dir = tempfile.mkdtemp(prefix="musetalk_avatar_frames_")
        try:
            frame_paths = []
            for i, frame in enumerate(raw_frames):
                p = os.path.join(tmp_dir, f"{i:06d}.jpg")
                if not cv2.imwrite(p, frame):
                    raise RuntimeError(f"Could not write temp frame: {p}")
                frame_paths.append(p)

            coord_list, _ = get_landmark_and_bbox(frame_paths, 0)

            materials = []
            skipped = 0
            for frame, bbox in zip(raw_frames, coord_list):
                if bbox == coord_placeholder:
                    skipped += 1
                    continue
                materials.append(self._build_frame_material(frame, bbox))

            if not materials:
                raise RuntimeError("No face detected in any frame of avatar video")
            if skipped:
                print(f"[musetalk_worker] skipped {skipped}/{len(raw_frames)} "
                      f"video frames with no detected face")

            return {"mode": "video", "frames": materials}
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _prepare_avatar(self, image_path: str, coord_cache: str | None) -> dict:
        if coord_cache and os.path.exists(coord_cache):
            try:
                with open(coord_cache, "rb") as f:
                    cached = pickle.load(f)
                if isinstance(cached, dict) and "frames" in cached and cached["frames"]:
                    return cached
                print(f"[musetalk_worker] cache at {coord_cache} is in an old/invalid "
                      f"format, recomputing")
            except Exception:
                print(f"[musetalk_worker] cache at {coord_cache} unreadable, recomputing")

        is_video = Path(image_path).suffix.lower() in VIDEO_SUFFIXES
        material = (
            self._prepare_avatar_video(image_path)
            if is_video
            else self._prepare_avatar_image(image_path)
        )

        if coord_cache:
            os.makedirs(os.path.dirname(coord_cache), exist_ok=True)
            with open(coord_cache, "wb") as f:
                pickle.dump(material, f)

        return material

    # ── inference for one (avatar, audio) pair ──────────────────────────────

    @torch.no_grad()
    def infer(self, image_path: str, audio_path: str, output_path: str, coord_cache: str | None) -> None:
        material = self._prepare_avatar(image_path, coord_cache)
        frames_material = material["frames"]

        # Ping-pong cycle (forward then reverse) so a short/looping source
        # video still lines up with however many audio-driven frames we need.
        # For a still image this list is just [material] repeated, i.e. the
        # old single-frame behaviour falls out of the same code path.
        if len(frames_material) > 1:
            cycle = frames_material + frames_material[::-1]
        else:
            cycle = frames_material
        latent_cycle = [m["latent"] for m in cycle]

        whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(
            audio_path, weight_dtype=self.weight_dtype
        )
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=FPS,
            audio_padding_length_left=AUDIO_PAD_LEFT,
            audio_padding_length_right=AUDIO_PAD_RIGHT,
        )
        video_num = len(whisper_chunks)
        if video_num == 0:
            raise RuntimeError("No audio frames extracted (empty/too-short audio?)")

        # get_whisper_chunk right-pads the whisper feature array with zeros
        # "to prevent out of bounds" indexing. Each frame's context window is
        # audio_feature_length_per_frame wide, so the last AUDIO_PAD_RIGHT
        # frames' windows straddle into that fabricated-silence padding
        # instead of getting a fully real window like every earlier frame —
        # that partial-fake context is what shows up as a brief mouth
        # flutter right as the voice ends. Freeze those frames on the latest
        # chunk that had a fully real window instead of predicting each from
        # a partially-fake one. video_num is unchanged, so muxed audio length
        # (and hence which words are heard) is untouched — only which
        # whisper features drive the last couple of frames changes.
        print(f"[musetalk_worker] TAIL_HOLD_FRAMES={TAIL_HOLD_FRAMES}", file=sys.stderr)
        if TAIL_HOLD_FRAMES > 0 and video_num > TAIL_HOLD_FRAMES:
            last_good = whisper_chunks[video_num - TAIL_HOLD_FRAMES - 1]
            whisper_chunks[video_num - TAIL_HOLD_FRAMES:] = last_good.unsqueeze(0).expand(
                TAIL_HOLD_FRAMES, *last_good.shape
            )

        tmp_dir = tempfile.mkdtemp(prefix="musetalk_")
        try:
            gen = datagen(whisper_chunks, latent_cycle, BATCH_SIZE)
            frame_idx = 0
            for whisper_batch, latent_batch in gen:
                audio_feature_batch = self.pe(whisper_batch.to(self.device))
                latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
                pred_latents = self.unet.model(
                    latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch
                ).sample
                pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
                recon = self.vae.decode_latents(pred_latents)

                for res_frame in recon:
                    # datagen selects latents by `i % len(latent_cycle)` in
                    # the same order we're consuming output frames here, so
                    # indexing `cycle` the same way keeps the background
                    # frame/bbox/mask in sync with the latent that produced
                    # this output frame.
                    src = cycle[frame_idx % len(cycle)]
                    x1, y1, x2, y2 = src["bbox"]

                    res_frame_resized = cv2.resize(
                        res_frame.astype(np.uint8), (x2 - x1, y2 - y1)
                    )
                    combined = get_image_blending(
                        src["frame"].copy(), res_frame_resized, src["bbox"],
                        src["mask"], src["mask_crop_box"],
                    )
                    cv2.imwrite(os.path.join(tmp_dir, f"{frame_idx:08d}.png"), combined)
                    frame_idx += 1

            silent_video = os.path.join(tmp_dir, "silent.mp4")
            _run_silent(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-r", str(FPS),
                    "-f", "image2",
                    "-i", os.path.join(tmp_dir, "%08d.png"),
                    "-vcodec", "libx264",
                    "-vf", "format=yuv420p",
                    "-crf", "18",
                    silent_video,
                ]
            )

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            _run_silent(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-i", str(Path(audio_path).resolve()),
                    "-i", silent_video,
                    # AAC re-encodes audio in fixed 1024-sample frames while the video
                    # is copied frame-exact, so the encoded audio track ends up a hair
                    # shorter than the (already audio-driven) video length. -shortest
                    # can't trim mid-AAC-frame, so without this pad the video is
                    # occasionally left with one trailing silent frame after the audio
                    # ends. Padding audio guarantees video is always the shorter
                    # stream, so -shortest trims to its exact, frame-accurate length.
                    "-af", "apad=pad_dur=0.2", #Zeeshan: Added for testing
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    str(Path(output_path).resolve()),
                ]
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    init_line = sys.stdin.readline()
    if not init_line:
        return  # parent closed stdin before sending config

    cfg = json.loads(init_line)

    try:
        worker = Worker(cfg)
    except Exception:
        print("[musetalk_worker] FATAL during model load:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # No READY was sent, so the parent's readline() will time out and
        # surface this as a startup failure — that's the intended behaviour.
        return

    _send_ready()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
            t0 = time.time()
            worker.infer(
                job["image"], job["audio"], job["output"], job.get("coord_cache")
            )
            print(f"[musetalk_worker] job done in {time.time() - t0:.2f}s -> {job['output']}")
            _send({"status": "ok"})
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            _send({"status": "error", "msg": str(e)})


if __name__ == "__main__":
    main()