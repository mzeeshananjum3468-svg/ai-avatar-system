# Backend map (FastAPI)

Read `../CLAUDE.md` first for the whole-system picture. This file is the
backend service-by-service detail.

## Layout

```
app/
  main.py              -- not present as a single file at this path from repo root;
                          entrypoint is app/ (uvicorn app.main:app) — see api/v1/*.py for routes
  config.py            Settings (pydantic-settings, reads .env at repo root)
  models.py            SQLAlchemy models — schema source of truth
  schemas.py            Pydantic request/response models
  database.py           AsyncSessionLocal / engine setup
  websocket.py           ConnectionManager — the real-time chat/animation pipeline
  telemetry.py           span() context manager — no-op unless OTEL_ENABLED
  celery_app.py          Celery app (background jobs beyond the WS pipeline)
  api/v1/
    avatars.py           upload/list/get/rename/delete avatar, metadata, voice assignment
    sessions.py          create/list/get/end/delete session
    messages.py          REST send/list/edit/delete (WS is the primary path; REST exists too)
    conversations.py     conversation list/rename/summarize/delete (auto-titled per session)
    voices.py             voice cloning (clone/list/delete/preview/synthesize)
    users.py              register/login/logout/profile (JWT + httpOnly cookie)
  services/
    animator.py           MuseTalk (persistent worker) / simple ffmpeg fallback
    tts.py                 Chatterbox TTS + edge-tts/gTTS fallback chain
    stt.py                 faster-whisper
    llm.py                 Anthropic/OpenAI/Ollama facade, streaming + non-streaming
    liveportrait.py        idle/thinking loop + reversed-companion generation
    avatar_processor.py    uploaded image/video → processed still + thumbnail + metadata
    storage.py              Local filesystem or S3, behind one interface
    cache.py                (see file — misc caching helpers)
  middleware/
    rate_limiter.py, security.py
alembic/versions/      migrations, applied in filename-number order
models/MuseTalk/       tracked copy of MuseTalk + scripts/musetalk_worker.py
tests/                  pytest suite
```

Migrations aren't run automatically — after editing `app/models.py` and adding
a migration, remember `alembic upgrade head` actually has to be run for the
schema change to take effect.

## WebSocket pipeline (`app/websocket.py`) — the real hot path

`ConnectionManager` holds all per-session state in `self.session_data` (a
plain dict keyed by session_id) plus `self._active_turns` (task handles, for
barge-in) and `self._send_locks` (one lock per session so the turn task and
the receive loop can't interleave a send).

**Connect flow**: `connect()` accepts the socket, loads session/avatar/voice
state from Postgres (`_load_session_data`), then **hard-gates** on running a
throwaway warmup job for both MuseTalk (`avatar_animator.warmup_avatar`) and
Chatterbox (`tts_service.warmup`) concurrently — the caller (the WS route)
doesn't start its receive loop until `connect()` returns, so the first *real*
turn never eats model-load/CUDA-warmup/face-landmark-cache latency. The
client is told `warmup_start` then `session_ready`, unconditionally (even
with no avatar yet), because the frontend flips to "warming up" optimistically
as soon as the socket opens.

**Turn flow** (`handle_text_input` → `_handle_text_input_inner`):
1. Any in-flight turn for the session is cancelled (`interrupt_active_turn`) —
   this is barge-in. The cancelled task's `finally`/`except CancelledError`
   blocks re-raise, so cleanup still runs.
2. `asyncio.gather(_llm_producer(...), _animate_from_queue(...))` — a
   bounded `asyncio.Queue(maxsize=4)` connects them. The LLM producer streams
   tokens (each forwarded live as `type: 'token'`), and calls `_drain_chunks`
   on a text buffer to detect chunk boundaries.
3. **Chunking rule** (`_drain_chunks`, `_SENTENCE_RE`/`_CLAUSE_RE`): the
   opening chunk is cut at the first CLAUSE boundary (comma/semicolon/
   colon/dash) once ≥ `_MIN_FIRST_CHUNK_LEN` (40) chars — ships fast. Every
   chunk after that is cut at SENTENCE boundaries once ≥
   `_MIN_SENTENCE_LEN` (35) chars — smoother prosody, fewer TTS calls. A
   run-on with no punctuation force-flushes at `_MAX_CHUNK_CHARS` (200) so
   nothing stalls forever waiting for a boundary. The 35/40 floors exist
   because Chatterbox's EOS-forcing heuristic
   (`AlignmentStreamAnalyzer`) is unreliable on very short text (see
   `tts.py`'s `_patch_alignment_analyzer` docstring) and over-generates
   trailing gibberish below ~5 tokens.
4. `_animate_from_queue` consumes each chunk: TTS → MuseTalk animate() →
   upload to storage → send `type: 'video_chunk'` with a URL. It never sends
   `total_chunks` up front (`-1` = streaming, unknown count) — the client
   relies on `video_chunk_end` (server done producing) + its own drained-queue
   check, not a count, to know a turn is fully over.
5. Both user and assistant messages are persisted to Postgres as they
   complete (`_persist_message`), and a `Conversation` row is lazily created
   and auto-titled from the first user turn (`_ensure_conversation_title`).

**Barge-in mechanics**: `_spawn_turn` creates the task but never awaits it —
the WS receive loop returns to `receive_json()` immediately so a fresh
message (or explicit interrupt) can cancel the in-flight task at any point.
`interrupt_active_turn` also sends `type: 'interrupted'` so the client drops
its queued/downloading chunks (see `frontend/CLAUDE.md`'s `resetPlayback`).

**Stale session reaping**: `cleanup_stale()` runs every 5 min
(`start_cleanup_task`), tearing down sessions whose socket is gone or whose
`last_activity` exceeds `STALE_SESSION_TTL_SECS` (2h). Snapshotting the
candidate list happens under `_mutation_lock` so it can't race a fresh
`connect()` for the same session id.

## MuseTalk (`services/animator.py`)

`AVATAR_ENGINE=musetalk` (default) or `simple` (ffmpeg still-image + audio,
no lip-sync — automatic fallback if MuseTalk isn't found, or if any MuseTalk
call raises).

MuseTalk is **not** imported in-process (its own conda env/torch build) —
it runs as a **persistent worker subprocess**
(`models/MuseTalk/scripts/musetalk_worker.py` or the checkout's own copy,
`_resolve_worker_script` prefers the checkout), communicating over
stdin/stdout as line-delimited JSON: one `{unet_model_path, ...}` init
message, then one `{image, audio, output, coord_cache, bbox_shift}` job per
line, one `{status, ...}` result line back. This keeps model weights loaded
across every chunk/turn instead of paying load time per call. `_worker_lock`
serializes job submission (one worker, one GPU). If the worker dies
(OOM/segfault/timeout) the code resets `_worker_proc = None` so the *next*
job respawns cleanly instead of writing into a dead pipe forever.

`coord_cache` is a per-avatar pickle of face-detection/landmark results,
stored next to the avatar's own source file (`{source}.musetalk_cache.pkl`)
so it survives redeploys and gets cleaned up when the avatar is deleted.
`bbox_shift` is a manual per-avatar face-crop tuning knob
(`AvatarMetadataUpdate.bbox_shift`, -30..30) — there's no way to compute it
automatically; it's set by comparing generated video quality by eye.

If the source avatar is a **video** (not a still image), MuseTalk animates
every frame of it (head motion + lip-sync together) — the animate() path
deliberately does NOT extract a single frame before calling MuseTalk (that
would silently downgrade every video avatar to flat-photo lip-sync); only the
`simple` ffmpeg fallback needs a still frame, and extracts one itself.

## TTS (`services/tts.py`)

Fallback chain: **Chatterbox** (Resemble AI, GPU, zero-shot voice cloning,
23 languages) → **edge-tts** (Microsoft neural voices, free, no GPU/key,
much better prosody than gTTS) → **gTTS** (last resort, network-only).
`SynthResult.fallback`/`.engine`/`.voice_cloned` let the WS pipeline notify
the client exactly once per turn when cloning silently dropped
(`type: 'tts_fallback'`).

`_patch_alignment_analyzer` monkeypatches a bug in the installed
`chatterbox-tts==0.1.4` package: its repetition-based EOS forcing fires on
just 2 identical consecutive tokens (comment claims 3, code slices `[-2:]`)
at ANY point in generation, not just the tail — this truncated audio
mid-sentence on ordinary held vowels/plosives. Patched to require 3-in-a-row
AND only once the utterance is otherwise complete.

`clean_audio()` trims trailing near-silence from Chatterbox's output (it
tails off into a few hundred ms of digital silence every call) — harmless
for playback, but every silent frame otherwise gets fed into MuseTalk's
lip-sync model (never trained on long silence) and produces a visible mouth
flutter. `_trim_trailing_silence` is the pydub equivalent for the
edge-tts/gTTS paths.

## LivePortrait idle/thinking loops (`services/liveportrait.py`)

Generates two ~10s silent loop videos per avatar at upload time (via
`_generate_idle_thinking_videos` in `api/v1/avatars.py`, run as a background
job): **idle** (neutral resting driving video) and **thinking** (a distinct
driving video shown while a turn is in flight). Also generates a
frame-reversed companion of each (`_reverse()`, `ffmpeg -vf reverse`, a full
re-encode — reversing can't stream-copy) so the frontend can ping-pong
between forward/reversed instead of seeking (see root CLAUDE.md's "idle-loop
stuck frame" section). Reversed-clip generation is wrapped in its own
try/except — failure there doesn't fail the whole avatar, it just leaves
that field `None` and the frontend falls back to a plain forward loop.

LivePortrait itself, like MuseTalk, is invoked as a subprocess in its own
conda env — but as a one-shot process per video (`_run_inference`), not a
persistent worker, since this only runs once per avatar upload rather than
once per chat turn. Runs are sequential (idle then thinking), not
concurrent, to avoid two GPU model loads racing for VRAM on a
single-GPU box.

Auto-discovery (`initialize()`): checkout path defaults to
`../LivePortrait` (sibling of `ai-avatar-system`), python to
`../conda_envs/LivePortrait/bin/python`, idle driving video to MuseTalk's
`yongen.mp4`, thinking driving video to `ai-avatar-system/source.mp4`. All
overridable via `.env` (`LIVEPORTRAIT_PATH`, `LIVEPORTRAIT_PYTHON`,
`IDLE_DRIVING_VIDEO`, `THINKING_DRIVING_VIDEO`). If any piece is missing,
`available` is `False` and new avatars simply don't get loop videos
(`idle_video_url`/etc. stay NULL, frontend shows a static image instead).

`avatar.status` stays `"processing"` (sessions.create_session rejects
non-"ready" avatars) until the loop videos are attached, flips to `"failed"`
if LivePortrait isn't configured or generation errors.

## Storage (`services/storage.py`)

One interface (`upload_file`, `download_file`, `get_local_path`,
`delete_file`, `get_url`, `serving_url`), two implementations selected by
`USE_LOCAL_STORAGE` (default `True`):
- **Local**: files under `LOCAL_STORAGE_PATH` (`uploads/`), served by
  FastAPI's StaticFiles at `/uploads/...`. `serving_url` == `get_url`
  (everything's public locally).
- **S3**: objects uploaded `ACL=private`; `serving_url` returns a CloudFront
  URL if `CLOUDFRONT_DOMAIN` is set, else a time-limited presigned URL —
  callers must always go through `serving_url`, never assume the raw
  `get_url` is fetchable by a browser.

`websocket.py`'s `_resolve_local_image` prefers, in order: the raw uploaded
**video** source (real head motion) → the LivePortrait **idle loop** →
the static processed **image** — downloading+caching from remote storage
into `TMPDIR/avatars/` if not already local.

## Config (`app/config.py`)

Single `Settings` (pydantic-settings) reads `.env` at repo root. Notable
non-obvious ones: `MUSETALK_TAIL_HOLD_FRAMES` is injected into the MuseTalk
worker's `os.environ` explicitly at worker-spawn time
(`animator.py::initialize`) — pydantic-settings' `env_file` only populates
the `settings` object, never `os.environ`, so anything the worker
subprocess needs has to be forwarded manually. `_validate_secrets` refuses
to boot if `SECRET_KEY`/`JWT_SECRET_KEY` are left at the placeholder default
or under 32 chars.

## Gotchas / invariants worth knowing before touching this code

- Every service (`animator`, `tts`, `stt`, `llm`, `liveportrait`) is a module-
  level singleton instantiated at import time and lazily initializes its
  heavy model/subprocess on first real use — don't add per-request
  instantiation.
- `session_data` and `_active_turns`/`_send_locks` are **in-process** dicts —
  this backend is not horizontally scalable across multiple processes/pods
  as-is (a WS session must stay pinned to the process that accepted it).
- Temp files for an in-flight session live under
  `TMPDIR/avatar-session-{id}/` with `0o700`/`0o600` permissions
  (`_private_session_dir`/`_write_private_bytes`) — deliberate hardening
  against another user on a shared host reading raw mic audio or in-flight
  video chunks.
- `bbox_shift` flows: avatar metadata → session_data at connect/set →
  passed to both `warmup_avatar` and every `animate()` call — if you add a
  new per-avatar MuseTalk tuning knob, follow this exact path or it'll only
  apply inconsistently between warmup and real turns.
