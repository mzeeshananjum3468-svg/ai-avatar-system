# AvatarAI — architecture map

Real-time talking-head avatar app. Browser mic/text → FastAPI WebSocket →
Whisper STT → Claude/GPT/Ollama (streaming) → Chatterbox TTS → MuseTalk
lip-sync → chunked video streamed back and played seamlessly on a canvas.
Also generates silent LivePortrait idle/thinking loop videos per avatar for
the gaps between replies.

For file-level detail, read **`backend/CLAUDE.md`** and
**`frontend/CLAUDE.md`** — this file is only the cross-cutting map. See also
`README.md` (product framing) and `SETUP_GUIDE.md` (install/deploy steps).

## Directory map

```
backend/           FastAPI app (see backend/CLAUDE.md)
  app/                  application code
  alembic/versions/     migrations — DB schema source of truth is app/models.py
  models/MuseTalk/      tracked copy of the MuseTalk checkout + persistent worker script
  tests/                pytest suite
  uploads/               local-storage mode: avatar images, generated videos
frontend/          Next.js 14 app (see frontend/CLAUDE.md)
  app/                  routes (mostly just app/page.tsx — this is a single-page app)
  components/           ChatInterface.tsx is the one that matters; rest are supporting panels
  lib/                   api.ts (REST client), types.ts (shared TS types)
  store/useStore.ts     zustand — auth/session/theme, NOT chat state (that's all local to ChatInterface)
scripts/           setup + benchmarking scripts (setup_musetalk.sh, benchmark_pipeline_rtf.py, etc.)
infrastructure/    IaC for AWS deploy
nginx/             reverse proxy config for prod
```

Sibling checkouts this repo expects to find next to it (same parent dir,
`Zeeshan/`): `MuseTalk/` and `LivePortrait/`, each with their own conda env
under `conda_envs/`. Both backend services auto-discover these paths (see
`backend/CLAUDE.md`) with `.env` overrides available.

## The one turn, end to end

1. Browser sends `{type: 'text', text}` or `{type: 'audio', audio}` over the
   session WebSocket (`backend/app/websocket.py`, `ConnectionManager`).
2. Any prior in-flight turn for that session is cancelled first (barge-in —
   sub-100ms cutoff so a new user input always wins).
3. LLM streams tokens; each token is forwarded to the client immediately
   (`type: 'token'`) AND accumulated into a buffer that's chunked at
   clause/sentence boundaries (`_drain_chunks` in websocket.py).
4. Each chunk goes through TTS (Chatterbox, with edge-tts/gTTS fallback) then
   MuseTalk lip-sync (persistent GPU worker subprocess), sequentially — not
   pipelined ahead of the LLM. The resulting video is uploaded to storage and
   its URL sent as `type: 'video_chunk'`.
5. The frontend downloads each chunk into a blob, queues it, and plays chunks
   back-to-back on a **canvas-based frame scheduler** (see
   `frontend/CLAUDE.md`) that cross-dissolves between clips so there's never
   a black flash or frozen frame at a boundary.
6. Whenever nothing is actually playing/queued/downloading, the canvas shows
   a silent, LivePortrait-generated idle (or "thinking", while a turn is in
   flight) loop instead of a static image.

## Two loop-closing feedback threads that shaped the current design

- **Silence padding → cross-chunk audio context**: TTS chunks used to be
  synthesized independently with silence at the boundaries, which MuseTalk
  would render as a visible "mouth closes then reopens" seam between chunks.
  Fixed with real audio context carried across chunk boundaries plus a tail
  hold (`MUSETALK_TAIL_HOLD_FRAMES` in the worker) instead of silence.
- **Idle-loop stuck frame**: the idle/thinking loop needs to play a driving
  video's motion smoothly forever, but browsers don't support negative
  `playbackRate` and reverse-seeking a compressed H.264 file (`currentTime`
  scrubbing) stalls at every loop boundary (a decoder reverse-seek redecodes
  forward from the nearest keyframe on every step). Fixed by having
  LivePortrait generate a frame-reversed companion clip
  (`ffmpeg -vf reverse`, full re-encode, done once at avatar-upload time) and
  having the client ping-pong between the forward and reversed clips —
  **both always played natively forward**, never seeked. Falls back to a
  plain `loop=true` single clip when no reversed companion exists (older
  avatars uploaded before this feature, or if the reversal step failed).

## Known limitations / open items

- Existing avatars created before the reversed-clip migration have
  `idle_video_reversed_url`/`thinking_video_reversed_url` = NULL forever —
  there's no backfill job. They silently use the `loop=true` fallback
  (functionally fine, just has a hard jump-cut at the loop point instead of
  the smoother ping-pong).
- Chunking thresholds in `websocket.py`
  (`_MIN_SENTENCE_LEN`/`_MIN_FIRST_CHUNK_LEN`/`_MAX_CHUNK_CHARS`) are tuned
  for low first-chunk latency, not throughput. `scripts/benchmark_pipeline_rtf.py`
  measures the real TTS→MuseTalk hand-off cost (`combined_rtf`) if this ever
  needs re-tuning against a fixed-cost-per-job model.
- Backend "Option D" (server-side frame crossfade at MuseTalk chunk
  boundaries, as an alternative/complement to the client-side scheduler) was
  discussed and deliberately deferred — not implemented.
