# Frontend map (Next.js 14)

Read `../CLAUDE.md` first for the whole-system picture. This is almost
entirely a single-page app — nearly everything that matters is one file:

```
components/ChatInterface.tsx   (~2000 lines) — the entire chat/video experience
components/AvatarList.tsx, AvatarUpload.tsx, AuthModal.tsx, HistoryPanel.tsx,
  SettingsPanel.tsx, VoicePanel.tsx, WaveformVisualizer.tsx    supporting panels
lib/api.ts                     axios REST client + WS URL builder
lib/types.ts                   shared TS types incl. the WS message discriminated union
store/useStore.ts              zustand — auth/session-id/theme ONLY, not chat state
```

Chat state (messages, playback, WS lifecycle) is **all local to
ChatInterface.tsx** via `useState`/`useRef` — `useStore` only holds
cross-page state (auth token, selected avatar/session id, theme).

## `lib/api.ts`

Thin axios wrapper (`apiClient`) with an interceptor that attaches
`Authorization: Bearer <token>` from localStorage (via zustand's persisted
`avatar-system-storage` key) and `withCredentials: true` so the httpOnly
auth cookie also rides along. A 401 anywhere triggers a global
`auth:logout` custom event + localStorage wipe. `buildSessionWsUrl` builds
the `ws://.../ws/session/{id}?token=...` URL — the WS constructor can't set
headers, so the JWT rides as a query param instead.

## `lib/types.ts`

`WsMessage` is a discriminated union over every inbound WS event type —
extend this (and the backend's corresponding `send_message` call) together
whenever a new WS message type is added, so the handler in ChatInterface can
rely on field presence instead of optional-chaining. `Avatar` mirrors
`AvatarResponse` on the backend — includes `idle_video_url`/
`thinking_video_url` and their `_reversed` counterparts.

## `components/ChatInterface.tsx` — the frame scheduler

This is the part worth understanding deeply before touching video playback.
The core problem it solves: the server streams a sequence of independently-
generated video clips (reply chunks, plus idle/thinking loops in the gaps),
and naively swapping `<video src>` between them causes a black flash or
frozen-frame stutter at every boundary. The fix is a persistent
`<canvas>` that always has *something* painted on it, fed by hidden
decode-only `<video>` elements.

**Elements** (all hidden, `opacity-0 pointer-events-none`, decode-only):
- `videoARef` / `videoBRef` — a ping-pong **pair** used exclusively for
  reply chunks (`transitionTo`/`playChunk`/`commitPrebuffered`). One is
  always "active" (`activeIdxRef`), the other is "standby" and gets the
  *next* chunk preloaded into it while the active one is still playing.
- `videoIdleRef` / `videoIdleRevRef` — a **separate** pair used only for the
  idle/thinking loop (`playLoop`/`currentLoop`). Entirely independent of the
  A/B pair — there's no shared standby slot for these to race over.
- `canvasRef` — the only thing actually visible. A `requestAnimationFrame`
  loop (mounted once) draws whichever element `switchTo()` last pointed at
  onto the canvas every frame, cross-dissolving over
  `FRAME_SCHEDULER_FADE_MS` (180ms) when the target just changed. It
  deliberately never clears the canvas speculatively — if the current
  source momentarily has no decoded frame, the canvas just keeps showing its
  last-painted pixels instead of flashing black.

**`switchTo(el, {fade})`** is the *sole* authoritative "what's on screen"
function — every other function that wants to change visible content must
route through it rather than touching a `<video>`'s style/visibility
directly. It also derives `isSpeaking` from the target's
`dataset.mode` (`'chunk:N'` → true), so that piece of state can never
desync from what's actually painted.

**Reply-chunk pipeline**: `pumpDownloads()` fetches each announced chunk into
a Blob (a real network round-trip, which incidentally gives the
idle/thinking transition time to actually commit a frame before being
superseded) and pushes it onto `playQueueRef`. `tryPlayNext()` pops the
queue and either `commitPrebuffered()` (if `prebufferNext()` already warmed
the standby element with this exact item) or `playChunk()` → `transitionTo()`
(cold path: load into standby, wait for `canplay` + an actual painted frame
via `revealWhenPainted`, then swap). Both commit paths call `openGate()`
last, after every DOM/ref mutation — it flips on the text bubble for this
turn's reply the moment its first chunk is actually visible, not before.

**Idle/thinking loop**: `currentLoop()` decides idle vs thinking (thinking
only while `isRespondingRef` is true and a thinking clip exists) and returns
`{src, reversedSrc, mode}` — `reversedSrc` is null when this avatar has no
reversed companion clip. `playLoop()` branches:
- **No reversed companion** (older avatars, or server-side reversal
  failed): plain `<video loop=true>` on `videoIdleRef` alone. Simple,
  zero seeking, but has a visible jump-cut at the loop point since the
  driving video's first frame ≠ last frame.
- **Reversed companion exists**: both `videoIdleRef` (forward) and
  `videoIdleRevRef` (reversed) load with `loop=false`, and a pair of
  `'ended'` listeners (wired once in the mount `useEffect`, guarded on
  `frontElRef` so a stale event from an already-superseded element can't
  restart a hand-off) hand off to the *other* element on every `ended` —
  each side always plays **natively forward**, never seeked. This is the
  fix for a real bug: reverse-seeking compressed (H.264) video via
  `currentTime` scrubbing requires the decoder to redecode forward from the
  nearest prior keyframe on every single step, which stalled visibly at
  every loop boundary. (The old code did exactly that — a manual
  `requestAnimationFrame` "pendulum" scrub — and has been fully removed.)

**When the loop reveals vs. stays hidden behind a chunk** —
`nothingComingSoon()` (queue empty AND download queue empty AND not
currently downloading) is the single source of truth every call site must
use before calling `playLoop()`. A weaker check (e.g. just
"nothing playing right now") flips to idle/thinking during an ordinary
inter-chunk network gap and then immediately back — a visible flash to a
different clip and back, which is exactly the seam this whole file exists
to prevent. `pipelineDrained()` is the stricter cousin used by
`maybeFinishResponding()` to decide the *whole turn* is over (server sent
`video_chunk_end` AND nothing local is still outstanding).

**Barge-in**: `resetPlayback()` bumps `currentGenRef` (so any in-flight
`.then()` from a stale download discards itself), clears both queues,
revokes blob URLs, resets the A/B pair to a blank `src`, and calls
`playLoop({immediate: true})` — `immediate` skips the cross-dissolve so a
lingering fade from whatever chunk just got cut off doesn't read as a
stutter.

## Dev workflow notes

- `npm run dev` picks up branch changes on save — no rebuild needed to test
  a branch switch. A rebuild (`npm run build` + `npm run start`) is only
  needed to validate the production build path.
- Chat state resets are all manual (no page reload) — `resetPlayback()` is
  the pattern to follow for anything that needs to fully re-arm the video
  pipeline (barge-in today; any future "start a new turn cleanly" need
  should reuse it rather than partially resetting individual refs).
