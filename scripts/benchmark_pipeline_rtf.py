#!/usr/bin/env python3
"""
Measure end-to-end TTS -> MuseTalk Real-Time Factor (RTF): given a piece of
text, how many seconds of wall-clock compute does (a) Chatterbox TTS need to
speak it, (b) MuseTalk need to lip-sync the result, and (c) the two stages
combined?

scripts/benchmark_musetalk_rtf.py already answers this for MuseTalk alone,
fed with silent audio of a known duration. This script drives the real
production hand-off instead (see backend/app/celery_app.py's
generate_video_task and backend/app/websocket.py's _animate_from_queue):

    tts_service.synthesize(text, audio_path)          # TTS stage
    avatar_animator.animate(image, audio_path, video) # MuseTalk stage

For every run it reports THREE numbers, not just one:

  - tts_rtf       = tts_compute_s      / audio_seconds
  - musetalk_rtf  = musetalk_compute_s / audio_seconds
  - combined_rtf  = (tts_compute_s + musetalk_compute_s) / audio_seconds

All three use the same convention as benchmark_musetalk_rtf.py: <1 means
faster than real time. throughput_factor = audio_seconds / compute_s is
also reported for each (the ">1 means it outruns real time" number).

`combined_rtf` treats the two stages as strictly sequential (TTS finishes,
then MuseTalk starts) — the correct number for a "generate everything, then
play" pipeline. In the live websocket path the two stages instead run
sentence-by-sentence (TTS for sentence N+1 can overlap MuseTalk for sentence
N), so combined_rtf is a worst-case bound, not what a pipelined stream
actually pays per second — that's exactly why the per-stage numbers are kept
alongside it instead of only reporting the sum.

Usage:
    conda_envs/ai-avatar/bin/python scripts/benchmark_pipeline_rtf.py
    conda_envs/ai-avatar/bin/python scripts/benchmark_pipeline_rtf.py --sentence-counts 1 2 4 8 --repeats 5
    conda_envs/ai-avatar/bin/python scripts/benchmark_pipeline_rtf.py --speaker-wav backend/voice_profiles/xxx.wav --json out.json

Run against an otherwise-idle GPU (no backend server already up) for clean
numbers — a resident backend process competes for the same CUDA context.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Reuses BACKEND_DIR/sys.path/chdir + settings import, media_duration() (ffprobe
# wrapper) and find_default_source_image() from the MuseTalk-only benchmark —
# importing it doesn't spawn anything, main() only runs under __main__.
from benchmark_musetalk_rtf import (  # noqa: E402
    BACKEND_DIR,
    COMFORTABLE_MARGIN,
    find_default_source_image,
    media_duration,
)

from app.services.animator import avatar_animator  # noqa: E402
from app.services.tts import tts_service  # noqa: E402

BASE_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the sun sets slowly "
    "behind the mountains. "
)
DEFAULT_SENTENCE_COUNTS = [1, 2, 4, 8]


def find_default_speaker_wav() -> Optional[Path]:
    voice_dir = BACKEND_DIR / "voice_profiles"
    if not voice_dir.exists():
        return None
    for candidate in sorted(voice_dir.glob("*.wav")):
        return candidate
    return None


def make_text(sentence_count: int) -> str:
    """Content doesn't matter for RTF — only how much audio it produces — so
    repeating one sentence keeps the benchmark reproducible, same rationale
    as make_silence() in the MuseTalk-only script."""
    return (BASE_SENTENCE * sentence_count).strip()


async def run_once(
    text: str, source_image: Path, speaker_wav: Optional[Path], language: str, work_dir: Path, tag: str,
) -> dict:
    audio_path = work_dir / f"{tag}_audio.wav"
    video_path = work_dir / f"{tag}_video.mp4"

    t0 = time.time()
    await tts_service.synthesize(
        text=text,
        output_path=str(audio_path),
        speaker_wav=str(speaker_wav) if speaker_wav else None,
        language=language,
    )
    tts_compute_s = time.time() - t0

    audio_s = media_duration(audio_path)

    t0 = time.time()
    await avatar_animator.animate(str(source_image), str(audio_path), str(video_path))
    musetalk_compute_s = time.time() - t0

    combined_compute_s = tts_compute_s + musetalk_compute_s

    audio_path.unlink(missing_ok=True)
    video_path.unlink(missing_ok=True)

    return {
        "audio_duration_s": round(audio_s, 3),
        "tts_compute_s": round(tts_compute_s, 3),
        "musetalk_compute_s": round(musetalk_compute_s, 3),
        "combined_compute_s": round(combined_compute_s, 3),
        "tts_rtf": round(tts_compute_s / audio_s, 3),
        "musetalk_rtf": round(musetalk_compute_s / audio_s, 3),
        "combined_rtf": round(combined_compute_s / audio_s, 3),
        "tts_throughput_factor": round(audio_s / tts_compute_s, 3),
        "musetalk_throughput_factor": round(audio_s / musetalk_compute_s, 3),
        "combined_throughput_factor": round(audio_s / combined_compute_s, 3),
    }


def print_verdict(label: str, min_throughput: float) -> None:
    if min_throughput >= COMFORTABLE_MARGIN:
        print(
            f"  {label}: consistently faster than real time (worst case "
            f"{min_throughput:.2f}x) — comfortable margin for continuous streaming."
        )
    elif min_throughput >= 1.0:
        print(
            f"  {label}: faster than real time but not by a wide margin "
            f"(worst case {min_throughput:.2f}x) — streaming can work, but size "
            f"buffers for the worst case above, not the average."
        )
    else:
        print(
            f"  {label}: barely real time or slower in at least one case "
            f"(worst case {min_throughput:.2f}x) — continuous low-latency "
            f"streaming is not realistic as-is for this stage alone."
        )


async def main_async(args: argparse.Namespace) -> None:
    source_image = args.source_image or find_default_source_image()
    if source_image is None:
        raise SystemExit(
            "No avatar image found under backend/uploads/avatars/*/image.jpg — pass --source-image"
        )
    if not Path(source_image).exists():
        raise SystemExit(f"--source-image not found: {source_image}")
    source_image = Path(source_image)

    speaker_wav = args.speaker_wav or find_default_speaker_wav()
    if speaker_wav is not None and not Path(speaker_wav).exists():
        raise SystemExit(f"--speaker-wav not found: {speaker_wav}")

    texts = args.texts if args.texts else [make_text(n) for n in args.sentence_counts]

    work_dir = Path(args.work_dir) if args.work_dir else BACKEND_DIR / "uploads" / "_rtf_bench_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Avatar: {source_image}")
    print(f"Speaker WAV: {speaker_wav or '(default voice)'}")
    print(f"Scratch dir: {work_dir}\n")

    print("Warm-up (model loads + kernel warmup for both stages, not timed)...")
    await tts_service.warmup(speaker_wav=str(speaker_wav) if speaker_wav else None, language=args.language)
    await avatar_animator.warmup_avatar(str(source_image))
    print("Warm-up done.\n")

    rows: list[dict] = []
    header = (
        f"{'text#':>6}  {'run':>4}  {'audio_s':>8}  {'tts_s':>7}  {'tts_rtf':>8}  "
        f"{'mt_s':>7}  {'mt_rtf':>7}  {'total_s':>8}  {'combined_rtf':>13}"
    )
    print(header)
    for text_id, text in enumerate(texts, start=1):
        for run in range(1, args.repeats + 1):
            tag = f"t{text_id}_r{run}"
            result = await run_once(text, source_image, speaker_wav, args.language, work_dir, tag)
            result["text_id"] = text_id
            result["run"] = run
            result["sentence_count"] = (
                args.sentence_counts[text_id - 1] if not args.texts else None
            )
            rows.append(result)
            print(
                f"{text_id:>6}  {run:>4}  {result['audio_duration_s']:>7.2f}s  "
                f"{result['tts_compute_s']:>6.2f}s  {result['tts_rtf']:>8.2f}  "
                f"{result['musetalk_compute_s']:>6.2f}s  {result['musetalk_rtf']:>7.2f}  "
                f"{result['combined_compute_s']:>7.2f}s  {result['combined_rtf']:>13.2f}"
            )

    try:
        work_dir.rmdir()
    except OSError:
        pass  # leave stray files (e.g. a failed run's leftovers) for inspection

    # ── summary ──────────────────────────────────────────────────────────────
    def summarize(key: str) -> tuple[float, float]:
        vals = [r[key] for r in rows]
        return statistics.mean(vals), min(vals)

    mean_tts_tp, min_tts_tp = summarize("tts_throughput_factor")
    mean_mt_tp, min_mt_tp = summarize("musetalk_throughput_factor")
    mean_combined_tp, min_combined_tp = summarize("combined_throughput_factor")
    mean_tts_rtf, _ = summarize("tts_rtf")
    mean_mt_rtf, _ = summarize("musetalk_rtf")
    mean_combined_rtf, _ = summarize("combined_rtf")

    print("\nOverall (mean / worst-case throughput factor, audio_s / compute_s):")
    print(f"  TTS:       mean RTF={mean_tts_rtf:.2f}       mean throughput={mean_tts_tp:.2f}x  worst={min_tts_tp:.2f}x")
    print(f"  MuseTalk:  mean RTF={mean_mt_rtf:.2f}       mean throughput={mean_mt_tp:.2f}x  worst={min_mt_tp:.2f}x")
    print(f"  Combined:  mean RTF={mean_combined_rtf:.2f}       mean throughput={mean_combined_tp:.2f}x  worst={min_combined_tp:.2f}x")

    print("\nVerdict:")
    print_verdict("TTS alone", min_tts_tp)
    print_verdict("MuseTalk alone", min_mt_tp)
    print_verdict("Combined (strictly sequential, no pipelining)", min_combined_tp)

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "rows": rows,
                    "summary": {
                        "tts": {
                            "mean_rtf": round(mean_tts_rtf, 3),
                            "mean_throughput_factor": round(mean_tts_tp, 3),
                            "worst_case_throughput_factor": round(min_tts_tp, 3),
                        },
                        "musetalk": {
                            "mean_rtf": round(mean_mt_rtf, 3),
                            "mean_throughput_factor": round(mean_mt_tp, 3),
                            "worst_case_throughput_factor": round(min_mt_tp, 3),
                        },
                        "combined": {
                            "mean_rtf": round(mean_combined_rtf, 3),
                            "mean_throughput_factor": round(mean_combined_tp, 3),
                            "worst_case_throughput_factor": round(min_combined_tp, 3),
                        },
                    },
                },
                indent=2,
            )
        )
        print(f"\nWrote raw results to {args.json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-image", type=Path, default=None, help="Avatar photo (or video) to drive")
    parser.add_argument("--speaker-wav", type=Path, default=None, help="Reference WAV for voice cloning")
    parser.add_argument("--language", type=str, default="en", help="Chatterbox language code (default: en)")
    parser.add_argument(
        "--sentence-counts", type=int, nargs="+", default=DEFAULT_SENTENCE_COUNTS,
        help=f"Repeats of a fixed base sentence per text sample (default: {DEFAULT_SENTENCE_COUNTS})",
    )
    parser.add_argument(
        "--texts", type=str, nargs="+", default=None,
        help="Explicit text samples to use instead of --sentence-counts",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timed runs per text sample (default: 3)")
    parser.add_argument("--work-dir", type=Path, default=None, help="Scratch dir for intermediate wav/mp4")
    parser.add_argument("--json", type=Path, default=None, help="Also write raw results as JSON")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
