"""
Text-to-Speech service backed by Chatterbox Multilingual (Resemble AI).

Replaces the deprecated Coqui XTTS v2. Voice profile WAVs in `voice_profiles/`
remain compatible — Chatterbox accepts any WAV reference for zero-shot cloning.

Fallback chain when Chatterbox can't load or fails mid-synthesis:

    chatterbox (cloned voice, GPU) → edge-tts (neural, free, no GPU) → gTTS

Edge TTS uses Microsoft's neural voices — dramatically better prosody than
gTTS — and needs no API key or GPU, so the degraded experience stays good.
gTTS remains as the last-ditch network fallback.

Synthesis result reports which engine produced the audio so the caller can
notify the user when voice cloning was silently dropped during fallback.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torchaudio

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SynthResult:
    output_path: str
    engine: str  # "chatterbox" or "gtts"
    fallback: bool  # True if the caller's preferred path was not taken
    voice_cloned: bool  # True if a speaker WAV was actually applied


# Microsoft neural voices for the Edge TTS fallback, one per supported
# language (the same 23-language set the voices API allows). Anything not
# listed falls back to the English voice. All female, so an un-cloned
# fallback voice stays consistent with the default female persona.
_EDGE_VOICES = {
    "ar": "ar-SA-ZariyahNeural",
    "da": "da-DK-ChristelNeural",
    "de": "de-DE-KatjaNeural",
    "el": "el-GR-AthinaNeural",
    "en": "en-US-JennyNeural",
    "es": "es-ES-ElviraNeural",
    "fi": "fi-FI-SelmaNeural",
    "fr": "fr-FR-DeniseNeural",
    "he": "he-IL-HilaNeural",
    "hi": "hi-IN-SwaraNeural",
    "it": "it-IT-ElsaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "ms": "ms-MY-YasminNeural",
    "nl": "nl-NL-ColetteNeural",
    "no": "nb-NO-IselinNeural",
    "pl": "pl-PL-AgnieszkaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "sv": "sv-SE-SofieNeural",
    "sw": "sw-KE-ZuriNeural",
    "tr": "tr-TR-EmelNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
}

# Older configs may still say "coqui"/"xtts" (the engine Chatterbox replaced).
# Treat them as chatterbox instead of breaking every synthesis call.
_LEGACY_PROVIDER_ALIASES = {"coqui": "chatterbox", "xtts": "chatterbox", "xtts_v2": "chatterbox"}

import torch

def clean_audio(wav):
    # Remove channel dimension if needed
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    # Remove DC offset
    wav = wav - wav.mean()

    # Normalize safely
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak * 0.95

    sr = 24000

    # Chatterbox (like most autoregressive TTS models) routinely tails off
    # into several hundred ms of near-total digital silence after the last
    # real phoneme — harmless for audio-only playback, but every one of
    # those silent frames gets fed straight into MuseTalk's lip-sync model,
    # which was never trained on long silent stretches and produces a
    # visible mouth flutter for that whole span instead of holding still.
    # Trim it here: playback loses the dead air, and lip-sync only ever
    # sees real, speech-driven audio.
    win = int(0.02 * sr)  # 20ms
    if wav.shape[1] > win:
        envelope = wav[0, : wav.shape[1] // win * win].reshape(-1, win).abs().mean(dim=1)
        threshold = max(envelope.max().item() * 0.02, 1e-4)
        above = (envelope > threshold).nonzero()
        if len(above) > 0:
            last_voiced_sample = (int(above[-1].item()) + 1) * win
            # Keep a small natural tail instead of a hard cut.
            tail_keep = int(0.15 * sr)
            cut_at = min(last_voiced_sample + tail_keep, wav.shape[1])
            wav = wav[:, :cut_at]

    # Short fade in/out to remove clicks
    fade_samples = min(int(0.01 * sr), wav.shape[1] // 2)
    wav[:, :fade_samples] *= torch.linspace(
        0, 1, fade_samples, device=wav.device
    )
    wav[:, -fade_samples:] *= torch.linspace(
        1, 0, fade_samples, device=wav.device
    )

    return wav

def _trim_trailing_silence(sound, keep_ms: int = 150):
    """pydub counterpart of the trailing-silence trim in clean_audio() above,
    for the edge-tts/gTTS fallback paths (mp3->wav via pydub, never touches
    clean_audio's torch tensor). Same reasoning: trailing silence is harmless
    for playback but every silent frame it spans gets fed straight into
    MuseTalk's lip-sync model, which produces a visible mouth flutter for
    silence it wasn't trained on. Edge/gTTS trail off far less than
    Chatterbox, but this keeps behavior consistent across all engines."""
    from pydub.silence import detect_leading_silence

    trailing_silence_ms = detect_leading_silence(sound.reverse())
    cut_at = len(sound) - trailing_silence_ms + keep_ms
    return sound[: max(cut_at, keep_ms)]


import re

def clean_tts_text(text: str) -> str:
    # Remove markdown
    text = re.sub(r"[*_~`#>]", "", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove emojis and unusual symbols
    text = re.sub(r"[^\w\s.,!?'-]", "", text)

    # Normalize repeated punctuation
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)

    # Convert long dashes to pauses
    text = text.replace("—", ", ")
    text = text.replace("–", ", ")

    # Remove brackets but keep words
    text = re.sub(r"[\[\]\(\)\{\}]", "", text)

    # Remove standalone punctuation lines
    text = re.sub(r"(?m)^[\s.,!?;:'\"-]+$", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

class TTSService:
    """Text-to-Speech service. Lazy-loads the model on first synthesis."""

    def __init__(self):
        provider = settings.TTS_PROVIDER
        if provider in _LEGACY_PROVIDER_ALIASES:
            logger.warning(
                f"TTS_PROVIDER={provider!r} is deprecated (Coqui XTTS was replaced "
                f"by Chatterbox) — using 'chatterbox'. Update your .env."
            )
            provider = _LEGACY_PROVIDER_ALIASES[provider]
        self.provider = provider
        self.model = None

    def _check_cuda(self) -> bool:
        try:
            return torch.cuda.is_available()
        except Exception:
            return False

    async def initialize(self):
        """Load the Chatterbox model (downloaded from HuggingFace on first run)."""
        if self.model is not None:
            return

        if self.provider != "chatterbox":
            raise ValueError(f"Unsupported TTS provider: {self.provider}")

        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            device = "cuda" if self._check_cuda() else "cpu"
            logger.info(f"Loading Chatterbox multilingual TTS on {device}...")
            # logger.info("Zeeshan: Forcing an exception to test fallback")
            # raise Exception("Demonstration exception for testing fallback")
            self.model = await asyncio.to_thread(
                ChatterboxMultilingualTTS.from_pretrained, device=device,
            )
            logger.info(f"Chatterbox loaded (sr={self.model.sr}, device={device})")

        except Exception :
            logger.exception("Failed to load Chatterbox")
            raise

    async def synthesize(
        self,
        text: str,
        output_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
    ) -> SynthResult:
        """
        Synthesize speech.

        Args:
            text: Text to speak.
            output_path: Destination WAV path.
            speaker_wav: Optional reference audio for voice cloning (≥10s recommended).
            language: 2-letter code from Chatterbox's 23-language set.

        Returns:
            SynthResult describing the WAV path and which engine was used.
            `fallback=True` indicates the preferred Chatterbox path failed
            and gTTS was used instead — voice cloning is lost in that case.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            if self.model is None:
                await self.initialize()

            logger.info(f"Synthesizing (chatterbox, lang={language}): {text[:80]}...")

            if speaker_wav and not Path(speaker_wav).exists():
                logger.warning(f"Speaker WAV not found: {speaker_wav!r} — using default voice")
                speaker_wav = None

            kwargs = {
                "language_id": language,
                "temperature": 0.6,        # default is likely higher; lower = more literal
                "repetition_penalty": 1.3, # discourage token loops/tails
                "cfg_weight": 0.4,         # if exposed — stronger text-conditioning
            }
            if speaker_wav:
                kwargs["audio_prompt_path"] = speaker_wav

            text = clean_tts_text(text)
            text = clean_tts_text(text)
            if text and text[-1] not in ".!?":
                text += "."

            wav = await asyncio.to_thread(self.model.generate, text, **kwargs)
            wav = clean_audio(wav)
            await asyncio.to_thread(torchaudio.save, output_path, wav, self.model.sr)

            logger.info(
                f"Synthesis complete{' (cloned voice)' if speaker_wav else ''}: {output_path}"
            )
            return SynthResult(
                output_path=output_path,
                engine="chatterbox",
                fallback=False,
                voice_cloned=bool(speaker_wav),
            )

        except Exception as e:
            if speaker_wav:
                logger.warning(
                    f"Chatterbox voice-clone failed — cloned voice NOT applied, "
                    f"falling back to a default neural voice. Error: {e}"
                )
            else:
                logger.warning(f"Chatterbox failed ({e}), trying Edge TTS fallback")

            # Prefer Edge TTS (Microsoft neural voices — much better prosody
            # than gTTS, still free and CPU-only); gTTS is the last resort.
            try:
                await self._edge_fallback(text, output_path, language)
                engine = "edge-tts"
            except Exception as edge_err:
                logger.warning(f"Edge TTS failed ({edge_err}), falling back to gTTS")
                await self._gtts_fallback(text, output_path, language)
                engine = "gtts"

            return SynthResult(
                output_path=output_path,
                engine=engine,
                fallback=True,
                voice_cloned=False,
            )

    async def _edge_fallback(self, text: str, output_path: str, language: str = "en") -> str:
        """Free neural-voice fallback via Microsoft Edge TTS (no key, no GPU)."""
        import edge_tts
        from pydub import AudioSegment

        voice = _EDGE_VOICES.get(language, _EDGE_VOICES["en"])
        logger.info(f"Synthesizing (edge-tts, {voice}): {text[:80]}...")
        mp3_path = output_path.replace(".wav", "_edge.mp3")

        await edge_tts.Communicate(text, voice).save(mp3_path)
        await asyncio.to_thread(
            lambda: _trim_trailing_silence(AudioSegment.from_mp3(mp3_path)).export(
                output_path, format="wav"
            )
        )
        Path(mp3_path).unlink(missing_ok=True)

        logger.info(f"Edge TTS synthesis complete: {output_path}")
        return output_path

    async def _gtts_fallback(self, text: str, output_path: str, language: str = "en") -> str:
        """Network-only fallback using Google TTS — no GPU/local model required."""
        try:
            from gtts import gTTS
            from pydub import AudioSegment

            logger.info(f"Synthesizing (gTTS): {text[:80]}...")
            mp3_path = output_path.replace(".wav", "_gtts.mp3")

            await asyncio.to_thread(
                lambda: gTTS(text=text, lang=language, slow=False).save(mp3_path)
            )
            await asyncio.to_thread(
                lambda: _trim_trailing_silence(AudioSegment.from_mp3(mp3_path)).export(
                    output_path, format="wav"
                )
            )
            Path(mp3_path).unlink(missing_ok=True)

            logger.info(f"gTTS synthesis complete: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"gTTS also failed: {e}")
            raise

    async def synthesize_bytes(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
    ) -> bytes:
        """Synthesize and return WAV bytes (used by REST callers)."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            await self.synthesize(text, tmp_path, speaker_wav, language)
            return Path(tmp_path).read_bytes()
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# Suppress unused-name warning — re-exported for type hints elsewhere
__all__ = ["TTSService", "SynthResult", "tts_service"]


# Global instance
tts_service = TTSService()
