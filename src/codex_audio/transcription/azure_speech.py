from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:  # pragma: no cover - dependency handled via optional install
    speechsdk = None  # type: ignore[assignment]

try:
    from pydub import AudioSegment
    from pydub.silence import detect_silence
except ImportError:  # pragma: no cover - dependency handled via optional install
    AudioSegment = None  # type: ignore[assignment]
    detect_silence = None  # type: ignore[assignment]

try:
    from thefuzz import fuzz
except ImportError:  # pragma: no cover - dependency handled via optional install
    fuzz = None  # type: ignore[assignment]

_WORD_TIMING_SCALE = 10_000_000  # Azure offset/duration unit is 100-ns


@dataclass
class TranscriptWord:
    text: str
    start_s: float
    end_s: float
    speaker_id: Optional[str] = None


@dataclass
class TranscriptionOutput:
    words: List[TranscriptWord] = field(default_factory=list)
    model: str = "azure-stt"
    language: str = "en-US"
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "language": self.language,
            "words": [
                {
                    "text": word.text,
                    "start": word.start_s,
                    "end": word.end_s,
                    "speaker": word.speaker_id,
                }
                for word in self.words
            ],
            "raw": self.raw,
        }


class TranscriptionError(RuntimeError):
    """Raised when Azure STT fails to return a valid transcript."""


def transcribe_audio(
    audio_path: Path,
    *,
    key: Optional[str] = None,
    region: Optional[str] = None,
    language: str = "en-US",
) -> TranscriptionOutput:
    if speechsdk is None:
        raise RuntimeError(
            "azure-cognitiveservices-speech is not installed. Install it to enable transcription."
        )

    audio_path = audio_path.expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    subscription_key = key or os.getenv("AZURE_SPEECH_KEY")
    service_region = region or os.getenv("AZURE_SPEECH_REGION")
    if not subscription_key or not service_region:
        raise ValueError("Azure Speech key/region must be provided via args or environment")

    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=service_region)
    speech_config.speech_recognition_language = language
    speech_config.request_word_level_timestamps()
    speech_config.output_format = speechsdk.OutputFormat.Detailed

    audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
        language=language,
    )
    _enable_diarization(recognizer)

    result = recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.Canceled:
        details = speechsdk.CancellationDetails(result)
        raise TranscriptionError(f"Transcription canceled: {details.reason}")
    if result.reason != speechsdk.ResultReason.RecognizedSpeech:
        raise TranscriptionError(f"Unexpected transcription result: {result.reason}")

    payload = _parse_payload(result.json)
    words = _extract_words(payload)
    return TranscriptionOutput(words=words, model="azure-stt", language=language, raw=payload)


def _parse_payload(raw_json: str) -> Dict[str, Any]:
    if not raw_json:
        return {}
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise TranscriptionError("Failed to parse Azure transcription payload") from exc


def _extract_words(payload: Dict[str, Any]) -> List[TranscriptWord]:
    words: List[TranscriptWord] = []
    nbest = payload.get("NBest") or []
    if not nbest:
        return words
    top = nbest[0]
    for item in top.get("Words", []):
        text = item.get("Word", "")
        offset = float(item.get("Offset", 0)) / _WORD_TIMING_SCALE
        duration = float(item.get("Duration", 0)) / _WORD_TIMING_SCALE
        speaker_raw = item.get("SpeakerId")
        speaker_id = str(speaker_raw) if speaker_raw is not None else None
        words.append(
            TranscriptWord(text=text, start_s=offset, end_s=offset + duration, speaker_id=speaker_id)
        )
    return words


def match_quote_to_timestamps(
    quote: str,
    transcript_words: Sequence[TranscriptWord],
    *,
    min_ratio: int = 80,
    window_expansion: int = 5,
) -> Optional[Tuple[int, int]]:
    """Return the best (start_ms, end_ms) that matches the quote via fuzzy matching."""

    if not quote.strip() or not transcript_words:
        return None

    normalized_quote = _normalize_text(quote)
    if not normalized_quote:
        return None

    quote_tokens = normalized_quote.split()
    target_window = max(1, len(quote_tokens))
    normalized_words = [_normalize_text(word.text) for word in transcript_words]

    best: Tuple[int, int, int] | None = None
    total_words = len(transcript_words)
    for start_idx in range(total_words):
        builder: list[str] = []
        for offset in range(target_window + window_expansion):
            idx = start_idx + offset
            if idx >= total_words:
                break
            builder.append(normalized_words[idx])
            candidate = " ".join(filter(None, builder)).strip()
            if not candidate:
                continue
            if candidate == normalized_quote:
                start_ms = int(transcript_words[start_idx].start_s * 1000)
                end_ms = int(transcript_words[idx].end_s * 1000)
                return start_ms, end_ms
            ratio = _fuzzy_ratio(candidate, normalized_quote)
            if ratio < min_ratio:
                continue
            start_ms = int(transcript_words[start_idx].start_s * 1000)
            end_ms = int(transcript_words[idx].end_s * 1000)
            if best is None:
                best = (start_ms, end_ms, ratio)
                continue
            best_duration = best[1] - best[0]
            duration = end_ms - start_ms
            if ratio > best[2] or (
                ratio == best[2]
                and (
                    duration > best_duration
                    or (duration == best_duration and start_ms < best[0])
                )
            ):
                best = (start_ms, end_ms, ratio)

    if best is None:
        return None
    return best[0], best[1]


def refine_range_with_silence(
    audio_path: Path,
    match_range_ms: Tuple[int, int],
    *,
    lookaround_ms: int = 500,
    min_gap_ms: int = 120,
    silence_margin_db: float = 16.0,
) -> Tuple[int, int]:
    """Snap the provided range to the nearest silence gap using +/- lookaround_ms."""

    if AudioSegment is None or detect_silence is None:
        raise RuntimeError("pydub is not installed; install it to refine timestamps")

    audio = AudioSegment.from_file(audio_path)
    duration_ms = len(audio)
    start_ms = max(0, min(match_range_ms[0], duration_ms))
    end_ms = max(start_ms + 1, min(match_range_ms[1], duration_ms))
    silence_thresh = audio.dBFS - silence_margin_db

    start_ms = _snap_to_silence(
        audio,
        target_ms=start_ms,
        lookaround_ms=lookaround_ms,
        min_gap_ms=min_gap_ms,
        silence_thresh=silence_thresh,
        prefer="end",
    )
    end_ms = _snap_to_silence(
        audio,
        target_ms=end_ms,
        lookaround_ms=lookaround_ms,
        min_gap_ms=min_gap_ms,
        silence_thresh=silence_thresh,
        prefer="start",
    )
    if end_ms <= start_ms:
        end_ms = min(duration_ms, start_ms + max(lookaround_ms, 50))
    return start_ms, end_ms


def _snap_to_silence(
    audio: "AudioSegment",
    *,
    target_ms: int,
    lookaround_ms: int,
    min_gap_ms: int,
    silence_thresh: float,
    prefer: str,
) -> int:
    window_start = max(0, target_ms - lookaround_ms)
    window_end = min(len(audio), target_ms + lookaround_ms)
    if window_start >= window_end:
        return target_ms

    segment = audio[window_start:window_end]
    gaps = detect_silence(segment, min_silence_len=min_gap_ms, silence_thresh=silence_thresh)
    if not gaps:
        return target_ms

    window_center = target_ms - window_start
    best_offset = None
    best_distance = math.inf
    for gap_start, gap_end in gaps:
        anchor = gap_end if prefer == "end" else gap_start
        distance = abs(anchor - window_center)
        if distance < best_distance:
            best_distance = distance
            best_offset = anchor

    if best_offset is None:
        return target_ms
    return max(0, min(len(audio), window_start + int(best_offset)))


def _fuzzy_ratio(candidate: str, quote: str) -> int:
    if fuzz is not None:
        return int(fuzz.token_set_ratio(candidate, quote))
    return int(SequenceMatcher(None, candidate, quote).ratio() * 100)


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9']+", " ", text.lower()).strip()


def _enable_diarization(recognizer: Any) -> None:
    if speechsdk is None:
        return
    diarization_cls = getattr(speechsdk, "SpeakerDiarizationConfig", None)
    if diarization_cls is None:
        return
    try:
        diarization_config = diarization_cls()
    except Exception:  # pragma: no cover - SDK specific failure
        return
    set_count = getattr(diarization_config, "set_speaker_count", None)
    if callable(set_count):
        try:
            set_count(2)
        except Exception:
            pass
    setattr(recognizer, "speaker_diarization_config", diarization_config)
