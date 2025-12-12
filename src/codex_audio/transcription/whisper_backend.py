from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:  # pragma: no cover - dependency handled via optional install
    speechsdk = None  # type: ignore[assignment]

_WORD_TIMING_SCALE = 10_000_000  # Azure offset/duration unit is 100-ns


@dataclass
class TranscriptWord:
    text: str
    start_s: float
    end_s: float


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
                {"text": word.text, "start": word.start_s, "end": word.end_s}
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
        words.append(TranscriptWord(text=text, start_s=offset, end_s=offset + duration))
    return words
