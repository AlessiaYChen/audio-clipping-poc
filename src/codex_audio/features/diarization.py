from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:  # pragma: no cover - dependency managed via extras
    speechsdk = None  # type: ignore[assignment]

_WORD_SCALE = 10_000_000


@dataclass
class DiarizationSegment:
    speaker: str
    start_s: float
    end_s: float


class DiarizationError(RuntimeError):
    """Raised when diarization fails."""


def run_diarization(
    audio_path: Path,
    *,
    key: Optional[str] = None,
    region: Optional[str] = None,
    language: str = "en-US",
) -> List[DiarizationSegment]:
    audio_path = audio_path.expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if speechsdk is None:
        raise RuntimeError(
            "azure-cognitiveservices-speech is not installed. Install dependency to enable diarization."
        )

    subscription_key = key or os.getenv("AZURE_SPEECH_KEY")
    service_region = region or os.getenv("AZURE_SPEECH_REGION")
    if not subscription_key or not service_region:
        raise ValueError("Azure Speech key/region must be provided via args or environment")

    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=service_region)
    speech_config.speech_recognition_language = language
    timestamps_property = getattr(
        speechsdk.PropertyId, "SpeechServiceResponse_RequestWordLevelTimestamps", None
    )
    if timestamps_property is not None:
        speech_config.set_property(timestamps_property, "true")
    diarization_property = getattr(
        speechsdk.PropertyId, "SpeechServiceResponse_DiarizeIntermediateResults", None
    )
    if diarization_property is not None:
        speech_config.set_property(diarization_property, "true")
    audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
        language=language,
    )

    result = recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.Canceled:
        details = speechsdk.CancellationDetails(result)
        raise DiarizationError(f"Diarization canceled: {details.reason}")
    if result.reason != speechsdk.ResultReason.RecognizedSpeech:
        raise DiarizationError(f"Unexpected diarization result: {result.reason}")

    payload = _parse_payload(result.json)
    return _words_to_segments(payload)


def _parse_payload(raw_json: str) -> dict:
    if not raw_json:
        return {}
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise DiarizationError("Failed to parse Azure Speech payload") from exc


def _words_to_segments(payload: dict) -> List[DiarizationSegment]:
    words = []
    nbest = payload.get("NBest") or []
    if nbest:
        words = nbest[0].get("Words", [])
    elif "Words" in payload:
        words = payload.get("Words", [])

    segments: List[DiarizationSegment] = []
    current_speaker: Optional[str] = None
    seg_start = 0.0
    seg_end = 0.0

    for word in words:
        speaker = str(word.get("SpeakerId", "unknown"))
        offset = float(word.get("Offset", 0)) / _WORD_SCALE
        duration = float(word.get("Duration", 0)) / _WORD_SCALE
        word_start = offset
        word_end = offset + duration
        if current_speaker is None:
            current_speaker = speaker
            seg_start = word_start
            seg_end = word_end
            continue
        if speaker == current_speaker:
            seg_end = max(seg_end, word_end)
        else:
            segments.append(
                DiarizationSegment(speaker=current_speaker, start_s=seg_start, end_s=seg_end)
            )
            current_speaker = speaker
            seg_start = word_start
            seg_end = word_end
    if current_speaker is not None:
        segments.append(DiarizationSegment(speaker=current_speaker, start_s=seg_start, end_s=seg_end))
    return segments
