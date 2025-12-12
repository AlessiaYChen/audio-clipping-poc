from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_audio.transcription.whisper_backend import (
    TranscriptWord,
    TranscriptionError,
    _extract_words,
    _parse_payload,
    transcribe_audio,
)


def test_transcribe_audio_handles_azure_response(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"audio")

    class FakeResult:
        reason = "RecognizedSpeech"

        def __init__(self, payload: dict[str, object]) -> None:
            self.json = json.dumps(payload)

    class FakeRecognizer:
        def __init__(self, *_, **__):
            pass

        def recognize_once_async(self):  # type: ignore[no-untyped-def]
            return self

        def get(self):  # type: ignore[no-untyped-def]
            return FakeResult(
                {
                    "NBest": [
                        {
                            "Words": [
                                {"Word": "Hello", "Offset": 1_000_000, "Duration": 500_000},
                                {"Word": "world", "Offset": 1_800_000, "Duration": 400_000},
                            ]
                        }
                    ]
                }
            )

    class FakeSpeechSDK:
        class SpeechConfig:
            OutputFormat = type("OutputFormat", (), {"Detailed": "detailed"})

            def __init__(self, subscription: str, region: str) -> None:
                self.subscription = subscription
                self.region = region

            def request_word_level_timestamps(self) -> None:
                pass

        class ResultReason:
            RecognizedSpeech = "RecognizedSpeech"
            Canceled = "Canceled"

        class CancellationDetails:
            def __init__(self, result):
                self.reason = "canceled"

        class audio:
            class AudioConfig:
                def __init__(self, filename: str) -> None:
                    self.filename = filename

        def __init__(self) -> None:
            self.OutputFormat = type("OutputFormat", (), {"Detailed": "detailed"})

        SpeechRecognizer = FakeRecognizer

    fake_sdk = FakeSpeechSDK()
    fake_sdk.SpeechRecognizer = FakeRecognizer
    fake_sdk.OutputFormat = type("OutputFormat", (), {"Detailed": "detailed"})

    monkeypatch.setattr(
        "codex_audio.transcription.whisper_backend.speechsdk",
        fake_sdk,
        raising=False,
    )

    result = transcribe_audio(
        audio_path,
        key="test",
        region="westeurope",
    )

    assert [word.text for word in result.words] == ["Hello", "world"]
    assert result.words[0].start_s == pytest.approx(0.1)


def test_transcribe_audio_missing_dependency(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"audio")

    monkeypatch.setattr(
        "codex_audio.transcription.whisper_backend.speechsdk",
        None,
        raising=False,
    )
    with pytest.raises(RuntimeError):
        transcribe_audio(audio_path, key="k", region="r")


def test_parse_and_extract_helpers() -> None:
    payload = _parse_payload("{\"NBest\": []}")
    assert payload["NBest"] == []
    words = _extract_words(
        {
            "NBest": [
                {
                    "Words": [
                        {"Word": "a", "Offset": 0, "Duration": 100_000},
                        {"Word": "b", "Offset": 100_000, "Duration": 200_000},
                    ]
                }
            ]
        }
    )
    assert isinstance(words[0], TranscriptWord)

    with pytest.raises(TranscriptionError):
        _parse_payload("not-json")
