from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from codex_audio.transcription.azure_speech import (
    TranscriptWord,
    TranscriptionError,
    match_quote_to_timestamps,
    refine_range_with_silence,
    transcribe_audio,
    _extract_words,
    _parse_payload,
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

        OutputFormat = type("OutputFormat", (), {"Detailed": "detailed"})
        SpeechRecognizer = FakeRecognizer

    fake_sdk = FakeSpeechSDK()

    monkeypatch.setattr(
        "codex_audio.transcription.azure_speech.speechsdk",
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
        "codex_audio.transcription.azure_speech.speechsdk",
        None,
        raising=False,
    )
    with pytest.raises(RuntimeError):
        transcribe_audio(audio_path, key="k", region="r")


def test_transcription_uses_conversation_transcriber_when_diarization_enabled(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"audio")

    class FakeResult:
        def __init__(self, payload: dict[str, object]) -> None:
            self.json = json.dumps(payload)

    class FakeFuture:
        def __init__(self, action):
            self._action = action

        def get(self):
            return self._action()

    class FakeEventSignal:
        def __init__(self) -> None:
            self._handlers: list = []

        def connect(self, handler):  # type: ignore[no-untyped-def]
            self._handlers.append(handler)

        def fire(self, evt):
            for handler in list(self._handlers):
                handler(evt)

    class FakeTranscriber:
        def __init__(self, *_):
            self.transcribed = FakeEventSignal()
            self.session_stopped = FakeEventSignal()
            self.canceled = FakeEventSignal()

        def start_transcribing_async(self):  # type: ignore[no-untyped-def]
            def _action():
                payload = {
                    "NBest": [
                        {
                            "Words": [
                                {"Word": "Hi", "Offset": 0, "Duration": 100_000, "SpeakerId": 1},
                                {"Word": "there", "Offset": 100_000, "Duration": 200_000, "SpeakerId": 1},
                            ]
                        }
                    ]
                }
                evt = type("Evt", (), {"result": FakeResult(payload)})
                self.transcribed.fire(evt)
                self.session_stopped.fire(object())

            return FakeFuture(_action)

        def stop_transcribing_async(self):  # type: ignore[no-untyped-def]
            return FakeFuture(lambda: None)

    class FakeTranscriptionModule:
        ConversationTranscriber = FakeTranscriber

    class FakeRecognizer:
        def __init__(self, *_, **__):
            pass

        def recognize_once_async(self):  # type: ignore[no-untyped-def]
            return self

        def get(self):  # type: ignore[no-untyped-def]
            raise AssertionError("Should not reach speech recognizer path")

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

        transcription = FakeTranscriptionModule()
        OutputFormat = type("OutputFormat", (), {"Detailed": "detailed"})
        SpeechRecognizer = FakeRecognizer

    fake_sdk = FakeSpeechSDK()

    monkeypatch.setattr(
        "codex_audio.transcription.azure_speech.speechsdk",
        fake_sdk,
        raising=False,
    )

    result = transcribe_audio(
        audio_path,
        key="test",
        region="westus",
        diarization_enabled=True,
        max_speakers=3,
    )

    assert [word.speaker_id for word in result.words] == ["1", "1"]
    assert result.raw["segments"]


def test_parse_and_extract_helpers() -> None:
    payload = _parse_payload("{\"NBest\": []}")
    assert payload["NBest"] == []
    words = _extract_words(
        {
            "SpeakerId": "Guest-9",
            "NBest": [
                {
                    "Words": [
                        {"Word": "a", "Offset": 0, "Duration": 100_000},
                        {"Word": "b", "Offset": 100_000, "Duration": 200_000, "SpeakerId": 1},
                    ]
                }
            ]
        }
    )
    assert isinstance(words[0], TranscriptWord)
    assert words[0].speaker_id == "Guest-9"
    assert words[1].speaker_id == "1"

    with pytest.raises(TranscriptionError):
        _parse_payload("not-json")


def test_match_quote_to_timestamps_finds_window() -> None:
    words = [
        TranscriptWord("Good", 0.0, 0.5),
        TranscriptWord("afternoon", 0.5, 1.0),
        TranscriptWord("radio", 1.0, 1.5),
        TranscriptWord("world", 1.5, 2.0),
    ]
    start_ms, end_ms = match_quote_to_timestamps("afternoon radio world", words)
    assert start_ms == 500
    assert end_ms == 2000


def test_refine_range_with_silence_snaps_to_gap(tmp_path: Path) -> None:
    tone = Sine(440).to_audio_segment(duration=400)
    audio = AudioSegment.silent(duration=300) + tone + AudioSegment.silent(duration=400)
    path = tmp_path / "tone.wav"
    audio.export(path, format="wav")

    refined = refine_range_with_silence(path, (320, 780))
    assert refined[0] == 300
    assert refined[1] == pytest.approx(700, abs=5)
