from __future__ import annotations

import json

import pytest

from codex_audio.features.diarization import DiarizationError, DiarizationSegment, run_diarization


def test_run_diarization_requires_dependency(monkeypatch, tmp_path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"data")

    monkeypatch.setattr("codex_audio.features.diarization.speechsdk", None)
    with pytest.raises(RuntimeError):
        run_diarization(audio_path, key="k", region="r")


def test_run_diarization_invokes_azure(monkeypatch, tmp_path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"data")

    class FakeResult:
        reason = "RecognizedSpeech"

        def __init__(self) -> None:
            self.json = json.dumps(
                {
                    "NBest": [
                        {
                            "Words": [
                                {"Offset": 0, "Duration": 5_000_000, "SpeakerId": "A"},
                                {"Offset": 5_000_000, "Duration": 5_000_000, "SpeakerId": "B"},
                            ]
                        }
                    ]
                }
            )

    class FakeRecognizer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        class Awaiter:
            @staticmethod
            def get():
                return FakeResult()

        def recognize_once_async(self):  # type: ignore[no-untyped-def]
            return FakeRecognizer.Awaiter()

    class FakeSpeechSDK:
        class SpeechConfig:
            def __init__(self, subscription: str, region: str) -> None:
                self.subscription = subscription
                self.region = region

            def set_property(self, *_, **__):
                pass

        class PropertyId:
            SpeechServiceResponse_RequestWordLevelTimestamps = "timestamps"
            SpeechServiceResponse_DiarizationEnabled = "diarization"

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

        SpeechRecognizer = FakeRecognizer

    monkeypatch.setattr(
        "codex_audio.features.diarization.speechsdk",
        FakeSpeechSDK,
    )

    segments = run_diarization(audio_path, key="k", region="r")

    assert len(segments) == 2
    assert segments[0].speaker == "A"
    assert segments[0].end_s == pytest.approx(0.5)


def test_run_diarization_handles_cancel(monkeypatch, tmp_path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"data")

    class FakeResult:
        reason = "Canceled"

        def __init__(self) -> None:
            self.json = "{}"

    class FakeSpeechSDK:
        class SpeechConfig:
            def __init__(self, subscription: str, region: str) -> None:
                pass

            def set_property(self, *_, **__):
                pass

        class PropertyId:
            SpeechServiceResponse_RequestWordLevelTimestamps = "timestamps"
            SpeechServiceResponse_DiarizationEnabled = "diarization"

        class ResultReason:
            RecognizedSpeech = "RecognizedSpeech"
            Canceled = "Canceled"

        class CancellationDetails:
            def __init__(self, result):
                self.reason = "canceled"

        class audio:
            class AudioConfig:
                def __init__(self, filename: str) -> None:
                    pass

        class SpeechRecognizer:
            def __init__(self, *args, **kwargs) -> None:
                pass

            class Awaiter:
                @staticmethod
                def get():
                    return FakeResult()

            def recognize_once_async(self):  # type: ignore[no-untyped-def]
                return FakeSpeechSDK.SpeechRecognizer.Awaiter()

    monkeypatch.setattr(
        "codex_audio.features.diarization.speechsdk",
        FakeSpeechSDK,
    )

    with pytest.raises(DiarizationError):
        run_diarization(audio_path, key="k", region="r")
