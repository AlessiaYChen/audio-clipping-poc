from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
from pydub.generators import Sine

from codex_audio.features import vad


def _write_audio(tmp_path: Path, duration_ms: int) -> Path:
    audio_path = tmp_path / "tone.wav"
    Sine(440).to_audio_segment(duration=duration_ms).set_channels(1).export(audio_path, format="wav")
    return audio_path


def test_run_vad_respects_backend_decisions(monkeypatch, tmp_path: Path) -> None:
    audio_path = _write_audio(tmp_path, duration_ms=900)
    speech_pattern: List[bool] = [False] * 5 + [True] * 10 + [False] * 5 + [True] * 10

    class FakeVad:
        def __init__(self, aggressiveness: int) -> None:
            self.calls = 0

        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            if self.calls < len(speech_pattern):
                value = speech_pattern[self.calls]
            else:
                value = False
            self.calls += 1
            return value

    monkeypatch.setattr(vad.webrtcvad, "Vad", FakeVad)

    segments = vad.run_vad(audio_path, frame_duration_ms=30)

    assert [segment.label for segment in segments] == [
        vad.SILENCE_LABEL,
        vad.SPEECH_LABEL,
        vad.SILENCE_LABEL,
        vad.SPEECH_LABEL,
    ]
    starts = [segment.start_s for segment in segments]
    ends = [segment.end_s for segment in segments]
    assert starts == pytest.approx([0.0, 0.15, 0.45, 0.6], abs=1e-6)
    assert ends == pytest.approx([0.15, 0.45, 0.6, 0.9], abs=1e-6)


def test_run_vad_missing_audio(tmp_path: Path) -> None:
    missing = tmp_path / "missing.wav"
    with pytest.raises(FileNotFoundError):
        vad.run_vad(missing)
