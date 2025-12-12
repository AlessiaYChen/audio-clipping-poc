from __future__ import annotations

from pathlib import Path

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from codex_audio.ingest import load_and_normalize_audio


def _create_stereo_tone(tmp_path: Path, duration_ms: int = 750) -> tuple[Path, AudioSegment]:
    tone = Sine(440).to_audio_segment(duration=duration_ms)
    tone = tone.set_frame_rate(44_100).set_channels(2)
    source_path = tmp_path / "tone.wav"
    tone.export(source_path, format="wav")
    return source_path, tone


def test_load_and_normalize_audio_creates_normalized_copy(tmp_path: Path) -> None:
    source_path, source_segment = _create_stereo_tone(tmp_path)
    work_dir = tmp_path / "artifacts"

    metadata, normalized_path = load_and_normalize_audio(
        source_path, work_dir=work_dir, target_sample_rate=16_000
    )

    assert normalized_path.exists()
    assert normalized_path.parent == work_dir

    normalized_segment = AudioSegment.from_file(normalized_path)

    assert metadata.source_path == source_path.resolve()
    assert metadata.duration_s == pytest.approx(len(source_segment) / 1000, rel=0.01)
    assert metadata.sample_rate == source_segment.frame_rate
    assert metadata.channels == 2
    assert metadata.normalized_sample_rate == 16_000
    assert metadata.sample_width == source_segment.sample_width

    assert normalized_segment.frame_rate == 16_000
    assert normalized_segment.channels == 1
    assert normalized_segment.duration_seconds == pytest.approx(metadata.duration_s, rel=0.01)
    assert normalized_segment.max_dBFS == pytest.approx(0.0, abs=0.5)


def test_load_and_normalize_audio_missing_source_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.wav"
    with pytest.raises(FileNotFoundError):
        load_and_normalize_audio(missing_path)
