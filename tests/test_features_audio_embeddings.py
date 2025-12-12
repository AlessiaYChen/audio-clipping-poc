from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from codex_audio.features.embeddings import AudioEmbedding, get_audio_embeddings


def _write_tone(path, duration_s=6.0, sample_rate=16_000, freq=220.0) -> None:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    samples = np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(path, samples, sample_rate)


def test_get_audio_embeddings_generates_overlapping_windows(tmp_path) -> None:
    audio_path = tmp_path / "tone.wav"
    _write_tone(audio_path)

    embeddings = get_audio_embeddings(audio_path, window_s=3.0, hop_ratio=0.5)

    assert len(embeddings) >= 4
    assert isinstance(embeddings[0], AudioEmbedding)
    assert embeddings[0].start_s == pytest.approx(0.0)
    assert embeddings[1].start_s == pytest.approx(1.5, rel=0.01)
    assert len(embeddings[0].vector) == 8
    assert all(abs(value) <= 1 for value in embeddings[0].vector)


def test_get_audio_embeddings_validates_inputs(tmp_path) -> None:
    audio_path = tmp_path / "tone.wav"
    _write_tone(audio_path, duration_s=1.0)

    with pytest.raises(ValueError):
        get_audio_embeddings(audio_path, window_s=0.0)
    with pytest.raises(ValueError):
        get_audio_embeddings(audio_path, hop_ratio=0.0)
    with pytest.raises(FileNotFoundError):
        get_audio_embeddings(tmp_path / "missing.wav")
