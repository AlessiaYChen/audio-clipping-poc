from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import soundfile as sf

DEFAULT_WINDOW_S = 3.0
DEFAULT_HOP_RATIO = 0.5


@dataclass
class AudioEmbedding:
    start_s: float
    end_s: float
    vector: List[float]


def get_audio_embeddings(
    audio_path: Path,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    hop_ratio: float = DEFAULT_HOP_RATIO,
) -> List[AudioEmbedding]:
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    if hop_ratio <= 0 or hop_ratio > 1:
        raise ValueError("hop_ratio must be in (0, 1]")

    audio_path = audio_path.expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    samples, sample_rate = sf.read(str(audio_path), always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32)

    window_samples = max(1, int(window_s * sample_rate))
    hop_samples = max(1, int(window_samples * hop_ratio))
    total_samples = len(samples)

    embeddings: List[AudioEmbedding] = []
    start_idx = 0
    while start_idx < total_samples:
        end_idx = start_idx + window_samples
        segment = samples[start_idx:end_idx]
        if len(segment) < window_samples:
            segment = np.pad(segment, (0, window_samples - len(segment)), mode="constant")
        start_s = start_idx / sample_rate
        end_s = start_s + window_s
        vector = _compute_embedding(segment, sample_rate)
        embeddings.append(AudioEmbedding(start_s=start_s, end_s=end_s, vector=vector))
        start_idx += hop_samples

    if not embeddings and total_samples:
        vector = _compute_embedding(samples, sample_rate)
        embeddings.append(AudioEmbedding(start_s=0.0, end_s=window_s, vector=vector))

    return embeddings


def _compute_embedding(segment: np.ndarray, sample_rate: int) -> List[float]:
    if not sample_rate:
        return [0.0] * 7
    windowed = segment * np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sample_rate)

    total_energy = float(np.sum(spectrum) + 1e-8)
    band_edges = [200, 1000, 4000]
    band_energies = []
    prev = 0.0
    for cutoff in band_edges:
        mask = (freqs >= prev) & (freqs < cutoff)
        band_energies.append(float(np.sum(spectrum[mask])))
        prev = cutoff
    band_energies.append(float(np.sum(spectrum[freqs >= prev])))

    stats = [
        float(np.mean(np.abs(segment))),
        float(np.std(segment)),
        float(np.max(np.abs(segment))),
        total_energy,
    ]
    vector = stats + band_energies
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = [value / norm for value in vector]
    return vector
