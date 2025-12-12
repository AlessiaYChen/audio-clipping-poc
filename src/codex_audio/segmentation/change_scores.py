from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

from codex_audio.boundary.candidates import BoundaryCandidate
from codex_audio.features.diarization import DiarizationSegment
from codex_audio.features.embeddings import AudioEmbedding
from codex_audio.features.patterns import find_anchor_return_candidates
from codex_audio.features.vad import SILENCE_LABEL, VadSegment
from codex_audio.text_features.embeddings import ChunkEmbedding

DEFAULT_SILENCE_WINDOW_S = 1.0
DEFAULT_SILENCE_NORM_S = 1.0
DEFAULT_ANCHOR_TOLERANCE_S = 0.5
DEFAULT_AUDIO_THRESHOLD = 0.6
DEFAULT_TEXT_THRESHOLD = 0.7


@dataclass
class ChangePoint:
    time_s: float
    audio_change: float = 0.0
    text_change: float = 0.0
    silence_change: float = 0.0
    anchor_flag: float = 0.0

    def combined(self, weights: Mapping[str, float]) -> float:
        return (
            self.audio_change * weights.get("audio", 1.0)
            + self.text_change * weights.get("text", 1.0)
            + self.silence_change * weights.get("silence", 1.0)
            + self.anchor_flag * weights.get("anchor", 1.0)
        )


def compute_change_points(
    *,
    audio_embeddings: Sequence[AudioEmbedding] | None = None,
    text_embeddings: Sequence[ChunkEmbedding] | None = None,
    vad_segments: Sequence[VadSegment] | None = None,
    diarization_segments: Sequence[DiarizationSegment] | None = None,
    silence_window_s: float = DEFAULT_SILENCE_WINDOW_S,
    silence_norm_s: float = DEFAULT_SILENCE_NORM_S,
    anchor_tolerance_s: float = DEFAULT_ANCHOR_TOLERANCE_S,
    audio_threshold: float = DEFAULT_AUDIO_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
) -> List[ChangePoint]:
    boundary_times = _derive_boundary_times(audio_embeddings, text_embeddings)
    points = [ChangePoint(time_s=time) for time in boundary_times]
    if not points:
        return points

    if audio_embeddings and len(audio_embeddings) > 1:
        audio_changes = _boundary_changes_from_embeddings(audio_embeddings, audio_threshold)
        _apply_component(points, audio_changes, "audio_change")

    if text_embeddings and len(text_embeddings) > 1:
        text_changes = _boundary_changes_from_embeddings(text_embeddings, text_threshold)
        _apply_component(points, text_changes, "text_change")

    if vad_segments:
        silence_changes = _silence_changes(boundary_times, vad_segments, silence_window_s, silence_norm_s)
        _apply_component(points, silence_changes, "silence_change")

    if diarization_segments:
        anchor_flags = _anchor_flags(boundary_times, diarization_segments, anchor_tolerance_s)
        _apply_component(points, anchor_flags, "anchor_flag")

    return points


def smooth_scores(scores: Sequence[float], window_size: int = 1) -> List[float]:
    if window_size <= 0:
        return list(scores)
    smoothed: List[float] = []
    for idx in range(len(scores)):
        start = max(0, idx - window_size)
        end = min(len(scores), idx + window_size + 1)
        smoothed.append(sum(scores[start:end]) / (end - start))
    return smoothed


def find_peak_candidates(
    times: Sequence[float],
    scores: Sequence[float],
    *,
    min_score: float,
    reason: str = "change_peak",
) -> List[BoundaryCandidate]:
    candidates: List[BoundaryCandidate] = []
    if len(scores) < 3:
        for time, score in zip(times, scores):
            if score >= min_score:
                candidates.append(BoundaryCandidate(time_s=time, score=score, reason=reason))
        return candidates
    for idx in range(1, len(scores) - 1):
        prev_score = scores[idx - 1]
        cur_score = scores[idx]
        next_score = scores[idx + 1]
        if cur_score >= prev_score and cur_score >= next_score and cur_score >= min_score:
            candidates.append(BoundaryCandidate(time_s=times[idx], score=cur_score, reason=reason))
    return candidates


def _derive_boundary_times(
    audio_embeddings: Sequence[AudioEmbedding] | None,
    text_embeddings: Sequence[ChunkEmbedding] | None,
) -> List[float]:
    if audio_embeddings and len(audio_embeddings) > 1:
        return [emb.start_s for emb in audio_embeddings[1:]]
    if text_embeddings and len(text_embeddings) > 1:
        return [chunk.text.start_s for chunk in text_embeddings[1:]]
    return []


def _boundary_changes_from_embeddings(
    embeddings: Sequence[AudioEmbedding] | Sequence[ChunkEmbedding],
    threshold: float,
) -> List[tuple[float, float]]:
    if threshold <= -1.0 or threshold > 1.0:
        raise ValueError("threshold must be within (-1, 1]")
    changes: List[tuple[float, float]] = []
    for left, right in zip(embeddings, embeddings[1:]):
        left_vec = getattr(left, "vector", None)
        right_vec = getattr(right, "vector", None)
        if left_vec is None or right_vec is None:
            continue
        similarity = _cosine_similarity(left_vec, right_vec)
        change = max(0.0, 1.0 - similarity)
        if hasattr(right, "start_s"):
            time_s = getattr(right, "start_s")  # AudioEmbedding
        else:
            time_s = getattr(right.text, "start_s")  # ChunkEmbedding
        changes.append((time_s, change))
    return changes


def _silence_changes(
    times: Sequence[float],
    vad_segments: Sequence[VadSegment],
    window_s: float,
    norm_s: float,
) -> List[tuple[float, float]]:
    silence_spans = [seg for seg in vad_segments if seg.label == SILENCE_LABEL]
    changes: List[tuple[float, float]] = []
    for time in times:
        window_start = time - window_s
        window_end = time + window_s
        longest = 0.0
        for seg in silence_spans:
            overlap = min(seg.end_s, window_end) - max(seg.start_s, window_start)
            if overlap > longest:
                longest = max(0.0, overlap)
        normalized = max(0.0, min(1.0, longest / norm_s if norm_s else longest))
        changes.append((time, normalized))
    return changes


def _anchor_flags(
    times: Sequence[float],
    diarization_segments: Sequence[DiarizationSegment],
    tolerance: float,
) -> List[tuple[float, float]]:
    anchor_candidates = find_anchor_return_candidates(diarization_segments)
    change_map: List[tuple[float, float]] = []
    for time in times:
        flag = 0.0
        for candidate in anchor_candidates:
            if abs(candidate.time_s - time) <= tolerance:
                flag = 1.0
                break
        change_map.append((time, flag))
    return change_map


def _apply_component(
    points: List[ChangePoint],
    values: Iterable[tuple[float, float]],
    attribute: str,
) -> None:
    point_map = {point.time_s: point for point in points}
    for time, value in values:
        point = point_map.get(time)
        if point is None:
            continue
        setattr(point, attribute, value)


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
