
from __future__ import annotations

from typing import Iterable, List, Sequence

from codex_audio.boundary.candidates import BoundaryCandidate
from codex_audio.features.diarization import DiarizationSegment
from codex_audio.features.embeddings import AudioEmbedding
from codex_audio.features.patterns import find_anchor_return_candidates
from codex_audio.features.vad import SILENCE_LABEL, VadSegment
from codex_audio.text_features.change_points import find_text_change_candidates
from codex_audio.text_features.embeddings import ChunkEmbedding

DEFAULT_MIN_SILENCE_S = 1.5
DEFAULT_AUDIO_THRESHOLD = 0.6
DEFAULT_TEXT_THRESHOLD = 0.7
DEFAULT_MERGE_THRESHOLD = 0.5
AUDIO_SHIFT_REASON = "audio_shift"
AUDIO_SHIFT_SCORE = 1.5


def from_vad(
    segments: Sequence[VadSegment], *, min_silence_s: float = DEFAULT_MIN_SILENCE_S
) -> List[BoundaryCandidate]:
    if min_silence_s <= 0:
        raise ValueError("min_silence_s must be positive")

    ordered = sorted(segments, key=lambda seg: seg.start_s)
    candidates: List[BoundaryCandidate] = []

    for segment in ordered:
        if segment.label != SILENCE_LABEL:
            continue
        duration = segment.duration()
        if duration < min_silence_s:
            continue
        midpoint = segment.start_s + duration / 2
        normalized = min(1.0, duration / (min_silence_s * 2))
        reason = f"silence_gap_{duration:.2f}s"
        candidates.append(
            BoundaryCandidate(time_s=midpoint, score=normalized, reason=reason)
        )

    return _merge_close_candidates(candidates, threshold=DEFAULT_MERGE_THRESHOLD)


def build_boundary_candidates(
    *,
    vad_segments: Sequence[VadSegment] | None = None,
    text_embeddings: Sequence[ChunkEmbedding] | None = None,
    audio_embeddings: Sequence[AudioEmbedding] | None = None,
    diarization_segments: Sequence[DiarizationSegment] | None = None,
    extra_candidates: Sequence[BoundaryCandidate] | None = None,
    min_silence_s: float = DEFAULT_MIN_SILENCE_S,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    audio_threshold: float = DEFAULT_AUDIO_THRESHOLD,
    merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
) -> List[BoundaryCandidate]:
    """Combine boundary candidates from all available modalities."""

    combined: List[BoundaryCandidate] = []

    if vad_segments:
        combined.extend(from_vad(vad_segments, min_silence_s=min_silence_s))

    if text_embeddings:
        combined.extend(
            find_text_change_candidates(text_embeddings, threshold=text_threshold)
        )

    if audio_embeddings:
        combined.extend(
            _from_audio_embeddings(audio_embeddings, threshold=audio_threshold)
        )

    if diarization_segments:
        combined.extend(find_anchor_return_candidates(diarization_segments))

    if extra_candidates:
        combined.extend(extra_candidates)

    combined.sort(key=lambda candidate: candidate.time_s)
    return _merge_close_candidates(combined, threshold=merge_threshold)


def _from_audio_embeddings(
    embeddings: Sequence[AudioEmbedding], *, threshold: float = DEFAULT_AUDIO_THRESHOLD
) -> List[BoundaryCandidate]:
    if threshold <= -1.0 or threshold > 1.0:
        raise ValueError("threshold must be within (-1, 1]")

    candidates: List[BoundaryCandidate] = []
    for left, right in zip(embeddings, embeddings[1:]):
        similarity = _cosine_similarity(left.vector, right.vector)
        if similarity < threshold:
            midpoint = (left.end_s + right.start_s) / 2
            candidates.append(
                BoundaryCandidate(
                    time_s=midpoint,
                    score=AUDIO_SHIFT_SCORE,
                    reason=f"{AUDIO_SHIFT_REASON}_{similarity:.2f}",
                )
            )
    return candidates


def _merge_close_candidates(
    candidates: Iterable[BoundaryCandidate], *, threshold: float = DEFAULT_MERGE_THRESHOLD
) -> List[BoundaryCandidate]:
    result: List[BoundaryCandidate] = []
    for candidate in candidates:
        if not result:
            result.append(candidate)
            continue
        previous = result[-1]
        if abs(candidate.time_s - previous.time_s) <= threshold:
            better = candidate if candidate.score >= previous.score else previous
            result[-1] = better
        else:
            result.append(candidate)
    return result


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
