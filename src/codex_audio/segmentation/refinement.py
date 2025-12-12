from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Sequence

from codex_audio.boundary.candidates import BoundaryCandidate
from codex_audio.features.vad import SILENCE_LABEL, VadSegment
from codex_audio.segmentation.change_scores import (
    ChangePoint,
    find_peak_candidates,
    smooth_scores,
)
from codex_audio.segmentation.planner import SegmentPlan
from codex_audio.segmentation.selection import SegmentConstraint, greedy_select_boundaries
from codex_audio.transcription import TranscriptWord

DEFAULT_CHANGE_WEIGHTS: Mapping[str, float] = {
    "audio": 1.0,
    "text": 1.3,
    "silence": 0.7,
    "anchor": 1.5,
}


@dataclass
class RefinementParams:
    constraints: SegmentConstraint = field(
        default_factory=lambda: SegmentConstraint(min_len=30.0, max_len=210.0)
    )
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_CHANGE_WEIGHTS))
    candidate_min_score: float = 0.8
    hard_min_cut_score: float = 1.2
    smoothing_window: int = 1
    snap_window_s: float = 1.0


_FULL_LABEL = "chunk_full"
_TAIL_LABEL = "chunk_tail"


def refine_chunk_segments(
    chunk_start: float,
    chunk_end: float,
    *,
    change_points: Sequence[ChangePoint],
    params: RefinementParams | None = None,
    vad_segments: Sequence[VadSegment] | None = None,
    transcript_words: Sequence[TranscriptWord] | None = None,
    peak_reason: str = "change_peak",
) -> List[SegmentPlan]:
    if params is None:
        params = RefinementParams()
    if chunk_end <= chunk_start:
        raise ValueError("chunk_end must be greater than chunk_start")

    window_points = [
        point for point in change_points if chunk_start < point.time_s < chunk_end
    ]
    if not window_points:
        return [SegmentPlan(chunk_start, chunk_end, _FULL_LABEL)]

    scores = [point.combined(params.weights) for point in window_points]
    smoothed = smooth_scores(scores, window_size=params.smoothing_window)
    times = [point.time_s for point in window_points]
    candidates = find_peak_candidates(
        times,
        smoothed,
        min_score=params.candidate_min_score,
        reason=peak_reason,
    )
    if not candidates:
        return [SegmentPlan(chunk_start, chunk_end, _FULL_LABEL)]

    selected = greedy_select_boundaries(
        candidates,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        constraints=params.constraints,
        hard_min_score=params.hard_min_cut_score,
    )
    if not selected:
        return [SegmentPlan(chunk_start, chunk_end, _FULL_LABEL)]

    snapped = _snap_candidates(
        selected,
        vad_segments=vad_segments,
        transcript_words=transcript_words,
        window_s=params.snap_window_s,
    )

    return _segments_from_boundaries(
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        boundaries=snapped,
        min_len=params.constraints.min_len,
    )


def _snap_candidates(
    candidates: Sequence[BoundaryCandidate],
    *,
    vad_segments: Sequence[VadSegment] | None,
    transcript_words: Sequence[TranscriptWord] | None,
    window_s: float,
) -> List[BoundaryCandidate]:
    snapped: List[BoundaryCandidate] = []
    for candidate in candidates:
        snapped_time = _snap_time(
            candidate.time_s,
            vad_segments=vad_segments,
            transcript_words=transcript_words,
            window_s=window_s,
        )
        snapped.append(
            BoundaryCandidate(
                time_s=snapped_time,
                score=candidate.score,
                reason=candidate.reason,
            )
        )
    snapped.sort(key=lambda c: c.time_s)
    deduped: List[BoundaryCandidate] = []
    for candidate in snapped:
        if deduped and abs(candidate.time_s - deduped[-1].time_s) < 1e-3:
            if candidate.score > deduped[-1].score:
                deduped[-1] = candidate
            continue
        deduped.append(candidate)
    return deduped


def _snap_time(
    time_s: float,
    *,
    vad_segments: Sequence[VadSegment] | None,
    transcript_words: Sequence[TranscriptWord] | None,
    window_s: float,
) -> float:
    if window_s <= 0:
        return time_s
    best_time = time_s
    best_delta = window_s

    if vad_segments:
        for segment in vad_segments:
            if segment.label != SILENCE_LABEL:
                continue
            if segment.end_s < time_s - window_s or segment.start_s > time_s + window_s:
                continue
            midpoint = segment.start_s + segment.duration() / 2
            delta = abs(midpoint - time_s)
            if delta < best_delta:
                best_delta = delta
                best_time = midpoint

    if transcript_words:
        for word in transcript_words:
            for boundary in (word.start_s, word.end_s):
                delta = abs(boundary - time_s)
                if delta <= window_s and delta < best_delta:
                    best_delta = delta
                    best_time = boundary

    return best_time


def _segments_from_boundaries(
    *,
    chunk_start: float,
    chunk_end: float,
    boundaries: Sequence[BoundaryCandidate],
    min_len: float,
) -> List[SegmentPlan]:
    if not boundaries:
        return [SegmentPlan(chunk_start, chunk_end, _FULL_LABEL)]

    segments: List[SegmentPlan] = []
    current = chunk_start
    for idx, boundary in enumerate(boundaries):
        boundary_time = max(chunk_start, min(boundary.time_s, chunk_end))
        if (boundary_time - current) < min_len:
            continue
        label = boundary.reason or f"segment_{idx}"
        segments.append(SegmentPlan(current, boundary_time, label))
        current = boundary_time

    if (chunk_end - current) >= min_len or not segments:
        segments.append(SegmentPlan(current, chunk_end, _TAIL_LABEL))
    else:
        segments[-1].end_s = chunk_end
    return segments
