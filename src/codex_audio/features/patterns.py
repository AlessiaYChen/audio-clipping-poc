from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, List, Sequence

from codex_audio.boundary.candidates import BoundaryCandidate
from codex_audio.features.diarization import DiarizationSegment

ANCHOR_RETURN_SCORE = 3.0
ANCHOR_REASON_START = "anchor_return_start"
ANCHOR_REASON_END = "anchor_return_end"


def find_anchor_return_candidates(
    segments: Sequence[DiarizationSegment], *, score: float = ANCHOR_RETURN_SCORE
) -> List[BoundaryCandidate]:
    if len(segments) < 3:
        return []

    anchor_speaker = _find_anchor_speaker(segments)
    if anchor_speaker is None:
        return []

    candidates: List[BoundaryCandidate] = []
    for idx in range(1, len(segments) - 1):
        left = segments[idx - 1]
        middle = segments[idx]
        right = segments[idx + 1]
        if (
            left.speaker == anchor_speaker
            and right.speaker == anchor_speaker
            and middle.speaker != anchor_speaker
        ):
            start_time = (left.end_s + middle.start_s) / 2
            end_time = (middle.end_s + right.start_s) / 2
            candidates.append(
                BoundaryCandidate(time_s=start_time, score=score, reason=ANCHOR_REASON_START)
            )
            candidates.append(
                BoundaryCandidate(time_s=end_time, score=score, reason=ANCHOR_REASON_END)
            )
    return candidates


def _find_anchor_speaker(segments: Sequence[DiarizationSegment]) -> str | None:
    durations: DefaultDict[str, float] = defaultdict(float)
    for segment in segments:
        durations[segment.speaker] += max(0.0, segment.end_s - segment.start_s)
    if not durations:
        return None
    return max(durations.items(), key=lambda item: item[1])[0]
