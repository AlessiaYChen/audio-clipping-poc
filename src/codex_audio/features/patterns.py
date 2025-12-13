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
    if not segments:
        return None

    ordered = sorted(segments, key=lambda seg: seg.start_s)
    start_speaker = ordered[0].speaker
    end_speaker = ordered[-1].speaker

    if start_speaker == end_speaker:
        return start_speaker

    scores: DefaultDict[str, float] = defaultdict(float)
    counts: DefaultDict[str, int] = defaultdict(int)

    for segment in ordered:
        duration = max(0.0, segment.end_s - segment.start_s)
        speaker = segment.speaker
        scores[speaker] += duration
        counts[speaker] += 1

    if not scores:
        return None

    return max(scores.keys(), key=lambda speaker: scores[speaker] * counts[speaker])
