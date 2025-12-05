from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SpeakerTurn:
    speaker: str
    start_s: float
    end_s: float


def diarize(audio_uri: str) -> List[SpeakerTurn]:
    return []
