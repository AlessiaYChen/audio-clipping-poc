from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class VoiceActivityEvent:
    start_s: float
    end_s: float
    confidence: float


def detect_activity(audio_path: Path) -> List[VoiceActivityEvent]:
    return []
