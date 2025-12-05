from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Word:
    text: str
    start_s: float
    end_s: float


@dataclass
class TranscriptionResult:
    words: List[Word] = field(default_factory=list)
    model: str = "whisper"


def transcribe(audio_path: Path, model: str = "base") -> TranscriptionResult:
    return TranscriptionResult(model=model)
