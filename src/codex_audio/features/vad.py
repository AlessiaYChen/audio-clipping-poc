from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Literal

from pydub import AudioSegment
import webrtcvad

FRAME_DURATION_OPTIONS = (10, 20, 30)
SPEECH_LABEL: Literal["speech"] = "speech"
SILENCE_LABEL: Literal["silence"] = "silence"


@dataclass
class VadSegment:
    start_s: float
    end_s: float
    label: Literal["speech", "silence"]

    def duration(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def _yield_frames(raw_data: bytes, frame_size: int) -> Iterator[bytes]:
    for idx in range(0, len(raw_data), frame_size):
        chunk = raw_data[idx : idx + frame_size]
        if len(chunk) == frame_size:
            yield chunk


def run_vad(
    audio_path: Path,
    *,
    aggressiveness: int = 2,
    frame_duration_ms: int = 30,
) -> List[VadSegment]:
    audio_path = audio_path.expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if aggressiveness < 0 or aggressiveness > 3:
        raise ValueError("Aggressiveness must be between 0 and 3")
    if frame_duration_ms not in FRAME_DURATION_OPTIONS:
        raise ValueError(
            f"frame_duration_ms must be one of {FRAME_DURATION_OPTIONS}, got {frame_duration_ms}"
        )

    audio = AudioSegment.from_file(audio_path)
    mono = audio.set_channels(1).set_frame_rate(16_000).set_sample_width(2)
    frame_size = int(mono.frame_rate * (frame_duration_ms / 1000.0) * mono.sample_width)
    raw_data = mono.raw_data

    if len(raw_data) < frame_size:
        frame_padding = frame_size - len(raw_data)
        raw_data += b"\0" * frame_padding

    vad = webrtcvad.Vad(aggressiveness)
    segments: List[VadSegment] = []
    current_label: Literal["speech", "silence"] | None = None
    segment_start = 0.0

    for idx, frame in enumerate(_yield_frames(raw_data, frame_size)):
        frame_start = idx * (frame_duration_ms / 1000.0)
        is_speech = vad.is_speech(frame, mono.frame_rate)
        label: Literal["speech", "silence"] = SPEECH_LABEL if is_speech else SILENCE_LABEL
        if current_label is None:
            current_label = label
            segment_start = frame_start
            continue
        if label != current_label:
            segments.append(VadSegment(segment_start, frame_start, current_label))
            current_label = label
            segment_start = frame_start

    total_duration = len(mono) / 1000.0
    if current_label is None:
        return []
    segments.append(VadSegment(segment_start, total_duration, current_label))

    consolidated = _merge_zero_length_segments(segments)
    return consolidated


def _merge_zero_length_segments(segments: Iterable[VadSegment]) -> List[VadSegment]:
    result: List[VadSegment] = []
    for segment in segments:
        if not result:
            result.append(segment)
            continue
        previous = result[-1]
        if segment.label == previous.label:
            result[-1] = VadSegment(previous.start_s, segment.end_s, previous.label)
        elif segment.duration() <= 0:
            continue
        else:
            result.append(segment)
    return result

