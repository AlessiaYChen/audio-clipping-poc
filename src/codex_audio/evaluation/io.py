from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Segment:
    start_s: float
    end_s: float
    label: str = ""


@dataclass
class EvaluationExample:
    audio_path: Path
    annotation_path: Path
    prediction_path: Path
    station: str


def load_manifest(manifest_path: Path) -> List[EvaluationExample]:
    rows = _read_csv(manifest_path)
    examples: List[EvaluationExample] = []
    for row in rows:
        audio = Path(row["audio_path"]).expanduser()
        annotation = Path(row["annotation_path"]).expanduser()
        prediction = Path(row.get("prediction_path", "")).expanduser()
        station = row.get("station", "default")
        if not prediction:
            raise ValueError("Manifest missing prediction_path column")
        examples.append(
            EvaluationExample(
                audio_path=audio,
                annotation_path=annotation,
                prediction_path=prediction,
                station=station,
            )
        )
    return examples


def load_reference_segments(annotation_path: Path) -> List[Segment]:
    rows = _read_csv(annotation_path)
    segments: List[Segment] = []
    for row in rows:
        segments.append(
            Segment(
                start_s=float(row["start_s"]),
                end_s=float(row["end_s"]),
                label=row.get("label", ""),
            )
        )
    return sorted(segments, key=lambda seg: seg.start_s)


def load_prediction_segments(prediction_path: Path) -> List[Segment]:
    payload = json.loads(prediction_path.read_text())
    segments_data = payload.get("segments", [])
    segments: List[Segment] = []
    for item in segments_data:
        segments.append(
            Segment(
                start_s=float(item["start"]),
                end_s=float(item["end"]),
                label=str(item.get("label", "")),
            )
        )
    return sorted(segments, key=lambda seg: seg.start_s)


def _read_csv(path: Path) -> List[dict[str, str]]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        return list(reader)
