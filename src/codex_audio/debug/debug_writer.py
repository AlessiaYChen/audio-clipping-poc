from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence


def write_story_debug_bundle(
    out_dir: Path,
    station: str,
    input_basename: str,
    segments: list[dict[str, Any]],
    transcript_words: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write index.json, story_XXX.(json|txt), and stories.ndjson files."""

    bundle_dir = out_dir / station / input_basename
    bundle_dir.mkdir(parents=True, exist_ok=True)

    index_payload = {
        "station": station,
        "input": input_basename,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "story_count": len(segments),
        "meta": meta or {},
    }

    ordered_words = _sorted_words(transcript_words)
    stories = _ensure_ids(segments)
    stories_ndjson = bundle_dir / "stories.ndjson"
    with stories_ndjson.open("w", encoding="utf-8") as fh:
        for story in stories:
            story_words = _slice_story_words(story, ordered_words)
            record = _story_record(story, story_words)
            summary = {
                "story_id": record.get("story_id"),
                "start_s": record.get("start_s"),
                "end_s": record.get("end_s"),
                "label": record.get("label"),
                "clip_path": record.get("clip_path"),
            }
            fh.write(json.dumps(summary) + "\n")
            _write_file(bundle_dir / f"{record['id']}.json", record)
            readable = _format_readable_transcript(story_words)
            (bundle_dir / f"{record['id']}.txt").write_text(
                _story_text(record, readable)
            )

    _write_file(bundle_dir / "transcript_words.json", transcript_words)

    index_payload.update(
        {
            "stories_file": str(stories_ndjson),
            "transcript_words_file": str(bundle_dir / "transcript_words.json"),
        }
    )
    _write_file(bundle_dir / "index.json", index_payload)
    return bundle_dir


def _ensure_ids(segments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for idx, story in enumerate(segments):
        record = dict(story)
        record.setdefault("id", f"story_{idx:03d}")
        record.setdefault("index", idx)
        record.setdefault("story_id", idx + 1)
        output.append(record)
    return output


def _story_record(
    story: dict[str, Any], words: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    record = dict(story)
    start = float(story.get("start_s", 0.0))
    end = float(story.get("end_s", 0.0))
    duration = max(0.0, end - start)
    record["duration_s"] = duration
    record["num_words"] = len(words)
    record["speakers"] = sorted({(word.get("speaker") or "?") for word in words})
    record.setdefault("story_id", record.get("story_id") or record.get("id"))
    record.setdefault("clip_path", story.get("clip_path"))
    record["transcript_words"] = words
    return record


def _story_text(story: Mapping[str, Any], readable: str) -> str:
    start = float(story.get("start_s", 0.0))
    end = float(story.get("end_s", 0.0))
    label = story.get("label")
    clip = story.get("clip_path")
    boundary = story.get("boundary") or {}
    duration = max(0.0, end - start)
    header = (
        f"STORY {story.get('id', '???')}\n"
        f"range = {_format_timestamp(start)}  {_format_timestamp(end)} (dur={duration:.2f}s)\n"
        f"label = {label}\n"
        f"clip_path = {clip}\n"
    )
    lines = [
        header,
    ]
    boundary_parts = []
    if boundary:
        reason = boundary.get("reason")
        b_type = boundary.get("type")
        confidence = boundary.get("confidence")
        quote = boundary.get("quote")
        if reason:
            boundary_parts.append(f"reason={reason}")
        if b_type:
            boundary_parts.append(f"type={b_type}")
        if confidence is not None:
            boundary_parts.append(f"confidence={confidence:.2f}")
        if quote:
            boundary_parts.append(f"quote=\"{quote}\"")
    if boundary_parts:
        lines.append("Boundary: " + ", ".join(boundary_parts))
    if readable:
        lines.append("")
        lines.append(readable)
    return "\n".join(lines).strip() + "\n"


def _sorted_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        words,
        key=lambda w: (
            float(w.get("start_s", w.get("start", 0.0))),
            float(w.get("end_s", w.get("end", 0.0))),
        ),
    )


def _slice_story_words(
    story: dict[str, Any], words: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    start = story.get("start_s")
    end = story.get("end_s")
    if start is None or end is None:
        return []
    result = []
    for word in words:
        word_start = float(word.get("start_s", word.get("start", 0.0)))
        if word_start < start:
            continue
        if word_start >= end:
            break
        result.append(word)
    return result


def _format_readable_transcript(words: List[dict[str, Any]]) -> str:
    if not words:
        return ""
    lines: List[str] = []
    current: List[str] = []
    current_len = 0
    last_end = None
    last_speaker = None

    def flush_line() -> None:
        nonlocal current, current_len
        if not current:
            return
        lines.append(" ".join(current))
        current = []
        current_len = 0

    for word in words:
        speaker = word.get("speaker") or "?"
        start = float(word.get("start_s", word.get("start", 0.0)))
        gap = start - last_end if last_end is not None else None
        if (
            last_speaker is not None
            and (speaker != last_speaker or (gap is not None and gap >= 1.2) or current_len >= 120)
        ):
            flush_line()
        if not current:
            timestamp = _format_timestamp(start)
            current.append(f"[{timestamp}][spk={speaker}]")
            current_len = len(current[-1])
        token = word.get("text", "")
        current.append(token)
        current_len += len(token) + 1
        last_speaker = speaker
        last_end = float(word.get("end_s", word.get("end", start)))
    flush_line()
    return "\n".join(lines)


def _write_file(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:06.3f}"


__all__ = ["write_story_debug_bundle"]
