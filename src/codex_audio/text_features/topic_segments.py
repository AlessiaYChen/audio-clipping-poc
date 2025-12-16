from __future__ import annotations

import json
import os
import re
from typing import Callable, List, Sequence, Tuple

from openai import AzureOpenAI

from codex_audio.boundary.candidates import BoundaryCandidate
from codex_audio.transcription import TranscriptWord

DEFAULT_LLM_MODEL = os.getenv("AZURE_OPENAI_LLM_MODEL", "gpt-4o-mini")
DEFAULT_SYSTEM_PROMPT = (
    "You segment radio news transcripts into distinct stories. "
    "Return JSON with a 'boundaries' array of objects {\"time\": float, \"reason\": str}."
)
MAX_WORDS = 1200

ResponseProvider = Callable[[str, str, str | None, str | None, str | None, str], str]
ParsedBoundary = Tuple[float, str | None]


def detect_topic_boundaries(
    words: Sequence[TranscriptWord],
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    key: str | None = None,
    endpoint: str | None = None,
    api_version: str | None = None,
    response_provider: ResponseProvider | None = None,
) -> List[BoundaryCandidate]:
    if not words:
        return []

    prompt = _build_prompt(words)
    provider = response_provider or _call_chat_completion
    response_text = provider(
        prompt,
        model or DEFAULT_LLM_MODEL,
        key or os.getenv("AZURE_OPENAI_KEY"),
        endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        system_prompt or DEFAULT_SYSTEM_PROMPT,
    )
    entries = _parse_boundaries(response_text)
    candidates = [
        BoundaryCandidate(time_s=time, score=3.0, reason="llm_topic_change", quote=quote)
        for time, quote in entries
    ]
    return candidates


def _build_prompt(words: Sequence[TranscriptWord]) -> str:
    limited = list(words[:MAX_WORDS])
    lines = []
    for word in limited:
        lines.append(f"[{word.start_s:.2f}] {word.text}")
    transcript = "\n".join(lines)
    instructions = (
        "Analyze the following radio news transcript. "
        "Identify timestamps (in seconds) where the topic changes. "
        "Respond with JSON: {\"boundaries\": [{\"time\": number, \"reason\": string}, ...]}."
    )
    return f"{instructions}\n\nTranscript:\n{transcript}"


def _call_chat_completion(
    prompt: str,
    model: str,
    key: str | None,
    endpoint: str | None,
    api_version: str,
    system_prompt: str,
) -> str:
    if not key or not endpoint:
        raise ValueError("Azure OpenAI key/endpoint must be configured")
    client = AzureOpenAI(api_key=key, api_version=api_version, azure_endpoint=endpoint)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    message = response.choices[0].message
    return message.content or ""


def _parse_boundaries(response_text: str) -> List[ParsedBoundary]:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        pattern = re.compile(r"\[(\d+(?:\.\d+)?)\]")
        matches = pattern.findall(response_text)
        if not matches:
            matches = re.findall(r"(\d+(?:\.\d+)?)", response_text)
        return [(float(value), None) for value in matches if _valid_time(value)]

    entries = payload.get("boundaries", []) if isinstance(payload, dict) else []
    parsed: List[ParsedBoundary] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        time_value = item.get("time")
        if not _valid_time(time_value):
            continue
        quote = item.get("quote") or item.get("reason")
        if quote is not None:
            quote = str(quote)
        parsed.append((float(time_value), quote))
    return parsed


def _valid_time(value: object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


__all__ = ["detect_topic_boundaries"]
