from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

from openai import AzureOpenAI

from codex_audio.text_features import TextChunk

DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_API_VERSION = "2024-02-01"


@dataclass
class ChunkEmbedding:
    text: TextChunk
    vector: List[float]


def embed_chunks(
    chunks: Sequence[TextChunk],
    *,
    model: str = DEFAULT_EMBED_MODEL,
    api_version: Optional[str] = None,
    key: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> List[ChunkEmbedding]:
    if not chunks:
        return []

    api_key = key or os.getenv("AZURE_OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    azure_endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    if not api_key or not azure_endpoint:
        raise ValueError("Azure OpenAI key/endpoint must be provided via args or environment")

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
        azure_endpoint=azure_endpoint,
    )

    inputs = [chunk.text for chunk in chunks]
    response = client.embeddings.create(model=model, input=inputs)
    vectors = [data.embedding for data in response.data]
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding count does not match chunk count")
    return [ChunkEmbedding(text=chunk, vector=vector) for chunk, vector in zip(chunks, vectors)]
