from __future__ import annotations

import pytest

from codex_audio.text_features import TextChunk
from codex_audio.text_features.embeddings import ChunkEmbedding, embed_chunks


def _chunk(start: float, end: float, text: str) -> TextChunk:
    return TextChunk(start_s=start, end_s=end, text=text)


def test_embed_chunks_invokes_openai(monkeypatch) -> None:
    chunks = [_chunk(0.0, 5.0, "hello world"), _chunk(5.0, 10.0, "news story")]

    class FakeEmbeddingsResponse:
        def __init__(self) -> None:
            self.data = [
                type("Item", (), {"embedding": [0.1, 0.2]}),
                type("Item", (), {"embedding": [0.3, 0.4]}),
            ]

    class FakeEmbeddingsAPI:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(kwargs)
            return FakeEmbeddingsResponse()

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.embeddings = FakeEmbeddingsAPI()

    created_clients = {}

    def fake_ctor(**kwargs):  # type: ignore[no-untyped-def]
        client = FakeClient(**kwargs)
        created_clients["client"] = client
        return client

    monkeypatch.setenv("AZURE_OPENAI_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setattr(
        "codex_audio.text_features.embeddings.AzureOpenAI",
        fake_ctor,
    )

    embeddings = embed_chunks(chunks, model="text-embedding-3-small")

    assert len(embeddings) == 2
    assert isinstance(embeddings[0], ChunkEmbedding)
    assert embeddings[0].vector == [0.1, 0.2]
    assert created_clients["client"].kwargs["azure_endpoint"].startswith("https://")
    assert created_clients["client"].embeddings.calls[0]["input"][0] == "hello world"


def test_embed_chunks_requires_credentials(monkeypatch) -> None:
    monkeypatch.delenv("AZURE_OPENAI_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(ValueError):
        embed_chunks([_chunk(0.0, 1.0, "text")])
