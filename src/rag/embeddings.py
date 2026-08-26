"""Embedding-provider boundary for indexing and query retrieval."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from openai import OpenAI


class EmbeddingProvider(Protocol):
    """Minimal interface required by the vector store."""

    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, query: str) -> np.ndarray: ...


class OpenAIEmbeddingProvider:
    """Batch OpenAI embeddings while preserving input order."""

    def __init__(
        self,
        *,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 64,
        dimensions: int | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your shell environment before "
                "building or querying the production vector store."
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self.dimensions = dimensions
        self._client = OpenAI(api_key=resolved_api_key)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("At least one document is required for embedding.")

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            request = {
                "model": self.model_name,
                "input": batch,
                "encoding_format": "float",
            }
            if self.dimensions is not None:
                request["dimensions"] = self.dimensions
            response = self._client.embeddings.create(**request)
            vectors.extend(
                item.embedding for item in sorted(response.data, key=lambda item: item.index)
            )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")
        return self.embed_documents([query])[0]
