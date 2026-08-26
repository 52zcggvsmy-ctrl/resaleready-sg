"""Small public retrieval interface for later UI and RAG orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from .embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from .models import RetrievedChunk
from .vector_store import FaissRetriever, STORE_MANIFEST_FILE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"


def retrieve(
    query: str,
    *,
    top_k: int = 5,
    vector_store_dir: Path = DEFAULT_VECTOR_STORE_DIR,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[RetrievedChunk]:
    """Return the top relevant chunks and provenance for a user query."""

    if embedding_provider is None:
        manifest = json.loads(
            (vector_store_dir / STORE_MANIFEST_FILE).read_text(encoding="utf-8")
        )
        embedding_provider = OpenAIEmbeddingProvider(
            model_name=manifest["embedding_model"]
        )
    return FaissRetriever.load(vector_store_dir, embedding_provider).retrieve(
        query, top_k=top_k
    )
