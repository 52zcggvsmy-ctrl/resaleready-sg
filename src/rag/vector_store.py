"""FAISS vector-store construction, persistence, and retrieval."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .embeddings import EmbeddingProvider
from .models import Chunk, RetrievedChunk

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.jsonl"
STORE_MANIFEST_FILE = "store_manifest.json"
SCHEMA_VERSION = 1


def _validate_matrix(vectors: np.ndarray, expected_rows: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] < 1:
        raise ValueError(
            f"Expected an embedding matrix with {expected_rows} rows; got {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix contains non-finite values.")
    if np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise ValueError("Embedding provider returned a zero-length vector.")
    return np.ascontiguousarray(matrix)


def _temp_path(output_dir: Path, suffix: str) -> Path:
    handle = tempfile.NamedTemporaryFile(delete=False, dir=output_dir, suffix=suffix)
    handle.close()
    return Path(handle.name)


def build_vector_store(
    chunks: Sequence[Chunk],
    embedding_provider: EmbeddingProvider,
    output_dir: Path,
    *,
    build_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Embed chunks and atomically write an aligned cosine-similarity store."""

    if not chunks:
        raise ValueError("At least one chunk is required to build the vector store.")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = _validate_matrix(
        embedding_provider.embed_documents([chunk.embedding_text for chunk in chunks]),
        len(chunks),
    )
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    store_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "embedding_model": embedding_provider.model_name,
        "embedding_dimension": matrix.shape[1],
        "similarity_metric": "cosine",
        "chunk_count": len(chunks),
    }
    if build_metadata:
        store_manifest.update(build_metadata)

    index_tmp = _temp_path(output_dir, ".faiss")
    chunks_tmp = _temp_path(output_dir, ".jsonl")
    manifest_tmp = _temp_path(output_dir, ".json")
    try:
        faiss.write_index(index, str(index_tmp))
        with chunks_tmp.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                record = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata,
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        manifest_tmp.write_text(
            json.dumps(store_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(index_tmp, output_dir / INDEX_FILE)
        os.replace(chunks_tmp, output_dir / CHUNKS_FILE)
        os.replace(manifest_tmp, output_dir / STORE_MANIFEST_FILE)
    finally:
        for path in (index_tmp, chunks_tmp, manifest_tmp):
            path.unlink(missing_ok=True)
    return store_manifest


class FaissRetriever:
    """Load an aligned FAISS index and return top matching source chunks."""

    def __init__(
        self,
        index: faiss.Index,
        chunks: list[Chunk],
        store_manifest: dict[str, Any],
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.index = index
        self.chunks = chunks
        self.store_manifest = store_manifest
        self.embedding_provider = embedding_provider

    @classmethod
    def load(cls, output_dir: Path, embedding_provider: EmbeddingProvider) -> "FaissRetriever":
        manifest_path = output_dir / STORE_MANIFEST_FILE
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_model = manifest.get("embedding_model")
        if expected_model != embedding_provider.model_name:
            raise ValueError(
                f"Vector store uses {expected_model!r}, but provider uses "
                f"{embedding_provider.model_name!r}."
            )

        index = faiss.read_index(str(output_dir / INDEX_FILE))
        chunks: list[Chunk] = []
        with (output_dir / CHUNKS_FILE).open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                chunks.append(
                    Chunk(
                        chunk_id=record["chunk_id"],
                        text=record["text"],
                        token_count=record["token_count"],
                        metadata=record["metadata"],
                    )
                )
        if index.ntotal != len(chunks) or manifest.get("chunk_count") != len(chunks):
            raise ValueError("FAISS index, chunk metadata, and store manifest are out of sync.")
        return cls(index, chunks, manifest, embedding_provider)

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        query_vector = np.asarray(self.embedding_provider.embed_query(query), dtype=np.float32)
        if query_vector.ndim != 1 or query_vector.shape[0] != self.index.d:
            raise ValueError(
                f"Query embedding dimension {query_vector.shape} does not match index dimension {self.index.d}."
            )
        query_matrix = np.ascontiguousarray(query_vector.reshape(1, -1))
        if not np.isfinite(query_matrix).all() or np.linalg.norm(query_matrix) == 0:
            raise ValueError("Query embedding must be finite and non-zero.")
        faiss.normalize_L2(query_matrix)
        scores, positions = self.index.search(query_matrix, min(top_k, len(self.chunks)))
        return [
            RetrievedChunk(
                text=self.chunks[position].text,
                score=float(score),
                metadata=dict(self.chunks[position].metadata),
            )
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        ]
