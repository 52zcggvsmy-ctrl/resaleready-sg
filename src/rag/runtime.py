"""Runtime assembly for the production ResaleReady Q&A service."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from src.openai_client import OpenAIResponsesClient

from .embeddings import OpenAIEmbeddingProvider
from .ingestion import chunk_documents
from .qa import ResaleReadyQA
from .retrieval import DEFAULT_VECTOR_STORE_DIR
from .vector_store import (
    CHUNKS_FILE,
    INDEX_FILE,
    STORE_MANIFEST_FILE,
    FaissRetriever,
    build_vector_store,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "data" / "rag_sources"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
_BUILD_LOCK = Lock()


def vector_store_is_ready(vector_store_dir: Path = DEFAULT_VECTOR_STORE_DIR) -> bool:
    return all(
        (vector_store_dir / filename).is_file()
        for filename in (INDEX_FILE, CHUNKS_FILE, STORE_MANIFEST_FILE)
    )


def ensure_vector_store(
    *,
    api_key: str,
    vector_store_dir: Path = DEFAULT_VECTOR_STORE_DIR,
) -> None:
    """Build the small curated store once when deployment artifacts are absent."""

    if vector_store_is_ready(vector_store_dir):
        return
    with _BUILD_LOCK:
        if vector_store_is_ready(vector_store_dir):
            return
        chunks = chunk_documents(SOURCE_DIR, SOURCE_MANIFEST)
        provider = OpenAIEmbeddingProvider(api_key=api_key)
        build_vector_store(
            chunks,
            provider,
            vector_store_dir,
            build_metadata={
                "chunking": {
                    "encoding": "cl100k_base",
                    "chunk_size_tokens": 850,
                    "overlap_tokens": 120,
                }
            },
        )


def create_qa_service(
    *,
    api_key: str,
    chat_model: str = "gpt-5.6-luna",
    vector_store_dir: Path = DEFAULT_VECTOR_STORE_DIR,
    top_k: int = 4,
) -> ResaleReadyQA:
    """Create a configured production Q&A service, building its store if needed."""

    ensure_vector_store(api_key=api_key, vector_store_dir=vector_store_dir)
    embedding_provider = OpenAIEmbeddingProvider(api_key=api_key)
    retriever = FaissRetriever.load(vector_store_dir, embedding_provider)
    text_generator = OpenAIResponsesClient(api_key=api_key, model=chat_model)
    return ResaleReadyQA(
        text_generator=text_generator,
        retrieve_chunks=lambda query, top_k: retriever.retrieve(query, top_k=top_k),
        top_k=top_k,
    )
