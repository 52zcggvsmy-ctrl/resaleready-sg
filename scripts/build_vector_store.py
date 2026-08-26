#!/usr/bin/env python3
"""Build the ResaleReady FAISS vector store from curated official documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.embeddings import OpenAIEmbeddingProvider
from src.rag.ingestion import chunk_documents, discover_sources, load_document_manifest
from src.rag.vector_store import build_vector_store


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract, chunk, embed, and index curated HDB PDF/TXT documents."
    )
    parser.add_argument(
        "--source-dir", type=Path, default=PROJECT_ROOT / "data" / "rag_sources"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "rag_sources" / "manifest.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "vector_store"
    )
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--chunk-size", type=int, default=850)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate extraction and chunking without calling the embeddings API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = chunk_documents(
        args.source_dir,
        args.manifest,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    specs = load_document_manifest(args.manifest)
    sources = discover_sources(args.source_dir, specs)
    total_tokens = sum(chunk.token_count for chunk in chunks)
    print(
        f"Validated {len(sources)} documents: {len(chunks)} chunks, "
        f"{total_tokens:,} chunk tokens."
    )
    if args.dry_run:
        print("Dry run complete; no embeddings were requested and no index was written.")
        return

    source_checksums = {path.name: sha256_file(path) for path in sources}
    metadata = {
        "chunking": {
            "encoding": "cl100k_base",
            "chunk_size_tokens": args.chunk_size,
            "overlap_tokens": args.overlap,
        },
        "source_manifest_sha256": sha256_file(args.manifest),
        "source_sha256": source_checksums,
    }
    provider = OpenAIEmbeddingProvider(
        model_name=args.model,
        batch_size=args.batch_size,
    )
    manifest = build_vector_store(
        chunks,
        provider,
        args.output_dir,
        build_metadata=metadata,
    )
    print(
        f"Wrote {manifest['chunk_count']} vectors "
        f"({manifest['embedding_dimension']} dimensions) to {args.output_dir}."
    )


if __name__ == "__main__":
    main()
