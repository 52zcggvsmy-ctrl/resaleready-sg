from __future__ import annotations

import hashlib
import re
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.rag.ingestion import chunk_documents
from src.rag.vector_store import (
    CHUNKS_FILE,
    INDEX_FILE,
    STORE_MANIFEST_FILE,
    FaissRetriever,
    build_vector_store,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "rag_sources"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"


class DeterministicTestEmbedder:
    """Small lexical embedder used only to test storage without network access."""

    model_name = "deterministic-test-embedder"

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for term in re.findall(r"[a-z0-9]+", text.casefold()):
            digest = hashlib.sha256(term.encode("utf-8")).digest()
            position = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[position] += 1.0
        return vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._embed(text) for text in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed(query)


class RagIngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chunks = chunk_documents(SOURCE_DIR, SOURCE_MANIFEST)

    def test_all_curated_documents_have_complete_provenance(self) -> None:
        filenames = {chunk.metadata["local_filename"] for chunk in self.chunks}
        self.assertEqual(6, len(filenames))
        for chunk in self.chunks:
            self.assertTrue(chunk.metadata["document_title"])
            self.assertEqual(
                "Housing & Development Board (HDB)",
                chunk.metadata["source_organization"],
            )
            self.assertTrue(chunk.metadata["source_url"].startswith("https://www.hdb.gov.sg/"))
            self.assertEqual(1, chunk.metadata["page"])
            self.assertLessEqual(chunk.token_count, 850)

    def test_export_navigation_is_removed(self) -> None:
        combined = "\n".join(chunk.text for chunk in self.chunks)
        self.assertNotIn("A Singapore Government Agency Website", combined)
        self.assertNotIn("BuyingandSellingaResaleFlat", combined)

    def test_chunk_ids_are_reproducible(self) -> None:
        second_run = chunk_documents(SOURCE_DIR, SOURCE_MANIFEST)
        self.assertEqual(
            [chunk.chunk_id for chunk in self.chunks],
            [chunk.chunk_id for chunk in second_run],
        )

    def test_faiss_build_load_and_retrieval(self) -> None:
        provider = DeterministicTestEmbedder()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            manifest = build_vector_store(self.chunks, provider, output_dir)
            self.assertEqual(len(self.chunks), manifest["chunk_count"])
            for filename in (INDEX_FILE, CHUNKS_FILE, STORE_MANIFEST_FILE):
                self.assertTrue((output_dir / filename).is_file())

            retriever = FaissRetriever.load(output_dir, provider)
            cases = {
                "How much is the Option Fee?": "hdb_03_option_to_purchase.pdf",
                "When should I submit a Request for Value?": "hdb_04_request_for_value.pdf",
                "fire insurance resale completion appointment": "hdb_06_resale_completion.pdf",
            }
            for query, expected_filename in cases.items():
                with self.subTest(query=query):
                    results = retriever.retrieve(query, top_k=3)
                    self.assertTrue(results)
                    self.assertIn(
                        expected_filename,
                        {result.metadata["local_filename"] for result in results},
                    )
                    self.assertTrue(all(result.metadata["source_url"] for result in results))


if __name__ == "__main__":
    unittest.main()
