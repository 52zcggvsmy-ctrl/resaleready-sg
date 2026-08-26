"""Validation, extraction, chunking, and in-memory retrieval for demo uploads."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import tiktoken

from .embeddings import EmbeddingProvider
from .ingestion import extract_document
from .models import Chunk, DocumentSpec, RetrievedChunk

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_UPLOAD_SUFFIXES = {".pdf", ".txt"}
SESSION_UPLOAD_STORE_KEY = "resaleready_uploaded_knowledge_store"
SESSION_UPLOAD_RESULTS_KEY = "resaleready_upload_results"
UPLOAD_SOURCE_KIND = "uploaded_demo"


class UploadValidationError(ValueError):
    """Raised when an uploaded file is unsafe or unsupported."""


class DuplicateUploadError(ValueError):
    """Raised when identical document bytes are already active in the session."""


@dataclass(frozen=True)
class ValidatedUpload:
    filename: str
    suffix: str
    data: bytes
    sha256: str


def _safe_filename(filename: str) -> str:
    cleaned = Path(filename.replace("\\", "/")).name.strip()
    if not cleaned or cleaned in {".", ".."}:
        raise UploadValidationError("The uploaded file must have a valid filename.")
    return cleaned


def validate_upload(
    filename: str,
    data: bytes,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> ValidatedUpload:
    """Validate size, extension, and basic file signature before parsing."""

    safe_name = _safe_filename(filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise UploadValidationError("Only PDF and TXT files are supported.")
    if not data:
        raise UploadValidationError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise UploadValidationError(
            f"The file exceeds the {max_bytes / (1024 * 1024):g} MB upload limit."
        )

    if suffix == ".pdf":
        if not data[:1024].lstrip().startswith(b"%PDF-"):
            raise UploadValidationError("The file does not contain a valid PDF signature.")
    else:
        try:
            decoded = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise UploadValidationError("TXT files must use UTF-8 encoding.") from error
        if "\x00" in decoded:
            raise UploadValidationError("The TXT file appears to contain binary data.")
        if not decoded.strip():
            raise UploadValidationError("The TXT file does not contain usable text.")

    return ValidatedUpload(
        filename=safe_name,
        suffix=suffix,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _document_title(filename: str) -> str:
    title = Path(filename).stem.replace("_", " ").replace("-", " ")
    return " ".join(title.split()).title() or filename


def chunk_uploaded_document(
    upload: ValidatedUpload,
    *,
    chunk_size: int = 850,
    overlap: int = 120,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Extract and chunk one validated upload without persisting its original bytes."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    spec = DocumentSpec(
        filename=upload.filename,
        document_title=_document_title(upload.filename),
        source_organization="Uploaded demo document (unverified)",
        source_url="",
    )
    with tempfile.NamedTemporaryFile(suffix=upload.suffix) as handle:
        handle.write(upload.data)
        handle.flush()
        units = extract_document(Path(handle.name), spec)

    encoding = tiktoken.get_encoding(encoding_name)
    step = chunk_size - overlap
    chunks: list[Chunk] = []
    for unit in units:
        tokens = encoding.encode(unit.text)
        for chunk_index, start in enumerate(range(0, len(tokens), step)):
            token_slice = tokens[start : start + chunk_size]
            if not token_slice:
                continue
            text = encoding.decode(token_slice).strip()
            if not text:
                continue
            identity = "|".join(
                (
                    upload.sha256,
                    str(unit.page or ""),
                    unit.section or "",
                    str(chunk_index),
                    text,
                )
            )
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            metadata = {
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "document_title": spec.document_title,
                "source_organization": spec.source_organization,
                "source_url": "",
                "source_kind": UPLOAD_SOURCE_KIND,
                "section": unit.section,
                "page": unit.page,
                "local_filename": upload.filename,
                "last_updated": None,
                "upload_sha256": upload.sha256,
                "token_start": start,
                "token_end": start + len(token_slice),
            }
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    token_count=len(token_slice),
                    metadata=metadata,
                )
            )
            if start + chunk_size >= len(tokens):
                break

    if not chunks:
        raise UploadValidationError("No usable text chunks could be extracted.")
    return chunks


def _normalised_matrix(vectors: np.ndarray, expected_rows: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] < 1:
        raise ValueError("The embedding provider returned an unexpected matrix shape.")
    if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise ValueError("The embedding provider returned invalid vectors.")
    matrix = np.ascontiguousarray(matrix)
    faiss.normalize_L2(matrix)
    return matrix


class UploadedVectorStore:
    """Session-scoped FAISS index for unverified administrative demo uploads."""

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        self.model_name = model_name
        self.chunks: list[Chunk] = []
        self.document_hashes: set[str] = set()
        self._index: faiss.Index | None = None

    @property
    def document_count(self) -> int:
        return len(self.document_hashes)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def filenames(self) -> list[str]:
        return sorted(
            {str(chunk.metadata["local_filename"]) for chunk in self.chunks}
        )

    def add_document(
        self,
        upload: ValidatedUpload,
        chunks: list[Chunk],
        embedding_provider: EmbeddingProvider,
    ) -> int:
        if upload.sha256 in self.document_hashes:
            raise DuplicateUploadError("This exact document is already active.")
        if not chunks:
            raise ValueError("At least one chunk is required before embedding.")
        if embedding_provider.model_name != self.model_name:
            raise ValueError("Upload and active knowledge-base embedding models differ.")
        matrix = _normalised_matrix(
            embedding_provider.embed_documents([chunk.embedding_text for chunk in chunks]),
            len(chunks),
        )
        if self._index is None:
            self._index = faiss.IndexFlatIP(matrix.shape[1])
        elif self._index.d != matrix.shape[1]:
            raise ValueError("Uploaded embedding dimensions do not match the active index.")
        self._index.add(matrix)
        self.chunks.extend(chunks)
        self.document_hashes.add(upload.sha256)
        return len(chunks)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        embedding_provider: EmbeddingProvider,
    ) -> list[RetrievedChunk]:
        if top_k < 1 or not self.chunks or self._index is None:
            return []
        if embedding_provider.model_name != self.model_name:
            raise ValueError("Query and uploaded-store embedding models differ.")
        vector = np.asarray(embedding_provider.embed_query(query), dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self._index.d:
            raise ValueError("Query embedding dimension does not match uploaded index.")
        matrix = np.ascontiguousarray(vector.reshape(1, -1))
        if not np.isfinite(matrix).all() or np.linalg.norm(matrix) == 0:
            raise ValueError("Query embedding must be finite and non-zero.")
        faiss.normalize_L2(matrix)
        scores, positions = self._index.search(matrix, min(top_k, len(self.chunks)))
        return [
            RetrievedChunk(
                text=self.chunks[position].text,
                score=float(score),
                metadata=dict(self.chunks[position].metadata),
            )
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        ]
