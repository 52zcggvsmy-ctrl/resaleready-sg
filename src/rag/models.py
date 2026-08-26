"""Shared data models for the ResaleReady retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentSpec:
    """Curated metadata and optional cleaning boundaries for one source file."""

    filename: str
    document_title: str
    source_organization: str
    source_url: str
    content_start: str | None = None
    content_end: str | None = None
    last_updated: str | None = None


@dataclass(frozen=True)
class TextUnit:
    """Clean text from a page or logical text section."""

    text: str
    page: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit and its provenance."""

    chunk_id: str
    text: str
    token_count: int
    metadata: dict[str, Any]

    @property
    def embedding_text(self) -> str:
        """Add concise context that helps short chunks retrieve accurately."""

        title = self.metadata.get("document_title", "")
        organization = self.metadata.get("source_organization", "")
        section = self.metadata.get("section")
        prefix = " | ".join(item for item in (title, organization, section) if item)
        return f"{prefix}\n\n{self.text}" if prefix else self.text


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk with a cosine-similarity score."""

    text: str
    score: float
    metadata: dict[str, Any]
