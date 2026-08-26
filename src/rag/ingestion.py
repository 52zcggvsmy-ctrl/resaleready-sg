"""Deterministic extraction, cleaning, and token-aware chunking."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import tiktoken
from pypdf import PdfReader

from .models import Chunk, DocumentSpec, TextUnit

SUPPORTED_SUFFIXES = {".pdf", ".txt"}


def load_document_manifest(path: Path) -> dict[str, DocumentSpec]:
    """Load and validate the curated source manifest, keyed by filename."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"Manifest must contain a non-empty 'documents' list: {path}")

    specs: dict[str, DocumentSpec] = {}
    required = {"filename", "document_title", "source_organization", "source_url"}
    for position, item in enumerate(documents, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest document {position} must be an object.")
        missing = required - item.keys()
        if missing:
            raise ValueError(
                f"Manifest document {position} is missing: {', '.join(sorted(missing))}"
            )
        spec = DocumentSpec(**item)
        if spec.filename in specs:
            raise ValueError(f"Duplicate manifest filename: {spec.filename}")
        if Path(spec.filename).name != spec.filename:
            raise ValueError(f"Manifest filenames must not contain directories: {spec.filename}")
        specs[spec.filename] = spec
    return specs


def _normalise_lines(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text.replace("\u00a0", " "))
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        # Web-to-PDF image descriptions can become thousands of joined characters.
        if len(line) > 300 and line.count(" ") / len(line) < 0.03:
            continue
        if line:
            lines.append(line)
    return lines


def _slice_curated_content(lines: list[str], spec: DocumentSpec) -> list[str]:
    start = 0
    if spec.content_start:
        marker = spec.content_start.casefold()
        start = next(
            (index for index, line in enumerate(lines) if marker in line.casefold()),
            -1,
        )
        if start < 0:
            raise ValueError(
                f"Start marker not found in {spec.filename}: {spec.content_start!r}"
            )

    end = len(lines)
    if spec.content_end:
        marker = spec.content_end.casefold()
        end = next(
            (
                index
                for index, line in enumerate(lines[start + 1 :], start=start + 1)
                if marker in line.casefold()
            ),
            -1,
        )
        if end < 0:
            raise ValueError(
                f"End marker not found in {spec.filename}: {spec.content_end!r}"
            )
    return lines[start:end]


def clean_extracted_text(text: str, spec: DocumentSpec) -> str:
    """Remove known export chrome while preserving the curated source wording."""

    lines = _slice_curated_content(_normalise_lines(text), spec)
    cleaned = "\n".join(lines).strip()
    if not cleaned:
        raise ValueError(f"No usable text remained after cleaning {spec.filename}.")
    return cleaned


def _extract_pdf(path: Path, spec: DocumentSpec) -> list[TextUnit]:
    reader = PdfReader(path)
    units: list[TextUnit] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        if not raw_text.strip():
            continue
        cleaned = clean_extracted_text(raw_text, spec)
        units.append(TextUnit(text=cleaned, page=page_number, section=None))
    if not units:
        raise ValueError(f"No extractable text found in PDF: {path}")
    return units


def _extract_txt(path: Path, spec: DocumentSpec) -> list[TextUnit]:
    cleaned = clean_extracted_text(path.read_text(encoding="utf-8"), spec)
    heading_pattern = re.compile(r"(?m)^#{1,6}\s+(.+?)\s*$")
    matches = list(heading_pattern.finditer(cleaned))
    if not matches:
        return [TextUnit(text=cleaned, section=None)]

    units: list[TextUnit] = []
    preamble = cleaned[: matches[0].start()].strip()
    if preamble:
        units.append(TextUnit(text=preamble, section=None))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        section_text = cleaned[match.end() : end].strip()
        if section_text:
            units.append(TextUnit(text=section_text, section=match.group(1).strip()))
    return units


def extract_document(path: Path, spec: DocumentSpec) -> list[TextUnit]:
    """Extract one supported document into clean page/section units."""

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path, spec)
    if suffix == ".txt":
        return _extract_txt(path, spec)
    raise ValueError(f"Unsupported document type: {path.suffix}")


def discover_sources(source_dir: Path, specs: dict[str, DocumentSpec]) -> list[Path]:
    """Return a stable source list and reject undocumented or missing files."""

    source_paths = sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    discovered = {path.name for path in source_paths}
    undocumented = discovered - specs.keys()
    missing = specs.keys() - discovered
    if undocumented:
        raise ValueError(f"Sources missing from manifest: {', '.join(sorted(undocumented))}")
    if missing:
        raise FileNotFoundError(f"Manifest sources not found: {', '.join(sorted(missing))}")
    return source_paths


def chunk_documents(
    source_dir: Path,
    manifest_path: Path,
    *,
    chunk_size: int = 850,
    overlap: int = 120,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Extract all curated files and return stable token-window chunks."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    specs = load_document_manifest(manifest_path)
    encoding = tiktoken.get_encoding(encoding_name)
    chunks: list[Chunk] = []

    for path in discover_sources(source_dir, specs):
        spec = specs[path.name]
        for unit in extract_document(path, spec):
            tokens = encoding.encode(unit.text)
            step = chunk_size - overlap
            for chunk_index, start in enumerate(range(0, len(tokens), step)):
                token_slice = tokens[start : start + chunk_size]
                if not token_slice:
                    continue
                text = encoding.decode(token_slice).strip()
                if not text:
                    continue
                identity = "|".join(
                    (
                        spec.filename,
                        str(unit.page or ""),
                        unit.section or "",
                        str(chunk_index),
                        text,
                    )
                )
                metadata = {
                    "chunk_id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                    "chunk_index": chunk_index,
                    "document_title": spec.document_title,
                    "source_organization": spec.source_organization,
                    "source_url": spec.source_url,
                    "section": unit.section,
                    "page": unit.page,
                    "local_filename": spec.filename,
                    "last_updated": spec.last_updated,
                    "token_start": start,
                    "token_end": start + len(token_slice),
                }
                chunks.append(
                    Chunk(
                        chunk_id=metadata["chunk_id"],
                        text=text,
                        token_count=len(token_slice),
                        metadata=metadata,
                    )
                )
                if start + chunk_size >= len(tokens):
                    break

    if not chunks:
        raise ValueError("No chunks were produced from the curated sources.")
    return chunks
