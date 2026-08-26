"""Safeguarded retrieval and grounded-answer orchestration."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from src.openai_client import TextGenerator
from src.prompts import (
    ANSWER_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    build_grounded_answer_input,
    build_rewrite_input,
)

from .models import RetrievedChunk
from .safeguards import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    MAX_QUESTION_CHARACTERS,
    SECURITY_BLOCK_MESSAGE,
    SENSITIVE_OUTPUT_MESSAGE,
    SafeguardResult,
    normalize_user_text,
    screen_model_output,
    validate_question,
    validate_retrieval_query,
)

_ANSWER_PROTOCOL_PATTERN = re.compile(
    r"^\s*(SUPPORTED|INSUFFICIENT)\s*:\s*", re.IGNORECASE
)
_CITATION_PATTERN = re.compile(r"\[Source (\d+)\]", re.IGNORECASE)


@dataclass(frozen=True)
class AnswerSource:
    title: str
    organization: str
    url: str
    page: int | None = None
    section: str | None = None
    source_kind: str = "curated_hdb"
    local_filename: str = ""


@dataclass(frozen=True)
class QAResult:
    answer: str
    retrieval_query: str
    sources: tuple[AnswerSource, ...]
    blocked: bool = False


def _deduplicate_sources(chunks: Sequence[RetrievedChunk]) -> tuple[AnswerSource, ...]:
    sources: list[AnswerSource] = []
    seen: set[tuple[str, str, int | None, str | None, str]] = set()
    for chunk in chunks:
        metadata = chunk.metadata
        source_kind = str(metadata.get("source_kind", "curated_hdb"))
        key = (
            str(metadata.get("document_title", "Untitled source")),
            str(metadata.get("source_url", "")),
            metadata.get("page"),
            metadata.get("section"),
            source_kind,
        )
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            AnswerSource(
                title=key[0],
                organization=str(metadata.get("source_organization", "")),
                url=key[1],
                page=key[2],
                section=key[3],
                source_kind=source_kind,
                local_filename=str(metadata.get("local_filename", "")),
            )
        )
    return tuple(sources)


def _cited_positions(answer: str) -> set[int]:
    return {int(position) for position in _CITATION_PATTERN.findall(answer)}


def _sources_for_positions(
    positions: set[int], chunks: Sequence[RetrievedChunk]
) -> tuple[AnswerSource, ...]:
    cited_chunks = [
        chunk
        for position, chunk in enumerate(chunks, start=1)
        if position in positions
    ]
    return _deduplicate_sources(cited_chunks)


class ResaleReadyQA:
    """Run validation, rewriting, retrieval, grounded generation, and output checks."""

    def __init__(
        self,
        *,
        text_generator: TextGenerator,
        retrieve_chunks: Callable[[str, int], list[RetrievedChunk]],
        top_k: int = 4,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        self.text_generator = text_generator
        self.retrieve_chunks = retrieve_chunks
        self.top_k = top_k

    def _rewrite_query(
        self, question: str, history: Sequence[dict[str, Any]]
    ) -> str:
        if not history:
            return normalize_user_text(question)
        rewritten = self.text_generator.generate(
            instructions=REWRITE_SYSTEM_PROMPT,
            input_text=build_rewrite_input(question, history),
            max_output_tokens=160,
        )
        rewritten = normalize_user_text(rewritten)[:MAX_QUESTION_CHARACTERS]
        rewrite_guard = validate_retrieval_query(rewritten)
        if not rewrite_guard.allowed:
            return normalize_user_text(question)
        return rewrite_guard.normalized_text or normalize_user_text(question)

    def answer(
        self,
        question: str,
        *,
        history: Sequence[dict[str, Any]] = (),
    ) -> QAResult:
        safeguard = validate_question(question, history=history)
        if not safeguard.allowed:
            return QAResult(
                answer=safeguard.message or SECURITY_BLOCK_MESSAGE,
                retrieval_query="",
                sources=(),
                blocked=True,
            )

        retrieval_query = self._rewrite_query(safeguard.normalized_text, history)
        chunks = self.retrieve_chunks(retrieval_query, self.top_k)
        if not chunks:
            return QAResult(
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                retrieval_query=retrieval_query,
                sources=(),
            )

        raw_answer = self.text_generator.generate(
            instructions=ANSWER_SYSTEM_PROMPT,
            input_text=build_grounded_answer_input(
                safeguard.normalized_text,
                retrieval_query,
                chunks,
            ),
            max_output_tokens=900,
        )
        output_guard = screen_model_output(raw_answer)
        if not output_guard.allowed:
            return QAResult(
                answer=output_guard.message or SENSITIVE_OUTPUT_MESSAGE,
                retrieval_query=retrieval_query,
                sources=(),
                blocked=True,
            )

        answer = output_guard.normalized_text
        protocol_match = _ANSWER_PROTOCOL_PATTERN.match(answer)
        if protocol_match:
            status = protocol_match.group(1).upper()
            answer = answer[protocol_match.end() :].strip()
            if status == "INSUFFICIENT":
                return QAResult(
                    answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                    retrieval_query=retrieval_query,
                    sources=(),
                )

        positions = _cited_positions(answer)
        valid_positions = set(range(1, len(chunks) + 1))
        if not positions or not positions.issubset(valid_positions):
            return QAResult(
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                retrieval_query=retrieval_query,
                sources=(),
            )

        return QAResult(
            answer=answer,
            retrieval_query=retrieval_query,
            sources=_sources_for_positions(positions, chunks),
        )


__all__ = [
    "AnswerSource",
    "QAResult",
    "ResaleReadyQA",
    "SafeguardResult",
    "validate_question",
]
