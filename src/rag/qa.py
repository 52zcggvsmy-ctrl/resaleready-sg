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

MAX_QUESTION_CHARACTERS = 1_000
MIN_QUESTION_CHARACTERS = 3

_PROMPT_INJECTION_PATTERNS = (
    r"ignore (?:all |the )?(?:previous|prior|system) instructions",
    r"reveal (?:the )?(?:system|developer) prompt",
    r"show (?:me )?(?:your )?(?:hidden|system|developer) instructions",
    r"jailbreak",
)
_VALUATION_PATTERNS = (
    r"(?:what is|what's|estimate|calculate|tell me) (?:my |this |the )?flat(?:'s)? (?:value|valuation|worth)",
    r"how much is (?:my|this|the) flat worth",
    r"predict (?:the )?(?:resale )?price",
)
_ADVICE_PATTERNS = (
    r"(?:give|provide) (?:me )?(?:financial|legal) advice",
    r"which (?:bank |home )?(?:loan|mortgage) should i (?:choose|take)",
    r"should i (?:buy|sell|sign|sue|borrow)",
)
_NRIC_PATTERN = re.compile(r"\b[STFGM]\d{7}[A-Z]\b", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+65[ -]?)?[689]\d{7}(?!\d)")


@dataclass(frozen=True)
class SafeguardResult:
    allowed: bool
    message: str | None = None


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


def validate_question(question: str) -> SafeguardResult:
    """Apply deterministic input, privacy, and scope safeguards before any API call."""

    normalized = " ".join(question.split())
    if len(normalized) < MIN_QUESTION_CHARACTERS:
        return SafeguardResult(False, "Please enter a complete question.")
    if len(normalized) > MAX_QUESTION_CHARACTERS:
        return SafeguardResult(
            False,
            f"Please shorten your question to {MAX_QUESTION_CHARACTERS:,} characters or fewer.",
        )
    if any(
        pattern.search(normalized)
        for pattern in (_NRIC_PATTERN, _EMAIL_PATTERN, _PHONE_PATTERN)
    ):
        return SafeguardResult(
            False,
            "Please remove personal information such as NRIC numbers, phone numbers, or email addresses before asking.",
        )
    if any(
        re.search(pattern, normalized, re.IGNORECASE)
        for pattern in _PROMPT_INJECTION_PATTERNS
    ):
        return SafeguardResult(
            False,
            "I can help with the buyer-side HDB resale journey, but I cannot reveal or override system instructions.",
        )
    if any(
        re.search(pattern, normalized, re.IGNORECASE)
        for pattern in _VALUATION_PATTERNS
    ):
        return SafeguardResult(
            False,
            "ResaleReady cannot value a property or predict its price. You can use the Market Explorer to review historical transactions, but those records are not a valuation.",
        )
    if any(
        re.search(pattern, normalized, re.IGNORECASE)
        for pattern in _ADVICE_PATTERNS
    ):
        return SafeguardResult(
            False,
            "ResaleReady cannot provide financial or legal advice or recommend a purchase, loan, or legal decision. Please consult an appropriately qualified professional or official channel.",
        )
    return SafeguardResult(True)


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


def _sources_cited_in_answer(
    answer: str, chunks: Sequence[RetrievedChunk]
) -> tuple[AnswerSource, ...]:
    """Display only retrieved chunks that the grounded answer actually cites."""

    cited_positions = {
        int(position)
        for position in re.findall(r"\[Source (\d+)\]", answer, re.IGNORECASE)
    }
    cited_chunks = [
        chunk
        for position, chunk in enumerate(chunks, start=1)
        if position in cited_positions
    ]
    return _deduplicate_sources(cited_chunks)


class ResaleReadyQA:
    """Run input checks, follow-up rewriting, retrieval, and grounded generation."""

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
            return question.strip()
        rewritten = self.text_generator.generate(
            instructions=REWRITE_SYSTEM_PROMPT,
            input_text=build_rewrite_input(question, history),
            max_output_tokens=160,
        )
        rewritten = " ".join(rewritten.split())[:MAX_QUESTION_CHARACTERS].strip()
        return rewritten or question.strip()

    def answer(
        self,
        question: str,
        *,
        history: Sequence[dict[str, Any]] = (),
    ) -> QAResult:
        safeguard = validate_question(question)
        if not safeguard.allowed:
            return QAResult(
                answer=safeguard.message or "I cannot help with that request.",
                retrieval_query="",
                sources=(),
                blocked=True,
            )

        retrieval_query = self._rewrite_query(question, history)
        chunks = self.retrieve_chunks(retrieval_query, self.top_k)
        answer = self.text_generator.generate(
            instructions=ANSWER_SYSTEM_PROMPT,
            input_text=build_grounded_answer_input(question, retrieval_query, chunks),
            max_output_tokens=900,
        )
        return QAResult(
            answer=answer,
            retrieval_query=retrieval_query,
            sources=_sources_cited_in_answer(answer, chunks),
        )
