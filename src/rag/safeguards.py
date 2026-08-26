"""Deterministic safeguards around the ResaleReady RAG model boundary."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

MAX_QUESTION_CHARACTERS = 1_000
MIN_QUESTION_CHARACTERS = 3
MAX_QUESTION_LINES = 12
INTERNAL_POLICY_MARKER = "RESALEREADY-INTERNAL-POLICY-2026"

OUT_OF_SCOPE_MESSAGE = (
    "I can help only with the buyer-side HDB resale journey. Try asking about the "
    "HFE letter, Option to Purchase, Request for Value, resale application, or resale "
    "completion."
)
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I could not find enough relevant information in the ResaleReady knowledge base "
    "to answer that reliably. Please check the official HDB website or contact HDB "
    "for confirmation."
)
SECURITY_BLOCK_MESSAGE = (
    "I can help with buyer-side HDB resale questions, but I cannot follow requests "
    "to override safeguards or change my assigned role."
)
DISCLOSURE_BLOCK_MESSAGE = (
    "I cannot reveal system instructions, API keys, credentials, secrets, or internal "
    "configuration. I can still help with a buyer-side HDB resale question."
)
SENSITIVE_OUTPUT_MESSAGE = (
    "I could not provide that response safely. Please rephrase your buyer-side HDB "
    "resale question or verify the matter through official HDB channels."
)

_PROMPT_INJECTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\b(?:ignore|disregard|forget|override|bypass|discard)\b.{0,80}\b(?:instructions?|rules?|prompts?|safeguards?)\b",
        r"\bpretend\b.{0,80}\b(?:no longer|not)\b.{0,80}\b(?:hdb|resaleready|assistant)\b",
        r"\b(?:act|role[ -]?play)\s+as\b.{0,80}\b(?:unrestricted|unfiltered|different|system|developer)\b",
        r"\b(?:developer|god|dan|debug)\s+mode\b",
        r"\b(?:new|replacement)\s+(?:system\s+)?instructions?\b",
        r"\bfollow\b.{0,50}\b(?:these|my)\s+instructions?\s+instead\b",
        r"\b(?:do not|don't)\s+follow\b.{0,50}\b(?:system|developer|previous|prior)\b",
        r"\bswitch\b.{0,30}\b(?:role|persona|identity)\b",
    )
)

_DISCLOSURE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\b(?:show|reveal|print|display|expose|repeat|quote|give|tell)\b.{0,70}\b(?:system|developer|hidden|initial|internal)\s+(?:prompt|message|instructions?|rules?)\b",
        r"\b(?:show|reveal|print|display|expose|give|tell)\b.{0,70}\b(?:api[ _-]?key|access[ _-]?token|secret|credentials?|password|environment variables?|configuration)\b",
        r"\b(?:what|where)\b.{0,40}\b(?:api[ _-]?key|access[ _-]?token|password|secret)\b",
        r"\b(?:openai_api_key|resaleready_password|st\.secrets|os\.environ)\b",
    )
)

_DOMAIN_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bhdb\s+resale\b",
        r"\bresale\s+(?:flat|market|transaction|price|process|journey|buyer|seller|purchase|application|completion)\b",
        r"\b(?:buy|buying|purchase|purchasing)\b.{0,45}\b(?:resale\s+flat|hdb\s+resale)\b",
        r"\b(?:hfe|otp)\b",
        r"\boption\s+(?:to\s+purchase|fee|exercise|period)\b",
        r"\brequest\s+for\s+value\b",
        r"\b(?:resale\s+)?completion\s+appointment\b",
        r"\btemporary\s+extension(?:\s+of\s+stay)?\b",
        r"\bextension\s+of\s+stay\b",
        r"\b(?:ethnic\s+integration\s+policy|eip|spr\s+quota)\b",
        r"\b(?:minimum\s+occupation\s+period|mop)\b",
        r"\bremaining\s+lease\b",
        r"\b(?:cpf|housing\s+loan|housing\s+grant|stamp\s+duty|legal\s+fees?)\b.{0,60}\b(?:hdb|resale|flat)\b",
        r"\b(?:hdb|resale|flat)\b.{0,60}\b(?:cpf|housing\s+loan|housing\s+grant|stamp\s+duty|legal\s+fees?)\b",
    )
)

_FOLLOW_UP_PATTERN = re.compile(
    r"(?:\b(?:it|that|this|they|them|those|these)\b|"
    r"\b(?:amount|fee|deadline|period|step|documents?|costs?|application|buyer|seller)\b|"
    r"^(?:and|what about|how about|what if|then|who must|when does|how long|how much)\b)",
    re.IGNORECASE,
)

_VALUATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:what is|what's|estimate|calculate|tell me) (?:my |this |the )?flat(?:'s)? (?:value|valuation|worth)",
        r"how much is (?:my|this|the) flat worth",
        r"predict (?:the )?(?:resale )?price",
    )
)
_ADVICE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:give|provide) (?:me )?(?:financial|legal) advice",
        r"which (?:bank |home )?(?:loan|mortgage) should i (?:choose|take)",
        r"should i (?:buy|sell|sign|sue|borrow)",
    )
)
_NRIC_PATTERN = re.compile(r"\b[STFGM]\d{7}[A-Z]\b", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+65[ -]?)?[689]\d{7}(?!\d)")
_REPEATED_CHARACTER_PATTERN = re.compile(r"(.)\1{49,}", re.DOTALL)

_SENSITIVE_OUTPUT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}\b",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"\b(?:OPENAI_API_KEY|RESALEREADY_PASSWORD)\s*[:=]\s*\S+",
        re.escape(INTERNAL_POLICY_MARKER),
    )
)


@dataclass(frozen=True)
class SafeguardResult:
    allowed: bool
    category: str
    message: str | None = None
    normalized_text: str = ""


def normalize_user_text(text: str) -> str:
    """Canonicalise Unicode and spacing before all deterministic checks."""

    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    )
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _matches_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def is_hdb_resale_domain(text: str) -> bool:
    return _matches_any(text, _DOMAIN_PATTERNS)


def _history_supports_follow_up(history: Sequence[dict[str, object]]) -> bool:
    recent = history[-6:]
    return any(
        not message.get("blocked")
        and is_hdb_resale_domain(normalize_user_text(str(message.get("content", ""))))
        for message in recent
    )


def validate_question(
    question: str,
    *,
    history: Sequence[dict[str, object]] = (),
) -> SafeguardResult:
    """Apply deterministic validation before rewriting, retrieval, or generation."""

    raw_text = str(question)
    normalized = normalize_user_text(raw_text)
    if len(normalized) < MIN_QUESTION_CHARACTERS:
        return SafeguardResult(False, "invalid_input", "Please enter a complete question.", normalized)
    if len(normalized) > MAX_QUESTION_CHARACTERS:
        return SafeguardResult(
            False,
            "invalid_input",
            f"Please shorten your question to {MAX_QUESTION_CHARACTERS:,} characters or fewer.",
            normalized,
        )
    if len(raw_text.splitlines()) > MAX_QUESTION_LINES:
        return SafeguardResult(
            False,
            "invalid_input",
            f"Please keep your question to {MAX_QUESTION_LINES} lines or fewer.",
            normalized,
        )
    if any(
        unicodedata.category(character) == "Cc" and character not in "\n\r\t"
        for character in raw_text
    ) or _REPEATED_CHARACTER_PATTERN.search(normalized):
        return SafeguardResult(
            False,
            "invalid_input",
            "Please remove control characters or unusually repeated text and try again.",
            normalized,
        )
    if any(pattern.search(normalized) for pattern in (_NRIC_PATTERN, _EMAIL_PATTERN, _PHONE_PATTERN)):
        return SafeguardResult(
            False,
            "personal_information",
            "Please remove personal information such as NRIC numbers, phone numbers, or email addresses before asking.",
            normalized,
        )
    if _matches_any(normalized, _DISCLOSURE_PATTERNS):
        return SafeguardResult(False, "sensitive_disclosure", DISCLOSURE_BLOCK_MESSAGE, normalized)
    if _matches_any(normalized, _PROMPT_INJECTION_PATTERNS):
        return SafeguardResult(False, "prompt_injection", SECURITY_BLOCK_MESSAGE, normalized)
    if _matches_any(normalized, _VALUATION_PATTERNS):
        return SafeguardResult(
            False,
            "property_valuation",
            "ResaleReady cannot value a property or predict its price. You can use the Market Explorer to review historical transactions, but those records are not a valuation.",
            normalized,
        )
    if _matches_any(normalized, _ADVICE_PATTERNS):
        return SafeguardResult(
            False,
            "financial_or_legal_advice",
            "ResaleReady cannot provide financial or legal advice or recommend a purchase, loan, or legal decision. Please consult an appropriately qualified professional or official channel.",
            normalized,
        )

    in_domain = is_hdb_resale_domain(normalized)
    is_contextual_follow_up = (
        len(normalized) <= 240
        and bool(_FOLLOW_UP_PATTERN.search(normalized))
        and _history_supports_follow_up(history)
    )
    if not in_domain and not is_contextual_follow_up:
        return SafeguardResult(False, "out_of_scope", OUT_OF_SCOPE_MESSAGE, normalized)
    return SafeguardResult(True, "allowed", normalized_text=normalized)


def validate_retrieval_query(query: str) -> SafeguardResult:
    """Reject unsafe text unexpectedly emitted by the query-rewrite model."""

    normalized = normalize_user_text(query)
    if not normalized or len(normalized) > MAX_QUESTION_CHARACTERS:
        return SafeguardResult(False, "unsafe_rewrite", SECURITY_BLOCK_MESSAGE, normalized)
    if _matches_any(normalized, _DISCLOSURE_PATTERNS) or _matches_any(
        normalized, _PROMPT_INJECTION_PATTERNS
    ):
        return SafeguardResult(False, "unsafe_rewrite", SECURITY_BLOCK_MESSAGE, normalized)
    return SafeguardResult(True, "allowed", normalized_text=normalized)


def screen_model_output(text: str) -> SafeguardResult:
    """Fail closed if generated text resembles a secret or leaks the policy canary."""

    normalized = str(text).strip()
    if not normalized:
        return SafeguardResult(False, "empty_model_output", SENSITIVE_OUTPUT_MESSAGE)
    if _matches_any(normalized, _SENSITIVE_OUTPUT_PATTERNS):
        return SafeguardResult(False, "sensitive_model_output", SENSITIVE_OUTPUT_MESSAGE)
    return SafeguardResult(True, "allowed", normalized_text=normalized)
