"""Statistics-only boundary for optional Market Explorer explanations."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from src.openai_client import TextGenerator
from src.prompts import (
    MARKET_EXPLANATION_SYSTEM_PROMPT,
    build_market_explanation_input,
)
from src.rag.safeguards import screen_model_output

_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])\d(?:[\d,]*\d)?(?:\.\d+)?")
_PROHIBITED_OUTPUT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\byou should (?:buy|purchase|proceed|borrow|choose|take)\b",
        r"\bi (?:recommend|advise) (?:buying|purchasing|proceeding|borrowing)\b",
        r"\b(?:good|bad) (?:buy|deal|investment)\b",
        r"\bprices? (?:will|are likely to|are expected to) "
        r"(?:rise|increase|grow|fall|decrease|drop)\b",
        r"\b(?:this|the) (?:specific )?(?:flat|property) is worth\b",
    )
)


class MarketExplanationError(RuntimeError):
    """Raised when an AI market explanation cannot be safely displayed."""


def _money(value: float) -> str:
    return f"S${value:,.0f}"


@dataclass(frozen=True)
class MarketExplanationContext:
    """Selected filters and Pandas-computed statistics permitted at the model boundary."""

    town: str
    flat_type: str
    period_start: str
    period_end: str
    budget: float
    transaction_count: int
    median_resale_price: float
    median_floor_area_sqm: float
    within_budget_count: int
    within_budget_percentage: float

    def __post_init__(self) -> None:
        text_values = (self.town, self.flat_type, self.period_start, self.period_end)
        if any(not value.strip() for value in text_values):
            raise ValueError("Market explanation filters must not be blank.")
        numeric_values = (
            self.budget,
            self.median_resale_price,
            self.median_floor_area_sqm,
            self.within_budget_percentage,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("Market explanation statistics must be finite.")
        if self.transaction_count < 1:
            raise ValueError("At least one transaction is required.")
        if not 0 <= self.within_budget_count <= self.transaction_count:
            raise ValueError("Within-budget count must fit the transaction count.")
        if not 0 <= self.within_budget_percentage <= 100:
            raise ValueError("Within-budget percentage must be between 0 and 100.")
        if self.budget <= 0 or self.median_resale_price <= 0 or self.median_floor_area_sqm <= 0:
            raise ValueError("Market explanation values must be positive.")

    @property
    def selected_filters(self) -> dict[str, str]:
        return {
            "town": self.town,
            "flat_type": self.flat_type,
            "transaction_period": f"{self.period_start} to {self.period_end}",
            "approximate_budget": _money(self.budget),
        }

    @property
    def computed_statistics(self) -> dict[str, str]:
        return {
            "matching_transactions": f"{self.transaction_count:,}",
            "median_resale_price": _money(self.median_resale_price),
            "median_floor_area": f"{self.median_floor_area_sqm:,.0f} m²",
            "budget_comparison": (
                f"{self.within_budget_count:,} of {self.transaction_count:,} matching "
                f"transactions ({self.within_budget_percentage:.1f}%) sold at or below "
                f"the selected budget of {_money(self.budget)}"
            ),
        }

    @property
    def input_text(self) -> str:
        return build_market_explanation_input(
            self.selected_filters,
            self.computed_statistics,
        )

    @property
    def signature(self) -> str:
        return hashlib.sha256(self.input_text.encode("utf-8")).hexdigest()


def _validate_explanation_output(text: str, input_text: str) -> str:
    output_guard = screen_model_output(text)
    if not output_guard.allowed:
        raise MarketExplanationError("The generated explanation did not pass output checks.")

    explanation = output_guard.normalized_text
    if any(pattern.search(explanation) for pattern in _PROHIBITED_OUTPUT_PATTERNS):
        raise MarketExplanationError("The generated explanation exceeded its permitted scope.")

    allowed_numbers = set(_NUMBER_PATTERN.findall(input_text))
    output_numbers = set(_NUMBER_PATTERN.findall(explanation))
    if not output_numbers.issubset(allowed_numbers):
        raise MarketExplanationError("The generated explanation introduced a new figure.")
    return explanation


def generate_market_explanation(
    text_generator: TextGenerator,
    context: MarketExplanationContext,
) -> str:
    """Generate and validate an explanation without exposing transaction-level rows."""

    input_text = context.input_text
    response = text_generator.generate(
        instructions=MARKET_EXPLANATION_SYSTEM_PROMPT,
        input_text=input_text,
        max_output_tokens=350,
    )
    return _validate_explanation_output(response, input_text)


__all__ = [
    "MarketExplanationContext",
    "MarketExplanationError",
    "generate_market_explanation",
]
