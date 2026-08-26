from __future__ import annotations

import json
import unittest

from src.market_explanation import (
    MarketExplanationContext,
    MarketExplanationError,
    generate_market_explanation,
)
from src.prompts import MARKET_EXPLANATION_SYSTEM_PROMPT


class FakeTextGenerator:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []

    def generate(self, *, instructions: str, input_text: str, max_output_tokens: int) -> str:
        self.calls.append(
            {
                "instructions": instructions,
                "input_text": input_text,
                "max_output_tokens": max_output_tokens,
            }
        )
        return self.output


def sample_context() -> MarketExplanationContext:
    return MarketExplanationContext(
        town="BEDOK",
        flat_type="4 ROOM",
        period_start="January 2024",
        period_end="December 2024",
        budget=600_000,
        transaction_count=1_250,
        median_resale_price=550_000,
        median_floor_area_sqm=93,
        within_budget_count=780,
        within_budget_percentage=62.4,
    )


class MarketExplanationTests(unittest.TestCase):
    def test_payload_contains_only_filters_and_computed_statistics(self) -> None:
        context = sample_context()
        header, payload_text = context.input_text.split("\n", 1)
        payload = json.loads(payload_text)

        self.assertIn("UNTRUSTED MARKET SUMMARY", header)
        self.assertEqual(
            {"selected_filters", "computed_statistics"},
            set(payload),
        )
        self.assertEqual("BEDOK", payload["selected_filters"]["town"])
        self.assertEqual(
            "S$550,000",
            payload["computed_statistics"]["median_resale_price"],
        )
        self.assertNotIn("address", payload_text.casefold())
        self.assertNotIn("transaction_rows", payload_text.casefold())
        self.assertNotIn("resale_price_distribution", payload_text.casefold())

    def test_valid_explanation_uses_reusable_generator(self) -> None:
        generator = FakeTextGenerator(
            "These indicative historical results cover 1,250 matching transactions in "
            "BEDOK. The median resale price was S$550,000 and the median floor area was "
            "93 m².\n\nThe selected budget was S$600,000, and 780 of 1,250 matching "
            "transactions (62.4%) sold at or below it. Historical results are not a "
            "property valuation or prediction."
        )

        result = generate_market_explanation(generator, sample_context())

        self.assertIn("indicative historical", result)
        self.assertEqual(1, len(generator.calls))
        self.assertEqual(
            MARKET_EXPLANATION_SYSTEM_PROMPT,
            generator.calls[0]["instructions"],
        )
        self.assertEqual(350, generator.calls[0]["max_output_tokens"])

    def test_new_figure_is_rejected(self) -> None:
        generator = FakeTextGenerator(
            "These indicative historical transactions suggest S$700,000 as another figure."
        )
        with self.assertRaises(MarketExplanationError):
            generate_market_explanation(generator, sample_context())

    def test_purchase_recommendation_is_rejected(self) -> None:
        generator = FakeTextGenerator(
            "The historical figures are indicative, but you should buy this flat."
        )
        with self.assertRaises(MarketExplanationError):
            generate_market_explanation(generator, sample_context())

    def test_invalid_context_is_rejected_before_model_access(self) -> None:
        with self.assertRaises(ValueError):
            MarketExplanationContext(
                town="BEDOK",
                flat_type="4 ROOM",
                period_start="January 2024",
                period_end="December 2024",
                budget=600_000,
                transaction_count=0,
                median_resale_price=550_000,
                median_floor_area_sqm=93,
                within_budget_count=0,
                within_budget_percentage=0,
            )


if __name__ == "__main__":
    unittest.main()
