from __future__ import annotations

import unittest

from src.rag.qa import ResaleReadyQA, validate_question
from tests.test_rag_qa import FakeTextGenerator, source_chunk


class RagSafeguardTests(unittest.TestCase):
    def test_phone_number_is_rejected_as_personal_information(self) -> None:
        result = validate_question("Call me at 91234567 about my HFE letter")
        self.assertFalse(result.allowed)
        self.assertIn("personal information", result.message or "")

    def test_only_sources_cited_by_the_answer_are_returned(self) -> None:
        generator = FakeTextGenerator(["The relevant process is described here [Source 2]."])
        first = source_chunk("HFE Letter", "hfe-letter")
        second = source_chunk("Option to Purchase", "option-to-purchase")
        service = ResaleReadyQA(
            text_generator=generator,
            retrieve_chunks=lambda query, top_k: [first, second],
        )

        result = service.answer("Where is the Option Fee described?")

        self.assertEqual(1, len(result.sources))
        self.assertEqual("Option to Purchase", result.sources[0].title)


if __name__ == "__main__":
    unittest.main()
