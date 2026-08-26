from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.openai_client import OpenAIResponsesClient


class OpenAIClientTests(unittest.TestCase):
    @patch("src.openai_client.OpenAI")
    def test_api_key_is_not_in_model_input_and_safety_identifier_is_sent(
        self, openai_class: MagicMock
    ) -> None:
        responses_create = openai_class.return_value.responses.create
        responses_create.return_value = SimpleNamespace(output_text="Safe answer")
        api_key = "sk-proj-test-secret-value"

        client = OpenAIResponsesClient(api_key=api_key, model="gpt-4o-mini")
        result = client.generate(
            instructions="System instructions",
            input_text="HDB resale question",
            max_output_tokens=100,
        )

        self.assertEqual("Safe answer", result)
        openai_class.assert_called_once_with(api_key=api_key)
        request = responses_create.call_args.kwargs
        self.assertEqual("resaleready-demo-user", request["safety_identifier"])
        self.assertFalse(request["store"])
        self.assertNotIn(api_key, request["instructions"])
        self.assertNotIn(api_key, request["input"])

    def test_safety_identifier_is_bounded(self) -> None:
        with self.assertRaises(ValueError):
            OpenAIResponsesClient(
                api_key="test-key",
                safety_identifier="x" * 65,
            )


if __name__ == "__main__":
    unittest.main()
