"""Reusable OpenAI Responses API text-generation client."""

from __future__ import annotations

from typing import Protocol

from openai import OpenAI


class TextGenerator(Protocol):
    """Minimal text-generation interface used by the RAG chain."""

    def generate(
        self,
        *,
        instructions: str,
        input_text: str,
        max_output_tokens: int,
    ) -> str: ...


class OpenAIResponsesClient:
    """Small wrapper around the OpenAI Responses API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-5.6-luna",
        safety_identifier: str = "resaleready-demo-user",
    ) -> None:
        if not api_key.strip():
            raise ValueError("An OpenAI API key is required.")
        if not model.strip():
            raise ValueError("An OpenAI response model is required.")
        if not safety_identifier.strip() or len(safety_identifier) > 64:
            raise ValueError("safety_identifier must contain 1 to 64 characters.")
        self.model = model
        self.safety_identifier = safety_identifier
        self._client = OpenAI(api_key=api_key)

    def generate(
        self,
        *,
        instructions: str,
        input_text: str,
        max_output_tokens: int,
    ) -> str:
        response = self._client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_text,
            max_output_tokens=max_output_tokens,
            safety_identifier=self.safety_identifier,
            store=False,
        )
        output_text = response.output_text.strip()
        if not output_text:
            raise RuntimeError("OpenAI returned an empty text response.")
        return output_text
