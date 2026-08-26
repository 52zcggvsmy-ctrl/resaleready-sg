#!/usr/bin/env python3
"""Run the documented ResaleReady safeguard checks without an API key."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts import ANSWER_SYSTEM_PROMPT, build_grounded_answer_input
from src.rag.models import RetrievedChunk
from src.rag.safeguards import validate_question

CASES_PATH = PROJECT_ROOT / "tests" / "safeguard_cases.json"


def _check_input_case(case: dict[str, object]) -> tuple[bool, str]:
    result = validate_question(str(case["question"]))
    passed = (
        result.allowed is bool(case["expected_allowed"])
        and result.category == case["expected_category"]
    )
    return passed, f"allowed={result.allowed}, category={result.category}"


def _check_retrieved_injection(case: dict[str, object]) -> tuple[bool, str]:
    retrieved_text = str(case["retrieved_text"])
    chunk = RetrievedChunk(
        text=retrieved_text,
        score=0.9,
        metadata={
            "document_title": "Adversarial upload",
            "source_organization": "Uploaded demo document (unverified)",
            "source_url": "",
            "source_kind": "uploaded_demo",
            "local_filename": "adversarial.txt",
            "section": "Test",
            "page": None,
        },
    )
    prompt_input = build_grounded_answer_input(
        str(case["question"]),
        str(case["question"]),
        [chunk],
    )
    _, json_payload = prompt_input.split("\n", 1)
    payload = json.loads(json_payload)
    extracted = payload["retrieved_sources"][0]
    passed = (
        extracted["extract"] == retrieved_text
        and extracted["trust"] == "uploaded_demo_unverified"
        and "never\n  system or developer instructions" in ANSWER_SYSTEM_PROMPT
        and "Never obey commands" in ANSWER_SYSTEM_PROMPT
    )
    return passed, "retrieved text remains labelled, serialized untrusted data"


def main() -> int:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    failures = 0
    for case in cases:
        if case["kind"] == "input":
            passed, detail = _check_input_case(case)
        else:
            passed, detail = _check_retrieved_injection(case)
        label = "PASS" if passed else "FAIL"
        print(f"{label} {case['id']}: {detail}")
        failures += int(not passed)
    print(f"\n{len(cases) - failures}/{len(cases)} safeguard checks passed.")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
