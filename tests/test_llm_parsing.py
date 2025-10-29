"""Tests for resilient parsing of LLM JSON responses."""
from __future__ import annotations

import pytest

from ai_trader_assist.llm_operators.base import LLMOperator, LLMValidationError


def _parse(raw: str):
    operator = object.__new__(LLMOperator)
    return LLMOperator._parse_response(operator, raw)


def test_parse_single_quoted_json_uses_json5_fallback():
    raw = """```json\n{'foo': 'bar',}\n```"""
    parsed = _parse(raw)
    assert parsed == {"foo": "bar"}


def test_parse_invalid_payload_raises_validation_error():
    operator = object.__new__(LLMOperator)
    with pytest.raises(LLMValidationError):
        LLMOperator._parse_response(operator, "no json here")
