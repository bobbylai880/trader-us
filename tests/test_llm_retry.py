from __future__ import annotations

from pathlib import Path

import pytest

from ai_trader_assist.llm_operators.base import LLMOperatorConfig, LLMRunArtifacts
from ai_trader_assist.llm_operators.market_analyzer import MarketAnalyzerOperator
from ai_trader_assist.validators.json_schemas import MARKET_ANALYZER_SCHEMA


class DummyClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def chat(self, messages, stage, temperature, max_tokens):
        response = self.responses[self.calls]
        self.calls += 1
        return response, {"total_tokens": 5, "prompt_tokens": 3, "completion_tokens": 2}


@pytest.mark.parametrize("retries", [1])
def test_operator_retries_and_parses(tmp_path: Path, retries: int) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("请输出 JSON", encoding="utf-8")

    client = DummyClient([
        "not-json",
        '{"risk_level": "low", "bias": "bullish", "drivers": [], "summary": "ok", "data_gaps": []}',
    ])

    operator = MarketAnalyzerOperator(
        config=LLMOperatorConfig(prompt_file=prompt_file, retries=retries),
        client=client,
        schema=MARKET_ANALYZER_SCHEMA,
        base_prompt=None,
        logger=None,
    )

    artefacts = LLMRunArtifacts(stage_dir=tmp_path / "stage")

    result, usage = operator.execute(
        payload={
            "as_of": "2024-01-01",
            "features": {"market": {}},
        },
        artefacts=artefacts,
    )

    assert result["risk_level"] == "low"
    assert usage["total_tokens"] == 5
    assert client.calls == 2
