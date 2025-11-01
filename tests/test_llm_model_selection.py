import os

import pytest

from ai_trader_assist.llm.client import get_model_for_operator


@pytest.fixture(autouse=True)
def _clear_stage_env(monkeypatch):
    """Ensure stage-specific overrides are cleared between tests."""

    for key in list(os.environ):
        if key.startswith("DEEPSEEK_MODEL_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    yield


def test_stage_env_override_has_highest_priority(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL_REPORT_COMPOSER", "stage-model")
    monkeypatch.setenv("DEEPSEEK_MODEL", "global-model")

    resolved = get_model_for_operator(
        "report_composer",
        {"model": "config-model"},
        base_model="base-model",
    )

    assert resolved == "stage-model"


def test_operator_config_used_when_stage_env_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_MODEL_REPORT_COMPOSER", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    resolved = get_model_for_operator(
        "report_composer",
        {"model": "config-model"},
        base_model="base-model",
    )

    assert resolved == "config-model"


def test_global_env_used_when_no_stage_or_config(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL", "global-model")

    resolved = get_model_for_operator(
        "market_analyzer",
        {},
        base_model="base-model",
    )

    assert resolved == "global-model"


def test_base_model_used_when_no_env_or_config(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    resolved = get_model_for_operator(
        "sector_analyzer",
        {},
        base_model="base-model",
    )

    assert resolved == "base-model"


def test_default_model_used_when_everything_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    resolved = get_model_for_operator(
        "exposure_planner",
        {},
        base_model=None,
    )

    assert resolved == "deepseek-chat"
