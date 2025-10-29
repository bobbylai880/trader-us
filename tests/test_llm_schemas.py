from __future__ import annotations

import pytest
from jsonschema import ValidationError, validate

from ai_trader_assist.validators.json_schemas import (
    MARKET_ANALYZER_SCHEMA,
    STOCK_CLASSIFIER_SCHEMA,
)


def test_market_schema_accepts_valid_payload() -> None:
    payload = {
        "risk_level": "medium",
        "bias": "neutral",
        "drivers": ["CPI data"],
        "summary": "Market is balanced.",
        "premarket_flags": [],
        "news_sentiment": 0.1,
        "data_gaps": [],
    }
    validate(instance=payload, schema=MARKET_ANALYZER_SCHEMA)


def test_market_schema_rejects_invalid_risk_level() -> None:
    payload = {
        "risk_level": "extreme",
        "bias": "neutral",
        "drivers": [],
        "summary": "",
        "data_gaps": [],
    }
    with pytest.raises(ValidationError):
        validate(instance=payload, schema=MARKET_ANALYZER_SCHEMA)


def test_stock_classifier_requires_categories() -> None:
    payload = {
        "categories": {
            "Buy": [
                {
                    "symbol": "AAPL",
                    "premarket_score": 55.0,
                    "drivers": ["earnings"],
                    "risks": ["valuation"],
                }
            ],
            "Hold": [],
            "Reduce": [],
            "Avoid": [],
        },
        "notes": [],
        "data_gaps": [],
    }
    validate(instance=payload, schema=STOCK_CLASSIFIER_SCHEMA)


def test_stock_classifier_rejects_missing_required_field() -> None:
    payload = {
        "categories": {
            "Buy": [
                {
                    "symbol": "AAPL",
                    "drivers": [],
                    "risks": [],
                    "premarket_score": 120,
                }
            ],
            "Hold": [],
            "Reduce": [],
            "Avoid": [],
        },
        "notes": [],
        "data_gaps": [],
    }
    with pytest.raises(ValidationError):
        validate(instance=payload, schema=STOCK_CLASSIFIER_SCHEMA)
