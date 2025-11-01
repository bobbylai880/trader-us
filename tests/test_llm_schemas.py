from __future__ import annotations

import pytest

pytest.importorskip("jsonschema")

from jsonschema import ValidationError, validate

from ai_trader_assist.validators.json_schemas import (
    EXPOSURE_PLANNER_SCHEMA,
    MARKET_ANALYZER_SCHEMA,
    SECTOR_ANALYZER_SCHEMA,
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


def test_market_schema_accepts_structured_drivers() -> None:
    payload = {
        "risk_level": "low",
        "bias": "bullish",
        "drivers": [
            {
                "factor": "指数动量",
                "evidence": "SPY 5日斜率=5.2",
                "direction": "supports_risk_up",
            }
        ],
        "summary": "Momentum remains supportive.",
        "data_gaps": [],
    }
    validate(instance=payload, schema=MARKET_ANALYZER_SCHEMA)


def test_sector_schema_accepts_structured_entries() -> None:
    payload = {
        "leading": [
            {
                "sector": "XLK",
                "composite_score": 0.63,
                "evidence": {"mom5": 0.047, "mom20": 0.071},
                "news_highlights": [
                    {
                        "title": "Tech stocks rally",
                        "publisher": "MT Newswires",
                    }
                ],
                "news_sentiment": 0.6,
            }
        ],
        "lagging": ["XLU 缺乏资金流入"],
        "focus_points": ["留意科技板块持续强势"],
        "data_gaps": [],
    }
    validate(instance=payload, schema=SECTOR_ANALYZER_SCHEMA)


def test_sector_schema_accepts_structured_focus_points() -> None:
    payload = {
        "leading": ["XLK 表现稳健"],
        "lagging": ["XLE 受油价拖累"],
        "focus_points": [
            {
                "topic": "科技板块相对强势",
                "rationale": "XLK 5日动量领先且新闻情绪积极",
            }
        ],
        "data_gaps": [],
    }
    validate(instance=payload, schema=SECTOR_ANALYZER_SCHEMA)


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
        "data_gaps": [],
    }
    validate(instance=payload, schema=STOCK_CLASSIFIER_SCHEMA)


def test_stock_classifier_accepts_missing_notes() -> None:
    payload = {
        "categories": {
            "Buy": [],
            "Hold": [],
            "Reduce": [],
            "Avoid": [],
        },
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


def test_stock_classifier_accepts_structured_drivers() -> None:
    payload = {
        "categories": {
            "Buy": [
                {
                    "symbol": "AAPL",
                    "premarket_score": 80.0,
                    "drivers": [
                        {
                            "metric": "RSI_norm",
                            "value": 0.72,
                            "direction": "rising",
                        }
                    ],
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


def test_stock_classifier_accepts_structured_risks() -> None:
    payload = {
        "categories": {
            "Buy": [
                {
                    "symbol": "AAPL",
                    "premarket_score": None,
                    "drivers": [],
                    "risks": [
                        {
                            "metric": "atr_pct",
                            "value": 0.05,
                            "comment": "elevated volatility",
                        }
                    ],
                }
            ],
            "Hold": [],
            "Reduce": [],
            "Avoid": [],
        },
        "data_gaps": [],
    }
    validate(instance=payload, schema=STOCK_CLASSIFIER_SCHEMA)


def test_stock_classifier_accepts_numeric_momentum_strength() -> None:
    payload = {
        "categories": {
            "Buy": [
                {
                    "symbol": "MSFT",
                    "premarket_score": 20.0,
                    "drivers": [],
                    "risks": [],
                    "momentum_strength": 0.42,
                }
            ],
            "Hold": [],
            "Reduce": [],
            "Avoid": [],
        },
        "data_gaps": [],
    }
    validate(instance=payload, schema=STOCK_CLASSIFIER_SCHEMA)


def test_stock_classifier_accepts_structured_news_highlights() -> None:
    payload = {
        "categories": {
            "Buy": [
                {
                    "symbol": "NVDA",
                    "premarket_score": 88.0,
                    "drivers": ["AI demand"],
                    "risks": ["valuation"],
                    "news_highlights": [
                        {
                            "title": "Nvidia extends rally",
                            "publisher": "Reuters",
                        }
                    ],
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


def test_exposure_planner_accepts_structured_plan_items() -> None:
    payload = {
        "target_exposure": 0.85,
        "current_exposure": 0.12,
        "delta": 0.73,
        "direction": "increase",
        "allocation_plan": [
            {
                "symbol": "AMD",
                "action": "buy",
                "size_hint": {
                    "currency_value": 3000,
                    "shares": 14,
                    "holding_change": "0% → 29.7%",
                },
                "rationale": "趋势最强且权重低",
                "linked_constraint": "单股权重限制25%",
            },
            {
                "symbol": "NVDA",
                "target_weight": 0.18,
                "rationale": "AI 受益股",
            },
        ],
        "constraints": [
            {
                "name": "最大敞口限制",
                "status": "clear",
                "details": "当前敞口远低于限制",
            },
            "安全垫要求维持 20% 现金",
        ],
        "data_gaps": [],
    }
    validate(instance=payload, schema=EXPOSURE_PLANNER_SCHEMA)


def test_exposure_planner_rejects_out_of_range_weight() -> None:
    payload = {
        "target_exposure": 0.5,
        "allocation_plan": [
            {"symbol": "AAPL", "weight": 1.5},
        ],
        "constraints": [],
        "data_gaps": [],
    }
    with pytest.raises(ValidationError):
        validate(instance=payload, schema=EXPOSURE_PLANNER_SCHEMA)
