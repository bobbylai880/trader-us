"""JSON Schema definitions enforced across the LLM pipeline."""
from __future__ import annotations

MARKET_ANALYZER_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["risk_level", "bias", "drivers", "summary", "data_gaps"],
    "definitions": {
        "MarketDriver": {
            "type": "object",
            "required": ["factor", "evidence", "direction"],
            "properties": {
                "factor": {"type": "string"},
                "evidence": {"type": "string"},
                "direction": {"type": "string"},
            },
            "additionalProperties": True,
        }
    },
    "properties": {
        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "bias": {"type": "string", "enum": ["bearish", "neutral", "bullish"]},
        "drivers": {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "string"},
                    {"$ref": "#/definitions/MarketDriver"},
                ]
            },
        },
        "premarket_flags": {"type": "array", "items": {"type": "string"}},
        "news_sentiment": {"type": "number", "minimum": -1, "maximum": 1},
        "summary": {"type": "string"},
        "data_gaps": {"type": "array", "items": {"type": "string"}},
    },
}

SECTOR_ANALYZER_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["leading", "lagging", "focus_points", "data_gaps"],
    "properties": {
        "leading": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["symbol", "evidence"],
                "properties": {
                    "symbol": {"type": "string"},
                    "evidence": {"type": "string"},
                },
            },
        },
        "lagging": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["symbol", "evidence"],
                "properties": {
                    "symbol": {"type": "string"},
                    "evidence": {"type": "string"},
                },
            },
        },
        "focus_points": {"type": "array", "items": {"type": "string"}},
        "data_gaps": {"type": "array", "items": {"type": "string"}},
    },
}

STOCK_CLASSIFIER_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["categories", "notes", "data_gaps"],
    "properties": {
        "categories": {
            "type": "object",
            "required": ["Buy", "Hold", "Reduce", "Avoid"],
            "properties": {
                "Buy": {"type": "array", "items": {"$ref": "#/definitions/StockItem"}},
                "Hold": {"type": "array", "items": {"$ref": "#/definitions/StockItem"}},
                "Reduce": {"type": "array", "items": {"$ref": "#/definitions/StockItem"}},
                "Avoid": {"type": "array", "items": {"$ref": "#/definitions/StockItem"}},
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
        "data_gaps": {"type": "array", "items": {"type": "string"}},
    },
    "definitions": {
        "StockDriver": {
            "type": "object",
            "required": ["metric"],
            "properties": {
                "metric": {"type": "string"},
                "value": {"type": ["number", "string"]},
                "direction": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "additionalProperties": True,
        },
        "StockItem": {
            "type": "object",
            "required": ["symbol", "premarket_score", "drivers", "risks"],
            "properties": {
                "symbol": {"type": "string"},
                "premarket_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                },
                "trend_change": {"type": "string"},
                "momentum_strength": {
                    "type": "string",
                    "enum": ["weak", "neutral", "strong"],
                },
                "trend_explanation": {"type": "string"},
                "drivers": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "string"},
                            {"$ref": "#/definitions/StockDriver"},
                        ]
                    },
                },
                "risks": {"type": "array", "items": {"type": "string"}},
                "news_highlights": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    },
}

EXPOSURE_PLANNER_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["target_exposure", "allocation_plan", "constraints"],
    "properties": {
        "target_exposure": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "allocation_plan": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["symbol", "weight"],
                "properties": {
                    "symbol": {"type": "string"},
                    "weight": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "rationale": {"type": "string"},
                },
            },
        },
        "constraints": {"type": "array", "items": {"type": "string"}},
        "data_gaps": {"type": "array", "items": {"type": "string"}},
    },
}

REPORT_COMPOSER_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["markdown", "sections", "data_gaps"],
    "properties": {
        "markdown": {"type": "string"},
        "sections": {"type": "object"},
        "data_gaps": {"type": "array", "items": {"type": "string"}},
    },
}

SCHEMAS = {
    "market_analyzer": MARKET_ANALYZER_SCHEMA,
    "sector_analyzer": SECTOR_ANALYZER_SCHEMA,
    "stock_classifier": STOCK_CLASSIFIER_SCHEMA,
    "exposure_planner": EXPOSURE_PLANNER_SCHEMA,
    "report_composer": REPORT_COMPOSER_SCHEMA,
}
