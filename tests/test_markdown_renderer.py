from __future__ import annotations

from typing import Dict, List

from ai_trader_assist.report_builder.markdown_renderer import (
    MarkdownRenderConfig,
    MarkdownRenderer,
)


def build_sample_report() -> Dict:
    return {
        "as_of": "2025-10-30T06:10:00-07:00",
        "snapshot_id": "c68b",
        "input_hash": "sha256:abc123",
        "config_profile": "baseline",
        "appendix": {
            "report_json_path": "storage/daily_2025-10-30/report.json",
            "reproduction_command": "python -m ai_trader_assist.jobs.run_daily --config configs/base.json --output-dir storage/daily_2025-10-30 --date 2025-10-30",
        },
        "artefact_summary": [
            {
                "name": "operations.jsonl",
                "path": "storage/operations.jsonl",
                "entries": 12,
            }
        ],
        "market": {
            "risk_level": "medium",
            "bias": "slightly_bullish",
            "summary": "Tech momentum improves while rates stabilize.",
            "drivers": ["earnings beats", "AI demand"],
            "premarket_flags": ["NVDA:gap_up"],
            "news_sentiment": 0.12,
            "news_highlights": [
                {"title": "Fed holds", "source": "WSJ", "ts": 1730286000},
                {"title": "AI boom", "source": "Bloomberg"},
            ],
        },
        "exposure": {
            "current": 0.63,
            "target": 0.72,
            "delta": 0.09,
            "constraints": {"max_exposure": 0.8, "max_single_weight": 0.2},
        },
        "allocation_plan": [
            {
                "symbol": "NVDA",
                "action": "BUY",
                "weight": 0.18,
                "budget": 12000,
                "shares": 11,
                "price_ref": 1090.2,
                "stops": {"atr_k1": 950.12},
                "targets": {"atr_k2": 1250.6},
                "reasons": ["trend_strength↑", "sector_leading"],
            },
            {
                "symbol": "AAPL",
                "action": "REDUCE",
                "weight": -0.05,
                "budget": -5000,
                "shares": -22,
                "price_ref": 228.5,
                "stops": {},
                "targets": {},
                "reasons": ["rebalance"],
            },
        ],
        "positions": {
            "cash": 18240.0,
            "equity_value": 159846.0,
            "exposure": 0.85,
            "items": [
                {"symbol": "NVDA", "shares": 130, "avg_cost": 1090.2},
                {"symbol": "AAPL", "shares": 80, "avg_cost": 226.5},
            ],
        },
        "sectors": [
            {"symbol": "XLK", "score": 0.74, "state": "leading", "news_highlight": "AI"}
        ],
        "stocks": [
            {
                "symbol": "NVDA",
                "category": "BUY",
                "premarket_score": 0.81,
                "trend_strength": 0.76,
                "momentum_10d": 0.12,
                "volatility_trend": 1.3,
                "trend_explanation": "uptrend / strong",
                "news_highlight": "Record guidance",
                "risks": ["earnings next week"],
                "flags": ["gap_up"],
            }
        ],
        "data_gaps": ["missing_yield_curve"],
        "ai_summary": {"text": "市场风险中性", "key_points": ["重点关注科技"]},
    }


def test_markdown_renderer_renders_full_report() -> None:
    report = build_sample_report()
    renderer = MarkdownRenderer(MarkdownRenderConfig())
    markdown = renderer.render(report)

    assert "# 盘前报告（2025-10-30T06:10:00-07:00）" in markdown
    assert "snapshot_id: c68b" in markdown
    assert "风险等级：**medium**" in markdown
    assert "组合敞口" in markdown
    assert "NVDA" in markdown and "AAPL" in markdown
    assert "AI 总结" in markdown
    assert "无额外字段" in markdown
    assert "## 产出摘要" in markdown
    assert "operations.jsonl" in markdown
    assert "复现命令" in markdown


def test_markdown_renderer_fold_behavior() -> None:
    report = build_sample_report()
    many_items: List[Dict] = []
    for idx in range(8):
        many_items.append(
            {
                "symbol": f"SYM{idx}",
                "action": "BUY",
                "weight": 0.01,
                "budget": 1000 + idx,
                "shares": 10,
                "price_ref": 10.0,
                "stops": {"atr_k1": 8.0},
                "targets": {"atr_k2": 12.0},
                "reasons": ["test"],
            }
        )
    report["allocation_plan"] = many_items
    config = MarkdownRenderConfig(max_rows_per_section=5)
    markdown = MarkdownRenderer(config).render(report)

    assert "已折叠 3 项" in markdown


def test_markdown_renderer_hidden_weight() -> None:
    report = build_sample_report()
    report["allocation_plan"].append(
        {
            "symbol": "LOW",
            "action": "BUY",
            "weight": 0.0005,
            "budget": 10,
            "shares": 1,
            "price_ref": 10,
            "stops": {},
            "targets": {},
            "reasons": ["micro"],
        }
    )
    config = MarkdownRenderConfig(hide_below_weight=0.001)
    markdown = MarkdownRenderer(config).render(report)

    assert "已隐藏 1 项" in markdown
    assert "LOW" in markdown


def test_markdown_renderer_missing_fields_warning() -> None:
    config = MarkdownRenderConfig()
    markdown = MarkdownRenderer(config).render({})

    assert "数据校验失败" in markdown
    assert "字段缺失：market" in markdown


def test_markdown_renderer_additional_fields_section() -> None:
    report = build_sample_report()
    report["trend_overview"] = [{"symbol": "NVDA", "trend": "up"}]
    markdown = MarkdownRenderer(MarkdownRenderConfig()).render(report)

    assert "## 附加信息" in markdown
    assert "### trend_overview" in markdown
    assert '"trend": "up"' in markdown
