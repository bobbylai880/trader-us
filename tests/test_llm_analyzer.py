from datetime import date
from pathlib import Path

from ai_trader_assist.llm.analyzer import DeepSeekAnalyzer
from ai_trader_assist.portfolio_manager.state import Position, PortfolioState


def _prompt_paths() -> dict:
    base = Path("configs/prompts")
    return {
        "market_overview": base / "deepseek_market_overview.md",
        "sector_analysis": base / "deepseek_sector_analysis.md",
        "stock_actions": base / "deepseek_stock_actions.md",
        "exposure_check": base / "deepseek_exposure_check.md",
        "report_compose": base / "deepseek_report_compose.md",
    }


def test_analyzer_emits_structured_outputs():
    analyzer = DeepSeekAnalyzer(prompt_files=_prompt_paths())
    portfolio = PortfolioState(
        cash=10000.0,
        positions=[Position(symbol="AMD", shares=4, avg_cost=120.0, last_price=130.0)],
    )

    risk = {"risk_level": "low", "bias": "bullish", "target_exposure": 0.74}
    sector_scores = [
        {
            "symbol": "XLK",
            "score": 0.25,
            "features": {
                "momentum_5d": 0.05,
                "momentum_20d": 0.10,
                "rs": 0.08,
                "volume_trend": 0.15,
                "news_score": 0.0,
            },
        },
        {
            "symbol": "XLF",
            "score": -0.10,
            "features": {
                "momentum_5d": -0.04,
                "momentum_20d": -0.02,
                "rs": -0.05,
                "volume_trend": -0.01,
                "news_score": 0.0,
            },
        },
    ]
    stock_scores = [
        {
            "symbol": "AAPL",
            "score": 0.82,
            "action": "buy",
            "confidence": 0.82,
            "atr_pct": 0.02,
            "price": 190.0,
            "position_shares": 0.0,
            "features": {
                "rsi_norm": 0.7,
                "macd_signal": 0.05,
                "trend_slope": 0.01,
                "volume_score": 0.2,
                "structure_score": 0.03,
                "risk_modifier": 0.0,
                "trend_strength": 0.4,
                "momentum_state": "strengthening",
                "momentum_10d": 0.06,
                "volatility_trend": 0.9,
                "trend_score": 0.7,
                "position_shares": 0.0,
            },
        },
        {
            "symbol": "MSFT",
            "score": 0.4,
            "action": "reduce",
            "confidence": 0.4,
            "atr_pct": 0.04,
            "price": 320.0,
            "position_shares": 0.0,
            "features": {
                "rsi_norm": 0.5,
                "macd_signal": 0.01,
                "trend_slope": 0.002,
                "volume_score": -0.05,
                "structure_score": 0.01,
                "risk_modifier": 0.0,
                "trend_strength": -0.2,
                "momentum_state": "weakening",
                "momentum_10d": -0.03,
                "volatility_trend": 1.3,
                "trend_score": 0.3,
                "position_shares": 0.0,
            },
        },
        {
            "symbol": "AMD",
            "score": 0.35,
            "action": "reduce",
            "confidence": 0.35,
            "atr_pct": 0.06,
            "price": 130.0,
            "position_shares": 4.0,
            "features": {
                "rsi_norm": 0.55,
                "macd_signal": -0.01,
                "trend_slope": -0.003,
                "volume_score": 0.05,
                "structure_score": -0.02,
                "risk_modifier": 0.0,
                "trend_strength": -0.3,
                "momentum_state": "rolling_over",
                "momentum_10d": -0.04,
                "volatility_trend": 1.4,
                "trend_score": 0.25,
                "position_shares": 4.0,
            },
        },
    ]

    orders = {
        "buy": [
            {
                "symbol": "AAPL",
                "shares": 5,
                "price": 190.0,
                "notional": 950.0,
                "confidence": 0.82,
            }
        ],
        "sell": [],
    }

    market_features = {
        "RS_SPY": 0.05,
        "RS_QQQ": 0.08,
        "VIX_Z": -0.9,
        "PUTCALL_Z": -0.3,
        "BREADTH": 0.6,
    }
    premarket_flags = {
        "AAPL": {"dev": 0.04, "vol_ratio": 1.8, "score": 0.35},
        "MSFT": {"dev": 0.01, "vol_ratio": 0.9, "score": 0.05},
    }

    news_bundle = {
        "market": {
            "headlines": [
                {
                    "symbol": "SPY",
                    "title": "指数高开，科技板块领涨",
                    "publisher": "Newswire",
                    "published": "2025-10-27T12:00:00+00:00",
                }
            ],
            "sentiment": 0.4,
        },
        "sectors": {
            "XLK": {
                "headlines": [
                    {
                        "symbol": "XLK",
                        "title": "半导体资本开支预期上调",
                        "publisher": "TechDaily",
                        "published": "2025-10-27T11:00:00+00:00",
                    }
                ],
                "sentiment": 0.6,
            }
        },
        "stocks": {
            "AAPL": {
                "headlines": [
                    {
                        "title": "苹果发布新品引发预订热潮",
                        "publisher": "MarketWatch",
                        "published": "2025-10-26T20:00:00+00:00",
                    }
                ],
                "sentiment": 0.5,
            },
            "MSFT": {
                "headlines": [],
                "sentiment": -0.1,
            },
            "AMD": {
                "headlines": [
                    {
                        "title": "AMD faces competitive pricing pressure",
                        "publisher": "SemiNews",
                        "published": "2025-10-25T18:00:00+00:00",
                    }
                ],
                "sentiment": -0.3,
            },
        },
    }

    payload = analyzer.run(
        trading_day=date(2025, 10, 27),
        risk=risk,
        sector_scores=sector_scores,
        stock_scores=stock_scores,
        orders=orders,
        portfolio_state=portfolio,
        market_features=market_features,
        premarket_flags=premarket_flags,
        news=news_bundle,
    )

    market_view = payload["market_overview"]
    assert market_view["risk_level"] == "low"
    assert market_view["drivers"]
    for driver in market_view["drivers"]:
        assert {"factor", "evidence", "direction"} <= driver.keys()
    assert market_view["news_highlights"]

    sector_view = payload["sector_analysis"]
    assert all("evidence" in entry for entry in sector_view["leading"])
    assert sector_view["leading"][0]["news_highlights"]

    stock_view = payload["stock_actions"]
    assert stock_view["categories"]["Buy"]
    assert stock_view["categories"]["Buy"][0]["drivers"]
    assert stock_view["categories"]["Buy"][0]["news_highlights"]
    assert "trend_change" in stock_view["categories"]["Buy"][0]
    assert all(item["position_shares"] > 0 for item in stock_view["categories"]["Reduce"])
    assert all(
        item["symbol"] != "MSFT"
        for item in stock_view["categories"]["Reduce"]
    ), "Non-held symbols should not appear in Reduce"

    exposure_view = payload["exposure_check"]
    assert exposure_view["direction"] in {"increase", "maintain", "decrease"}
    assert isinstance(exposure_view["allocation_plan"], list)

    report_view = payload["report_compose"]
    assert report_view["sections"]["market"].startswith("风险")
    assert report_view["markdown"].startswith("📆 2025-10-27")
    assert report_view["sections"]["news"] == [
        {
            "symbol": "SPY",
            "title": "指数高开，科技板块领涨",
            "publisher": "Newswire",
            "published": "2025-10-27T12:00:00+00:00",
        }
    ]
