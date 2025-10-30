import pytest

from ai_trader_assist.decision_engine.stock_scoring import StockDecisionEngine
from ai_trader_assist.feature_engineering.pipeline import _premarket_score, _structure_score


def test_premarket_score_clamped_to_unit_interval() -> None:
    assert _premarket_score(0.0, 0.0, 0.0) == pytest.approx(0.1)
    assert _premarket_score(0.2, 10.0, -1.0) == pytest.approx(1.0)
    assert _premarket_score(-0.5, -3.0, 2.0) == pytest.approx(0.0)


def test_stock_scoring_uses_normalised_premarket_penalty() -> None:
    engine = StockDecisionEngine()
    symbol = "TEST"
    base_features = {
        "rsi_norm": 0.8,
        "macd_signal": 0.05,
        "trend_slope": 0.02,
        "volume_score": 0.5,
        "structure_score": 0.3,
        "risk_modifier": 0.0,
        "atr_pct": 0.02,
        "price": 50.0,
        "news_score": 0.2,
        "recent_news": [],
        "trend_slope_5d": 0.01,
        "trend_slope_20d": 0.02,
        "momentum_10d": 0.1,
        "volatility_trend": 1.0,
        "moving_avg_cross": 1,
        "trend_strength": 0.3,
        "trend_state": "uptrend",
        "momentum_state": "strengthening",
        "position_shares": 0.0,
        "position_value": 0.0,
    }

    neutral = engine.score_stocks({symbol: base_features}, {symbol: {"score": 0.0}})[0]
    penalised = engine.score_stocks({symbol: base_features}, {symbol: {"score": 0.8}})[0]

    assert neutral["symbol"] == symbol
    assert penalised["symbol"] == symbol

    assert 0.0 <= neutral["score"] <= 1.0
    assert 0.0 <= penalised["score"] <= 1.0

    expected_penalised = neutral["score"] * (1 - 0.8 * 0.25)
    assert penalised["score"] == pytest.approx(expected_penalised, rel=1e-5)
    assert penalised["score"] < neutral["score"]
    assert penalised["premarket"] == pytest.approx(0.8)


def test_structure_score_ignores_nan_components() -> None:
    price = 100.0
    ma_values = [float("nan"), 95.0, float("inf"), 0.0]

    score = _structure_score(price, ma_values)

    expected = price / 95.0 - 1.0
    assert score == pytest.approx(expected)

    zero_score = _structure_score(float("nan"), [95.0])
    assert zero_score == 0.0
