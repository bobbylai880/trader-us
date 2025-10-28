import numpy as np
import pandas as pd

from ai_trader_assist.feature_engineering.trend_features import compute_trend_features


def test_trend_features_detect_uptrend():
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    close = pd.Series(np.linspace(100, 180, len(dates)), index=dates)
    frame = pd.DataFrame({"Close": close})

    result = compute_trend_features({"AAPL": frame})
    features = result["AAPL"]

    assert features["trend_state"] == "uptrend"
    assert features["trend_strength"] > 0
    assert features["momentum_state"] == "strengthening"
    assert features["momentum_10d"] > 0


def test_trend_features_handle_missing_data():
    result = compute_trend_features({"AAPL": pd.DataFrame()})
    features = result["AAPL"]

    assert features["trend_state"] == "flat"
    assert features["trend_strength"] == 0.0
    assert features["volatility_trend"] == 1.0
