import pandas as pd
import pytest

from ai_trader_assist.feature_engineering import indicators
from ai_trader_assist.feature_engineering.pipeline import (
    _compute_vix_metrics,
    _default_macro_series,
    _normalize_macro_series,
)


def test_compute_vix_metrics_returns_value_and_zscore():
    close = pd.Series(range(10, 36))
    frame = pd.DataFrame({"Close": close})

    metrics = _compute_vix_metrics(frame)

    expected_z = indicators.zscore(frame["Close"], window=20).iloc[-1]
    assert metrics["vix_value"] == pytest.approx(float(close.iloc[-1]))
    assert metrics["vix_zscore"] == pytest.approx(float(expected_z))


def test_compute_vix_metrics_handles_missing_history():
    frame = pd.DataFrame()
    metrics = _compute_vix_metrics(frame)
    assert metrics == {"vix_value": 0.0, "vix_zscore": 0.0}


def test_normalize_macro_series_supports_strings_and_mappings():
    config = {"ABC": {"label": "Alpha"}, "XYZ": "Beta"}
    normalized = _normalize_macro_series(config)

    assert normalized["ABC"]["label"] == "Alpha"
    assert normalized["XYZ"]["label"] == "Beta"


def test_default_macro_series_not_empty():
    default_series = _default_macro_series()
    assert "CPIAUCSL" in default_series
    assert all(isinstance(meta, dict) for meta in default_series.values())
