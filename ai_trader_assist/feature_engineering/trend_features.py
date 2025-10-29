"""Trend feature computation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import pandas as pd


@dataclass
class TrendConfig:
    """Configuration for trend indicator windows."""

    window_short: int = 5
    window_long: int = 20
    momentum_days: int = 10
    ma_fast: int = 10
    ma_slow: int = 30
    volatility_short: int = 5
    volatility_long: int = 20

    @classmethod
    def from_dict(cls, payload: Mapping[str, int] | None) -> "TrendConfig":
        if not payload:
            return cls()
        fields: Dict[str, int] = {}
        for field in (
            "window_short",
            "window_long",
            "momentum_days",
            "ma_fast",
            "ma_slow",
            "volatility_short",
            "volatility_long",
        ):
            if field in payload:
                fields[field] = int(payload[field])
        return cls(**fields)


def _linreg_slope(series: pd.Series, window: int) -> float:
    if series is None or series.empty or len(series.dropna()) < window:
        return 0.0
    tail = series.tail(window).to_numpy(dtype=float)
    if np.isnan(tail).any():
        return 0.0
    x = np.arange(window, dtype=float)
    slope, _ = np.polyfit(x, tail, 1)
    return float(slope)


def _momentum(series: pd.Series, days: int) -> float:
    if series is None or series.empty or len(series.dropna()) <= days:
        return 0.0
    base = float(series.iloc[-days - 1])
    latest = float(series.iloc[-1])
    if base == 0:
        return 0.0
    return float(latest / base - 1.0)


def _volatility_ratio(series: pd.Series, short: int, long: int) -> float:
    if series is None or series.empty or len(series.dropna()) < long:
        return 1.0
    tail = series.tail(long)
    short_std = float(tail.tail(short).pct_change().dropna().std(ddof=0) or 0.0)
    long_std = float(tail.pct_change().dropna().std(ddof=0) or 0.0)
    if long_std == 0:
        return 1.0
    return float(short_std / long_std)


def _moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _moving_average_cross(series: pd.Series, fast: int, slow: int) -> int:
    if series is None or series.empty:
        return 0
    fast_ma = _moving_average(series, fast)
    slow_ma = _moving_average(series, slow)
    if fast_ma.empty or slow_ma.empty:
        return 0
    diff = fast_ma - slow_ma
    if diff.empty or len(diff.dropna()) < 2:
        return 0
    latest = float(diff.iloc[-1])
    previous = float(diff.iloc[-2])
    if latest > 0 and previous <= 0:
        return 1
    if latest < 0 and previous >= 0:
        return -1
    return 0


def _momentum_state(short_slope: float, long_slope: float, momentum: float) -> str:
    threshold = 1e-4
    if momentum > 0.02:
        return "strengthening"
    if momentum < -0.02:
        return "weakening"
    if short_slope - long_slope > threshold:
        return "strengthening"
    if long_slope - short_slope > threshold:
        return "weakening"
    return "stable"


def _trend_strength(
    slope_short: float,
    slope_long: float,
    momentum: float,
    price: float,
    volatility_ratio: float,
) -> float:
    if price <= 0:
        return 0.0
    slope_short_pct = slope_short / price
    slope_long_pct = slope_long / price
    composite = (slope_short_pct * 5 + slope_long_pct * 5 + momentum) / 3
    penalty = np.log(volatility_ratio) if volatility_ratio > 1 else 0.0
    strength = composite - penalty
    return float(np.clip(strength, -1.0, 1.0))


def _trend_state(strength: float) -> str:
    if strength >= 0.05:
        return "uptrend"
    if strength <= -0.05:
        return "downtrend"
    return "flat"


def compute_trend_features(
    price_history: Dict[str, pd.DataFrame],
    config: Mapping[str, int] | None = None,
) -> Dict[str, Dict[str, float | int | str]]:
    """Compute trend metrics for each symbol in ``price_history``."""

    trend_config = TrendConfig.from_dict(config)
    results: Dict[str, Dict[str, float | int | str]] = {}

    for symbol, frame in price_history.items():
        if frame is None or frame.empty or "Close" not in frame:
            results[symbol] = {
                "trend_slope_5d": 0.0,
                "trend_slope_20d": 0.0,
                "momentum_10d": 0.0,
                "volatility_trend": 1.0,
                "moving_avg_cross": 0,
                "trend_strength": 0.0,
                "trend_state": "flat",
                "momentum_state": "stable",
            }
            continue

        close = frame["Close"].dropna()
        slope_short = _linreg_slope(close, trend_config.window_short)
        slope_long = _linreg_slope(close, trend_config.window_long)
        momentum = _momentum(close, trend_config.momentum_days)
        vol_ratio = _volatility_ratio(
            close, trend_config.volatility_short, trend_config.volatility_long
        )
        ma_cross = _moving_average_cross(close, trend_config.ma_fast, trend_config.ma_slow)
        price = float(close.iloc[-1]) if not close.empty else 0.0
        strength = _trend_strength(slope_short, slope_long, momentum, price, vol_ratio)
        state = _trend_state(strength)
        momentum_state = _momentum_state(slope_short, slope_long, momentum)

        results[symbol] = {
            "trend_slope_5d": slope_short,
            "trend_slope_20d": slope_long,
            "momentum_10d": momentum,
            "volatility_trend": vol_ratio,
            "moving_avg_cross": ma_cross,
            "trend_strength": strength,
            "trend_state": state,
            "momentum_state": momentum_state,
        }

    return results
