"""Indicator computations for the decision engines."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": histogram}
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    volume = df["Volume"].replace(0, np.nan)
    cumulative_tp_vol = (typical_price * volume).cumsum()
    cumulative_vol = volume.cumsum().replace(0, np.nan)
    return cumulative_tp_vol / cumulative_vol


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std(ddof=0)
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def slope(series: pd.Series, window: int = 5) -> pd.Series:
    def _calc(x: np.ndarray) -> float:
        y = x
        x_idx = np.arange(len(x))
        if len(x_idx) < 2:
            return 0.0
        slope, _ = np.polyfit(x_idx, y, 1)
        return slope

    return series.rolling(window=window, min_periods=2).apply(_calc, raw=True)


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Calculate Bollinger Bands.
    
    Parameters
    ----------
    series : pd.Series
        Price series (typically Close).
    window : int
        Lookback window for moving average, default 20.
    num_std : float
        Number of standard deviations for bands, default 2.0.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: middle, upper, lower, pct_b, bandwidth
        - middle: SMA (middle band)
        - upper: upper band (SMA + num_std * std)
        - lower: lower band (SMA - num_std * std)
        - pct_b: %B indicator ((price - lower) / (upper - lower))
        - bandwidth: band width ((upper - lower) / middle)
    """
    middle = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std(ddof=0)
    
    upper = middle + num_std * std
    lower = middle - num_std * std
    
    # %B: Position within bands (0 = lower, 1 = upper, can exceed)
    band_range = (upper - lower).replace(0, np.nan)
    pct_b = (series - lower) / band_range
    
    # Bandwidth: Volatility measure
    bandwidth = band_range / middle.replace(0, np.nan)
    
    return pd.DataFrame({
        "middle": middle,
        "upper": upper,
        "lower": lower,
        "pct_b": pct_b,
        "bandwidth": bandwidth,
    })


def bollinger_position_score(series: pd.Series, window: int = 20) -> pd.Series:
    """Calculate mean reversion score based on Bollinger Band position.
    
    Returns a score from 0 to 1:
    - Near lower band (oversold): score approaches 1 (buy signal)
    - Near upper band (overbought): score approaches 0 (sell signal)
    - At middle band: score = 0.5 (neutral)
    
    Parameters
    ----------
    series : pd.Series
        Price series (typically Close).
    window : int
        Lookback window for Bollinger Bands.
    
    Returns
    -------
    pd.Series
        Mean reversion score (0-1, higher = more oversold/buy opportunity)
    """
    bb = bollinger_bands(series, window=window)
    pct_b = bb["pct_b"]
    
    # Invert %B: low %B (oversold) -> high score, high %B (overbought) -> low score
    # Clip to 0-1 range for extreme values
    score = (1 - pct_b).clip(0, 1)
    
    return score
