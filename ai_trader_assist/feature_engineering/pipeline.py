"""Feature preparation helpers for the daily job."""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd

from ..data_collector.fred_client import FredClient
from ..data_collector.yf_client import YahooFinanceClient
from ..portfolio_manager.state import PortfolioState
from . import indicators


def _fetch_history(
    client: YahooFinanceClient, symbol: str, start: datetime, end: datetime
) -> pd.DataFrame:
    data = client.fetch_history(symbol, start=start, end=end, interval="1d")
    if data is None or data.empty:
        return pd.DataFrame()
    # yfinance may return multi-index columns (Price x Ticker). Reduce to a
    # single ticker frame so downstream lookups like df["Close"] return a
    # Series instead of a nested frame, avoiding ambiguous truth-value checks.
    if isinstance(data.columns, pd.MultiIndex):
        level_names = list(data.columns.names)
        if "Ticker" in level_names:
            ticker_level = level_names.index("Ticker")
        else:
            ticker_level = len(level_names) - 1
        try:
            data = data.xs(symbol, axis=1, level=ticker_level)
        except KeyError:
            # Fall back to dropping the last level if the symbol key is missing.
            data = data.droplevel(ticker_level, axis=1)
    return data


def _latest_close(df: pd.DataFrame) -> float:
    if "Close" not in df or df["Close"].empty:
        return 0.0
    return float(df["Close"].iloc[-1])


def _pct_change(series: pd.Series, periods: int) -> float:
    if series is None or series.empty or len(series) <= periods:
        return 0.0
    base = series.iloc[-periods - 1]
    if base == 0:
        return 0.0
    return float(series.iloc[-1] / base - 1.0)


def _relative_to_ma(series: pd.Series, window: int) -> float:
    if series is None or series.empty:
        return 0.0
    ma = series.rolling(window=window, min_periods=window).mean()
    if ma.empty:
        return 0.0
    ma_latest = float(ma.iloc[-1])
    if math.isclose(ma_latest, 0.0):
        return 0.0
    return float(series.iloc[-1] / ma_latest - 1.0)


def _volume_trend(volume: pd.Series) -> float:
    if volume is None or volume.empty:
        return 0.0
    vol5 = volume.tail(5).mean()
    vol20 = volume.tail(20).mean()
    if vol20 in (0, np.nan) or pd.isna(vol20):
        return 0.0
    return float(vol5 / vol20 - 1.0)


def _atr_pct(df: pd.DataFrame) -> float:
    if any(column not in df for column in ("High", "Low", "Close")):
        return 0.02
    atr_series = indicators.atr(df["High"], df["Low"], df["Close"], window=14)
    if atr_series.empty:
        return 0.02
    price = df["Close"].iloc[-1]
    if price == 0:
        return 0.02
    return float((atr_series.iloc[-1] or 0.0) / price)


def _premarket_score(dev: float, vol_ratio: float, sentiment: float) -> float:
    s = (1 - sentiment) / 2
    return 0.5 * min(dev, 0.1) + 0.3 * min(vol_ratio / 5, 1) + 0.2 * s


def _trading_day_date(trading_day: Union[date, datetime]) -> date:
    if isinstance(trading_day, datetime):
        return trading_day.date()
    return trading_day


POSITIVE_KEYWORDS = {
    "beat",
    "growth",
    "up",
    "surge",
    "record",
    "upgrade",
    "strong",
    "positive",
    "rally",
    "bullish",
}


NEGATIVE_KEYWORDS = {
    "miss",
    "down",
    "cut",
    "drop",
    "lawsuit",
    "negative",
    "downgrade",
    "weak",
    "bearish",
    "loss",
}


def _news_sentiment_score(articles: List[Dict[str, str]]) -> float:
    if not articles:
        return 0.0

    signal = 0
    total = 0
    for article in articles:
        text = " ".join(
            filter(
                None,
                [
                    str(article.get("title", "")),
                    str(article.get("summary", "")),
                    str(article.get("content", "")),
                ],
            )
        ).lower()
        pos_hits = sum(keyword in text for keyword in POSITIVE_KEYWORDS)
        neg_hits = sum(keyword in text for keyword in NEGATIVE_KEYWORDS)
        if pos_hits or neg_hits:
            signal += pos_hits - neg_hits
            total += pos_hits + neg_hits
    if total == 0:
        return 0.0
    score = signal / max(total, 1)
    return float(np.clip(score, -1.0, 1.0))


def _aggregate_headlines(news_map: Dict[str, List[Dict]], limit: int = 6) -> List[Dict]:
    entries: List[Dict] = []
    for symbol, articles in news_map.items():
        for article in articles:
            entries.append(
                {
                    "symbol": symbol,
                    "title": article.get("title", ""),
                    "summary": article.get("summary", ""),
                    "publisher": article.get("publisher", ""),
                    "published": article.get("published", ""),
                    "link": article.get("link", ""),
                    "content": article.get("content", ""),
                }
            )
    entries.sort(key=lambda item: item.get("published", ""), reverse=True)
    return entries[:limit]


def prepare_feature_sets(
    config: Dict,
    state: PortfolioState,
    yf_client: YahooFinanceClient,
    fred_client: FredClient,
    trading_day: Union[date, datetime],
) -> Tuple[
    Dict,
    Dict[str, Dict],
    Dict[str, Dict],
    Dict[str, Dict],
    Dict[str, Dict],
]:
    day = _trading_day_date(trading_day)
    end = datetime.combine(day, datetime.min.time()) + timedelta(days=1)
    start = end - timedelta(days=250)

    spy_history = _fetch_history(yf_client, "SPY", start, end)
    qqq_history = _fetch_history(yf_client, "QQQ", start, end)
    vix_history = _fetch_history(yf_client, "^VIX", start, end)

    market_symbols = ["SPY", "QQQ"]
    market_news = {
        symbol: yf_client.fetch_news(symbol, lookback_days=7)
        for symbol in market_symbols
    }

    sector_symbols: Iterable[str] = config.get("universe", {}).get("sectors", [])
    sector_data = {
        symbol: _fetch_history(yf_client, symbol, start, end)
        for symbol in sector_symbols
    }
    sector_news = {
        symbol: yf_client.fetch_news(symbol, lookback_days=7)
        for symbol in sector_symbols
    }

    watchlist = set(config.get("universe", {}).get("watchlist", []))
    watchlist.update(position.symbol for position in state.positions)
    sorted_watchlist = sorted(watchlist)
    stock_data = {
        symbol: _fetch_history(yf_client, symbol, start, end)
        for symbol in sorted_watchlist
    }
    stock_news = {
        symbol: yf_client.fetch_news(symbol, lookback_days=7)
        for symbol in sorted_watchlist
    }

    rs_spy = _relative_to_ma(spy_history.get("Close"), 50)
    rs_qqq = _relative_to_ma(qqq_history.get("Close"), 50)

    vix_z = 0.0
    if not vix_history.empty and "Close" in vix_history:
        vix_z = float(indicators.zscore(vix_history["Close"], window=20).iloc[-1])

    put_call_z = 0.0
    start_str = (start.date()).isoformat()
    put_call_df = fred_client.fetch_series("PUTCALL", start=start_str)
    if not put_call_df.empty and "value" in put_call_df:
        put_call_df["value"] = pd.to_numeric(put_call_df["value"], errors="coerce")
        put_call_df = put_call_df.dropna(subset=["value"])
        if not put_call_df.empty:
            put_call_z = float(indicators.zscore(put_call_df["value"], window=30).iloc[-1])

    breadth_samples = []
    for df in sector_data.values():
        if "Close" not in df or df["Close"].empty:
            continue
        above = df["Close"].iloc[-1] > df["Close"].rolling(window=50, min_periods=50).mean().iloc[-1]
        breadth_samples.append(1.0 if above else 0.0)
    breadth = float(sum(breadth_samples) / len(breadth_samples)) if breadth_samples else 0.0

    market_features = {
        "RS_SPY": rs_spy,
        "RS_QQQ": rs_qqq,
        "VIX_Z": vix_z,
        "PUTCALL_Z": put_call_z,
        "BREADTH": breadth,
    }

    spy_close = _latest_close(spy_history)
    sector_features: Dict[str, Dict] = {}
    for symbol, df in sector_data.items():
        close = df.get("Close")
        volume = df.get("Volume")
        momentum5 = _pct_change(close, 5)
        momentum20 = _pct_change(close, 20)
        relative_strength = 0.0
        if spy_close:
            latest = _latest_close(df)
            relative_strength = float((latest / spy_close) - 1.0) if latest else 0.0
        news_articles = sector_news.get(symbol, [])
        sector_features[symbol] = {
            "momentum_5d": momentum5,
            "momentum_20d": momentum20,
            "rs": relative_strength,
            "volume_trend": _volume_trend(volume),
            "news_score": _news_sentiment_score(news_articles),
            "news": news_articles[:5],
        }

    stock_features: Dict[str, Dict] = {}
    premarket_flags: Dict[str, Dict] = {}
    for symbol, df in stock_data.items():
        close = df.get("Close")
        volume = df.get("Volume")
        price = _latest_close(df)
        news_articles = stock_news.get(symbol, [])
        if close is None or close.empty or price == 0:
            stock_features[symbol] = {
                "rsi_norm": 0.5,
                "macd_signal": 0.0,
                "trend_slope": 0.0,
                "volume_score": 0.0,
                "structure_score": 0.0,
                "risk_modifier": 0.0,
                "atr_pct": 0.02,
                "price": 0.0,
                "news_score": 0.0,
                "recent_news": news_articles[:5],
            }
            continue

        rsi_series = indicators.rsi(close, window=14)
        rsi_norm = float((rsi_series.iloc[-1] if not rsi_series.empty else 50) / 100)

        macd_df = indicators.macd(close)
        macd_val = float(macd_df["macd"].iloc[-1] / price) if not macd_df.empty else 0.0

        slope_val = indicators.slope(close, window=10)
        trend_slope = float((slope_val.iloc[-1] if not slope_val.empty else 0.0) / price)

        volume_score = 0.0
        if volume is not None and not volume.empty:
            vol20 = volume.tail(20).mean()
            if vol20:
                volume_score = float(volume.iloc[-1] / vol20 - 1.0)

        ma50 = close.rolling(window=50, min_periods=50).mean()
        ma200 = close.rolling(window=200, min_periods=200).mean()
        structure_components = []
        if not ma50.empty and not math.isclose(float(ma50.iloc[-1]), 0.0):
            structure_components.append(float(price / ma50.iloc[-1] - 1.0))
        if not ma200.empty and not math.isclose(float(ma200.iloc[-1]), 0.0):
            structure_components.append(float(price / ma200.iloc[-1] - 1.0))
        structure_score = float(np.mean(structure_components)) if structure_components else 0.0

        risk_modifier = -0.05 if structure_components and structure_components[0] < 0 else 0.0

        stock_features[symbol] = {
            "rsi_norm": rsi_norm,
            "macd_signal": macd_val,
            "trend_slope": trend_slope,
            "volume_score": volume_score,
            "structure_score": structure_score,
            "risk_modifier": risk_modifier,
            "atr_pct": _atr_pct(df),
            "price": price,
            "news_score": _news_sentiment_score(news_articles),
            "recent_news": news_articles[:5],
        }

        if len(close) > 1:
            prev_close = float(close.iloc[-2])
            dev = abs(price - prev_close) / prev_close if prev_close else 0.0
            vol_ratio = 0.0
            if volume is not None and not volume.empty:
                vol20 = volume.tail(20).mean()
                if vol20:
                    vol_ratio = float(volume.iloc[-1] / vol20)
            sentiment = 0.0
            premarket_flags[symbol] = {
                "dev": dev,
                "vol_ratio": vol_ratio,
                "sentiment": sentiment,
                "score": _premarket_score(dev, vol_ratio, sentiment),
            }

    news_bundle = {
        "market": {
            "symbols": market_symbols,
            "headlines": _aggregate_headlines(market_news),
            "sentiment": _news_sentiment_score(
                [article for articles in market_news.values() for article in articles]
            ),
        },
        "sectors": {
            symbol: {
                "headlines": _aggregate_headlines({symbol: sector_news.get(symbol, [])}, 3),
                "sentiment": _news_sentiment_score(sector_news.get(symbol, [])),
            }
            for symbol in sector_symbols
        },
        "stocks": {
            symbol: {
                "headlines": stock_news.get(symbol, [])[:5],
                "sentiment": _news_sentiment_score(stock_news.get(symbol, [])),
            }
            for symbol in sorted_watchlist
        },
    }

    return market_features, sector_features, stock_features, premarket_flags, news_bundle
