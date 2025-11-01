"""Feature preparation helpers for the daily job."""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta, timezone
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..data_collector.cboe_client import CboeClient
from ..data_collector.fred_client import FredClient
from ..data_collector.yf_client import YahooFinanceClient
from ..portfolio_manager.state import PortfolioState
from ..utils import log_ok, log_result, log_step
from . import indicators
from .trend_features import compute_trend_features


def _is_finite(value: Any) -> bool:
    """Return True if *value* can be represented as a finite float."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert to float while guarding against None/NaN/inf values."""

    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _default_macro_series() -> Dict[str, Dict[str, object]]:
    return {
        "CPIAUCSL": {"label": "CPI YoY"},
        "T10Y2Y": {"label": "10Y-2Y Spread"},
        "FEDFUNDS": {"label": "Fed Funds Rate"},
        "M2SL": {"label": "M2 Money Supply"},
        "UNRATE": {"label": "Unemployment Rate"},
        "INDPRO": {"label": "Industrial Production"},
    }


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
    series = df["Close"].dropna()
    if series.empty:
        return 0.0
    return float(series.iloc[-1])


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
    """Convert premarket signals into a 0-1 risk score."""

    dev_clamped = min(max(dev, 0.0), 0.1)
    vol_clamped = min(max(vol_ratio, 0.0), 5.0)
    sentiment_clamped = max(-1.0, min(1.0, sentiment))

    dev_component = dev_clamped / 0.1  # cap 10% gaps
    vol_component = vol_clamped / 5.0  # cap at 5x typical volume
    sentiment_component = (1 - sentiment_clamped) / 2  # map [-1,1] -> [1,0]

    score = (dev_component * 0.4) + (vol_component * 0.4) + (sentiment_component * 0.2)
    return round(score, 4)


def _structure_score(price: float, moving_averages: Iterable[float]) -> float:
    if not _is_finite(price) or math.isclose(price, 0.0):
        return 0.0

    components: List[float] = []
    for ma in moving_averages:
        if not _is_finite(ma):
            continue
        if math.isclose(float(ma), 0.0):
            continue
        components.append(float(price / float(ma) - 1.0))

    if not components:
        return 0.0

    return float(np.mean(components))


def _compute_vix_metrics(vix_history: pd.DataFrame) -> Dict[str, float]:
    value = 0.0
    zscore = 0.0
    if "Close" in vix_history and not vix_history.empty:
        close = vix_history["Close"].dropna()
        if not close.empty:
            value = float(close.iloc[-1])
            if len(close) >= 20:
                z_values = indicators.zscore(close, window=20)
                if not z_values.empty:
                    zscore = float(z_values.iloc[-1])
    return {"vix_value": value, "vix_zscore": zscore}


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


def _normalize_macro_series(
    series_cfg: Optional[Mapping[str, Any]] | Iterable[str],
) -> Dict[str, Dict[str, object]]:
    if series_cfg is None:
        return {key: dict(meta) for key, meta in _default_macro_series().items()}

    normalized: Dict[str, Dict[str, object]] = {}
    if isinstance(series_cfg, Mapping):
        for series_id, meta in series_cfg.items():
            if isinstance(meta, Mapping):
                normalized[str(series_id)] = {str(k): v for k, v in meta.items()}
            else:
                normalized[str(series_id)] = {"label": str(meta)}
    else:
        for series_id in series_cfg:
            normalized[str(series_id)] = {}

    return normalized or {key: dict(meta) for key, meta in _default_macro_series().items()}


def _normalize_news_payload(articles: List[Dict], limit_chars: int = 400) -> List[Dict]:
    """Trim long news fields before passing to downstream consumers."""

    trimmed: List[Dict] = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        entry = {
            "title": article.get("title", ""),
            "summary": (article.get("summary") or "")[:limit_chars],
            "content": (article.get("content") or "")[:limit_chars],
            "publisher": article.get("publisher", ""),
            "published": article.get("published", ""),
            "link": article.get("link", ""),
        }
        trimmed.append(entry)
    return trimmed


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
    cboe_client: Optional[CboeClient] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[
    Dict,
    Dict[str, Dict],
    Dict[str, Dict],
    Dict[str, Dict],
    Dict[str, Dict],
    Dict[str, Dict],
    Dict[str, Dict],
    Dict[str, Dict],
]:
    """Collect data, engineer features, and emit news/trend bundles."""

    day = _trading_day_date(trading_day)
    end = datetime.combine(day, datetime.min.time()) + timedelta(days=1)
    as_of = datetime.combine(day, datetime.max.time()).replace(tzinfo=timezone.utc)
    start = end - timedelta(days=250)

    market_symbols = ["SPY", "QQQ"]
    sector_symbols: List[str] = list(config.get("universe", {}).get("sectors", []))
    watchlist = set(config.get("universe", {}).get("watchlist", []))
    watchlist.update(position.symbol for position in state.positions)
    sorted_watchlist = sorted(watchlist)

    history_symbols = market_symbols + ["^VIX"] + sector_symbols + sorted_watchlist

    metrics: Dict[str, Dict[str, object]] = {
        "data_collector": {},
        "feature_engineering": {},
    }

    data_start = perf_counter()
    if logger:
        log_step(
            logger,
            "data_collector",
            "Fetching price history and news for %d symbols (lookback=250d, news_window=7d)"
            % len(history_symbols),
        )

    spy_history = _fetch_history(yf_client, "SPY", start, end)
    qqq_history = _fetch_history(yf_client, "QQQ", start, end)
    vix_history = _fetch_history(yf_client, "^VIX", start, end)

    market_history = {"SPY": spy_history, "QQQ": qqq_history}
    market_news = {
        symbol: _normalize_news_payload(
            yf_client.fetch_news(symbol, lookback_days=7, as_of=as_of)
        )
        for symbol in market_symbols
    }

    sector_data = {
        symbol: _fetch_history(yf_client, symbol, start, end)
        for symbol in sector_symbols
    }
    sector_news = {
        symbol: _normalize_news_payload(
            yf_client.fetch_news(symbol, lookback_days=7, as_of=as_of)
        )
        for symbol in sector_symbols
    }

    stock_data = {
        symbol: _fetch_history(yf_client, symbol, start, end)
        for symbol in sorted_watchlist
    }
    stock_news = {
        symbol: _normalize_news_payload(
            yf_client.fetch_news(symbol, lookback_days=7, as_of=as_of)
        )
        for symbol in sorted_watchlist
    }
    premarket_snapshot = yf_client.fetch_premarket_snapshot(sorted_watchlist)

    rs_spy = _relative_to_ma(spy_history.get("Close"), 50)
    rs_qqq = _relative_to_ma(qqq_history.get("Close"), 50)

    vix_metrics = _compute_vix_metrics(vix_history)
    vix_z = float(vix_metrics["vix_zscore"])
    vix_value = float(vix_metrics["vix_value"])

    pcr_total: Optional[float] = None
    pcr_index: Optional[float] = None
    pcr_equity: Optional[float] = None
    pcr_trade_date: Optional[str] = None
    market_data_gaps: List[str] = []
    cboe_stats: Dict[str, object] = {}

    if cboe_client:
        try:
            trade_dt, sections = cboe_client.fetch_put_call_ratios()
        except Exception as exc:  # pragma: no cover - network failure path
            market_data_gaps.append("put/call ratio具体数值缺失")
            if logger:
                log_result(
                    logger,
                    "data_collector",
                    f"cboe: failed to fetch put/call ratios ({exc})",
                )
        else:
            if trade_dt:
                pcr_trade_date = trade_dt.date().isoformat()

            def _extract_ratio(record: Optional[Mapping[str, object]]) -> Optional[float]:
                if not isinstance(record, Mapping):
                    return None
                value = record.get("pc_ratio")
                try:
                    return float(value) if value is not None else None
                except (TypeError, ValueError):
                    return None

            pcr_total = _extract_ratio(sections.get("total"))
            pcr_index = _extract_ratio(sections.get("index"))
            pcr_equity = _extract_ratio(sections.get("equity"))

            if any(ratio is None for ratio in (pcr_total, pcr_index, pcr_equity)):
                market_data_gaps.append("put/call ratio具体数值缺失")

            if logger:
                log_result(
                    logger,
                    "data_collector",
                    "cboe: trade_date=%s, total=%s, index=%s, equity=%s"
                    % (
                        pcr_trade_date or "n/a",
                        "%.2f" % pcr_total if isinstance(pcr_total, (int, float)) else "n/a",
                        "%.2f" % pcr_index if isinstance(pcr_index, (int, float)) else "n/a",
                        "%.2f" % pcr_equity if isinstance(pcr_equity, (int, float)) else "n/a",
                    ),
                )
        finally:
            cboe_stats = cboe_client.snapshot_stats()
    else:
        market_data_gaps.append("put/call ratio具体数值缺失")

    put_call_z = 0.0
    start_str = (start.date()).isoformat()
    put_call_df = fred_client.fetch_series("PUTCALL", start=start_str)
    if not put_call_df.empty and "value" in put_call_df:
        put_call_df["value"] = pd.to_numeric(put_call_df["value"], errors="coerce")
        put_call_df = put_call_df.dropna(subset=["value"])
        if not put_call_df.empty:
            put_call_z = float(indicators.zscore(put_call_df["value"], window=30).iloc[-1])

    breadth_samples = []
    breadth_details: Dict[str, Dict[str, object]] = {}
    for symbol, df in sector_data.items():
        if "Close" not in df or df["Close"].empty:
            continue
        close_series = df["Close"]
        ma50 = (
            close_series.rolling(window=50, min_periods=50).mean().iloc[-1]
            if len(close_series) >= 50
            else None
        )
        above_ma = False
        distance = 0.0
        if ma50 and not math.isclose(float(ma50), 0.0):
            latest_close = float(close_series.iloc[-1])
            ma_val = float(ma50)
            above_ma = latest_close > ma_val
            distance = float(latest_close / ma_val - 1.0)
            breadth_samples.append(1.0 if above_ma else 0.0)
        breadth_details[symbol] = {
            "above_ma50": above_ma,
            "distance_ma50": distance,
            "momentum_5d": _pct_change(close_series, 5),
            "momentum_20d": _pct_change(close_series, 20),
        }
    breadth = float(sum(breadth_samples) / len(breadth_samples)) if breadth_samples else 0.0

    macro_cfg = config.get("macro", {}) if isinstance(config, Mapping) else {}
    macro_series_map = _normalize_macro_series(macro_cfg.get("series"))
    macro_ids = list(macro_series_map.keys())
    macro_lookback = int(macro_cfg.get("lookback_days", 730) or 730)
    macro_start = (day - timedelta(days=macro_lookback)).isoformat()
    macro_raw = fred_client.fetch_macro_indicators(
        macro_ids,
        start_date=macro_start,
        logger=logger,
    )
    macro_flags: Dict[str, Dict[str, object]] = {}
    for series_id, data in macro_raw.items():
        entry = dict(macro_series_map.get(series_id, {}))
        entry.update(data)
        macro_flags[series_id] = entry

    yf_stats = yf_client.snapshot_stats()
    fred_stats = fred_client.snapshot_stats()
    data_duration = perf_counter() - data_start

    history_stats = yf_stats.get("history", {})
    news_stats = yf_stats.get("news", {})

    metrics["data_collector"] = {
        "history_requests": history_stats.get("requests", 0),
        "history_cache_hits": history_stats.get("cache_hits", 0),
        "history_rows": history_stats.get("rows", 0),
        "history_symbols": history_stats.get("symbol_count", 0),
        "history_synthetic_fallbacks": history_stats.get("synthetic_fallbacks", 0),
        "news_requests": news_stats.get("requests", 0),
        "news_cache_hits": news_stats.get("cache_hits", 0),
        "news_articles": news_stats.get("articles", 0),
        "news_symbols": news_stats.get("symbol_count", 0),
        "news_content_downloads": news_stats.get("content_downloads", 0),
        "news_content_requests": news_stats.get("content_requests", 0),
        "news_content_failures": news_stats.get("content_failures", 0),
        "news_synthetic": news_stats.get("synthetic_articles", 0),
        "fred_series": fred_stats.get("series_count", 0),
        "fred_cache_hits": fred_stats.get("cache_hits", 0),
        "cboe_requests": cboe_stats.get("requests", 0),
        "cboe_success": cboe_stats.get("success", 0),
        "cboe_failures": cboe_stats.get("failures", 0),
        "duration": data_duration,
    }

    if logger:
        history_requests = history_stats.get("requests", 0)
        history_hits = history_stats.get("cache_hits", 0)
        history_rate = (history_hits / history_requests) if history_requests else 0.0
        log_result(
            logger,
            "data_collector",
            "history: symbols=%d, rows=%d, cache_hit=%d/%d (%.0f%%), synthetic=%d"
            % (
                history_stats.get("symbol_count", 0),
                history_stats.get("rows", 0),
                history_hits,
                history_requests,
                history_rate * 100,
                history_stats.get("synthetic_fallbacks", 0),
            ),
        )

        news_requests = news_stats.get("requests", 0)
        news_hits = news_stats.get("cache_hits", 0)
        news_rate = (news_hits / news_requests) if news_requests else 0.0
        content_requests = news_stats.get("content_requests", 0)
        content_downloads = news_stats.get("content_downloads", 0)
        log_result(
            logger,
            "data_collector",
            "news: symbols=%d, articles=%d, cache_hit=%d/%d (%.0f%%), content=%d/%d, synthetic=%d"
            % (
                news_stats.get("symbol_count", 0),
                news_stats.get("articles", 0),
                news_hits,
                news_requests,
                news_rate * 100,
                content_downloads,
                content_requests,
                news_stats.get("synthetic_articles", 0),
            ),
        )
        if fred_stats.get("series_count", 0):
            log_result(
                logger,
                "data_collector",
                "fred: series=%d, cache_hits=%d"
                % (
                    fred_stats.get("series_count", 0),
                    fred_stats.get("cache_hits", 0),
                ),
            )
        if cboe_stats.get("requests"):
            log_result(
                logger,
                "data_collector",
                "cboe: requests=%d, success=%d, failures=%d"
                % (
                    cboe_stats.get("requests", 0),
                    cboe_stats.get("success", 0),
                    cboe_stats.get("failures", 0),
                ),
            )
        log_ok(logger, "data_collector", f"Completed in {data_duration:.2f}s")

    if logger:
        log_step(
            logger,
            "feature_engineering",
            "Calculating indicators and trend features (stocks=%d, sectors=%d)"
            % (len(sorted_watchlist), len(sector_symbols)),
        )
    feature_start = perf_counter()

    trend_config = config.get("trend")

    market_trends = compute_trend_features(market_history, config=trend_config)
    sector_trends = compute_trend_features(sector_data, config=trend_config)
    stock_trends = compute_trend_features(stock_data, config=trend_config)

    market_data_gaps = list(dict.fromkeys(market_data_gaps))

    market_features = {
        "RS_SPY": rs_spy,
        "RS_QQQ": rs_qqq,
        "VIX_Z": vix_z,
        "PUTCALL_Z": put_call_z,
        "BREADTH": breadth,
        "trend": market_trends,
        "vix_value": vix_value,
        "vix_zscore": vix_z,
        "breadth_details": breadth_details,
        "pcr_total": pcr_total,
        "pcr_index": pcr_index,
        "pcr_equity": pcr_equity,
        "pcr_trade_date": pcr_trade_date,
        "data_gaps": market_data_gaps,
    }
    if macro_flags:
        market_features["macro_flags"] = macro_flags

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
        trend_meta = sector_trends.get(symbol, {})
        sector_features[symbol] = {
            "momentum_5d": momentum5,
            "momentum_20d": momentum20,
            "rs": relative_strength,
            "volume_trend": _volume_trend(volume),
            "news_score": _news_sentiment_score(news_articles),
            "news": news_articles[:3],
            "trend_slope_5d": float(trend_meta.get("trend_slope_5d", 0.0)),
            "trend_slope_20d": float(trend_meta.get("trend_slope_20d", 0.0)),
            "momentum_10d": float(trend_meta.get("momentum_10d", 0.0)),
            "volatility_trend": float(trend_meta.get("volatility_trend", 1.0)),
            "moving_avg_cross": int(trend_meta.get("moving_avg_cross", 0)),
            "trend_strength": float(trend_meta.get("trend_strength", 0.0)),
            "trend_state": trend_meta.get("trend_state", "flat"),
            "momentum_state": trend_meta.get("momentum_state", "stable"),
        }

    stock_features: Dict[str, Dict] = {}
    premarket_flags: Dict[str, Dict] = {}
    for symbol, df in stock_data.items():
        raw_close = df.get("Close")
        raw_volume = df.get("Volume")
        close = raw_close.dropna() if isinstance(raw_close, pd.Series) else pd.Series(dtype=float)
        volume = raw_volume.dropna() if isinstance(raw_volume, pd.Series) else pd.Series(dtype=float)
        price = _latest_close(df)
        news_articles = stock_news.get(symbol, [])
        news_sentiment = _news_sentiment_score(news_articles)
        trend_meta = stock_trends.get(symbol, {})
        position = state.get_position(symbol)
        held_shares = float(position.shares) if position else 0.0
        held_value = float(position.market_value) if position else 0.0
        premarket_meta = premarket_snapshot.get(symbol, {}) or {}
        premarket_price = _safe_float(premarket_meta.get("price"))
        premarket_volume = _safe_float(premarket_meta.get("volume"))
        premarket_prev_close = _safe_float(premarket_meta.get("prev_close"))
        premarket_change_pct = _safe_float(premarket_meta.get("change_pct"))
        premarket_timestamp = premarket_meta.get("timestamp")
        if close.empty or not _is_finite(price) or math.isclose(price, 0.0):
            dev = 0.0
            basis_close = 0.0
            if _is_finite(premarket_price) and _is_finite(premarket_prev_close) and not math.isclose(premarket_prev_close, 0.0):
                basis_close = premarket_prev_close
                dev = abs(premarket_price - basis_close) / basis_close
            vol_ratio = 0.0
            premarket_flags[symbol] = {
                "dev": dev,
                "vol_ratio": vol_ratio,
                "sentiment": news_sentiment,
                "score": _premarket_score(dev, vol_ratio, news_sentiment),
                "premarket_price": premarket_price,
                "prev_close": basis_close,
                "change_pct": premarket_change_pct,
                "volume": premarket_volume,
                "timestamp": premarket_timestamp,
            }
            stock_features[symbol] = {
                "rsi_norm": 0.5,
                "macd_signal": 0.0,
                "trend_slope": 0.0,
                "volume_score": 0.0,
                "structure_score": 0.0,
                "risk_modifier": 0.0,
                "atr_pct": 0.02,
                "price": 0.0,
                "news_score": news_sentiment,
                "recent_news": news_articles[:3],
                "trend_slope_5d": float(trend_meta.get("trend_slope_5d", 0.0)),
                "trend_slope_20d": float(trend_meta.get("trend_slope_20d", 0.0)),
                "momentum_10d": float(trend_meta.get("momentum_10d", 0.0)),
                "volatility_trend": float(trend_meta.get("volatility_trend", 1.0)),
                "moving_avg_cross": int(trend_meta.get("moving_avg_cross", 0)),
                "trend_strength": float(trend_meta.get("trend_strength", 0.0)),
                "trend_state": trend_meta.get("trend_state", "flat"),
                "momentum_state": trend_meta.get("momentum_state", "stable"),
                "position_shares": held_shares,
                "position_value": held_value,
                "premarket_price": premarket_price,
                "premarket_change_pct": premarket_change_pct,
                "premarket_timestamp": premarket_timestamp,
            }
            continue

        rsi_series = indicators.rsi(close, window=14)
        rsi_norm = float((rsi_series.iloc[-1] if not rsi_series.empty else 50) / 100)

        macd_df = indicators.macd(close)
        macd_val = float(macd_df["macd"].iloc[-1] / price) if not macd_df.empty else 0.0

        slope_val = indicators.slope(close, window=10)
        trend_slope = float((slope_val.iloc[-1] if not slope_val.empty else 0.0) / price)

        volume_score = 0.0
        if not volume.empty:
            vol20_series = volume.tail(20)
            vol20 = _safe_float(vol20_series.mean())
            latest_volume = _safe_float(volume.iloc[-1])
            if vol20 > 0 and latest_volume > 0:
                volume_score = float(latest_volume / vol20 - 1.0)

        ma50 = close.rolling(window=50, min_periods=50).mean()
        ma200 = close.rolling(window=200, min_periods=200).mean()
        ma50_latest = float(ma50.iloc[-1]) if not ma50.empty else float("nan")
        ma200_latest = float(ma200.iloc[-1]) if not ma200.empty else float("nan")
        structure_score = _structure_score(price, [ma50_latest, ma200_latest])

        structure_50d = 0.0
        if _is_finite(ma50_latest) and not math.isclose(ma50_latest, 0.0):
            structure_50d = float(price / ma50_latest - 1.0)

        risk_modifier = -0.05 if structure_50d < 0 else 0.0

        stock_features[symbol] = {
            "rsi_norm": rsi_norm,
            "macd_signal": macd_val,
            "trend_slope": trend_slope,
            "volume_score": volume_score,
            "structure_score": structure_score,
            "risk_modifier": risk_modifier,
            "atr_pct": _atr_pct(df),
            "price": price,
            "news_score": news_sentiment,
            "recent_news": news_articles[:3],
            "trend_slope_5d": float(trend_meta.get("trend_slope_5d", 0.0)),
            "trend_slope_20d": float(trend_meta.get("trend_slope_20d", 0.0)),
            "momentum_10d": float(trend_meta.get("momentum_10d", 0.0)),
            "volatility_trend": float(trend_meta.get("volatility_trend", 1.0)),
            "moving_avg_cross": int(trend_meta.get("moving_avg_cross", 0)),
            "trend_strength": float(trend_meta.get("trend_strength", 0.0)),
            "trend_state": trend_meta.get("trend_state", "flat"),
            "momentum_state": trend_meta.get("momentum_state", "stable"),
            "position_shares": held_shares,
            "position_value": held_value,
            "premarket_price": premarket_price,
            "premarket_change_pct": premarket_change_pct,
            "premarket_timestamp": premarket_timestamp,
        }

        dev = 0.0
        basis_close = 0.0
        if _is_finite(premarket_price) and _is_finite(premarket_prev_close) and not math.isclose(premarket_prev_close, 0.0):
            basis_close = premarket_prev_close
            dev = abs(premarket_price - basis_close) / basis_close
        elif len(close) > 1:
            prev_close = _safe_float(close.iloc[-2])
            if prev_close > 0:
                basis_close = prev_close
                dev = abs(price - basis_close) / basis_close

        vol_ratio = 0.0
        vol20 = 0.0
        if not volume.empty:
            vol20_series = volume.tail(20) if len(volume) >= 20 else volume
            vol20 = _safe_float(vol20_series.mean())
        if premarket_volume > 0 and vol20 > 0:
            baseline = max(vol20 / 10.0, 1.0)
            vol_ratio = float(premarket_volume / baseline)
        elif not volume.empty and vol20 > 0:
            latest_volume = _safe_float(volume.iloc[-1])
            if latest_volume > 0:
                vol_ratio = float(latest_volume / vol20)

        sentiment = news_sentiment
        premarket_flags[symbol] = {
            "dev": dev,
            "vol_ratio": vol_ratio,
            "sentiment": sentiment,
            "score": _premarket_score(dev, vol_ratio, sentiment),
            "premarket_price": premarket_price,
            "prev_close": basis_close,
            "change_pct": premarket_change_pct,
            "volume": premarket_volume,
            "timestamp": premarket_timestamp,
        }

    feature_duration = perf_counter() - feature_start

    stock_trend_states = [meta.get("trend_state", "flat") for meta in stock_trends.values()]
    up_count = sum(1 for state in stock_trend_states if state == "uptrend")
    down_count = sum(1 for state in stock_trend_states if state == "downtrend")
    flat_count = len(stock_trend_states) - up_count - down_count
    avg_momentum = float(
        np.mean([meta.get("momentum_10d", 0.0) for meta in stock_trends.values()])
    ) if stock_trends else 0.0

    metrics["feature_engineering"] = {
        "market_fields": len(market_features),
        "sector_symbols": len(sector_features),
        "stock_symbols": len(stock_features),
        "trend_states": {
            "uptrend": up_count,
            "downtrend": down_count,
            "flat": flat_count,
        },
        "avg_momentum_10d": avg_momentum,
        "duration": feature_duration,
    }

    if logger:
        log_result(
            logger,
            "feature_engineering",
            "features: market_fields=%d, sectors=%d, stocks=%d"
            % (len(market_features), len(sector_features), len(stock_features)),
        )
        log_result(
            logger,
            "feature_engineering",
            "stock_trend_states: up=%d, down=%d, flat=%d, avg_momentum_10d=%.1f%%"
            % (up_count, down_count, flat_count, avg_momentum * 100),
        )
        log_ok(logger, "feature_engineering", f"Completed in {feature_duration:.2f}s")

    news_bundle = {
        "market": {
            "symbols": market_symbols,
            "headlines": _aggregate_headlines(market_news, limit=4),
            "sentiment": _news_sentiment_score(
                [article for articles in market_news.values() for article in articles]
            ),
        },
        "sectors": {
            symbol: {
                "headlines": _aggregate_headlines({symbol: sector_news.get(symbol, [])}, 2),
                "sentiment": _news_sentiment_score(sector_news.get(symbol, [])),
            }
            for symbol in sector_symbols
        },
        "stocks": {
            symbol: {
                "headlines": stock_news.get(symbol, [])[:3],
                "sentiment": _news_sentiment_score(stock_news.get(symbol, [])),
            }
            for symbol in sorted_watchlist
        },
    }

    trend_bundle = {
        "market": market_trends,
        "sectors": sector_trends,
        "stocks": stock_trends,
    }

    return (
        market_features,
        sector_features,
        stock_features,
        premarket_flags,
        news_bundle,
        trend_bundle,
        macro_flags,
        metrics,
    )
