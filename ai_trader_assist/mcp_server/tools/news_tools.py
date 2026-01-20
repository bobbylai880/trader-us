"""News and macro data tools for MCP Server."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP


def register_news_tools(mcp: FastMCP, config: Dict[str, Any], project_root: Path) -> None:
    """Register news and macro data MCP tools."""

    cache_dir = project_root / "storage" / "cache"
    yf_cache = cache_dir / "yf"
    fred_cache = cache_dir / "fred"

    @mcp.tool()
    def get_news(
        symbol: str,
        max_items: int = 5,
        lookback_days: int = 7,
    ) -> Dict[str, Any]:
        """获取股票相关新闻。

        Args:
            symbol: 股票代码（如 NVDA, AAPL, SPY）
            max_items: 最大新闻条数，默认 5
            lookback_days: 回溯天数，默认 7

        Returns:
            包含新闻列表和情绪分数的字典
        """
        from ai_trader_assist.data_collector.yf_client import YahooFinanceClient

        client = YahooFinanceClient(cache_dir=yf_cache)
        articles = client.fetch_news(
            symbol.upper(),
            max_items=max_items,
            lookback_days=lookback_days,
        )

        # 计算简单情绪分数
        positive_keywords = {"beat", "growth", "up", "surge", "record", "upgrade", "strong", "positive", "rally", "bullish"}
        negative_keywords = {"miss", "down", "cut", "drop", "lawsuit", "negative", "downgrade", "weak", "bearish", "loss"}

        sentiment_score = 0.0
        total_hits = 0
        for article in articles:
            text = " ".join([
                str(article.get("title", "")),
                str(article.get("summary", "")),
            ]).lower()
            pos_hits = sum(1 for kw in positive_keywords if kw in text)
            neg_hits = sum(1 for kw in negative_keywords if kw in text)
            if pos_hits or neg_hits:
                sentiment_score += pos_hits - neg_hits
                total_hits += pos_hits + neg_hits

        normalized_sentiment = sentiment_score / max(total_hits, 1) if total_hits > 0 else 0.0

        return {
            "symbol": symbol.upper(),
            "news": [
                {
                    "title": a.get("title", ""),
                    "summary": a.get("summary", "")[:200],
                    "publisher": a.get("publisher", ""),
                    "published": a.get("published", ""),
                }
                for a in articles
            ],
            "count": len(articles),
            "sentiment_score": round(normalized_sentiment, 2),
            "sentiment_label": "positive" if normalized_sentiment > 0.2 else ("negative" if normalized_sentiment < -0.2 else "neutral"),
        }

    @mcp.tool()
    def get_macro() -> Dict[str, Any]:
        """获取宏观经济指标。

        Returns:
            包含各宏观指标的字典：CPI, 利率, 失业率, VIX 等
        """
        from ai_trader_assist.data_collector.fred_client import FredClient
        from ai_trader_assist.data_collector.yf_client import YahooFinanceClient

        fred_api_key = os.getenv("FRED_API_KEY")
        fred_client = FredClient(api_key=fred_api_key, cache_dir=fred_cache)

        # 获取 FRED 指标
        macro_series = config.get("macro", {}).get("series", {})
        series_ids = list(macro_series.keys()) if macro_series else [
            "CPIAUCSL", "T10Y2Y", "FEDFUNDS", "UNRATE"
        ]

        lookback_days = config.get("macro", {}).get("lookback_days", 730)
        start_date = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        indicators = fred_client.fetch_macro_indicators(series_ids, start_date=start_date)

        # 获取 VIX
        yf_client = YahooFinanceClient(cache_dir=yf_cache)
        end = datetime.utcnow()
        start = end - timedelta(days=60)
        vix_history = yf_client.fetch_history("^VIX", start=start, end=end)

        vix_data = {}
        if not vix_history.empty and "Close" in vix_history:
            close = vix_history["Close"].dropna()
            if not close.empty:
                vix_value = float(close.iloc[-1])
                vix_mean = float(close.mean())
                vix_std = float(close.std())
                vix_zscore = (vix_value - vix_mean) / vix_std if vix_std > 0 else 0.0
                vix_data = {
                    "value": round(vix_value, 2),
                    "zscore": round(vix_zscore, 2),
                    "mean_60d": round(vix_mean, 2),
                    "interpretation": "高恐慌" if vix_zscore > 1.5 else ("低恐慌" if vix_zscore < -1.0 else "正常"),
                }

        return {
            "indicators": indicators,
            "vix": vix_data,
            "timestamp": datetime.utcnow().isoformat(),
        }

    @mcp.tool()
    def get_pcr() -> Dict[str, Any]:
        """获取 Put/Call Ratio（看跌/看涨比率）。

        Returns:
            包含 PCR 数据的字典：total, index, equity 比率
        """
        from ai_trader_assist.data_collector.cboe_client import CboeClient

        cboe_client = CboeClient(cache_dir=cache_dir / "cboe")

        try:
            trade_date, sections = cboe_client.fetch_put_call_ratios()
        except Exception as e:
            return {
                "error": f"无法获取 PCR 数据: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
            }

        def extract_ratio(record):
            if not record:
                return None
            ratio = record.get("pc_ratio")
            return float(ratio) if ratio is not None else None

        pcr_total = extract_ratio(sections.get("total"))
        pcr_index = extract_ratio(sections.get("index"))
        pcr_equity = extract_ratio(sections.get("equity"))

        # PCR 解释
        interpretation = "中性"
        if pcr_total:
            if pcr_total > 1.2:
                interpretation = "极度恐慌（可能超卖）"
            elif pcr_total > 1.0:
                interpretation = "偏空"
            elif pcr_total < 0.7:
                interpretation = "极度乐观（可能超买）"
            elif pcr_total < 0.85:
                interpretation = "偏多"

        return {
            "trade_date": trade_date.isoformat() if trade_date else None,
            "total": pcr_total,
            "index": pcr_index,
            "equity": pcr_equity,
            "interpretation": interpretation,
            "timestamp": datetime.utcnow().isoformat(),
        }
