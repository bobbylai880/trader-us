"""Price data tools for MCP Server."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP


def register_price_tools(mcp: FastMCP, config: Dict[str, Any], project_root: Path) -> None:
    """Register price-related MCP tools."""

    cache_dir = project_root / "storage" / "cache" / "yf"

    @mcp.tool()
    def get_price(symbol: str) -> Dict[str, Any]:
        """获取股票最新价格。

        Args:
            symbol: 股票代码（如 NVDA, AAPL, SPY）

        Returns:
            包含价格信息的字典：price, change, change_pct, volume, timestamp
        """
        from ai_trader_assist.data_collector.yf_client import YahooFinanceClient

        client = YahooFinanceClient(cache_dir=cache_dir)
        price = client.latest_price(symbol.upper())

        if price is None:
            return {
                "symbol": symbol.upper(),
                "price": None,
                "error": "无法获取价格数据",
            }

        return {
            "symbol": symbol.upper(),
            "price": price,
            "timestamp": datetime.utcnow().isoformat(),
        }

    @mcp.tool()
    def get_history(
        symbol: str,
        days: int = 60,
        interval: str = "1d",
    ) -> Dict[str, Any]:
        """获取股票历史行情数据。

        Args:
            symbol: 股票代码（如 NVDA, AAPL, SPY）
            days: 回溯天数，默认 60 天
            interval: 数据间隔，默认 "1d"（日线）

        Returns:
            包含历史数据的字典：symbol, interval, data_points, latest, earliest, summary
        """
        from ai_trader_assist.data_collector.yf_client import YahooFinanceClient

        client = YahooFinanceClient(cache_dir=cache_dir)
        end = datetime.utcnow()
        start = end - timedelta(days=days)

        df = client.fetch_history(symbol.upper(), start=start, end=end, interval=interval)

        if df.empty:
            return {
                "symbol": symbol.upper(),
                "interval": interval,
                "data_points": 0,
                "error": "无法获取历史数据",
            }

        # 计算摘要统计
        close = df["Close"] if "Close" in df else None
        summary = {}
        if close is not None and not close.empty:
            summary = {
                "latest_close": float(close.iloc[-1]),
                "earliest_close": float(close.iloc[0]),
                "high": float(df["High"].max()) if "High" in df else None,
                "low": float(df["Low"].min()) if "Low" in df else None,
                "avg_volume": float(df["Volume"].mean()) if "Volume" in df else None,
                "price_change": float(close.iloc[-1] - close.iloc[0]),
                "price_change_pct": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
            }

        return {
            "symbol": symbol.upper(),
            "interval": interval,
            "data_points": len(df),
            "date_range": {
                "start": df.index[0].isoformat() if len(df) > 0 else None,
                "end": df.index[-1].isoformat() if len(df) > 0 else None,
            },
            "summary": summary,
        }

    @mcp.tool()
    def get_quotes(symbols: List[str]) -> Dict[str, Any]:
        """获取多只股票的盘前/盘后报价。

        Args:
            symbols: 股票代码列表（如 ["NVDA", "AAPL", "MSFT"]）

        Returns:
            包含各股票报价的字典
        """
        from ai_trader_assist.data_collector.yf_client import YahooFinanceClient

        client = YahooFinanceClient(cache_dir=cache_dir)
        symbols_upper = [s.upper() for s in symbols]

        snapshot = client.fetch_premarket_snapshot(symbols_upper)

        result = {}
        for symbol, data in snapshot.items():
            result[symbol] = {
                "premarket_price": data.get("price"),
                "prev_close": data.get("prev_close"),
                "change": data.get("change"),
                "change_pct": data.get("change_pct"),
                "volume": data.get("volume"),
                "timestamp": data.get("timestamp"),
            }

        return {
            "quotes": result,
            "count": len(result),
            "timestamp": datetime.utcnow().isoformat(),
        }
