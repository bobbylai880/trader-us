"""Analysis tools for MCP Server."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP


def register_analysis_tools(mcp: FastMCP, config: Dict[str, Any], project_root: Path) -> None:
    """Register analysis MCP tools."""

    cache_dir = project_root / "storage" / "cache" / "yf"
    storage_path = project_root / "storage"

    @mcp.tool()
    def calc_indicators(
        symbol: str,
        indicators: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """计算股票技术指标。

        Args:
            symbol: 股票代码（如 NVDA, AAPL）
            indicators: 要计算的指标列表，默认 ["rsi", "macd", "atr"]
                       可选: rsi, macd, atr, sma_50, sma_200, zscore

        Returns:
            包含各技术指标的字典
        """
        from ai_trader_assist.data_collector.yf_client import YahooFinanceClient
        from ai_trader_assist.feature_engineering import indicators as ind

        if indicators is None:
            indicators = ["rsi", "macd", "atr"]

        client = YahooFinanceClient(cache_dir=cache_dir)
        end = datetime.utcnow()
        start = end - timedelta(days=250)
        df = client.fetch_history(symbol.upper(), start=start, end=end)

        if df.empty:
            return {
                "symbol": symbol.upper(),
                "error": "无法获取历史数据",
            }

        close = df["Close"] if "Close" in df else None
        high = df["High"] if "High" in df else None
        low = df["Low"] if "Low" in df else None

        if close is None or close.empty:
            return {
                "symbol": symbol.upper(),
                "error": "收盘价数据缺失",
            }

        result = {
            "symbol": symbol.upper(),
            "price": float(close.iloc[-1]),
            "date": df.index[-1].isoformat() if len(df) > 0 else None,
        }

        indicators_lower = [i.lower() for i in indicators]

        if "rsi" in indicators_lower:
            rsi_series = ind.rsi(close, window=14)
            if not rsi_series.empty:
                rsi_value = float(rsi_series.iloc[-1])
                result["rsi"] = {
                    "value": round(rsi_value, 2),
                    "interpretation": "超买" if rsi_value > 70 else ("超卖" if rsi_value < 30 else "中性"),
                }

        if "macd" in indicators_lower:
            macd_df = ind.macd(close)
            if not macd_df.empty:
                macd_value = float(macd_df["macd"].iloc[-1])
                signal_value = float(macd_df["signal"].iloc[-1])
                histogram = float(macd_df["histogram"].iloc[-1])
                result["macd"] = {
                    "macd": round(macd_value, 4),
                    "signal": round(signal_value, 4),
                    "histogram": round(histogram, 4),
                    "interpretation": "看多" if histogram > 0 else "看空",
                }

        if "atr" in indicators_lower and high is not None and low is not None:
            atr_series = ind.atr(high, low, close, window=14)
            if not atr_series.empty:
                atr_value = float(atr_series.iloc[-1])
                atr_pct = atr_value / float(close.iloc[-1]) * 100
                result["atr"] = {
                    "value": round(atr_value, 2),
                    "pct": round(atr_pct, 2),
                    "interpretation": "高波动" if atr_pct > 3 else ("低波动" if atr_pct < 1.5 else "正常"),
                }

        if "sma_50" in indicators_lower:
            sma = close.rolling(window=50, min_periods=50).mean()
            if not sma.empty and len(sma.dropna()) > 0:
                sma_value = float(sma.iloc[-1])
                above = float(close.iloc[-1]) > sma_value
                result["sma_50"] = {
                    "value": round(sma_value, 2),
                    "price_above": above,
                    "distance_pct": round((float(close.iloc[-1]) / sma_value - 1) * 100, 2),
                }

        if "sma_200" in indicators_lower:
            sma = close.rolling(window=200, min_periods=200).mean()
            if not sma.empty and len(sma.dropna()) > 0:
                sma_value = float(sma.iloc[-1])
                above = float(close.iloc[-1]) > sma_value
                result["sma_200"] = {
                    "value": round(sma_value, 2),
                    "price_above": above,
                    "distance_pct": round((float(close.iloc[-1]) / sma_value - 1) * 100, 2),
                }

        if "zscore" in indicators_lower:
            zscore_series = ind.zscore(close, window=20)
            if not zscore_series.empty:
                z = float(zscore_series.iloc[-1])
                result["zscore"] = {
                    "value": round(z, 2),
                    "interpretation": "极端高" if z > 2 else ("极端低" if z < -2 else "正常"),
                }

        return result

    @mcp.tool()
    def score_stocks(symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """对股票进行评分。

        Args:
            symbols: 股票代码列表，默认使用配置中的 watchlist

        Returns:
            各股票的评分和操作建议
        """
        from ai_trader_assist.data_collector.yf_client import YahooFinanceClient
        from ai_trader_assist.decision_engine.stock_scoring import StockDecisionEngine
        from ai_trader_assist.feature_engineering import indicators as ind

        if symbols is None:
            symbols = config.get("universe", {}).get("watchlist", [])

        if not symbols:
            return {"error": "未指定股票列表", "scores": []}

        client = YahooFinanceClient(cache_dir=cache_dir)
        end = datetime.utcnow()
        start = end - timedelta(days=250)

        # 构建特征
        stock_features = {}
        for symbol in symbols:
            symbol = symbol.upper()
            df = client.fetch_history(symbol, start=start, end=end)

            if df.empty or "Close" not in df:
                continue

            close = df["Close"].dropna()
            if close.empty:
                continue

            price = float(close.iloc[-1])

            # 计算指标
            rsi_series = ind.rsi(close, window=14)
            rsi_norm = float(rsi_series.iloc[-1] / 100) if not rsi_series.empty else 0.5

            macd_df = ind.macd(close)
            macd_signal = 0.0
            if not macd_df.empty:
                macd_signal = float(macd_df["macd"].iloc[-1] / price) if price > 0 else 0.0

            slope = ind.slope(close, window=10)
            trend_slope = float(slope.iloc[-1] / price) if not slope.empty and price > 0 else 0.0

            # Volume score
            volume = df["Volume"] if "Volume" in df else None
            volume_score = 0.0
            if volume is not None and not volume.empty:
                vol20 = volume.tail(20).mean()
                if vol20 > 0:
                    volume_score = float(volume.iloc[-1] / vol20 - 1)

            # ATR
            atr_pct = 0.02
            if "High" in df and "Low" in df:
                atr_series = ind.atr(df["High"], df["Low"], close, window=14)
                if not atr_series.empty:
                    atr_pct = float(atr_series.iloc[-1] / price) if price > 0 else 0.02

            stock_features[symbol] = {
                "rsi_norm": rsi_norm,
                "macd_signal": macd_signal,
                "trend_slope": trend_slope,
                "volume_score": volume_score,
                "structure_score": 0.0,
                "risk_modifier": 0.0,
                "atr_pct": atr_pct,
                "price": price,
                "news_score": 0.0,
                "trend_strength": 0.5,
                "trend_state": "flat",
                "momentum_10d": 0.0,
                "position_shares": 0,
            }

        # 评分
        engine = StockDecisionEngine()
        scores = engine.score_stocks(stock_features, premarket_flags={})

        return {
            "scores": [
                {
                    "symbol": s["symbol"],
                    "score": round(s["score"], 2),
                    "action": s["action"],
                    "price": round(s["price"], 2),
                    "atr_pct": round(s["atr_pct"] * 100, 2),
                }
                for s in scores
            ],
            "count": len(scores),
            "timestamp": datetime.utcnow().isoformat(),
        }

    @mcp.tool()
    def generate_orders(
        symbol: str,
        action: str,
        budget: Optional[float] = None,
        shares: Optional[int] = None,
    ) -> Dict[str, Any]:
        """生成订单建议（含止损止盈价位）。

        Args:
            symbol: 股票代码
            action: 操作类型（BUY 或 SELL）
            budget: 预算金额（与 shares 二选一）
            shares: 股数（与 budget 二选一）

        Returns:
            订单详情：股数、止损价、目标价、预计金额
        """
        from ai_trader_assist.data_collector.yf_client import YahooFinanceClient
        from ai_trader_assist.feature_engineering import indicators as ind

        if budget is None and shares is None:
            return {"error": "请提供 budget（预算）或 shares（股数）"}

        client = YahooFinanceClient(cache_dir=cache_dir)
        symbol = symbol.upper()
        action = action.upper()

        # 获取当前价格和 ATR
        price = client.latest_price(symbol)
        if price is None:
            return {"error": f"无法获取 {symbol} 的价格"}

        end = datetime.utcnow()
        start = end - timedelta(days=60)
        df = client.fetch_history(symbol, start=start, end=end)

        atr_pct = 0.02
        if not df.empty and "High" in df and "Low" in df and "Close" in df:
            atr_series = ind.atr(df["High"], df["Low"], df["Close"], window=14)
            if not atr_series.empty:
                atr_pct = float(atr_series.iloc[-1] / price) if price > 0 else 0.02

        # 从配置读取止损止盈系数
        sizer_config = config.get("sizer", {})
        k1_stop = sizer_config.get("k1_stop", 1.5)
        k2_target = sizer_config.get("k2_target", 2.5)

        # 计算止损止盈
        atr_value = price * atr_pct
        stop_loss = price - k1_stop * atr_value
        take_profit = price + k2_target * atr_value

        # 计算股数
        if shares is not None:
            calc_shares = shares
            notional = price * shares
        else:
            # 基于预算计算
            calc_shares = int(budget / price)
            notional = price * calc_shares

        # 风险金额（到止损的亏损）
        risk_per_share = price - stop_loss
        total_risk = risk_per_share * calc_shares

        return {
            "symbol": symbol,
            "action": action,
            "price": round(price, 2),
            "shares": calc_shares,
            "notional": round(notional, 2),
            "stop_loss": round(stop_loss, 2),
            "stop_loss_pct": round((1 - stop_loss / price) * 100, 1),
            "take_profit": round(take_profit, 2),
            "take_profit_pct": round((take_profit / price - 1) * 100, 1),
            "risk_amount": round(total_risk, 2),
            "reward_ratio": round(k2_target / k1_stop, 2),
            "atr_pct": round(atr_pct * 100, 2),
        }
