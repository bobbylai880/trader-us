"""V8.2 数据加载器 - PostgreSQL 市场数据访问层"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd
import psycopg2


class DataLoader:
    """PostgreSQL 数据加载器，提供价格和指标数据访问"""
    
    def __init__(self):
        self._conn: Optional[psycopg2.extensions.connection] = None
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def _get_conn(self) -> psycopg2.extensions.connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=os.getenv("PG_HOST", "192.168.10.11"),
                port=int(os.getenv("PG_PORT", "5432")),
                database=os.getenv("PG_DATABASE", "trader"),
                user=os.getenv("PG_USER", "trader"),
                password=os.getenv("PG_PASSWORD", ""),
            )
        return self._conn
    
    def load_prices(
        self,
        symbols: List[str],
        start: date,
        end: date,
        lookback: int = 100,
    ) -> Dict[str, pd.DataFrame]:
        """加载价格数据并计算常用指标
        
        Returns:
            Dict[symbol -> DataFrame with columns: close, sma5, sma20, sma50, mom5, mom20]
        """
        query = """
            SELECT symbol, trade_date, close
            FROM daily_prices
            WHERE trade_date BETWEEN %s AND %s
              AND symbol IN %s
            ORDER BY symbol, trade_date
        """
        actual_start = start - timedelta(days=lookback)
        conn = self._get_conn()
        df = pd.read_sql(query, conn, params=(actual_start, end, tuple(symbols)))
        
        result = {}
        for sym in df["symbol"].unique():
            sdf = df[df["symbol"] == sym].copy()
            sdf.set_index("trade_date", inplace=True)
            sdf["sma5"] = sdf["close"].rolling(5).mean()
            sdf["sma20"] = sdf["close"].rolling(20).mean()
            sdf["sma50"] = sdf["close"].rolling(50).mean()
            sdf["mom5"] = sdf["close"].pct_change(5)
            sdf["mom20"] = sdf["close"].pct_change(20)
            result[sym] = sdf
        
        self._cache = result
        return result
    
    def get(self, symbol: str, dt: date, col: str) -> Optional[float]:
        """获取指定日期的数据值"""
        if symbol not in self._cache:
            return None
        df = self._cache[symbol]
        valid = df[df.index <= dt]
        if len(valid) == 0:
            return None
        val = valid[col].iloc[-1]
        return float(val) if pd.notna(val) else None
    
    def get_history(self, symbol: str, end_date: date, lookback: int = 200) -> pd.Series:
        """获取历史收盘价序列（用于宏观指标计算）"""
        if symbol in self._cache:
            series = self._cache[symbol]["close"]
            return series[series.index <= end_date].tail(lookback)
        
        # 从数据库加载
        query = "SELECT trade_date, close FROM daily_prices WHERE symbol = %s ORDER BY trade_date"
        conn = self._get_conn()
        df = pd.read_sql(query, conn, params=(symbol,))
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        df.set_index("trade_date", inplace=True)
        series = df["close"]
        return series[series.index <= end_date].tail(lookback)
    
    def get_trading_days(self, start: date, end: date) -> List[date]:
        """获取交易日列表"""
        if "SPY" in self._cache:
            df = self._cache["SPY"]
            days = [d for d in df.index if start <= d <= end]
            return sorted(days)
        return []
    
    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
