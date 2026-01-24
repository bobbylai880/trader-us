"""V8.2 宏观主题生成器 - 完全对齐原版 db_theme_generator.py"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import pandas as pd
import psycopg2


def get_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "192.168.10.11"),
        port=os.getenv("PG_PORT", "5432"),
        database=os.getenv("PG_DATABASE", "trader"),
        user=os.getenv("PG_USER", "trader"),
        password=os.getenv("PG_PASSWORD", ""),
    )


@dataclass
class ThemeConfig:
    """主题配置"""
    focus_sectors: List[str] = field(default_factory=list)
    focus_stocks: List[str] = field(default_factory=list)
    avoid_sectors: List[str] = field(default_factory=list)
    avoid_stocks: List[str] = field(default_factory=list)
    sector_bonus: Dict[str, float] = field(default_factory=dict)
    stock_bonus: Dict[str, float] = field(default_factory=dict)
    risk_level: str = "normal"
    theme_drivers: List[str] = field(default_factory=list)


class MacroTheme:
    """V8.2 客观宏观主题生成器 - 完全对齐原版
    
    基于真实市场数据计算主题信号，权重配置：
    - 动量 (30%): 个股相对强度
    - 风险偏好 (20%): HYG/TLT (高收益债/国债)
    - 经济前景 (15%): XLY/XLP (非必需/必需消费)
    - 恐慌指数 (15%): ^VIX
    - 长期利率 (10%): ^TNX (10年美债)
    - 美元汇率 (5%): UUP
    - 短期利率 (5%): ^IRX
    """
    
    WEIGHTS = {
        "momentum": 0.30,
        "risk_on": 0.20,
        "economic": 0.15,
        "vix": 0.15,
        "yields": 0.10,
        "dollar": 0.05,
        "rates": 0.05,
    }
    
    SECTOR_STOCKS = {
        "XLK": ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMD", "AVGO"],
        "XLF": ["JPM", "GS", "BAC", "WFC", "MS"],
        "XLE": ["XOM", "CVX", "COP"],
        "XLV": ["UNH", "LLY", "JNJ", "PFE"],
        "XLI": ["CAT", "HON", "BA", "GE"],
        "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
        "XLC": ["META", "GOOGL", "NFLX"],
    }
    
    def __init__(self, universe: Optional[List[str]] = None):
        self.universe = universe or [
            "NVDA", "META", "GOOGL", "AMZN", "MSFT", "AAPL",
            "AMD", "AVGO", "NFLX", "TSLA",
        ]
        self._conn = None
        self._cache: Dict[str, ThemeConfig] = {}
        self._macro_data: Dict[str, pd.Series] = {}
    
    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = get_connection()
        return self._conn
    
    def _fetch_history(self, symbol: str, end_date: date, lookback: int = 200) -> pd.Series:
        """获取单个指标的历史收盘价"""
        if symbol not in self._macro_data:
            conn = self._get_conn()
            query = "SELECT trade_date, close FROM daily_prices WHERE symbol = %s ORDER BY trade_date"
            df = pd.read_sql(query, conn, params=(symbol,))
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            df.set_index("trade_date", inplace=True)
            self._macro_data[symbol] = df["close"]
        
        series = self._macro_data[symbol]
        return series[series.index <= end_date].tail(lookback)
    
    def generate_theme(
        self,
        as_of: date,
        momentum_scores: Optional[Dict[str, float]] = None,
    ) -> ThemeConfig:
        """生成主题配置 - 方法名对齐原版"""
        cache_key = str(as_of)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if momentum_scores is None:
            momentum_scores = {}
        
        # 1. 计算宏观信号 (-1.0 到 1.0)
        risk_signal = self._calc_risk_appetite(as_of)
        econ_signal = self._calc_economic_outlook(as_of)
        vix_signal = self._calc_vix_signal(as_of)
        yield_signal = self._calc_yield_signal(as_of)
        dollar_signal = self._calc_dollar_signal(as_of)
        rate_signal = self._calc_short_rate_signal(as_of)
        
        # 2. 综合评分
        stock_scores = {}
        for symbol in self.universe:
            macro_score = (
                risk_signal * self.WEIGHTS["risk_on"] +
                econ_signal * self.WEIGHTS["economic"] +
                vix_signal * self.WEIGHTS["vix"] +
                yield_signal * self.WEIGHTS["yields"] +
                dollar_signal * self.WEIGHTS["dollar"] +
                rate_signal * self.WEIGHTS["rates"]
            )
            mom_score = momentum_scores.get(symbol, 0) * self.WEIGHTS["momentum"]
            total = max(-1.0, min(1.0, macro_score + mom_score))
            stock_scores[symbol] = total
        
        # 3. 生成配置
        theme = self._build_theme(stock_scores, vix_signal, risk_signal)
        self._cache[cache_key] = theme
        return theme
    
    def _calc_trend_score(self, series: pd.Series) -> float:
        """计算趋势得分: 当前价相对于 MA50 的位置"""
        if len(series) < 50:
            return 0.0
        current = float(series.iloc[-1])
        ma50 = float(series.rolling(50).mean().iloc[-1])
        if ma50 == 0:
            return 0.0
        return (current / ma50) - 1.0
    
    def _calc_risk_appetite(self, as_of: date) -> float:
        """HYG/TLT 比率: 上升代表风险偏好增强"""
        hyg = self._fetch_history("HYG", as_of)
        tlt = self._fetch_history("TLT", as_of)
        if len(hyg) < 50 or len(tlt) < 50:
            return 0.0
        ratio = hyg / tlt
        score = self._calc_trend_score(ratio)
        return max(-1.0, min(1.0, score * 10))
    
    def _calc_economic_outlook(self, as_of: date) -> float:
        """XLY/XLP 比率: 上升代表经济前景乐观"""
        xly = self._fetch_history("XLY", as_of)
        xlp = self._fetch_history("XLP", as_of)
        if len(xly) < 50 or len(xlp) < 50:
            return 0.0
        ratio = xly / xlp
        score = self._calc_trend_score(ratio)
        return max(-1.0, min(1.0, score * 5))
    
    def _calc_vix_signal(self, as_of: date) -> float:
        """VIX: 低于20看多，高于30看空"""
        vix = self._fetch_history("^VIX", as_of, 10)
        if len(vix) == 0:
            return 0.0
        curr = float(vix.iloc[-1])
        if curr < 15:
            return 0.5
        if curr < 20:
            return 0.2
        if curr > 30:
            return -0.8
        if curr > 25:
            return -0.4
        return 0.0
    
    def _calc_yield_signal(self, as_of: date) -> float:
        """^TNX (10年美债): 急升对科技股利空"""
        tnx = self._fetch_history("^TNX", as_of)
        score = self._calc_trend_score(tnx)
        return max(-1.0, min(1.0, -score * 5))
    
    def _calc_dollar_signal(self, as_of: date) -> float:
        """UUP (美元): 走强对跨国巨头利空"""
        uup = self._fetch_history("UUP", as_of)
        score = self._calc_trend_score(uup)
        return max(-1.0, min(1.0, -score * 3))
    
    def _calc_short_rate_signal(self, as_of: date) -> float:
        """^IRX (3月美债): 反映加息预期"""
        irx = self._fetch_history("^IRX", as_of)
        score = self._calc_trend_score(irx)
        return max(-1.0, min(1.0, -score * 2))
    
    def _build_theme(
        self,
        stock_scores: Dict[str, float],
        vix_signal: float,
        risk_signal: float,
    ) -> ThemeConfig:
        """构建主题配置"""
        sorted_stocks = sorted(stock_scores.items(), key=lambda x: -x[1])
        
        focus_stocks = [s for s, v in sorted_stocks if v > 0.05][:5]
        avoid_stocks = [s for s, v in sorted_stocks if v < -0.05]
        
        # 板块评分
        sector_scores: Dict[str, float] = {s: 0.0 for s in self.SECTOR_STOCKS.keys()}
        for sector, stocks in self.SECTOR_STOCKS.items():
            vals = [stock_scores.get(s, 0) for s in stocks if s in stock_scores]
            if vals:
                sector_scores[sector] = sum(vals) / len(vals)
        
        sorted_sectors = sorted(sector_scores.items(), key=lambda x: -x[1])
        focus_sectors = [s for s, v in sorted_sectors if v > 0.05][:2]
        
        # 风险定级
        if vix_signal < -0.3 or risk_signal < -0.5:
            risk_level = "high"
        elif vix_signal > 0.2 and risk_signal > 0.1:
            risk_level = "low"
        else:
            risk_level = "normal"
        
        stock_bonus = {s: round(v * 0.1, 3) for s, v in stock_scores.items() if abs(v) > 0.05}
        sector_bonus = {s: round(v * 0.1, 3) for s, v in sector_scores.items() if abs(v) > 0.05}
        
        drivers = []
        if risk_signal > 0.2:
            drivers.append("Risk-On")
        elif risk_signal < -0.2:
            drivers.append("Risk-Off")
        if vix_signal < 0:
            drivers.append("High Volatility")
        
        return ThemeConfig(
            focus_sectors=focus_sectors,
            focus_stocks=focus_stocks,
            avoid_sectors=[],
            avoid_stocks=avoid_stocks,
            sector_bonus=sector_bonus,
            stock_bonus=stock_bonus,
            risk_level=risk_level,
            theme_drivers=drivers,
        )
    
    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
