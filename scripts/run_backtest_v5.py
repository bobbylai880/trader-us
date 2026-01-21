#!/usr/bin/env python3
"""
3年回测 V5 - 融合策略 (V3 趋势跟踪 + V4 分层决策)

核心优化:
1. 放宽进攻阈值 - score >= 2 即进入 offensive (原来需要 >= 3)
2. 延长板块持有周期 - 从周度改为双周
3. 结合 V3 优点 - 宽松止损(18%) + 禁用止盈
4. offensive 模式聚焦科技龙头
5. 简化 neutral 模式 - 只有一档 70% 仓位

目标: 收益接近 V3 (+100%+)，回撤控制在 15% 以内
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 板块与个股映射
# ============================================================

SECTOR_STOCKS = {
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ADBE", "CRM", "ORCL", "CSCO", "INTC"],
    "XLC": ["META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "TGT"],
    "XLF": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "SCHW", "PNC"],
    "XLV": ["UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY"],
    "XLE": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY"],
    "XLI": ["CAT", "UNP", "HON", "UPS", "BA", "RTX", "DE", "LMT", "GE", "FDX"],
    "XLP": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO"],
    "XLU": ["NEE", "DUK", "SO", "D", "AEP"],
    "XLB": ["LIN", "APD", "SHW", "ECL", "DD"],
    "XLRE": ["AMT", "PLD", "CCI", "EQIX", "SPG"],
}

# V5 核心: offensive 模式下聚焦的科技龙头
TECH_LEADERS = ["NVDA", "META", "GOOGL", "AMZN", "MSFT", "AAPL", "AMD", "AVGO", "NFLX", "TSLA"]

DEFENSIVE_SECTORS = ["XLP", "XLV", "XLU"]
GROWTH_SECTORS = ["XLK", "XLC", "XLY"]


@dataclass
class MacroView:
    """宏观视图 - 月度更新"""
    date: str
    market_regime: str  # "offensive", "neutral", "defensive"
    target_exposure: float
    vix_level: float
    vix_trend: str
    news_sentiment: float
    spy_momentum: float
    score: int  # 新增: 保存原始分数用于调试
    reasoning: str


@dataclass
class SectorAllocation:
    """板块配置 - 双周更新 (V5改进)"""
    date: str
    top_sectors: List[str]
    sector_scores: Dict[str, float]


@dataclass
class Position:
    symbol: str
    shares: int
    avg_cost: float
    entry_date: str
    sector: str
    highest_price: float
    source: str  # "tech_leader" or "sector_rotation"


@dataclass
class Trade:
    date: str
    symbol: str
    action: str
    price: float
    shares: int
    sector: str
    source: str
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""


class HybridBacktestV5:
    """V5 融合策略回测引擎"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.conn = psycopg2.connect(
            host=os.getenv("PG_HOST", "192.168.10.11"),
            port=os.getenv("PG_PORT", "5432"),
            database=os.getenv("PG_DATABASE", "trader"),
            user=os.getenv("PG_USER", "trader"),
            password=os.getenv("PG_PASSWORD", "")
        )
        
        self._prices: Dict[str, pd.DataFrame] = {}
        self._news_sentiment: Dict[str, Dict[str, float]] = {}
        
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[str, float, float]] = []
        self.macro_history: List[MacroView] = []
        self.sector_history: List[SectorAllocation] = []
        
        self._current_macro: Optional[MacroView] = None
        self._current_sectors: List[str] = []
    
    def _load_data(self, start: date, end: date):
        print("  加载价格数据...")
        
        all_symbols = set(['SPY', 'VIX'])
        for etf in SECTOR_STOCKS:
            all_symbols.add(etf)
            all_symbols.update(SECTOR_STOCKS[etf])
        all_symbols.update(TECH_LEADERS)
        
        query = """
            SELECT symbol, trade_date, open, high, low, close, volume
            FROM daily_prices
            WHERE trade_date BETWEEN %s AND %s
              AND symbol IN %s
            ORDER BY symbol, trade_date
        """
        df = pd.read_sql(query, self.conn, params=(start - timedelta(days=100), end, tuple(all_symbols)))
        
        for sym in df['symbol'].unique():
            sdf = df[df['symbol'] == sym].copy()
            sdf.set_index('trade_date', inplace=True)
            sdf['sma20'] = sdf['close'].rolling(20).mean()
            sdf['sma50'] = sdf['close'].rolling(50).mean()
            sdf['mom5'] = sdf['close'].pct_change(5)
            sdf['mom20'] = sdf['close'].pct_change(20)
            sdf['vol_ratio'] = sdf['volume'] / sdf['volume'].rolling(20).mean()
            self._prices[sym] = sdf
        
        print(f"    已加载 {len(self._prices)} 只标的")
        
        print("  加载新闻情绪...")
        query2 = """
            SELECT symbol, DATE(published_at) as news_date, 
                   AVG(sentiment_score) as sentiment
            FROM news
            GROUP BY symbol, DATE(published_at)
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query2)
            for row in cur.fetchall():
                sym = row['symbol']
                dt = str(row['news_date'])
                if sym not in self._news_sentiment:
                    self._news_sentiment[sym] = {}
                self._news_sentiment[sym][dt] = float(row['sentiment'] or 0)
        
        print(f"    已加载 {len(self._news_sentiment)} 只标的新闻")
    
    def _get(self, sym: str, dt: date, col: str) -> Optional[float]:
        if sym not in self._prices:
            return None
        df = self._prices[sym]
        valid = df[df.index <= dt]
        if len(valid) == 0:
            return None
        val = valid[col].iloc[-1]
        return float(val) if pd.notna(val) else None
    
    def _get_news_sentiment(self, symbols: List[str], dt: date, lookback: int = 30) -> float:
        sentiments = []
        for sym in symbols:
            if sym not in self._news_sentiment:
                continue
            for i in range(lookback):
                check_dt = str(dt - timedelta(days=i))
                if check_dt in self._news_sentiment[sym]:
                    sentiments.append(self._news_sentiment[sym][check_dt])
        return np.mean(sentiments) if sentiments else 0.0
    
    # ================================================================
    # 第一层: 宏观趋势分析 (月度) - V5 优化版
    # ================================================================
    
    def _analyze_macro(self, dt: date) -> MacroView:
        """月度宏观分析 - V5 放宽进攻阈值"""
        
        vix = self._get('VIX', dt, 'close') or 20
        vix_20d_ago = self._get('VIX', dt - timedelta(days=20), 'close') or vix
        
        if vix > vix_20d_ago * 1.2:
            vix_trend = "rising"
        elif vix < vix_20d_ago * 0.8:
            vix_trend = "falling"
        else:
            vix_trend = "stable"
        
        spy_mom = self._get('SPY', dt, 'mom20') or 0
        spy_close = self._get('SPY', dt, 'close') or 0
        spy_sma50 = self._get('SPY', dt, 'sma50') or spy_close
        
        market_symbols = ['SPY', 'QQQ'] + list(SECTOR_STOCKS.keys())
        news_sentiment = self._get_news_sentiment(market_symbols, dt, 30)
        
        score = 0
        reasoning_parts = []
        
        # VIX 评分 (V5: 放宽阈值)
        if vix < 18:  # 原来是 15
            score += 2
            reasoning_parts.append("VIX低位(贪婪)")
        elif vix < 22:  # 原来是 20
            score += 1
            reasoning_parts.append("VIX正常")
        elif vix < 30:
            score -= 1
            reasoning_parts.append("VIX偏高(谨慎)")
        else:
            score -= 2
            reasoning_parts.append("VIX恐慌")
        
        # SPY 动量评分 (V5: 放宽阈值)
        if spy_close > spy_sma50 and spy_mom > 0.02:  # 原来是 0.03
            score += 2
            reasoning_parts.append("SPY强势上涨")
        elif spy_close > spy_sma50:
            score += 1
            reasoning_parts.append("SPY在均线上方")
        elif spy_close < spy_sma50 and spy_mom < -0.05:  # 原来是 -0.03
            score -= 2
            reasoning_parts.append("SPY弱势下跌")
        else:
            reasoning_parts.append("SPY在均线下方")
            # 不扣分，保持中性
        
        # 新闻情绪 (权重不变)
        if news_sentiment > 0.2:  # 原来是 0.3
            score += 1
            reasoning_parts.append("新闻情绪积极")
        elif news_sentiment < -0.3:
            score -= 1
            reasoning_parts.append("新闻情绪消极")
        
        # V5 核心改进: 放宽 offensive 阈值
        # 原来: score >= 3 才进入 offensive
        # 现在: score >= 2 即进入 offensive
        if score >= 2:  # 放宽阈值!
            regime = "offensive"
            target_exposure = 0.95
        elif score >= 0:  # 原来是 >= 1, 现在 >= 0
            regime = "neutral"
            target_exposure = 0.70  # V5: 只有一档 70%
        else:
            regime = "defensive"
            target_exposure = 0.30
        
        return MacroView(
            date=str(dt),
            market_regime=regime,
            target_exposure=target_exposure,
            vix_level=vix,
            vix_trend=vix_trend,
            news_sentiment=news_sentiment,
            spy_momentum=spy_mom,
            score=score,
            reasoning=" | ".join(reasoning_parts)
        )
    
    # ================================================================
    # 第二层: 板块轮动 (双周) - V5 延长周期
    # ================================================================
    
    def _analyze_sectors(self, dt: date, macro: MacroView) -> SectorAllocation:
        """双周板块分析 - V5 延长持有周期"""
        
        spy_mom20 = self._get('SPY', dt, 'mom20') or 0
        
        sector_scores = {}
        for etf in SECTOR_STOCKS.keys():
            mom20 = self._get(etf, dt, 'mom20') or 0
            mom5 = self._get(etf, dt, 'mom5') or 0
            
            rs_vs_spy = mom20 - spy_mom20
            
            sector_sentiment = self._get_news_sentiment(
                SECTOR_STOCKS.get(etf, [])[:5], dt, 14
            )
            
            # 评分权重: 长期动量权重更高
            score = 0.5 * mom20 + 0.25 * rs_vs_spy + 0.15 * mom5 + 0.1 * sector_sentiment
            sector_scores[etf] = score
        
        # 根据宏观状态调整板块偏好
        if macro.market_regime == "defensive":
            for s in DEFENSIVE_SECTORS:
                if s in sector_scores:
                    sector_scores[s] += 0.05
            for s in GROWTH_SECTORS:
                if s in sector_scores:
                    sector_scores[s] -= 0.03
        elif macro.market_regime == "offensive":
            for s in GROWTH_SECTORS:
                if s in sector_scores:
                    sector_scores[s] += 0.05  # 增加成长板块偏好
        
        ranked = sorted(sector_scores.items(), key=lambda x: -x[1])
        
        # V5: offensive 模式下更激进
        if macro.market_regime == "offensive":
            top_n = 4
        elif macro.market_regime == "defensive":
            top_n = 2
        else:
            top_n = 3
        
        top_sectors = [s[0] for s in ranked[:top_n] if s[1] > -0.05]
        
        return SectorAllocation(
            date=str(dt),
            top_sectors=top_sectors,
            sector_scores=sector_scores,
        )
    
    # ================================================================
    # 第三层: 选股逻辑 - V5 offensive 模式聚焦科技龙头
    # ================================================================
    
    def _select_stocks(self, dt: date, macro: MacroView, sectors: List[str]) -> List[Tuple[str, str, float, str]]:
        """
        选股逻辑 - V5 核心改进
        
        offensive 模式: 优先选择 TECH_LEADERS
        neutral/defensive 模式: 在板块内选股
        
        返回: [(symbol, sector, score, source), ...]
        """
        candidates = []
        
        # V5 核心: offensive 模式聚焦科技龙头
        if macro.market_regime == "offensive":
            # 优先从科技龙头中选股
            tech_candidates = []
            for sym in TECH_LEADERS:
                if sym not in self._prices:
                    continue
                
                mom20 = self._get(sym, dt, 'mom20')
                mom5 = self._get(sym, dt, 'mom5')
                close = self._get(sym, dt, 'close')
                sma20 = self._get(sym, dt, 'sma20')
                sma50 = self._get(sym, dt, 'sma50')
                vol_ratio = self._get(sym, dt, 'vol_ratio')
                
                if mom20 is None or close is None:
                    continue
                
                # V5: 放宽科技龙头筛选条件
                # 只要价格在 SMA50 之上，且动量不是负的
                if sma50 and close < sma50 * 0.95:  # 允许 5% 的容忍度
                    continue
                
                if mom20 < -0.05:  # 只过滤明显下跌的
                    continue
                
                # 评分: 动量 + 相对强度
                score = (mom20 or 0) * 0.6 + (mom5 or 0) * 0.3
                if vol_ratio and vol_ratio > 1.2:
                    score += 0.03
                
                # 确定板块
                sector = "XLK"  # 默认科技
                for sec, stocks in SECTOR_STOCKS.items():
                    if sym in stocks:
                        sector = sec
                        break
                
                tech_candidates.append((sym, sector, score, "tech_leader"))
            
            tech_candidates.sort(key=lambda x: -x[2])
            candidates.extend(tech_candidates[:6])  # 取 Top 6 科技龙头
        
        # 板块内选股 (所有模式都可以用)
        for sector in sectors:
            stocks = SECTOR_STOCKS.get(sector, [])
            stock_scores = []
            
            for sym in stocks:
                # 如果已经在科技龙头候选中，跳过
                if any(c[0] == sym for c in candidates):
                    continue
                
                if sym not in self._prices:
                    continue
                
                mom20 = self._get(sym, dt, 'mom20')
                mom5 = self._get(sym, dt, 'mom5')
                close = self._get(sym, dt, 'close')
                sma20 = self._get(sym, dt, 'sma20')
                vol_ratio = self._get(sym, dt, 'vol_ratio')
                
                if mom20 is None or close is None or sma20 is None:
                    continue
                
                if close < sma20:
                    continue
                
                if mom20 < 0:
                    continue
                
                score = mom20 * 0.5 + (mom5 or 0) * 0.3
                if vol_ratio and vol_ratio > 1.2:
                    score += 0.02
                
                stock_scores.append((sym, sector, score, "sector_rotation"))
            
            stock_scores.sort(key=lambda x: -x[2])
            candidates.extend(stock_scores[:2])  # 每板块 Top 2
        
        candidates.sort(key=lambda x: -x[2])
        return candidates
    
    # ================================================================
    # 交易执行
    # ================================================================
    
    def _portfolio_value(self, dt: date) -> float:
        pos_val = sum(
            p.shares * (self._get(s, dt, 'close') or p.avg_cost)
            for s, p in self.positions.items()
        )
        return self.cash + pos_val
    
    def _buy(self, sym: str, sector: str, source: str, dt: date, budget: float, reason: str) -> bool:
        price = self._get(sym, dt, 'close')
        if not price or budget < 1000:
            return False
        
        shares = int(budget / price)
        if shares <= 0:
            return False
        
        cost = shares * price
        if cost > self.cash:
            return False
        
        self.cash -= cost
        
        if sym in self.positions:
            p = self.positions[sym]
            total = p.shares + shares
            p.avg_cost = (p.avg_cost * p.shares + price * shares) / total
            p.shares = total
            p.highest_price = max(p.highest_price, price)
        else:
            self.positions[sym] = Position(sym, shares, price, str(dt), sector, price, source)
        
        self.trades.append(Trade(str(dt), sym, "BUY", price, shares, sector, source, reason=reason))
        return True
    
    def _sell(self, sym: str, dt: date, reason: str) -> float:
        if sym not in self.positions:
            return 0
        
        p = self.positions[sym]
        price = self._get(sym, dt, 'close') or p.avg_cost
        proceeds = p.shares * price
        pnl = proceeds - p.shares * p.avg_cost
        pnl_pct = pnl / (p.shares * p.avg_cost)
        
        self.cash += proceeds
        self.trades.append(Trade(str(dt), sym, "SELL", price, p.shares, p.sector, p.source, pnl, pnl_pct, reason))
        del self.positions[sym]
        return pnl
    
    def _check_stops(self, dt: date, macro: MacroView):
        """
        检查止损 - V5 采用 V3 的宽松止损
        
        - 跟踪止损: 18% (V4 是 15%)
        - 禁用止盈 (让利润奔跑)
        - 熊市保护: 硬止损 10%
        """
        to_sell = []
        
        for sym, pos in self.positions.items():
            price = self._get(sym, dt, 'close')
            if not price:
                continue
            
            pos.highest_price = max(pos.highest_price, price)
            
            drawdown = (pos.highest_price - price) / pos.highest_price
            
            # V5 核心: 宽松跟踪止损 (18%)
            if drawdown > 0.18:
                to_sell.append((sym, f"跟踪止损({drawdown:.1%})"))
                continue
            
            # 熊市/防御模式: 更严格的止损
            if macro.market_regime == "defensive":
                if price < pos.avg_cost * 0.90:  # 硬止损 10%
                    to_sell.append((sym, "防御模式止损"))
                    continue
            
            # 只在明显趋势破坏时卖出
            sma50 = self._get(sym, dt, 'sma50')
            mom20 = self._get(sym, dt, 'mom20')
            if sma50 and price < sma50 * 0.92 and mom20 and mom20 < -0.10:
                to_sell.append((sym, "趋势破坏"))
        
        for sym, reason in to_sell:
            self._sell(sym, dt, reason)
    
    def _rebalance(self, dt: date, macro: MacroView, candidates: List[Tuple[str, str, float, str]]):
        """再平衡组合 - V5 优化版"""
        pv = self._portfolio_value(dt)
        current_exposure = (pv - self.cash) / pv if pv > 0 else 0
        target_exposure = macro.target_exposure
        
        # V5: 不再因为板块轮出就卖出
        # 只在以下情况卖出:
        # 1. 止损触发 (在 _check_stops 中处理)
        # 2. 仓位需要降低 (defensive 模式)
        
        if macro.market_regime == "defensive" and current_exposure > target_exposure + 0.1:
            # 需要减仓，卖出表现最差的
            holdings = []
            for sym, pos in self.positions.items():
                price = self._get(sym, dt, 'close') or pos.avg_cost
                pnl_pct = (price - pos.avg_cost) / pos.avg_cost
                holdings.append((sym, pnl_pct))
            
            holdings.sort(key=lambda x: x[1])  # 按收益排序
            
            # 卖出表现最差的直到仓位达标
            for sym, _ in holdings:
                if current_exposure <= target_exposure + 0.1:
                    break
                self._sell(sym, dt, "防御减仓")
                current_exposure = (self._portfolio_value(dt) - self.cash) / self._portfolio_value(dt)
        
        # 加仓逻辑
        if current_exposure < target_exposure - 0.1:
            available = self.cash * 0.95
            
            # V5: offensive 模式最多 6 只，其他模式 5 只
            if macro.market_regime == "offensive":
                max_positions = 6
                position_pct = 0.16  # 每只约 16%
            else:
                max_positions = 5
                position_pct = 0.14
            
            for sym, sector, score, source in candidates:
                if len(self.positions) >= max_positions:
                    break
                if sym in self.positions:
                    continue
                
                budget = min(pv * position_pct, available)
                reason = f"{source}({sector}, score:{score:.3f})"
                if self._buy(sym, sector, source, dt, budget, reason):
                    available -= budget
    
    # ================================================================
    # 主运行循环
    # ================================================================
    
    def run(self, start: date, end: date) -> dict:
        print("\n" + "=" * 70)
        print("V5 融合策略回测 (V3趋势跟踪 + V4分层决策)")
        print("=" * 70)
        print("  核心优化:")
        print("    1. 放宽进攻阈值 (score >= 2 即进入 offensive)")
        print("    2. 延长板块持有周期 (双周)")
        print("    3. 宽松止损 18% + 禁用止盈")
        print("    4. offensive 模式聚焦科技龙头")
        
        self._load_data(start, end)
        
        if 'SPY' not in self._prices:
            raise ValueError("SPY 数据缺失")
        
        trading_days = sorted(self._prices['SPY'].index.tolist())
        trading_days = [d for d in trading_days if start <= d <= end]
        
        print(f"\n  回测区间: {start} ~ {end}")
        print(f"  交易日数: {len(trading_days)}")
        print(f"  初始资金: ${self.initial_capital:,.0f}")
        
        last_macro_month = None
        last_sector_biweek = 0  # V5: 改为双周
        
        for i, dt in enumerate(trading_days):
            current_month = dt.strftime("%Y-%m")
            current_week = dt.isocalendar()[1]
            current_biweek = current_week // 2  # 双周编号
            
            # 月度宏观分析
            if current_month != last_macro_month:
                self._current_macro = self._analyze_macro(dt)
                self.macro_history.append(self._current_macro)
                last_macro_month = current_month
                
                if i % 50 == 0 or len(self.macro_history) <= 3:
                    print(f"\n  📊 [{dt}] 月度宏观: {self._current_macro.market_regime} "
                          f"(分数:{self._current_macro.score}, 仓位:{self._current_macro.target_exposure:.0%}) "
                          f"- {self._current_macro.reasoning}")
            
            # V5: 双周板块分析 (原来是周度)
            if current_biweek != last_sector_biweek and self._current_macro:
                sector_alloc = self._analyze_sectors(dt, self._current_macro)
                self.sector_history.append(sector_alloc)
                self._current_sectors = sector_alloc.top_sectors
                last_sector_biweek = current_biweek
            
            # 每日止损检查
            if self._current_macro:
                self._check_stops(dt, self._current_macro)
            
            # 每 5 天再平衡
            if i % 5 == 0 and self._current_macro and self._current_sectors:
                candidates = self._select_stocks(dt, self._current_macro, self._current_sectors)
                self._rebalance(dt, self._current_macro, candidates)
            
            # 记录净值
            pv = self._portfolio_value(dt)
            spy_price = self._get('SPY', dt, 'close') or 0
            spy_base = self._get('SPY', start, 'close') or 1
            spy_val = self.initial_capital * spy_price / spy_base
            self.equity_curve.append((str(dt), pv, spy_val))
            
            if i % 150 == 0:
                print(f"  [{i+1}/{len(trading_days)}] {dt}: ${pv:,.0f} (SPY: ${spy_val:,.0f})")
        
        return self._calc_results(start, end)
    
    def _calc_results(self, start: date, end: date) -> dict:
        final = self.equity_curve[-1][1]
        spy_final = self.equity_curve[-1][2]
        
        total_ret = final / self.initial_capital - 1
        spy_ret = spy_final / self.initial_capital - 1
        
        years = (end - start).days / 365
        ann_ret = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
        
        values = [e[1] for e in self.equity_curve]
        peak = self.initial_capital
        max_dd = 0
        for v in values:
            peak = max(peak, v)
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)
        
        rets = pd.Series(values).pct_change().dropna()
        sharpe = np.sqrt(252) * rets.mean() / rets.std() if rets.std() > 0 else 0
        
        sells = [t for t in self.trades if t.action == "SELL"]
        wins = [t for t in sells if t.pnl > 0]
        win_rate = len(wins) / len(sells) if sells else 0
        
        total_win = sum(t.pnl for t in wins)
        total_loss = abs(sum(t.pnl for t in sells if t.pnl < 0))
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        
        # 统计
        regime_dist = {}
        for m in self.macro_history:
            regime_dist[m.market_regime] = regime_dist.get(m.market_regime, 0) + 1
        
        source_dist = {"tech_leader": 0, "sector_rotation": 0}
        for t in self.trades:
            if t.action == "BUY":
                source_dist[t.source] = source_dist.get(t.source, 0) + 1
        
        sector_counts = {}
        for t in self.trades:
            if t.action == "BUY":
                sector_counts[t.sector] = sector_counts.get(t.sector, 0) + 1
        
        return {
            "final_value": final,
            "total_return": total_ret,
            "annualized_return": ann_ret,
            "spy_return": spy_ret,
            "alpha": total_ret - spy_ret,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(self.trades),
            "avg_win": np.mean([t.pnl for t in wins]) if wins else 0,
            "avg_loss": np.mean([t.pnl for t in sells if t.pnl < 0]) if any(t.pnl < 0 for t in sells) else 0,
            "regime_distribution": regime_dist,
            "source_distribution": source_dist,
            "sector_distribution": sector_counts,
        }


def main():
    bt = HybridBacktestV5(100000.0)
    result = bt.run(date(2023, 1, 3), date(2026, 1, 16))
    
    print("\n" + "=" * 70)
    print("V5 融合策略回测结果")
    print("=" * 70)
    print(f"\n  最终价值: ${result['final_value']:,.0f}")
    print(f"  总收益率: {result['total_return']:+.2%}")
    print(f"  年化收益: {result['annualized_return']:+.2%}")
    print(f"  SPY收益:  {result['spy_return']:+.2%}")
    print(f"  超额收益: {result['alpha']:+.2%}")
    print(f"\n  最大回撤: {result['max_drawdown']:.2%}")
    print(f"  夏普比率: {result['sharpe']:.2f}")
    print(f"  胜率: {result['win_rate']:.1%}")
    print(f"  盈亏比: {result['profit_factor']:.2f}")
    print(f"  总交易: {result['total_trades']} 笔")
    print(f"  平均盈利: ${result['avg_win']:,.0f}")
    print(f"  平均亏损: ${result['avg_loss']:,.0f}")
    
    print(f"\n  宏观状态分布:")
    for regime, count in result['regime_distribution'].items():
        print(f"    {regime}: {count} 月")
    
    print(f"\n  交易来源分布:")
    for source, count in result['source_distribution'].items():
        print(f"    {source}: {count} 笔")
    
    print(f"\n  板块交易分布:")
    sorted_sectors = sorted(result['sector_distribution'].items(), key=lambda x: -x[1])
    for sector, count in sorted_sectors[:5]:
        print(f"    {sector}: {count} 笔")
    
    # 保存结果
    output = Path("storage/backtest_3y_v5")
    output.mkdir(parents=True, exist_ok=True)
    
    with open(output / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    trades_data = [
        {"date": t.date, "symbol": t.symbol, "action": t.action,
         "price": t.price, "shares": t.shares, "sector": t.sector,
         "source": t.source, "pnl": t.pnl, "pnl_pct": t.pnl_pct, "reason": t.reason}
        for t in bt.trades
    ]
    with open(output / "trades.json", "w") as f:
        json.dump(trades_data, f, indent=2)
    
    macro_data = [
        {"date": m.date, "regime": m.market_regime, "exposure": m.target_exposure,
         "score": m.score, "vix": m.vix_level, "sentiment": m.news_sentiment, 
         "reasoning": m.reasoning}
        for m in bt.macro_history
    ]
    with open(output / "macro_history.json", "w") as f:
        json.dump(macro_data, f, indent=2)
    
    equity_df = pd.DataFrame(bt.equity_curve, columns=['date', 'portfolio', 'spy'])
    equity_df.to_csv(output / "equity_curve.csv", index=False)
    
    print(f"\n📁 保存到: {output}")
    
    # 最大盈利交易
    print("\n【最大盈利交易】")
    top = sorted([t for t in bt.trades if t.action == "SELL"], key=lambda x: -x.pnl)[:5]
    for t in top:
        print(f"  {t.date} {t.symbol}({t.source}): ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
    
    # 最大亏损交易
    print("\n【最大亏损交易】")
    bottom = sorted([t for t in bt.trades if t.action == "SELL"], key=lambda x: x.pnl)[:5]
    for t in bottom:
        print(f"  {t.date} {t.symbol}({t.source}): ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
    
    # 宏观月度回顾
    print("\n【宏观月度回顾 (最近6个月)】")
    for m in bt.macro_history[-6:]:
        print(f"  {m.date}: {m.market_regime} (分数:{m.score}, VIX:{m.vix_level:.1f}, 仓位:{m.target_exposure:.0%})")
    
    # V3/V4 对比
    print("\n" + "=" * 70)
    print("策略对比 (V3 vs V4 vs V5)")
    print("=" * 70)
    print("""
    | 指标       | V3 趋势跟踪 | V4 分层决策 | V5 融合策略 |
    |------------|-------------|-------------|-------------|
    | 总收益率   | +117.02%    | +40.65%     | 待运行...   |
    | 年化收益   | +29.05%     | +11.88%     | 待运行...   |
    | Alpha      | +35.40%     | -40.98%     | 待运行...   |
    | 夏普比率   | 1.32        | 0.89        | 待运行...   |
    | 最大回撤   | 16.10%      | 13.69%      | 待运行...   |
    """)


if __name__ == "__main__":
    main()
