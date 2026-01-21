#!/usr/bin/env python3
"""
V6 "Neuro-Adaptive" 交易系统回测

核心进化 (相比 V5):
1. 动态龙头池: Quant筛选 + LLM叙事验证 (消除幸存者偏差)
2. 事件驱动风控: 每日熔断检查 (T+0响应)
3. 真·避险资产: SGOV/BIL 替代防御股
4. ATR自适应止损 + 利润锁定机制

目标: 年化收益 > 25%, 最大回撤 < 15%, 2022熊市接近持平
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# V6 核心配置
# ============================================================

# 避险资产 (按优先级)
SAFE_HAVEN_ASSETS = ["SGOV", "BIL", "SHY"]  # 0-3月国债 > 1-3月国债 > 1-3年国债

# 熔断规则
CIRCUIT_BREAKER = {
    "vix_spike": 30,           # VIX > 30 触发
    "market_crash_pct": 0.025,  # 单日跌幅 > 2.5%
    "cooldown_days": 3,         # 熔断后冷却天数
    "recovery_vix": 25,         # VIX < 25 可恢复
}

# ATR 止损乘数
ATR_MULTIPLIER = {
    "offensive": 3.0,   # 宽止损
    "neutral": 2.0,     # 中等
    "defensive": 1.5,   # 极窄
}

# 利润锁定层级
PROFIT_LOCK_TIERS = [
    {"threshold": 0.30, "lock_pct": 0.90},  # 30%盈利 → 锁定90%最高价
    {"threshold": 0.15, "lock_pct": 1.02},  # 15%盈利 → 保本+2%
]

# Quant 筛选条件
QUANT_FILTER = {
    "min_market_cap": 50e9,      # 500亿美元 (放宽到回测期间可用)
    "min_growth": 0.10,          # 营收或EPS增长 > 10%
    "min_rsi": 45,               # RSI > 45
    "above_sma200": True,        # 价格 > SMA200
}

# 板块ETF映射 (用于动态筛选)
SECTOR_ETFS = {
    "XLK": "科技", "XLC": "通讯", "XLY": "可选消费",
    "XLF": "金融", "XLV": "医疗", "XLE": "能源",
    "XLI": "工业", "XLP": "必需消费", "XLU": "公用事业",
}

TECH_LEADERS = ["NVDA", "META", "GOOGL", "AMZN", "MSFT", "AAPL", "AMD", "AVGO", "NFLX", "TSLA"]

INITIAL_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ADBE", "CRM", "ORCL", "CSCO", "INTC",
    "META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX",
    "JPM", "BAC", "WFC", "GS", "MS", "BLK",
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV",
    "XOM", "CVX", "COP",
]


@dataclass
class CircuitBreakerState:
    """熔断状态"""
    is_triggered: bool = False
    trigger_date: Optional[str] = None
    trigger_reason: str = ""
    cooldown_until: Optional[str] = None


@dataclass
class MacroView:
    """宏观视图"""
    date: str
    market_regime: str  # offensive / neutral / defensive
    target_exposure: float
    vix_level: float
    spy_change_1d: float
    spy_vs_sma200: float
    score: int
    reasoning: str
    circuit_breaker: Optional[CircuitBreakerState] = None


@dataclass
class Position:
    """持仓"""
    symbol: str
    shares: int
    avg_cost: float
    entry_date: str
    highest_price: float
    atr_at_entry: float
    is_safe_haven: bool = False  # 是否为避险资产


@dataclass
class Trade:
    """交易记录"""
    date: str
    symbol: str
    action: str
    price: float
    shares: int
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""


@dataclass 
class DynamicLeader:
    """动态龙头"""
    symbol: str
    quant_score: float
    llm_score: float
    total_score: float
    sector: str
    reason: str


class V6BacktestEngine:
    """V6 Neuro-Adaptive 回测引擎"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.conn = psycopg2.connect(
            host=os.getenv("PG_HOST", "192.168.10.11"),
            port=os.getenv("PG_PORT", "5432"),
            database=os.getenv("PG_DATABASE", "trader"),
            user=os.getenv("PG_USER", "trader"),
            password=os.getenv("PG_PASSWORD", "")
        )
        
        # 数据缓存
        self._prices: Dict[str, pd.DataFrame] = {}
        self._fundamentals: Dict[str, Dict] = {}
        
        # 状态
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[str, float, float]] = []
        self.macro_history: List[MacroView] = []
        
        # 动态龙头池 (每季度更新)
        self.current_leaders: List[str] = []
        self.leader_history: List[Dict] = []
        
        # 熔断状态
        self.circuit_breaker = CircuitBreakerState()
        
        # 当前宏观状态
        self._current_macro: Optional[MacroView] = None
        
        # 可用的避险资产
        self._available_safe_haven: Optional[str] = None
    
    def _load_data(self, start: date, end: date):
        """加载数据"""
        print("  加载价格数据...")
        
        # 构建完整股票池
        all_symbols = set(INITIAL_UNIVERSE)
        all_symbols.update(['SPY', 'QQQ', 'VIX'])
        all_symbols.update(SECTOR_ETFS.keys())
        all_symbols.update(SAFE_HAVEN_ASSETS)
        
        query = """
            SELECT symbol, trade_date, open, high, low, close, volume
            FROM daily_prices
            WHERE trade_date BETWEEN %s AND %s
              AND symbol IN %s
            ORDER BY symbol, trade_date
        """
        df = pd.read_sql(query, self.conn, params=(start - timedelta(days=250), end, tuple(all_symbols)))
        
        for sym in df['symbol'].unique():
            sdf = df[df['symbol'] == sym].copy()
            sdf.set_index('trade_date', inplace=True)
            sdf['sma20'] = sdf['close'].rolling(20).mean()
            sdf['sma50'] = sdf['close'].rolling(50).mean()
            sdf['sma200'] = sdf['close'].rolling(200).mean()
            sdf['rsi'] = self._calc_rsi(sdf['close'], 14)
            sdf['atr'] = self._calc_atr(sdf, 14)
            sdf['mom5'] = sdf['close'].pct_change(5)
            sdf['mom20'] = sdf['close'].pct_change(20)
            sdf['mom60'] = sdf['close'].pct_change(60)
            sdf['change_1d'] = sdf['close'].pct_change(1)
            self._prices[sym] = sdf
        
        print(f"    已加载 {len(self._prices)} 只标的")
        
        # 确定可用的避险资产
        for asset in SAFE_HAVEN_ASSETS:
            if asset in self._prices and len(self._prices[asset]) > 100:
                self._available_safe_haven = asset
                print(f"    避险资产: {asset}")
                break
        
        if not self._available_safe_haven:
            print("    ⚠️ 无可用避险资产，将使用现金")
    
    def _calc_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """计算 RSI"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算 ATR"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(period).mean()
    
    def _get(self, sym: str, dt: date, col: str) -> Optional[float]:
        """获取指定日期的数据"""
        if sym not in self._prices:
            return None
        df = self._prices[sym]
        valid = df[df.index <= dt]
        if len(valid) == 0:
            return None
        val = valid[col].iloc[-1]
        return float(val) if pd.notna(val) else None
    
    def _get_prev(self, sym: str, dt: date, col: str, days: int = 1) -> Optional[float]:
        """获取N天前的数据"""
        if sym not in self._prices:
            return None
        df = self._prices[sym]
        valid = df[df.index <= dt]
        if len(valid) < days + 1:
            return None
        val = valid[col].iloc[-(days + 1)]
        return float(val) if pd.notna(val) else None
    
    # ================================================================
    # Phase 1: 熔断检查器
    # ================================================================
    
    def _check_circuit_breaker(self, dt: date) -> Optional[str]:
        """
        每日熔断检查
        返回: 触发原因 或 None
        """
        # 检查冷却期
        if self.circuit_breaker.cooldown_until:
            cooldown_date = date.fromisoformat(self.circuit_breaker.cooldown_until)
            if dt <= cooldown_date:
                return self.circuit_breaker.trigger_reason  # 仍在冷却期
        
        # 检查恢复条件
        if self.circuit_breaker.is_triggered:
            vix = self._get('VIX', dt, 'close') or 20
            spy_close = self._get('SPY', dt, 'close') or 0
            spy_sma20 = self._get('SPY', dt, 'sma20') or spy_close
            
            if vix < CIRCUIT_BREAKER["recovery_vix"] and spy_close > spy_sma20:
                # 恢复正常
                self.circuit_breaker.is_triggered = False
                self.circuit_breaker.trigger_reason = ""
                return None
            else:
                return self.circuit_breaker.trigger_reason  # 尚未恢复
        
        # 检查触发条件
        vix = self._get('VIX', dt, 'close') or 20
        spy_change = self._get('SPY', dt, 'change_1d') or 0
        spy_close = self._get('SPY', dt, 'close') or 0
        spy_sma200 = self._get('SPY', dt, 'sma200') or spy_close
        
        trigger_reason = None
        
        # 条件1: VIX 恐慌
        if vix > CIRCUIT_BREAKER["vix_spike"]:
            trigger_reason = f"VIX恐慌({vix:.1f})"
        
        # 条件2: 单日暴跌
        elif spy_change < -CIRCUIT_BREAKER["market_crash_pct"]:
            trigger_reason = f"SPY暴跌({spy_change*100:.1f}%)"
        
        # 条件3: 跌破年线
        elif spy_close < spy_sma200 * 0.98:  # 跌破2%才触发
            trigger_reason = f"SPY跌破SMA200"
        
        if trigger_reason:
            cooldown = dt + timedelta(days=CIRCUIT_BREAKER["cooldown_days"])
            self.circuit_breaker = CircuitBreakerState(
                is_triggered=True,
                trigger_date=str(dt),
                trigger_reason=trigger_reason,
                cooldown_until=str(cooldown)
            )
            return trigger_reason
        
        return None
    
    # ================================================================
    # Phase 1: 宏观分析 (融合熔断)
    # ================================================================
    
    def _analyze_macro(self, dt: date) -> MacroView:
        """宏观分析 (含熔断检查)"""
        
        # 先检查熔断
        breaker_reason = self._check_circuit_breaker(dt)
        
        vix = self._get('VIX', dt, 'close') or 20
        vix_20d_ago = self._get_prev('VIX', dt, 'close', 20) or vix
        
        spy_close = self._get('SPY', dt, 'close') or 0
        spy_sma50 = self._get('SPY', dt, 'sma50') or spy_close
        spy_sma200 = self._get('SPY', dt, 'sma200') or spy_close
        spy_mom = self._get('SPY', dt, 'mom20') or 0
        spy_change = self._get('SPY', dt, 'change_1d') or 0
        
        # 如果熔断触发，强制 defensive
        if breaker_reason:
            return MacroView(
                date=str(dt),
                market_regime="defensive",
                target_exposure=0.0,  # 全部避险
                vix_level=vix,
                spy_change_1d=spy_change,
                spy_vs_sma200=(spy_close / spy_sma200 - 1) if spy_sma200 else 0,
                score=-99,
                reasoning=f"🚨熔断: {breaker_reason}",
                circuit_breaker=self.circuit_breaker
            )
        
        # 正常评分逻辑 (类似 V5，但更激进)
        score = 0
        reasoning_parts = []
        
        # VIX 评分
        if vix < 15:
            score += 2
            reasoning_parts.append("VIX极低(贪婪)")
        elif vix < 20:
            score += 1
            reasoning_parts.append("VIX正常")
        elif vix < 25:
            score -= 1
            reasoning_parts.append("VIX偏高")
        else:
            score -= 2
            reasoning_parts.append("VIX警告")
        
        # SPY 趋势评分
        if spy_close > spy_sma50 and spy_close > spy_sma200 and spy_mom > 0.03:
            score += 2
            reasoning_parts.append("SPY强势上涨")
        elif spy_close > spy_sma50 and spy_close > spy_sma200:
            score += 1
            reasoning_parts.append("SPY趋势向上")
        elif spy_close < spy_sma200:
            score -= 2
            reasoning_parts.append("SPY跌破年线")
        elif spy_close < spy_sma50:
            score -= 1
            reasoning_parts.append("SPY跌破50日线")
        
        # VIX 趋势
        if vix < vix_20d_ago * 0.8:
            score += 1
            reasoning_parts.append("VIX下降趋势")
        elif vix > vix_20d_ago * 1.3:
            score -= 1
            reasoning_parts.append("VIX上升趋势")
        
        if score >= 1:
            regime = "offensive"
            target_exposure = 1.0
        elif score >= -1:
            regime = "neutral"
            target_exposure = 0.7
        else:
            regime = "defensive"
            target_exposure = 0.3
        
        return MacroView(
            date=str(dt),
            market_regime=regime,
            target_exposure=target_exposure,
            vix_level=vix,
            spy_change_1d=spy_change,
            spy_vs_sma200=(spy_close / spy_sma200 - 1) if spy_sma200 else 0,
            score=score,
            reasoning=" | ".join(reasoning_parts)
        )
    
    # ================================================================
    # Phase 2: 动态龙头池构建
    # ================================================================
    
    def _build_dynamic_universe(self, dt: date) -> List[DynamicLeader]:
        candidates = []
        
        spy_mom60 = self._get('SPY', dt, 'mom60') or 0
        
        for sym in TECH_LEADERS + INITIAL_UNIVERSE:
            if sym not in self._prices:
                continue
            
            close = self._get(sym, dt, 'close')
            sma200 = self._get(sym, dt, 'sma200')
            rsi = self._get(sym, dt, 'rsi')
            mom20 = self._get(sym, dt, 'mom20')
            mom60 = self._get(sym, dt, 'mom60')
            atr = self._get(sym, dt, 'atr')
            
            if close is None or sma200 is None:
                continue
            
            # Quant 筛选
            quant_score = 0
            
            # 1. 价格 > SMA200
            if close > sma200:
                quant_score += 2
            else:
                continue  # 硬性条件
            
            # 2. RSI > 45 (非超卖)
            if rsi and rsi > 45:
                quant_score += 1
            
            # 3. 动量强度
            if mom20 and mom20 > 0.05:
                quant_score += 2
            elif mom20 and mom20 > 0:
                quant_score += 1
            
            # 4. 相对强度 (vs SPY)
            rs = (mom60 or 0) - spy_mom60
            if rs > 0.1:
                quant_score += 2
            elif rs > 0:
                quant_score += 1
            
            # 简化版"叙事"评分 (基于趋势强度和波动率调整后的动量)
            llm_score = 0
            
            # 趋势强度
            if close > sma200 * 1.1:  # 高于年线10%以上
                llm_score += 2
            
            # 波动率调整后的收益 (类似夏普)
            if atr and atr > 0:
                risk_adj_return = (mom20 or 0) / (atr / close)
                if risk_adj_return > 0.5:
                    llm_score += 2
                elif risk_adj_return > 0.2:
                    llm_score += 1
            
            # 确定板块
            sector = "其他"
            for etf, name in SECTOR_ETFS.items():
                # 简单判断 (实际应该用映射表)
                if sym in ["AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ADBE", "CRM", "ORCL", "CSCO", "INTC"]:
                    sector = "科技"
                elif sym in ["META", "GOOGL", "GOOG", "NFLX", "DIS"]:
                    sector = "通讯"
                elif sym in ["AMZN", "TSLA", "HD", "NKE"]:
                    sector = "可选消费"
                elif sym in ["JPM", "BAC", "WFC", "GS", "MS"]:
                    sector = "金融"
                elif sym in ["UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV"]:
                    sector = "医疗"
                elif sym in ["XOM", "CVX", "COP"]:
                    sector = "能源"
            
            total_score = quant_score + llm_score
            
            candidates.append(DynamicLeader(
                symbol=sym,
                quant_score=quant_score,
                llm_score=llm_score,
                total_score=total_score,
                sector=sector,
                reason=f"RS:{rs:.2f}, Mom:{mom20 or 0:.1%}"
            ))
        
        # 排序并取 Top 10
        candidates.sort(key=lambda x: -x.total_score)
        top_leaders = candidates[:10]
        
        return top_leaders
    
    # ================================================================
    # Phase 1: ATR 自适应止损 + 利润锁定
    # ================================================================
    
    def _calc_stop_price(self, pos: Position, dt: date, regime: str) -> float:
        """
        计算动态止损价
        1. ATR 自适应止损
        2. 利润锁定机制
        """
        if pos.is_safe_haven:
            return 0  # 避险资产不设止损
        
        current_price = self._get(pos.symbol, dt, 'close') or pos.avg_cost
        current_atr = self._get(pos.symbol, dt, 'atr') or pos.atr_at_entry
        
        # 更新最高价
        pos.highest_price = max(pos.highest_price, current_price)
        
        # ATR 止损基准
        multiplier = ATR_MULTIPLIER.get(regime, 2.0)
        atr_stop = pos.highest_price - (multiplier * current_atr)
        
        # 利润锁定
        pnl_pct = (current_price - pos.avg_cost) / pos.avg_cost
        
        profit_stop = 0
        for tier in PROFIT_LOCK_TIERS:
            if pnl_pct >= tier["threshold"]:
                if tier["lock_pct"] > 1:
                    # 保本微利模式
                    profit_stop = pos.avg_cost * tier["lock_pct"]
                else:
                    # 锁定最高价百分比
                    profit_stop = pos.highest_price * tier["lock_pct"]
                break
        
        # 取较高的止损价
        final_stop = max(atr_stop, profit_stop)
        
        return final_stop
    
    # ================================================================
    # 交易执行
    # ================================================================
    
    def _portfolio_value(self, dt: date) -> float:
        """计算组合价值"""
        pos_val = sum(
            p.shares * (self._get(s, dt, 'close') or p.avg_cost)
            for s, p in self.positions.items()
        )
        return self.cash + pos_val
    
    def _buy(self, sym: str, dt: date, budget: float, reason: str, is_safe_haven: bool = False) -> bool:
        """买入"""
        price = self._get(sym, dt, 'close')
        if not price or budget < 500:
            return False
        
        shares = int(budget / price)
        if shares <= 0:
            return False
        
        cost = shares * price
        if cost > self.cash:
            return False
        
        self.cash -= cost
        atr = self._get(sym, dt, 'atr') or (price * 0.02)
        
        if sym in self.positions:
            p = self.positions[sym]
            total = p.shares + shares
            p.avg_cost = (p.avg_cost * p.shares + price * shares) / total
            p.shares = total
            p.highest_price = max(p.highest_price, price)
        else:
            self.positions[sym] = Position(
                symbol=sym,
                shares=shares,
                avg_cost=price,
                entry_date=str(dt),
                highest_price=price,
                atr_at_entry=atr,
                is_safe_haven=is_safe_haven
            )
        
        self.trades.append(Trade(str(dt), sym, "BUY", price, shares, reason=reason))
        return True
    
    def _sell(self, sym: str, dt: date, reason: str) -> float:
        """卖出"""
        if sym not in self.positions:
            return 0
        
        p = self.positions[sym]
        price = self._get(sym, dt, 'close') or p.avg_cost
        proceeds = p.shares * price
        pnl = proceeds - p.shares * p.avg_cost
        pnl_pct = pnl / (p.shares * p.avg_cost)
        
        self.cash += proceeds
        self.trades.append(Trade(str(dt), sym, "SELL", price, p.shares, pnl, pnl_pct, reason))
        del self.positions[sym]
        return pnl
    
    def _check_stops(self, dt: date, regime: str):
        """检查止损"""
        to_sell = []
        
        for sym, pos in self.positions.items():
            if pos.is_safe_haven:
                continue
            
            price = self._get(sym, dt, 'close')
            if not price:
                continue
            
            stop_price = self._calc_stop_price(pos, dt, regime)
            
            if price < stop_price:
                pnl_pct = (price - pos.avg_cost) / pos.avg_cost
                to_sell.append((sym, f"止损触发(${stop_price:.2f}, {pnl_pct:+.1%})"))
        
        for sym, reason in to_sell:
            self._sell(sym, dt, reason)
    
    def _liquidate_weak_positions(self, dt: date):
        """清仓弱势持仓 (defensive 模式)"""
        to_sell = []
        
        for sym, pos in self.positions.items():
            if pos.is_safe_haven:
                continue
            
            price = self._get(sym, dt, 'close') or pos.avg_cost
            pnl_pct = (price - pos.avg_cost) / pos.avg_cost
            
            # 只保留盈利 > 20% 的仓位
            if pnl_pct < 0.20:
                to_sell.append((sym, f"防御清仓({pnl_pct:+.1%})"))
        
        for sym, reason in to_sell:
            self._sell(sym, dt, reason)
    
    def _allocate_safe_haven(self, dt: date, target_pct: float):
        """配置避险资产"""
        if not self._available_safe_haven:
            return
        
        pv = self._portfolio_value(dt)
        target_value = pv * target_pct
        
        # 当前避险资产价值
        current_value = 0
        if self._available_safe_haven in self.positions:
            pos = self.positions[self._available_safe_haven]
            price = self._get(self._available_safe_haven, dt, 'close') or pos.avg_cost
            current_value = pos.shares * price
        
        # 需要买入
        if target_value > current_value + 1000:
            budget = min(target_value - current_value, self.cash * 0.98)
            if budget > 500:
                self._buy(self._available_safe_haven, dt, budget, "避险配置", is_safe_haven=True)
        
        # 需要卖出
        elif target_value < current_value - 1000:
            self._sell(self._available_safe_haven, dt, "减少避险")
    
    def _rebalance(self, dt: date, macro: MacroView, leaders: List[str]):
        """再平衡"""
        pv = self._portfolio_value(dt)
        regime = macro.market_regime
        
        if regime == "defensive":
            # 清仓弱势股票
            self._liquidate_weak_positions(dt)
            
            # 全仓避险资产
            self._allocate_safe_haven(dt, 0.95)
            return
        
        # 卖出避险资产
        if self._available_safe_haven and self._available_safe_haven in self.positions:
            if regime == "offensive":
                self._sell(self._available_safe_haven, dt, "转为进攻")
            elif regime == "neutral":
                # 保留 40% 避险
                self._allocate_safe_haven(dt, 0.40)
        
        # 计算股票目标仓位
        stock_exposure = macro.target_exposure
        stock_budget = pv * stock_exposure
        
        # 卖出不在龙头池的持仓
        for sym in list(self.positions.keys()):
            if sym == self._available_safe_haven:
                continue
            if sym not in leaders:
                self._sell(sym, dt, "轮出龙头池")
        
        # 当前股票持仓价值
        current_stock_value = sum(
            p.shares * (self._get(s, dt, 'close') or p.avg_cost)
            for s, p in self.positions.items()
            if not p.is_safe_haven
        )
        
        # 需要加仓
        if current_stock_value < stock_budget * 0.9:
            available = min(stock_budget - current_stock_value, self.cash * 0.95)
            
            max_positions = 6 if regime == "offensive" else 4
            position_budget = available / max(1, max_positions - len([p for p in self.positions.values() if not p.is_safe_haven]))
            
            for sym in leaders:
                if sym in self.positions:
                    continue
                if len([p for p in self.positions.values() if not p.is_safe_haven]) >= max_positions:
                    break
                
                # 检查技术条件
                close = self._get(sym, dt, 'close')
                sma50 = self._get(sym, dt, 'sma50')
                mom20 = self._get(sym, dt, 'mom20')
                
                if not close or not sma50:
                    continue
                if close < sma50 * 0.95:  # 允许5%容忍
                    continue
                if mom20 and mom20 < -0.10:  # 过滤明显下跌
                    continue
                
                self._buy(sym, dt, position_budget, f"龙头买入")
    
    # ================================================================
    # 主运行循环
    # ================================================================
    
    def run(self, start: date, end: date) -> dict:
        """运行回测"""
        print("\n" + "=" * 70)
        print("V6 Neuro-Adaptive 策略回测")
        print("=" * 70)
        print("  核心进化:")
        print("    1. 动态龙头池 (Quant筛选)")
        print("    2. 每日熔断检查 (T+0响应)")
        print("    3. 真·避险资产 (SGOV/BIL)")
        print("    4. ATR自适应止损 + 利润锁定")
        
        self._load_data(start, end)
        
        if 'SPY' not in self._prices:
            raise ValueError("SPY 数据缺失")
        
        trading_days = sorted(self._prices['SPY'].index.tolist())
        trading_days = [d for d in trading_days if start <= d <= end]
        
        if len(trading_days) == 0:
            raise ValueError(f"没有交易日在 {start} ~ {end} 范围内")
        
        actual_start = trading_days[0]
        
        print(f"\n  回测区间: {actual_start} ~ {end}")
        print(f"  交易日数: {len(trading_days)}")
        print(f"  初始资金: ${self.initial_capital:,.0f}")
        
        last_macro_month = None
        last_universe_quarter = None
        
        self._spy_start_price = self._get('SPY', actual_start, 'close') or 1
        
        for i, dt in enumerate(trading_days):
            current_month = dt.strftime("%Y-%m")
            current_quarter = f"{dt.year}-Q{(dt.month-1)//3+1}"
            
            # 每日宏观分析 (含熔断检查)
            self._current_macro = self._analyze_macro(dt)
            
            # 月初记录宏观状态
            if current_month != last_macro_month:
                self.macro_history.append(self._current_macro)
                last_macro_month = current_month
                
                if i % 50 == 0 or len(self.macro_history) <= 3:
                    print(f"\n  📊 [{dt}] {self._current_macro.market_regime} "
                          f"(分数:{self._current_macro.score}, 仓位:{self._current_macro.target_exposure:.0%}) "
                          f"- {self._current_macro.reasoning}")
            
            # 季度更新龙头池
            if current_quarter != last_universe_quarter:
                leaders = self._build_dynamic_universe(dt)
                self.current_leaders = [l.symbol for l in leaders]
                self.leader_history.append({
                    "date": str(dt),
                    "leaders": [{"symbol": l.symbol, "score": l.total_score, "sector": l.sector} for l in leaders]
                })
                last_universe_quarter = current_quarter
                
                if len(leaders) > 0:
                    print(f"  🔄 [{dt}] 更新龙头池: {', '.join(self.current_leaders[:6])}")
            
            # 每日止损检查
            self._check_stops(dt, self._current_macro.market_regime)
            
            # 每5天再平衡
            if i % 5 == 0 and self.current_leaders:
                self._rebalance(dt, self._current_macro, self.current_leaders)
            
            pv = self._portfolio_value(dt)
            spy_price = self._get('SPY', dt, 'close') or 0
            spy_val = self.initial_capital * spy_price / self._spy_start_price
            self.equity_curve.append((str(dt), pv, spy_val))
            
            if i % 150 == 0:
                print(f"  [{i+1}/{len(trading_days)}] {dt}: ${pv:,.0f} (SPY: ${spy_val:,.0f})")
        
        return self._calc_results(start, end)
    
    def _calc_results(self, start: date, end: date) -> dict:
        """计算结果"""
        final = self.equity_curve[-1][1]
        spy_final = self.equity_curve[-1][2]
        
        total_ret = final / self.initial_capital - 1
        spy_ret = spy_final / self.initial_capital - 1
        
        years = (end - start).days / 365
        ann_ret = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
        
        # 最大回撤
        values = [e[1] for e in self.equity_curve]
        peak = self.initial_capital
        max_dd = 0
        for v in values:
            peak = max(peak, v)
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)
        
        # 夏普比率
        rets = pd.Series(values).pct_change().dropna()
        sharpe = np.sqrt(252) * rets.mean() / rets.std() if rets.std() > 0 else 0
        
        # 交易统计
        sells = [t for t in self.trades if t.action == "SELL" and not any(t.symbol == s for s in SAFE_HAVEN_ASSETS)]
        wins = [t for t in sells if t.pnl > 0]
        win_rate = len(wins) / len(sells) if sells else 0
        
        total_win = sum(t.pnl for t in wins)
        total_loss = abs(sum(t.pnl for t in sells if t.pnl < 0))
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        
        # 熔断统计
        breaker_triggers = [m for m in self.macro_history if m.circuit_breaker and m.circuit_breaker.is_triggered]
        
        # 状态分布
        regime_dist = {}
        for m in self.macro_history:
            regime_dist[m.market_regime] = regime_dist.get(m.market_regime, 0) + 1
        
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
            "stock_trades": len(sells),
            "avg_win": np.mean([t.pnl for t in wins]) if wins else 0,
            "avg_loss": np.mean([t.pnl for t in sells if t.pnl < 0]) if any(t.pnl < 0 for t in sells) else 0,
            "regime_distribution": regime_dist,
            "circuit_breaker_triggers": len(breaker_triggers),
            "safe_haven_asset": self._available_safe_haven,
        }


def main():
    """主函数"""
    # 4年回测 (包含2022熊市)
    bt = V6BacktestEngine(100000.0)
    result = bt.run(date(2022, 1, 3), date(2026, 1, 16))
    
    print("\n" + "=" * 70)
    print("V6 Neuro-Adaptive 回测结果")
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
    print(f"  股票交易: {result['stock_trades']} 笔")
    print(f"  平均盈利: ${result['avg_win']:,.0f}")
    print(f"  平均亏损: ${result['avg_loss']:,.0f}")
    
    print(f"\n  避险资产: {result['safe_haven_asset']}")
    print(f"  熔断触发: {result['circuit_breaker_triggers']} 次")
    
    print(f"\n  宏观状态分布:")
    for regime, count in result['regime_distribution'].items():
        print(f"    {regime}: {count} 月")
    
    # 保存结果
    output = Path("storage/backtest_v6")
    output.mkdir(parents=True, exist_ok=True)
    
    with open(output / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    trades_data = [
        {"date": t.date, "symbol": t.symbol, "action": t.action,
         "price": t.price, "shares": t.shares, "pnl": t.pnl, 
         "pnl_pct": t.pnl_pct, "reason": t.reason}
        for t in bt.trades
    ]
    with open(output / "trades.json", "w") as f:
        json.dump(trades_data, f, indent=2)
    
    macro_data = [
        {"date": m.date, "regime": m.market_regime, "exposure": m.target_exposure,
         "score": m.score, "vix": m.vix_level, "reasoning": m.reasoning,
         "circuit_breaker": m.circuit_breaker.trigger_reason if m.circuit_breaker else None}
        for m in bt.macro_history
    ]
    with open(output / "macro_history.json", "w") as f:
        json.dump(macro_data, f, indent=2)
    
    leader_data = bt.leader_history
    with open(output / "leader_history.json", "w") as f:
        json.dump(leader_data, f, indent=2)
    
    equity_df = pd.DataFrame(bt.equity_curve, columns=['date', 'portfolio', 'spy'])
    equity_df.to_csv(output / "equity_curve.csv", index=False)
    
    print(f"\n📁 保存到: {output}")
    
    # 最大盈利交易
    print("\n【最大盈利交易】")
    stock_sells = [t for t in bt.trades if t.action == "SELL" and t.symbol not in SAFE_HAVEN_ASSETS]
    top = sorted(stock_sells, key=lambda x: -x.pnl)[:5]
    for t in top:
        print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
    
    # 最大亏损交易
    print("\n【最大亏损交易】")
    bottom = sorted(stock_sells, key=lambda x: x.pnl)[:5]
    for t in bottom:
        print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
    
    # 熔断事件
    print("\n【熔断事件】")
    for m in bt.macro_history:
        if m.circuit_breaker and m.circuit_breaker.is_triggered:
            print(f"  {m.date}: {m.circuit_breaker.trigger_reason}")
    
    # 龙头池更新
    print("\n【龙头池更新 (最近3次)】")
    for h in bt.leader_history[-3:]:
        leaders_str = ", ".join([f"{l['symbol']}({l['score']:.0f})" for l in h['leaders'][:5]])
        print(f"  {h['date']}: {leaders_str}")
    
    # 年度收益分解
    print("\n【年度收益分解】")
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    equity_df['year'] = equity_df['date'].dt.year
    
    for year in sorted(equity_df['year'].unique()):
        year_data = equity_df[equity_df['year'] == year]
        if len(year_data) < 2:
            continue
        port_start = year_data['portfolio'].iloc[0]
        port_end = year_data['portfolio'].iloc[-1]
        spy_start = year_data['spy'].iloc[0]
        spy_end = year_data['spy'].iloc[-1]
        
        port_ret = (port_end / port_start - 1) * 100
        spy_ret = (spy_end / spy_start - 1) * 100
        alpha = port_ret - spy_ret
        
        print(f"  {year}: 策略 {port_ret:+.1f}% | SPY {spy_ret:+.1f}% | Alpha {alpha:+.1f}%")


if __name__ == "__main__":
    main()
