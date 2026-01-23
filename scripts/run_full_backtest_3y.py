#!/usr/bin/env python3
"""
3年完整日度回测脚本 - 整合新闻/板块/个股数据

功能:
1. 从 PostgreSQL 读取 3 年历史数据 (2023-01 ~ 2026-01)
2. 整合新闻情绪、板块轮动、个股技术指标
3. 市场状态识别 + 自适应策略切换
4. 动态选股 (Core/Rotation/Candidate)
5. 完整日度模拟交易

使用方法:
    PYTHONPATH=. python scripts/run_full_backtest_3y.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class TradeRecord:
    """交易记录"""
    date: str
    symbol: str
    action: str  # BUY, SELL
    price: float
    shares: int
    reason: str
    regime: str
    strategy: str
    pool: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    news_sentiment: float = 0.0


@dataclass
class Position:
    """持仓"""
    symbol: str
    shares: int
    avg_cost: float
    entry_date: str
    stop_loss: float
    take_profit: float
    pool: str = ""
    
    @property
    def market_value(self) -> float:
        return self.shares * self.avg_cost


@dataclass
class DailySnapshot:
    """每日快照"""
    date: str
    portfolio_value: float
    cash: float
    positions_value: float
    regime: str
    strategy: str
    drawdown: float
    spy_value: float
    news_sentiment: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    spy_return: float
    alpha: float
    beta: float
    trades: List[TradeRecord] = field(default_factory=list)
    daily_snapshots: List[DailySnapshot] = field(default_factory=list)
    regime_distribution: Dict[str, int] = field(default_factory=dict)
    pool_distribution: Dict[str, int] = field(default_factory=dict)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    sector_performance: Dict[str, float] = field(default_factory=dict)


# ============================================================
# 市场状态识别
# ============================================================

class MarketRegime:
    BULL_TREND = "bull_trend"
    BULL_PULLBACK = "bull_pullback"
    RANGE_BOUND = "range_bound"
    BEAR_RALLY = "bear_rally"
    BEAR_TREND = "bear_trend"


class MarketRegimeDetector:
    """市场状态识别器"""
    
    def __init__(self):
        self._prev_regime = MarketRegime.RANGE_BOUND
        self._regime_days = 0
    
    def detect(
        self,
        spy_prices: pd.DataFrame,
        vix_close: Optional[float] = None,
        trade_date: Optional[date] = None,
    ) -> str:
        """检测市场状态 - 优化版"""
        if spy_prices.empty or len(spy_prices) < 20:
            return MarketRegime.RANGE_BOUND
        
        close = spy_prices['close'].values
        
        # 计算指标
        sma_20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
        sma_50 = np.mean(close[-50:]) if len(close) >= 50 else sma_20
        current_price = close[-1]
        
        # 动量计算
        momentum_20d = (current_price / close[-20] - 1) if len(close) >= 20 else 0
        momentum_5d = (current_price / close[-5] - 1) if len(close) >= 5 else 0
        momentum_10d = (current_price / close[-10] - 1) if len(close) >= 10 else 0
        
        # 趋势强度
        above_sma20 = current_price > sma_20
        above_sma50 = current_price > sma_50
        sma20_above_sma50 = sma_20 > sma_50
        
        # VIX 水平 (如果有)
        vix_high = vix_close is not None and vix_close > 25
        vix_extreme = vix_close is not None and vix_close > 35
        vix_low = vix_close is not None and vix_close < 15
        
        # 波动率 (20日标准差)
        volatility = np.std(close[-20:]) / sma_20 if len(close) >= 20 else 0.02
        
        # 状态判断 - 放宽条件
        if vix_extreme or (momentum_20d < -0.08 and not above_sma50):
            # 极端恐慌或深度下跌
            regime = MarketRegime.BEAR_TREND
        elif above_sma20 and above_sma50 and sma20_above_sma50:
            # 价格在均线之上，均线多头排列
            if momentum_20d > 0.03:
                regime = MarketRegime.BULL_TREND
            elif momentum_5d < -0.015:
                regime = MarketRegime.BULL_PULLBACK
            else:
                regime = MarketRegime.BULL_TREND
        elif not above_sma20 and not above_sma50 and not sma20_above_sma50:
            # 价格在均线之下，均线空头排列
            if momentum_5d > 0.02:
                regime = MarketRegime.BEAR_RALLY
            else:
                regime = MarketRegime.BEAR_TREND
        elif above_sma20 and momentum_10d > 0:
            # 短期强势
            regime = MarketRegime.BULL_PULLBACK
        elif not above_sma20 and momentum_10d < -0.02:
            # 短期弱势
            regime = MarketRegime.BEAR_RALLY
        else:
            # 震荡市
            regime = MarketRegime.RANGE_BOUND
        
        # 状态平滑 - 避免频繁切换 (降低到1天)
        if regime != self._prev_regime:
            self._regime_days = 0
        else:
            self._regime_days += 1
        
        # 需要连续 1 天才确认状态变化
        if self._regime_days < 1 and regime != self._prev_regime:
            return self._prev_regime
        
        self._prev_regime = regime
        return regime


# ============================================================
# 策略配置
# ============================================================

STRATEGY_PARAMS = {
    MarketRegime.BULL_TREND: {
        "name": "趋势跟踪",
        "max_exposure": 0.95,
        "position_size": 0.18,
        "stop_loss_atr": 2.0,
        "take_profit_atr": 0,
        "min_momentum": 0.01,
        "prefer_sectors": ["XLK", "XLC", "XLY"],
    },
    MarketRegime.BULL_PULLBACK: {
        "name": "回调买入",
        "max_exposure": 0.85,
        "position_size": 0.15,
        "stop_loss_atr": 1.8,
        "take_profit_atr": 4.0,
        "min_momentum": -0.01,
        "prefer_sectors": ["XLK", "XLV", "XLF"],
    },
    MarketRegime.RANGE_BOUND: {
        "name": "均值回归",
        "max_exposure": 0.60,
        "position_size": 0.12,
        "stop_loss_atr": 1.5,
        "take_profit_atr": 2.0,
        "min_momentum": -0.02,
        "prefer_sectors": ["XLP", "XLV", "XLU"],
    },
    MarketRegime.BEAR_RALLY: {
        "name": "熊市反弹",
        "max_exposure": 0.40,
        "position_size": 0.10,
        "stop_loss_atr": 1.2,
        "take_profit_atr": 1.5,
        "min_momentum": 0.01,
        "prefer_sectors": ["XLP", "XLV", "XLU"],
    },
    MarketRegime.BEAR_TREND: {
        "name": "防御保守",
        "max_exposure": 0.30,
        "position_size": 0.08,
        "stop_loss_atr": 1.0,
        "take_profit_atr": 1.2,
        "min_momentum": 0.0,
        "prefer_sectors": ["XLP", "XLV", "XLU"],
    },
}


# ============================================================
# 主回测类
# ============================================================

class FullBacktester:
    """3年完整回测引擎"""
    
    CORE_SYMBOLS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD"]
    SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLB", "XLU", "XLRE"]
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_positions: int = 8,
        rebalance_days: int = 5,
    ):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.rebalance_days = rebalance_days
        
        self.conn = self._get_db_connection()
        self.regime_detector = MarketRegimeDetector()
        
        # 缓存
        self._prices_cache: Dict[str, pd.DataFrame] = {}
        self._indicators_cache: Dict[str, pd.DataFrame] = {}
        self._news_cache: Dict[str, Dict[str, float]] = {}
        self._all_symbols: List[str] = []
        
        # 状态
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[TradeRecord] = []
        self.daily_snapshots: List[DailySnapshot] = []
        self.peak_value = initial_capital
    
    def _get_db_connection(self):
        return psycopg2.connect(
            host=os.getenv("PG_HOST", "192.168.10.11"),
            port=os.getenv("PG_PORT", "5432"),
            database=os.getenv("PG_DATABASE", "trader"),
            user=os.getenv("PG_USER", "trader"),
            password=os.getenv("PG_PASSWORD", "")
        )
    
    def _load_all_prices(self, start_date: date, end_date: date):
        """加载所有价格数据到缓存"""
        print("  加载价格数据...")
        
        query = """
            SELECT symbol, trade_date, open, high, low, close, adj_close, volume
            FROM daily_prices
            WHERE trade_date BETWEEN %s AND %s
            ORDER BY symbol, trade_date
        """
        
        df = pd.read_sql(query, self.conn, params=(start_date, end_date))
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            symbol_df.set_index('trade_date', inplace=True)
            self._prices_cache[symbol] = symbol_df
            self._all_symbols.append(symbol)
        
        print(f"    已加载 {len(self._prices_cache)} 只股票价格数据")
    
    def _load_all_indicators(self, start_date: date, end_date: date):
        """加载所有技术指标到缓存"""
        print("  加载技术指标...")
        
        query = """
            SELECT symbol, trade_date, rsi_14, macd, macd_signal, atr_14,
                   sma_20, sma_50, sma_200, momentum_10d, volume_ratio
            FROM indicators
            WHERE trade_date BETWEEN %s AND %s
            ORDER BY symbol, trade_date
        """
        
        df = pd.read_sql(query, self.conn, params=(start_date, end_date))
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            symbol_df.set_index('trade_date', inplace=True)
            self._indicators_cache[symbol] = symbol_df
        
        print(f"    已加载 {len(self._indicators_cache)} 只股票技术指标")
    
    def _load_news_sentiment(self):
        """加载新闻情绪数据"""
        print("  加载新闻情绪...")
        
        query = """
            SELECT symbol, DATE(published_at) as news_date, 
                   AVG(sentiment_score) as avg_sentiment,
                   COUNT(*) as news_count
            FROM news
            GROUP BY symbol, DATE(published_at)
        """
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()
        
        for row in rows:
            symbol = row['symbol']
            news_date = str(row['news_date'])
            if symbol not in self._news_cache:
                self._news_cache[symbol] = {}
            self._news_cache[symbol][news_date] = float(row['avg_sentiment'] or 0)
        
        print(f"    已加载 {len(self._news_cache)} 只股票新闻情绪")
    
    def _get_price(self, symbol: str, trade_date: date) -> Optional[float]:
        """获取收盘价"""
        if symbol not in self._prices_cache:
            return None
        df = self._prices_cache[symbol]
        if trade_date in df.index:
            return float(df.loc[trade_date, 'close'])
        # 找最近的日期
        valid_dates = df.index[df.index <= trade_date]
        if len(valid_dates) > 0:
            return float(df.loc[valid_dates[-1], 'close'])
        return None
    
    def _get_atr(self, symbol: str, trade_date: date) -> Optional[float]:
        """获取 ATR"""
        if symbol not in self._indicators_cache:
            return None
        df = self._indicators_cache[symbol]
        if trade_date in df.index:
            return float(df.loc[trade_date, 'atr_14'] or 0)
        valid_dates = df.index[df.index <= trade_date]
        if len(valid_dates) > 0:
            return float(df.loc[valid_dates[-1], 'atr_14'] or 0)
        return None
    
    def _get_indicator(self, symbol: str, trade_date: date, indicator: str) -> Optional[float]:
        """获取技术指标"""
        if symbol not in self._indicators_cache:
            return None
        df = self._indicators_cache[symbol]
        if trade_date in df.index and indicator in df.columns:
            val = df.loc[trade_date, indicator]
            return float(val) if pd.notna(val) else None
        valid_dates = df.index[df.index <= trade_date]
        if len(valid_dates) > 0 and indicator in df.columns:
            val = df.loc[valid_dates[-1], indicator]
            return float(val) if pd.notna(val) else None
        return None
    
    def _get_news_sentiment(self, symbol: str, trade_date: date, lookback_days: int = 7) -> float:
        """获取近期新闻情绪"""
        if symbol not in self._news_cache:
            return 0.0
        
        sentiments = []
        for i in range(lookback_days):
            check_date = str(trade_date - timedelta(days=i))
            if check_date in self._news_cache[symbol]:
                sentiments.append(self._news_cache[symbol][check_date])
        
        return np.mean(sentiments) if sentiments else 0.0
    
    def _get_spy_prices(self, end_date: date, lookback: int = 60) -> pd.DataFrame:
        """获取 SPY 价格用于状态检测"""
        if 'SPY' not in self._prices_cache:
            return pd.DataFrame()
        
        df = self._prices_cache['SPY']
        valid = df[df.index <= end_date].tail(lookback)
        return valid
    
    def _get_vix(self, trade_date: date) -> Optional[float]:
        """获取 VIX"""
        return self._get_price('VIX', trade_date)
    
    def _calc_sector_scores(self, trade_date: date) -> List[Tuple[str, float]]:
        """计算板块评分"""
        scores = []
        spy_ret_20d = self._calc_return('SPY', trade_date, 20)
        
        for etf in self.SECTOR_ETFS:
            ret_20d = self._calc_return(etf, trade_date, 20)
            ret_5d = self._calc_return(etf, trade_date, 5)
            rs_vs_spy = ret_20d - spy_ret_20d
            
            # 板块评分 = 0.5*20日收益 + 0.3*相对强度 + 0.2*5日动量
            score = 0.5 * ret_20d + 0.3 * rs_vs_spy + 0.2 * ret_5d
            scores.append((etf, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def _calc_return(self, symbol: str, trade_date: date, days: int) -> float:
        """计算N日收益率"""
        if symbol not in self._prices_cache:
            return 0.0
        
        df = self._prices_cache[symbol]
        valid = df[df.index <= trade_date]
        
        if len(valid) < days + 1:
            return 0.0
        
        current = float(valid['close'].iloc[-1])
        past = float(valid['close'].iloc[-days-1])
        
        return (current / past - 1) if past > 0 else 0.0
    
    def _select_candidates(self, trade_date: date, regime: str, strategy: dict) -> List[Tuple[str, float, str]]:
        """选择交易候选股票"""
        candidates = []
        
        # 1. Core Pool - 核心股票
        for symbol in self.CORE_SYMBOLS:
            if symbol not in self._prices_cache:
                continue
            
            momentum = self._get_indicator(symbol, trade_date, 'momentum_10d') or 0
            rsi = self._get_indicator(symbol, trade_date, 'rsi_14') or 50
            news_sentiment = self._get_news_sentiment(symbol, trade_date)
            
            # 评分: 动量 + RSI正常化 + 新闻情绪
            score = momentum * 0.4 + (50 - abs(rsi - 50)) / 50 * 0.3 + news_sentiment * 0.3
            
            if momentum >= strategy['min_momentum']:
                candidates.append((symbol, score, 'core'))
        
        # 2. 板块轮动 - 选择强势板块中的股票
        sector_scores = self._calc_sector_scores(trade_date)
        top_sectors = [s[0] for s in sector_scores[:3] if s[1] > 0]
        
        # 3. Rotation Pool - 轮动股票
        for symbol in self._all_symbols:
            if symbol in self.CORE_SYMBOLS or symbol in self.SECTOR_ETFS:
                continue
            if symbol in ['SPY', 'QQQ', 'IWM', 'DIA', 'VIX']:
                continue
            
            momentum = self._get_indicator(symbol, trade_date, 'momentum_10d') or 0
            rsi = self._get_indicator(symbol, trade_date, 'rsi_14') or 50
            volume_ratio = self._get_indicator(symbol, trade_date, 'volume_ratio') or 1
            sma_50 = self._get_indicator(symbol, trade_date, 'sma_50')
            price = self._get_price(symbol, trade_date)
            news_sentiment = self._get_news_sentiment(symbol, trade_date)
            
            # 趋势过滤
            above_sma50 = price and sma_50 and price > sma_50
            
            if regime in [MarketRegime.BULL_TREND, MarketRegime.BULL_PULLBACK]:
                # 牛市: 动量 + 放量 + 趋势
                if momentum > 0.02 and volume_ratio >= 1.0 and above_sma50:
                    score = momentum * 0.4 + (volume_ratio - 1) * 0.3 + news_sentiment * 0.3
                    candidates.append((symbol, score, 'rotation'))
                elif momentum > 0 and volume_ratio >= 0.8:
                    score = momentum * 0.3 + news_sentiment * 0.3
                    candidates.append((symbol, score, 'candidate'))
            
            elif regime == MarketRegime.RANGE_BOUND:
                # 震荡市: RSI 低位 + 趋势
                if 30 <= rsi <= 50 and above_sma50:
                    score = (50 - rsi) / 50 * 0.4 + volume_ratio * 0.3 + news_sentiment * 0.3
                    candidates.append((symbol, score, 'rotation'))
            
            else:
                # 熊市: 防御 + 低波动
                if momentum > -0.02 and rsi < 60:
                    score = (60 - rsi) / 60 * 0.3 + news_sentiment * 0.3
                    candidates.append((symbol, score, 'candidate'))
        
        # 按评分排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:20]  # 返回前20个候选
    
    def _portfolio_value(self, trade_date: date) -> float:
        """计算组合总价值"""
        positions_value = 0.0
        for symbol, pos in self.positions.items():
            price = self._get_price(symbol, trade_date)
            if price:
                positions_value += pos.shares * price
        return self.cash + positions_value
    
    def _current_exposure(self, trade_date: date) -> float:
        """计算当前仓位比例"""
        total_value = self._portfolio_value(trade_date)
        if total_value <= 0:
            return 0.0
        return (total_value - self.cash) / total_value
    
    def _execute_buy(
        self,
        symbol: str,
        trade_date: date,
        price: float,
        budget: float,
        regime: str,
        strategy: dict,
        pool: str,
        reason: str,
    ) -> Optional[TradeRecord]:
        """执行买入"""
        if budget < 500:  # 最小交易金额
            return None
        
        shares = int(budget / price)
        if shares <= 0:
            return None
        
        cost = shares * price
        if cost > self.cash:
            shares = int(self.cash / price)
            cost = shares * price
        
        if shares <= 0:
            return None
        
        # 计算止损止盈
        atr = self._get_atr(symbol, trade_date) or price * 0.02
        stop_loss = price - strategy['stop_loss_atr'] * atr
        take_profit = price + strategy['take_profit_atr'] * atr if strategy['take_profit_atr'] > 0 else price * 2
        
        # 更新持仓
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_shares = pos.shares + shares
            pos.avg_cost = (pos.avg_cost * pos.shares + price * shares) / total_shares
            pos.shares = total_shares
            pos.stop_loss = stop_loss
            pos.take_profit = take_profit
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=shares,
                avg_cost=price,
                entry_date=str(trade_date),
                stop_loss=stop_loss,
                take_profit=take_profit,
                pool=pool,
            )
        
        self.cash -= cost
        
        trade = TradeRecord(
            date=str(trade_date),
            symbol=symbol,
            action="BUY",
            price=price,
            shares=shares,
            reason=reason,
            regime=regime,
            strategy=strategy['name'],
            pool=pool,
            news_sentiment=self._get_news_sentiment(symbol, trade_date),
        )
        self.trades.append(trade)
        return trade
    
    def _execute_sell(
        self,
        symbol: str,
        trade_date: date,
        price: float,
        shares: int,
        regime: str,
        strategy: dict,
        reason: str,
    ) -> Optional[TradeRecord]:
        """执行卖出"""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        sell_shares = min(shares, pos.shares)
        
        proceeds = sell_shares * price
        cost_basis = sell_shares * pos.avg_cost
        pnl = proceeds - cost_basis
        pnl_pct = pnl / cost_basis if cost_basis > 0 else 0
        
        self.cash += proceeds
        
        if sell_shares >= pos.shares:
            del self.positions[symbol]
        else:
            pos.shares -= sell_shares
        
        trade = TradeRecord(
            date=str(trade_date),
            symbol=symbol,
            action="SELL",
            price=price,
            shares=sell_shares,
            reason=reason,
            regime=regime,
            strategy=strategy['name'],
            pool=pos.pool,
            pnl=pnl,
            pnl_pct=pnl_pct,
            news_sentiment=self._get_news_sentiment(symbol, trade_date),
        )
        self.trades.append(trade)
        return trade
    
    def _check_exits(self, trade_date: date, regime: str, strategy: dict):
        """检查止损止盈"""
        symbols_to_exit = []
        
        for symbol, pos in self.positions.items():
            price = self._get_price(symbol, trade_date)
            if not price:
                continue
            
            # 止损
            if price <= pos.stop_loss:
                symbols_to_exit.append((symbol, price, "止损"))
            # 止盈
            elif price >= pos.take_profit and strategy['take_profit_atr'] > 0:
                symbols_to_exit.append((symbol, price, "止盈"))
            # 负面新闻
            elif self._get_news_sentiment(symbol, trade_date) < -0.5:
                symbols_to_exit.append((symbol, price, "负面新闻"))
        
        for symbol, price, reason in symbols_to_exit:
            self._execute_sell(
                symbol=symbol,
                trade_date=trade_date,
                price=price,
                shares=self.positions[symbol].shares,
                regime=regime,
                strategy=strategy,
                reason=reason,
            )
    
    def _rebalance(self, trade_date: date, regime: str, strategy: dict, candidates: List[Tuple[str, float, str]]):
        """再平衡组合"""
        current_exposure = self._current_exposure(trade_date)
        target_exposure = strategy['max_exposure']
        
        # 如果仓位过高，减仓
        if current_exposure > target_exposure + 0.1:
            excess = current_exposure - target_exposure
            total_value = self._portfolio_value(trade_date)
            reduce_amount = excess * total_value
            
            # 按盈亏排序，优先卖出亏损仓位
            positions_by_pnl = []
            for symbol, pos in self.positions.items():
                price = self._get_price(symbol, trade_date)
                if price:
                    pnl_pct = (price - pos.avg_cost) / pos.avg_cost
                    positions_by_pnl.append((symbol, pnl_pct, price))
            
            positions_by_pnl.sort(key=lambda x: x[1])
            
            for symbol, pnl_pct, price in positions_by_pnl:
                if reduce_amount <= 0:
                    break
                pos = self.positions.get(symbol)
                if not pos:
                    continue
                
                sell_value = min(pos.shares * price, reduce_amount)
                sell_shares = int(sell_value / price)
                
                if sell_shares > 0:
                    self._execute_sell(
                        symbol=symbol,
                        trade_date=trade_date,
                        price=price,
                        shares=sell_shares,
                        regime=regime,
                        strategy=strategy,
                        reason="减仓调整",
                    )
                    reduce_amount -= sell_shares * price
        
        # 如果仓位过低，加仓
        elif current_exposure < target_exposure - 0.1:
            available_budget = self.cash * 0.9  # 保留10%现金
            position_budget = self._portfolio_value(trade_date) * strategy['position_size']
            
            for symbol, score, pool in candidates:
                if len(self.positions) >= self.max_positions:
                    break
                if symbol in self.positions:
                    continue
                if available_budget < position_budget * 0.5:
                    break
                
                price = self._get_price(symbol, trade_date)
                if not price:
                    continue
                
                # 买入
                budget = min(position_budget, available_budget)
                trade = self._execute_buy(
                    symbol=symbol,
                    trade_date=trade_date,
                    price=price,
                    budget=budget,
                    regime=regime,
                    strategy=strategy,
                    pool=pool,
                    reason=f"新建仓位 (评分:{score:.3f})",
                )
                
                if trade:
                    available_budget -= trade.shares * trade.price
    
    def run(self, start_date: date, end_date: date) -> BacktestResult:
        """运行完整回测"""
        print("\n" + "=" * 70)
        print("3年完整日度回测")
        print("=" * 70)
        
        # 加载数据
        print("\n【1. 加载数据】")
        self._load_all_prices(start_date, end_date)
        self._load_all_indicators(start_date, end_date)
        self._load_news_sentiment()
        
        # 获取交易日列表
        if 'SPY' not in self._prices_cache:
            raise ValueError("SPY 数据缺失")
        
        trading_days = sorted(self._prices_cache['SPY'].index.tolist())
        trading_days = [d for d in trading_days if start_date <= d <= end_date]
        
        print(f"\n【2. 开始回测】")
        print(f"  回测区间: {start_date} ~ {end_date}")
        print(f"  交易日数: {len(trading_days)}")
        print(f"  初始资金: ${self.initial_capital:,.0f}")
        
        # 统计
        regime_counts = {}
        rebalance_counter = 0
        
        # 遍历每个交易日
        for i, trade_date in enumerate(trading_days):
            # 进度显示
            if i % 50 == 0:
                pv = self._portfolio_value(trade_date)
                print(f"  [{i+1}/{len(trading_days)}] {trade_date} - 组合价值: ${pv:,.0f}")
            
            # 1. 识别市场状态
            spy_prices = self._get_spy_prices(trade_date)
            vix = self._get_vix(trade_date)
            regime = self.regime_detector.detect(spy_prices, vix, trade_date)
            strategy = STRATEGY_PARAMS[regime]
            
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # 2. 检查止损止盈
            self._check_exits(trade_date, regime, strategy)
            
            # 3. 定期再平衡
            rebalance_counter += 1
            if rebalance_counter >= self.rebalance_days:
                rebalance_counter = 0
                candidates = self._select_candidates(trade_date, regime, strategy)
                self._rebalance(trade_date, regime, strategy, candidates)
            
            # 4. 记录每日快照
            pv = self._portfolio_value(trade_date)
            self.peak_value = max(self.peak_value, pv)
            drawdown = (self.peak_value - pv) / self.peak_value if self.peak_value > 0 else 0
            
            spy_price = self._get_price('SPY', trade_date) or 0
            spy_base = self._get_price('SPY', start_date) or 1
            spy_value = self.initial_capital * (spy_price / spy_base)
            
            avg_sentiment = np.mean([
                self._get_news_sentiment(s, trade_date)
                for s in self.positions.keys()
            ]) if self.positions else 0
            
            self.daily_snapshots.append(DailySnapshot(
                date=str(trade_date),
                portfolio_value=pv,
                cash=self.cash,
                positions_value=pv - self.cash,
                regime=regime,
                strategy=strategy['name'],
                drawdown=drawdown,
                spy_value=spy_value,
                news_sentiment=avg_sentiment,
            ))
        
        # 计算最终结果
        print("\n【3. 计算回测指标】")
        return self._calculate_results(start_date, end_date, regime_counts)
    
    def _calculate_results(self, start_date: date, end_date: date, regime_counts: dict) -> BacktestResult:
        """计算回测结果"""
        if not self.daily_snapshots:
            raise ValueError("无交易日数据")
        
        # 基础指标
        final_value = self.daily_snapshots[-1].portfolio_value
        total_return = (final_value / self.initial_capital - 1)
        
        # 年化收益
        days = (end_date - start_date).days
        years = days / 365.0
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 最大回撤
        max_drawdown = max(s.drawdown for s in self.daily_snapshots)
        
        # 收益率序列
        values = [s.portfolio_value for s in self.daily_snapshots]
        returns = pd.Series(values).pct_change().dropna()
        
        # 夏普比率
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0
        
        # 索提诺比率
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino_ratio = np.sqrt(252) * returns.mean() / negative_returns.std()
        else:
            sortino_ratio = sharpe_ratio
        
        # 胜率和盈亏比
        sell_trades = [t for t in self.trades if t.action == "SELL"]
        wins = [t for t in sell_trades if t.pnl > 0]
        losses = [t for t in sell_trades if t.pnl < 0]
        
        win_rate = len(wins) / len(sell_trades) if sell_trades else 0
        
        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        avg_trade_pnl = sum(t.pnl for t in sell_trades) / len(sell_trades) if sell_trades else 0
        
        # SPY 收益
        spy_return = (self.daily_snapshots[-1].spy_value / self.initial_capital - 1)
        alpha = total_return - spy_return
        
        # Beta
        spy_returns = pd.Series([s.spy_value for s in self.daily_snapshots]).pct_change().dropna()
        if len(returns) == len(spy_returns) and spy_returns.var() > 0:
            beta = returns.cov(spy_returns) / spy_returns.var()
        else:
            beta = 1.0
        
        # 池分布
        pool_distribution = {}
        for t in self.trades:
            if t.action == "BUY":
                pool_distribution[t.pool] = pool_distribution.get(t.pool, 0) + 1
        
        # 月度收益
        monthly_returns = {}
        for snapshot in self.daily_snapshots:
            month = snapshot.date[:7]
            monthly_returns[month] = snapshot.portfolio_value
        
        # 转换为收益率
        months = sorted(monthly_returns.keys())
        monthly_rets = {}
        for i, month in enumerate(months):
            if i == 0:
                monthly_rets[month] = monthly_returns[month] / self.initial_capital - 1
            else:
                monthly_rets[month] = monthly_returns[month] / monthly_returns[months[i-1]] - 1
        
        # 板块表现
        sector_performance = {}
        for etf in self.SECTOR_ETFS:
            start_price = self._get_price(etf, start_date)
            end_price = self._get_price(etf, end_date)
            if start_price and end_price:
                sector_performance[etf] = end_price / start_price - 1
        
        return BacktestResult(
            start_date=str(start_date),
            end_date=str(end_date),
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            avg_trade_pnl=avg_trade_pnl,
            spy_return=spy_return,
            alpha=alpha,
            beta=beta,
            trades=self.trades,
            daily_snapshots=self.daily_snapshots,
            regime_distribution=regime_counts,
            pool_distribution=pool_distribution,
            monthly_returns=monthly_rets,
            sector_performance=sector_performance,
        )


def print_results(result: BacktestResult):
    """打印回测结果"""
    print("\n" + "=" * 70)
    print("回测结果摘要")
    print("=" * 70)
    
    print(f"\n【基础信息】")
    print(f"  回测区间: {result.start_date} ~ {result.end_date}")
    print(f"  初始资金: ${result.initial_capital:,.0f}")
    print(f"  最终价值: ${result.final_value:,.0f}")
    
    print(f"\n【收益指标】")
    print(f"  总收益率: {result.total_return:+.2%}")
    print(f"  年化收益: {result.annualized_return:+.2%}")
    print(f"  SPY收益:  {result.spy_return:+.2%}")
    print(f"  超额收益: {result.alpha:+.2%}")
    
    print(f"\n【风险指标】")
    print(f"  最大回撤: {result.max_drawdown:.2%}")
    print(f"  夏普比率: {result.sharpe_ratio:.2f}")
    print(f"  索提诺比率: {result.sortino_ratio:.2f}")
    print(f"  Beta: {result.beta:.2f}")
    
    print(f"\n【交易统计】")
    print(f"  总交易次数: {result.total_trades}")
    print(f"  胜率: {result.win_rate:.1%}")
    print(f"  盈亏比: {result.profit_factor:.2f}")
    print(f"  平均每笔盈亏: ${result.avg_trade_pnl:,.0f}")
    
    print(f"\n【市场状态分布】")
    total_days = sum(result.regime_distribution.values())
    for regime, count in sorted(result.regime_distribution.items(), key=lambda x: -x[1]):
        pct = count / total_days * 100
        print(f"  {regime}: {count} 天 ({pct:.1f}%)")
    
    print(f"\n【交易来源分布】")
    total_buys = sum(result.pool_distribution.values())
    for pool, count in sorted(result.pool_distribution.items(), key=lambda x: -x[1]):
        pct = count / total_buys * 100 if total_buys > 0 else 0
        print(f"  {pool}: {count} 笔 ({pct:.1f}%)")
    
    print(f"\n【板块表现】")
    sorted_sectors = sorted(result.sector_performance.items(), key=lambda x: -x[1])
    for etf, ret in sorted_sectors[:5]:
        print(f"  {etf}: {ret:+.1%}")
    print("  ...")
    for etf, ret in sorted_sectors[-3:]:
        print(f"  {etf}: {ret:+.1%}")
    
    # 最大盈利交易
    sell_trades = [t for t in result.trades if t.action == "SELL" and t.pnl != 0]
    if sell_trades:
        print(f"\n【最大盈利交易 Top 5】")
        top_wins = sorted(sell_trades, key=lambda x: -x.pnl)[:5]
        for t in top_wins:
            print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
        
        print(f"\n【最大亏损交易 Top 5】")
        top_losses = sorted(sell_trades, key=lambda x: x.pnl)[:5]
        for t in top_losses:
            print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")


def save_results(result: BacktestResult, output_dir: Path):
    """保存回测结果"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存汇总 JSON
    summary = {
        "start_date": result.start_date,
        "end_date": result.end_date,
        "initial_capital": result.initial_capital,
        "final_value": result.final_value,
        "total_return": result.total_return,
        "annualized_return": result.annualized_return,
        "max_drawdown": result.max_drawdown,
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "total_trades": result.total_trades,
        "avg_trade_pnl": result.avg_trade_pnl,
        "spy_return": result.spy_return,
        "alpha": result.alpha,
        "beta": result.beta,
        "regime_distribution": result.regime_distribution,
        "pool_distribution": result.pool_distribution,
        "sector_performance": result.sector_performance,
    }
    
    with open(output_dir / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # 保存交易记录
    trades_data = [
        {
            "date": t.date,
            "symbol": t.symbol,
            "action": t.action,
            "price": t.price,
            "shares": t.shares,
            "reason": t.reason,
            "regime": t.regime,
            "strategy": t.strategy,
            "pool": t.pool,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "news_sentiment": t.news_sentiment,
        }
        for t in result.trades
    ]
    
    with open(output_dir / "trades.json", "w") as f:
        json.dump(trades_data, f, indent=2)
    
    # 保存每日快照
    snapshots_data = [
        {
            "date": s.date,
            "portfolio_value": s.portfolio_value,
            "cash": s.cash,
            "positions_value": s.positions_value,
            "regime": s.regime,
            "strategy": s.strategy,
            "drawdown": s.drawdown,
            "spy_value": s.spy_value,
            "news_sentiment": s.news_sentiment,
        }
        for s in result.daily_snapshots
    ]
    
    with open(output_dir / "daily_snapshots.json", "w") as f:
        json.dump(snapshots_data, f, indent=2)
    
    # 保存月度收益
    with open(output_dir / "monthly_returns.json", "w") as f:
        json.dump(result.monthly_returns, f, indent=2)
    
    print(f"\n📁 结果已保存到: {output_dir}")


def main():
    # 回测参数
    start_date = date(2023, 1, 3)  # 2023年1月3日开始
    end_date = date(2026, 1, 16)   # 2026年1月16日结束
    initial_capital = 100000.0
    
    # 创建回测器
    backtester = FullBacktester(
        initial_capital=initial_capital,
        max_positions=8,
        rebalance_days=5,
    )
    
    # 运行回测
    result = backtester.run(start_date, end_date)
    
    # 打印结果
    print_results(result)
    
    # 保存结果
    output_dir = Path("storage/backtest_3y_full")
    save_results(result, output_dir)
    
    print("\n" + "=" * 70)
    print("✅ 3年完整日度回测完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
