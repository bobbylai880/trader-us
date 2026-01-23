#!/usr/bin/env python3
"""
3年完整日度回测脚本 V2 - 激进动量策略

核心改进:
1. 简化市场状态识别 - 基于 SPY 趋势
2. 高仓位运行 - 牛市 95%, 震荡 80%
3. 聚焦强势板块 - XLK/XLC/XLY 优先
4. 宽松止损 + 禁用止盈 - 让利润奔跑
5. 动量选股 - 20日动量 > 5% 优先
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TradeRecord:
    date: str
    symbol: str
    action: str
    price: float
    shares: int
    reason: str
    regime: str
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class Position:
    symbol: str
    shares: int
    avg_cost: float
    entry_date: str
    stop_loss: float
    highest_price: float = 0.0


@dataclass
class DailySnapshot:
    date: str
    portfolio_value: float
    cash: float
    positions_count: int
    regime: str
    drawdown: float
    spy_value: float


class MomentumBacktester:
    
    CORE_SYMBOLS = ["NVDA", "META", "GOOGL", "AMZN", "MSFT", "AAPL", "AMD", "AVGO", "NFLX", "TSLA"]
    SECTOR_ETFS = ["XLK", "XLC", "XLY", "XLF", "XLV", "XLE", "XLI", "XLP", "XLB", "XLU", "XLRE"]
    
    def __init__(self, initial_capital: float = 100000.0, max_positions: int = 10):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.conn = self._get_db_connection()
        
        self._prices_cache: Dict[str, pd.DataFrame] = {}
        self._indicators_cache: Dict[str, pd.DataFrame] = {}
        self._all_symbols: List[str] = []
        
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
    
    def _load_data(self, start_date: date, end_date: date):
        print("  加载数据...")
        
        query = """
            SELECT symbol, trade_date, open, high, low, close, volume
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
        
        query2 = """
            SELECT symbol, trade_date, rsi_14, atr_14, sma_20, sma_50, momentum_10d
            FROM indicators
            WHERE trade_date BETWEEN %s AND %s
        """
        df2 = pd.read_sql(query2, self.conn, params=(start_date, end_date))
        
        for symbol in df2['symbol'].unique():
            symbol_df = df2[df2['symbol'] == symbol].copy()
            symbol_df.set_index('trade_date', inplace=True)
            self._indicators_cache[symbol] = symbol_df
        
        print(f"    已加载 {len(self._prices_cache)} 只股票")
    
    def _get_price(self, symbol: str, trade_date: date) -> Optional[float]:
        if symbol not in self._prices_cache:
            return None
        df = self._prices_cache[symbol]
        valid = df[df.index <= trade_date]
        if len(valid) > 0:
            return float(valid['close'].iloc[-1])
        return None
    
    def _get_atr(self, symbol: str, trade_date: date) -> float:
        if symbol not in self._indicators_cache:
            price = self._get_price(symbol, trade_date)
            return price * 0.03 if price else 1.0
        df = self._indicators_cache[symbol]
        valid = df[df.index <= trade_date]
        if len(valid) > 0:
            val = valid['atr_14'].iloc[-1]
            return float(val) if pd.notna(val) else 1.0
        return 1.0
    
    def _calc_momentum(self, symbol: str, trade_date: date, days: int = 20) -> float:
        if symbol not in self._prices_cache:
            return 0.0
        df = self._prices_cache[symbol]
        valid = df[df.index <= trade_date]
        if len(valid) < days + 1:
            return 0.0
        current = float(valid['close'].iloc[-1])
        past = float(valid['close'].iloc[-days-1])
        return (current / past - 1) if past > 0 else 0.0
    
    def _detect_regime(self, trade_date: date) -> str:
        spy_mom_20 = self._calc_momentum('SPY', trade_date, 20)
        spy_mom_50 = self._calc_momentum('SPY', trade_date, 50)
        
        if spy_mom_20 > 0.03 and spy_mom_50 > 0:
            return "bull"
        elif spy_mom_20 < -0.05 or spy_mom_50 < -0.08:
            return "bear"
        else:
            return "neutral"
    
    def _get_strategy_params(self, regime: str) -> dict:
        if regime == "bull":
            return {"max_exposure": 0.95, "position_pct": 0.15, "stop_atr": 3.0, "trail_atr": 2.5}
        elif regime == "bear":
            return {"max_exposure": 0.40, "position_pct": 0.08, "stop_atr": 1.5, "trail_atr": 1.0}
        else:
            return {"max_exposure": 0.80, "position_pct": 0.12, "stop_atr": 2.5, "trail_atr": 2.0}
    
    def _rank_candidates(self, trade_date: date, regime: str) -> List[Tuple[str, float]]:
        candidates = []
        
        for symbol in self.CORE_SYMBOLS:
            if symbol not in self._prices_cache:
                continue
            mom_20 = self._calc_momentum(symbol, trade_date, 20)
            mom_5 = self._calc_momentum(symbol, trade_date, 5)
            
            if regime == "bull":
                if mom_20 > 0.03:
                    score = mom_20 * 0.6 + mom_5 * 0.4
                    candidates.append((symbol, score))
            elif regime == "bear":
                if mom_20 > -0.02 and mom_5 > 0:
                    score = mom_5 * 0.7 + (0.1 - abs(mom_20)) * 0.3
                    candidates.append((symbol, score))
            else:
                if mom_20 > 0:
                    score = mom_20 * 0.5 + mom_5 * 0.5
                    candidates.append((symbol, score))
        
        for symbol in self._all_symbols:
            if symbol in self.CORE_SYMBOLS or symbol in self.SECTOR_ETFS:
                continue
            if symbol in ['SPY', 'QQQ', 'IWM', 'DIA', 'VIX']:
                continue
            
            mom_20 = self._calc_momentum(symbol, trade_date, 20)
            mom_5 = self._calc_momentum(symbol, trade_date, 5)
            
            if regime == "bull" and mom_20 > 0.08:
                score = mom_20 * 0.6 + mom_5 * 0.4
                candidates.append((symbol, score))
            elif regime == "neutral" and mom_20 > 0.05:
                score = mom_20 * 0.5 + mom_5 * 0.5
                candidates.append((symbol, score))
        
        candidates.sort(key=lambda x: -x[1])
        return candidates[:15]
    
    def _portfolio_value(self, trade_date: date) -> float:
        pos_value = sum(
            pos.shares * (self._get_price(sym, trade_date) or pos.avg_cost)
            for sym, pos in self.positions.items()
        )
        return self.cash + pos_value
    
    def _execute_buy(self, symbol: str, trade_date: date, price: float, budget: float, regime: str, reason: str) -> bool:
        if budget < 500:
            return False
        
        shares = int(budget / price)
        if shares <= 0:
            return False
        
        cost = shares * price
        if cost > self.cash:
            shares = int(self.cash / price)
            cost = shares * price
        
        if shares <= 0:
            return False
        
        params = self._get_strategy_params(regime)
        atr = self._get_atr(symbol, trade_date)
        stop_loss = price - params['stop_atr'] * atr
        
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_shares = pos.shares + shares
            pos.avg_cost = (pos.avg_cost * pos.shares + price * shares) / total_shares
            pos.shares = total_shares
            pos.highest_price = max(pos.highest_price, price)
        else:
            self.positions[symbol] = Position(
                symbol=symbol, shares=shares, avg_cost=price,
                entry_date=str(trade_date), stop_loss=stop_loss, highest_price=price
            )
        
        self.cash -= cost
        self.trades.append(TradeRecord(
            date=str(trade_date), symbol=symbol, action="BUY",
            price=price, shares=shares, reason=reason, regime=regime
        ))
        return True
    
    def _execute_sell(self, symbol: str, trade_date: date, price: float, reason: str, regime: str) -> float:
        if symbol not in self.positions:
            return 0.0
        
        pos = self.positions[symbol]
        proceeds = pos.shares * price
        cost_basis = pos.shares * pos.avg_cost
        pnl = proceeds - cost_basis
        pnl_pct = pnl / cost_basis if cost_basis > 0 else 0
        
        self.cash += proceeds
        del self.positions[symbol]
        
        self.trades.append(TradeRecord(
            date=str(trade_date), symbol=symbol, action="SELL",
            price=price, shares=pos.shares, reason=reason,
            regime=regime, pnl=pnl, pnl_pct=pnl_pct
        ))
        return pnl
    
    def _check_stops(self, trade_date: date, regime: str):
        params = self._get_strategy_params(regime)
        to_sell = []
        
        for symbol, pos in self.positions.items():
            price = self._get_price(symbol, trade_date)
            if not price:
                continue
            
            pos.highest_price = max(pos.highest_price, price)
            
            atr = self._get_atr(symbol, trade_date)
            trailing_stop = pos.highest_price - params['trail_atr'] * atr
            
            effective_stop = max(pos.stop_loss, trailing_stop)
            
            if price <= effective_stop:
                to_sell.append((symbol, price, "止损"))
            elif price < pos.avg_cost * 0.85:
                to_sell.append((symbol, price, "硬止损15%"))
        
        for symbol, price, reason in to_sell:
            self._execute_sell(symbol, trade_date, price, reason, regime)
    
    def _rebalance(self, trade_date: date, regime: str, candidates: List[Tuple[str, float]]):
        params = self._get_strategy_params(regime)
        pv = self._portfolio_value(trade_date)
        current_exposure = (pv - self.cash) / pv if pv > 0 else 0
        
        if regime == "bear" and current_exposure > params['max_exposure']:
            for symbol in list(self.positions.keys()):
                mom = self._calc_momentum(symbol, trade_date, 10)
                if mom < -0.03:
                    price = self._get_price(symbol, trade_date)
                    if price:
                        self._execute_sell(symbol, trade_date, price, "熊市减仓", regime)
        
        if current_exposure < params['max_exposure'] - 0.05:
            available = self.cash * 0.95
            position_budget = pv * params['position_pct']
            
            for symbol, score in candidates:
                if len(self.positions) >= self.max_positions:
                    break
                if symbol in self.positions:
                    continue
                if available < position_budget * 0.5:
                    break
                
                price = self._get_price(symbol, trade_date)
                if not price:
                    continue
                
                budget = min(position_budget, available)
                if self._execute_buy(symbol, trade_date, price, budget, regime, f"动量买入(score:{score:.2f})"):
                    available -= budget
    
    def run(self, start_date: date, end_date: date) -> dict:
        print("\n" + "=" * 70)
        print("3年动量策略回测 V2")
        print("=" * 70)
        
        self._load_data(start_date, end_date)
        
        if 'SPY' not in self._prices_cache:
            raise ValueError("SPY 数据缺失")
        
        trading_days = sorted(self._prices_cache['SPY'].index.tolist())
        trading_days = [d for d in trading_days if start_date <= d <= end_date]
        
        print(f"\n  回测区间: {start_date} ~ {end_date}")
        print(f"  交易日数: {len(trading_days)}")
        print(f"  初始资金: ${self.initial_capital:,.0f}")
        
        regime_counts = {"bull": 0, "bear": 0, "neutral": 0}
        rebalance_counter = 0
        
        for i, trade_date in enumerate(trading_days):
            if i % 100 == 0:
                pv = self._portfolio_value(trade_date)
                print(f"  [{i+1}/{len(trading_days)}] {trade_date} - ${pv:,.0f}")
            
            regime = self._detect_regime(trade_date)
            regime_counts[regime] += 1
            
            self._check_stops(trade_date, regime)
            
            rebalance_counter += 1
            if rebalance_counter >= 5:
                rebalance_counter = 0
                candidates = self._rank_candidates(trade_date, regime)
                self._rebalance(trade_date, regime, candidates)
            
            pv = self._portfolio_value(trade_date)
            self.peak_value = max(self.peak_value, pv)
            dd = (self.peak_value - pv) / self.peak_value if self.peak_value > 0 else 0
            
            spy_price = self._get_price('SPY', trade_date) or 0
            spy_base = self._get_price('SPY', start_date) or 1
            spy_value = self.initial_capital * (spy_price / spy_base)
            
            self.daily_snapshots.append(DailySnapshot(
                date=str(trade_date), portfolio_value=pv, cash=self.cash,
                positions_count=len(self.positions), regime=regime,
                drawdown=dd, spy_value=spy_value
            ))
        
        return self._calc_results(start_date, end_date, regime_counts)
    
    def _calc_results(self, start_date: date, end_date: date, regime_counts: dict) -> dict:
        final_value = self.daily_snapshots[-1].portfolio_value
        total_return = final_value / self.initial_capital - 1
        
        days = (end_date - start_date).days
        years = days / 365.0
        ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        max_dd = max(s.drawdown for s in self.daily_snapshots)
        
        values = [s.portfolio_value for s in self.daily_snapshots]
        returns = pd.Series(values).pct_change().dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        sell_trades = [t for t in self.trades if t.action == "SELL"]
        wins = [t for t in sell_trades if t.pnl > 0]
        win_rate = len(wins) / len(sell_trades) if sell_trades else 0
        
        total_pnl = sum(t.pnl for t in sell_trades)
        avg_pnl = total_pnl / len(sell_trades) if sell_trades else 0
        
        spy_return = self.daily_snapshots[-1].spy_value / self.initial_capital - 1
        alpha = total_return - spy_return
        
        return {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "annualized_return": ann_return,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
            "total_trades": len(self.trades),
            "avg_trade_pnl": avg_pnl,
            "spy_return": spy_return,
            "alpha": alpha,
            "regime_distribution": regime_counts,
        }


def main():
    start_date = date(2023, 1, 3)
    end_date = date(2026, 1, 16)
    
    backtester = MomentumBacktester(initial_capital=100000.0, max_positions=10)
    result = backtester.run(start_date, end_date)
    
    print("\n" + "=" * 70)
    print("回测结果")
    print("=" * 70)
    print(f"\n  初始资金: ${result['initial_capital']:,.0f}")
    print(f"  最终价值: ${result['final_value']:,.0f}")
    print(f"\n  总收益率: {result['total_return']:+.2%}")
    print(f"  年化收益: {result['annualized_return']:+.2%}")
    print(f"  SPY收益:  {result['spy_return']:+.2%}")
    print(f"  超额收益: {result['alpha']:+.2%}")
    print(f"\n  最大回撤: {result['max_drawdown']:.2%}")
    print(f"  夏普比率: {result['sharpe_ratio']:.2f}")
    print(f"  胜率: {result['win_rate']:.1%}")
    print(f"  总交易: {result['total_trades']} 笔")
    print(f"  平均盈亏: ${result['avg_trade_pnl']:,.0f}")
    
    print(f"\n  市场状态分布:")
    for regime, count in result['regime_distribution'].items():
        pct = count / sum(result['regime_distribution'].values()) * 100
        print(f"    {regime}: {count} 天 ({pct:.1f}%)")
    
    output_dir = Path("storage/backtest_3y_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    trades_data = [
        {"date": t.date, "symbol": t.symbol, "action": t.action,
         "price": t.price, "shares": t.shares, "pnl": t.pnl, "reason": t.reason}
        for t in backtester.trades
    ]
    with open(output_dir / "trades.json", "w") as f:
        json.dump(trades_data, f, indent=2)
    
    print(f"\n📁 结果已保存到: {output_dir}")
    
    if backtester.trades:
        print("\n【最大盈利交易 Top 5】")
        top_wins = sorted([t for t in backtester.trades if t.action == "SELL"], key=lambda x: -x.pnl)[:5]
        for t in top_wins:
            print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
