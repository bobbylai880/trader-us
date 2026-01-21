#!/usr/bin/env python3
"""
3年回测 V5b - V3趋势跟踪 + V4风控开关

核心设计思路:
┌─────────────────────────────────────────────────────────────┐
│  V3 趋势跟踪 = 主引擎 (100% 时间运行)                         │
│  - 科技龙头聚焦                                              │
│  - 动量选股                                                  │
│  - 跟踪止损 15%                                              │
├─────────────────────────────────────────────────────────────┤
│  V4 风控开关 = 刹车系统 (只在危险时介入)                      │
│  - VIX > 30: 强制减仓至 30%                                  │
│  - VIX > 25 + SPY破位: 减仓至 50%                            │
│  - 正常情况: 完全不干预                                       │
└─────────────────────────────────────────────────────────────┘

关键区别:
- 旧 V5: V4 分层框架控制仓位权重 (offensive/neutral/defensive)
- 新 V5b: V3 全权运行, V4 只是风控熔断器

目标: 保留 V3 的高收益 (+117%), 用 V4 降低极端回撤
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

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class Position:
    symbol: str
    shares: int
    avg_cost: float
    entry_date: str
    highest_price: float


@dataclass
class Trade:
    date: str
    symbol: str
    action: str
    price: float
    shares: int
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""


@dataclass
class RiskState:
    """风控状态 - 来自 V4 的简化版"""
    date: str
    vix_level: float
    spy_below_sma50: bool
    spy_momentum: float
    risk_mode: str  # "normal", "caution", "danger"
    max_exposure: float
    trigger_reason: str


class TrendFollowingWithRiskSwitch:
    """V5b: V3趋势跟踪 + V4风控开关"""
    
    # V3 的科技龙头股票池
    UNIVERSE = ["NVDA", "META", "GOOGL", "AMZN", "MSFT", "AAPL", "AMD", "AVGO", "NFLX", "TSLA"]
    
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
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[str, float, float]] = []
        self.risk_history: List[RiskState] = []
        
        self._current_risk: Optional[RiskState] = None
    
    def _load_data(self, start: date, end: date):
        query = """
            SELECT symbol, trade_date, close
            FROM daily_prices
            WHERE trade_date BETWEEN %s AND %s
              AND symbol IN %s
            ORDER BY symbol, trade_date
        """
        symbols = tuple(self.UNIVERSE + ['SPY', 'VIX'])
        df = pd.read_sql(query, self.conn, params=(start, end, symbols))
        
        for sym in df['symbol'].unique():
            sdf = df[df['symbol'] == sym].copy()
            sdf.set_index('trade_date', inplace=True)
            sdf['sma50'] = sdf['close'].rolling(50).mean()
            sdf['sma20'] = sdf['close'].rolling(20).mean()
            sdf['mom20'] = sdf['close'].pct_change(20)
            self._prices[sym] = sdf
    
    def _get(self, sym: str, dt: date, col: str) -> Optional[float]:
        if sym not in self._prices:
            return None
        df = self._prices[sym]
        valid = df[df.index <= dt]
        if len(valid) == 0:
            return None
        val = valid[col].iloc[-1]
        return float(val) if pd.notna(val) else None
    
    # ================================================================
    # V4 风控开关 (简化版 - 只做熔断)
    # ================================================================
    
    def _check_risk_switch(self, dt: date) -> RiskState:
        """
        风控开关逻辑 - 只在极端情况下触发
        
        正常模式 (normal): 不干预, max_exposure = 95%
        警戒模式 (caution): VIX偏高或SPY破位, max_exposure = 70%
        危险模式 (danger): VIX恐慌, max_exposure = 30%
        """
        vix = self._get('VIX', dt, 'close') or 20
        spy_close = self._get('SPY', dt, 'close') or 0
        spy_sma50 = self._get('SPY', dt, 'sma50') or spy_close
        spy_mom = self._get('SPY', dt, 'mom20') or 0
        
        spy_below_sma50 = spy_close < spy_sma50
        
        # 危险模式: VIX > 30 (恐慌)
        if vix > 30:
            return RiskState(
                date=str(dt),
                vix_level=vix,
                spy_below_sma50=spy_below_sma50,
                spy_momentum=spy_mom,
                risk_mode="danger",
                max_exposure=0.30,
                trigger_reason=f"VIX恐慌({vix:.1f}>30)"
            )
        
        # 警戒模式: VIX > 25 且 SPY 破位下跌
        if vix > 25 and spy_below_sma50 and spy_mom < -0.05:
            return RiskState(
                date=str(dt),
                vix_level=vix,
                spy_below_sma50=spy_below_sma50,
                spy_momentum=spy_mom,
                risk_mode="caution",
                max_exposure=0.50,
                trigger_reason=f"VIX偏高({vix:.1f}) + SPY破位({spy_mom:.1%})"
            )
        
        # 轻度警戒: VIX > 22 且 SPY 明显下跌
        if vix > 22 and spy_mom < -0.08:
            return RiskState(
                date=str(dt),
                vix_level=vix,
                spy_below_sma50=spy_below_sma50,
                spy_momentum=spy_mom,
                risk_mode="caution",
                max_exposure=0.70,
                trigger_reason=f"市场回调({spy_mom:.1%})"
            )
        
        # 正常模式: 不干预
        return RiskState(
            date=str(dt),
            vix_level=vix,
            spy_below_sma50=spy_below_sma50,
            spy_momentum=spy_mom,
            risk_mode="normal",
            max_exposure=0.95,
            trigger_reason="正常运行"
        )
    
    # ================================================================
    # V3 趋势跟踪核心逻辑 (完整保留)
    # ================================================================
    
    def _is_bull_market(self, dt: date) -> bool:
        """V3 原版: SPY 在 SMA50 之上"""
        spy_close = self._get('SPY', dt, 'close')
        spy_sma50 = self._get('SPY', dt, 'sma50')
        spy_mom = self._get('SPY', dt, 'mom20')
        
        if spy_close is None or spy_sma50 is None:
            return False
        
        return spy_close > spy_sma50 and (spy_mom is None or spy_mom > -0.05)
    
    def _portfolio_value(self, dt: date) -> float:
        pos_val = sum(
            p.shares * (self._get(s, dt, 'close') or p.avg_cost)
            for s, p in self.positions.items()
        )
        return self.cash + pos_val
    
    def _rank_stocks(self, dt: date) -> List[Tuple[str, float]]:
        """V3 原版: 动量排名选股"""
        ranked = []
        for sym in self.UNIVERSE:
            mom = self._get(sym, dt, 'mom20')
            close = self._get(sym, dt, 'close')
            sma20 = self._get(sym, dt, 'sma20')
            
            if mom is None or close is None or sma20 is None:
                continue
            
            if close > sma20 and mom > 0:
                ranked.append((sym, mom))
        
        ranked.sort(key=lambda x: -x[1])
        return ranked[:5]
    
    def _buy(self, sym: str, dt: date, budget: float) -> bool:
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
            self.positions[sym] = Position(sym, shares, price, str(dt), price)
        
        self.trades.append(Trade(str(dt), sym, "BUY", price, shares, reason="趋势买入"))
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
        self.trades.append(Trade(str(dt), sym, "SELL", price, p.shares, pnl, pnl_pct, reason))
        del self.positions[sym]
        return pnl
    
    def _check_stops(self, dt: date, is_bull: bool):
        """V3 原版止损逻辑 + 风控开关加强"""
        for sym in list(self.positions.keys()):
            p = self.positions[sym]
            price = self._get(sym, dt, 'close')
            if not price:
                continue
            
            p.highest_price = max(p.highest_price, price)
            
            # V3 原版: 跟踪止损 15%
            drawdown = (p.highest_price - price) / p.highest_price
            if drawdown > 0.15:
                self._sell(sym, dt, f"跟踪止损({drawdown:.1%})")
                continue
            
            # V3 原版: 熊市保护
            if not is_bull and price < p.avg_cost * 0.92:
                self._sell(sym, dt, "熊市保护")
    
    def _force_reduce_exposure(self, dt: date, target_exposure: float, reason: str):
        """风控开关触发时的强制减仓"""
        pv = self._portfolio_value(dt)
        current_exposure = (pv - self.cash) / pv if pv > 0 else 0
        
        if current_exposure <= target_exposure:
            return
        
        # 按收益率排序，卖出表现最差的
        holdings = []
        for sym, pos in self.positions.items():
            price = self._get(sym, dt, 'close') or pos.avg_cost
            pnl_pct = (price - pos.avg_cost) / pos.avg_cost
            holdings.append((sym, pnl_pct))
        
        holdings.sort(key=lambda x: x[1])  # 收益最差的先卖
        
        for sym, _ in holdings:
            if current_exposure <= target_exposure:
                break
            self._sell(sym, dt, f"风控减仓: {reason}")
            pv = self._portfolio_value(dt)
            current_exposure = (pv - self.cash) / pv if pv > 0 else 0
    
    # ================================================================
    # 主运行循环
    # ================================================================
    
    def run(self, start: date, end: date) -> dict:
        print("\n" + "=" * 70)
        print("V5b 策略: V3趋势跟踪 + V4风控开关")
        print("=" * 70)
        print("  核心设计:")
        print("    - V3 趋势跟踪 = 主引擎 (100% 时间运行)")
        print("    - V4 风控开关 = 刹车系统 (只在VIX>25/30时介入)")
        print("    - 目标: 保留 V3 高收益, 用 V4 降低极端回撤")
        
        self._load_data(start - timedelta(days=100), end)
        
        trading_days = sorted(self._prices['SPY'].index.tolist())
        trading_days = [d for d in trading_days if start <= d <= end]
        
        print(f"\n  回测区间: {start} ~ {end} ({len(trading_days)} 天)")
        print(f"  初始资金: ${self.initial_capital:,.0f}")
        
        rebal_count = 0
        peak = self.initial_capital
        last_risk_mode = "normal"
        
        for i, dt in enumerate(trading_days):
            pv = self._portfolio_value(dt)
            peak = max(peak, pv)
            
            spy_price = self._get('SPY', dt, 'close') or 0
            spy_base = self._get('SPY', start, 'close') or 1
            spy_val = self.initial_capital * spy_price / spy_base
            
            self.equity_curve.append((str(dt), pv, spy_val))
            
            if i % 150 == 0:
                print(f"  [{i+1}/{len(trading_days)}] {dt}: ${pv:,.0f} (SPY: ${spy_val:,.0f})")
            
            # ============ V4 风控开关检查 ============
            self._current_risk = self._check_risk_switch(dt)
            
            # 风控状态变化时记录
            if self._current_risk.risk_mode != last_risk_mode:
                self.risk_history.append(self._current_risk)
                if self._current_risk.risk_mode != "normal":
                    print(f"\n  ⚠️ [{dt}] 风控触发: {self._current_risk.risk_mode.upper()} "
                          f"- {self._current_risk.trigger_reason} "
                          f"(max_exposure: {self._current_risk.max_exposure:.0%})")
                else:
                    print(f"\n  ✅ [{dt}] 风控解除: 恢复正常运行")
                last_risk_mode = self._current_risk.risk_mode
            
            # 如果风控触发，强制减仓
            if self._current_risk.risk_mode != "normal":
                self._force_reduce_exposure(dt, self._current_risk.max_exposure, 
                                           self._current_risk.trigger_reason)
            
            # ============ V3 趋势跟踪主逻辑 ============
            is_bull = self._is_bull_market(dt)
            
            # 止损检查
            self._check_stops(dt, is_bull)
            
            # 再平衡 (每 10 天)
            rebal_count += 1
            if rebal_count >= 10:
                rebal_count = 0
                
                # 只有在正常模式 + 牛市才加仓
                if is_bull and self._current_risk.risk_mode == "normal":
                    candidates = self._rank_stocks(dt)
                    target_positions = 5
                    position_pct = 0.19
                    
                    for sym, _ in candidates:
                        if len(self.positions) >= target_positions:
                            break
                        if sym in self.positions:
                            continue
                        
                        budget = pv * position_pct
                        self._buy(sym, dt, min(budget, self.cash * 0.95))
                
                # 警戒模式下也可以小仓位操作
                elif is_bull and self._current_risk.risk_mode == "caution":
                    candidates = self._rank_stocks(dt)
                    # 警戒模式: 最多 3 只，每只 15%
                    target_positions = 3
                    position_pct = 0.15
                    
                    for sym, _ in candidates:
                        if len(self.positions) >= target_positions:
                            break
                        if sym in self.positions:
                            continue
                        
                        current_exposure = (pv - self.cash) / pv if pv > 0 else 0
                        if current_exposure >= self._current_risk.max_exposure:
                            break
                        
                        budget = pv * position_pct
                        self._buy(sym, dt, min(budget, self.cash * 0.95))
        
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
        
        # 风控统计
        risk_triggers = [r for r in self.risk_history if r.risk_mode != "normal"]
        
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
            "risk_triggers": len(risk_triggers),
        }


def main():
    bt = TrendFollowingWithRiskSwitch(100000.0)
    result = bt.run(date(2023, 1, 3), date(2026, 1, 16))
    
    print("\n" + "=" * 70)
    print("V5b 回测结果")
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
    print(f"\n  风控触发次数: {result['risk_triggers']} 次")
    
    # 保存结果
    output = Path("storage/backtest_3y_v5b")
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
    
    risk_data = [
        {"date": r.date, "mode": r.risk_mode, "vix": r.vix_level,
         "max_exposure": r.max_exposure, "reason": r.trigger_reason}
        for r in bt.risk_history
    ]
    with open(output / "risk_history.json", "w") as f:
        json.dump(risk_data, f, indent=2)
    
    equity_df = pd.DataFrame(bt.equity_curve, columns=['date', 'portfolio', 'spy'])
    equity_df.to_csv(output / "equity_curve.csv", index=False)
    
    print(f"\n📁 保存到: {output}")
    
    # 最大盈利交易
    print("\n【最大盈利交易】")
    top = sorted([t for t in bt.trades if t.action == "SELL"], key=lambda x: -x.pnl)[:5]
    for t in top:
        print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
    
    # 最大亏损交易
    print("\n【最大亏损交易】")
    bottom = sorted([t for t in bt.trades if t.action == "SELL"], key=lambda x: x.pnl)[:5]
    for t in bottom:
        print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
    
    # 风控触发记录
    if bt.risk_history:
        print("\n【风控触发记录】")
        for r in bt.risk_history:
            print(f"  {r.date}: {r.risk_mode} - {r.trigger_reason} (max: {r.max_exposure:.0%})")
    
    # 策略对比
    print("\n" + "=" * 70)
    print("策略对比 (V3 vs V5 vs V5b)")
    print("=" * 70)
    print("""
    | 指标       | V3 趋势跟踪 | V5 融合策略 | V5b 风控开关 |
    |------------|-------------|-------------|--------------|
    | 总收益率   | +117.02%    | +90.43%     | 待确认...    |
    | 年化收益   | +29.05%     | +23.61%     | 待确认...    |
    | Alpha      | +35.40%     | +8.80%      | 待确认...    |
    | 夏普比率   | 1.32        | 1.43        | 待确认...    |
    | 最大回撤   | 16.10%      | 12.56%      | 待确认...    |
    
    V5b 设计目标:
    - 收益接近 V3 (>100%)
    - 回撤接近 V5 (<13%)
    - 风控只在极端情况介入，不影响正常交易
    """)


if __name__ == "__main__":
    main()
