#!/usr/bin/env python3
"""
3年回测 V5d - V3趋势跟踪 + 预防性风控 + V3过滤器 + 硬性止损

核心改进 (相比 V5c):
┌─────────────────────────────────────────────────────────────┐
│  1. V3 过滤器 - 只有 SPY > SMA50 才允许建仓                   │
│     - 风控解除后也要等 V3 条件满足                            │
│                                                              │
│  2. 硬性止损 12% - 防止单笔亏损过大                           │
│     - 成本价 * 0.88 触发硬止损                               │
│                                                              │
│  3. 保留核心持仓 - 风控期间不完全清仓                         │
│     - danger 模式: 保留 1 只最强持仓                          │
│     - caution 模式: 保留 2 只最强持仓                         │
│                                                              │
│  4. 分批建仓 - 风控解除后渐进加仓                             │
│     - 第一次只建 3 只仓位                                     │
│     - 10 天后再补满 5 只                                      │
└─────────────────────────────────────────────────────────────┘

目标: 保留 V5c 的高收益 (+114%), 回撤降到 <15%
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
    date: str
    vix_level: float
    vix_sma5: float
    vix_sma20: float
    vix_trend: str
    spy_above_sma50: bool  # V3 过滤器
    spy_momentum: float
    risk_mode: str
    max_exposure: float
    min_positions: int  # 最少保留持仓数
    cooldown_days: int
    trigger_reason: str


class TrendFollowingWithV3Filter:
    """V5d: V3趋势跟踪 + 预防性风控 + V3过滤器 + 硬性止损"""
    
    UNIVERSE = ["NVDA", "META", "GOOGL", "AMZN", "MSFT", "AAPL", "AMD", "AVGO", "NFLX", "TSLA"]
    
    # 核心持仓 (风控期间优先保留)
    CORE_HOLDINGS = ["AAPL", "MSFT", "GOOGL"]
    
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
        self._cooldown_remaining: int = 0
        self._last_risk_mode: str = "normal"
        self._recovery_days: int = 0  # 风控解除后的恢复天数
    
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
            sdf['sma5'] = sdf['close'].rolling(5).mean()
            sdf['mom20'] = sdf['close'].pct_change(20)
            sdf['mom5'] = sdf['close'].pct_change(5)
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
    # V3 过滤器 (核心新增)
    # ================================================================
    
    def _v3_filter_pass(self, dt: date) -> bool:
        """V3 过滤器: SPY 必须在 SMA50 之上"""
        spy_close = self._get('SPY', dt, 'close')
        spy_sma50 = self._get('SPY', dt, 'sma50')
        spy_mom = self._get('SPY', dt, 'mom20')
        
        if spy_close is None or spy_sma50 is None:
            return False
        
        return spy_close > spy_sma50 and (spy_mom is None or spy_mom > -0.05)
    
    # ================================================================
    # 预防性风控 (改进版)
    # ================================================================
    
    def _check_risk_preventive(self, dt: date) -> RiskState:
        vix = self._get('VIX', dt, 'close') or 20
        vix_sma5 = self._get('VIX', dt, 'sma5') or vix
        vix_sma20 = self._get('VIX', dt, 'sma20') or vix
        vix_mom5 = self._get('VIX', dt, 'mom5') or 0
        
        spy_close = self._get('SPY', dt, 'close') or 0
        spy_sma50 = self._get('SPY', dt, 'sma50') or spy_close
        spy_mom = self._get('SPY', dt, 'mom20') or 0
        
        spy_above_sma50 = spy_close > spy_sma50
        
        if vix_sma5 > vix_sma20 * 1.1:
            vix_trend = "rising"
        elif vix_sma5 < vix_sma20 * 0.9:
            vix_trend = "falling"
        else:
            vix_trend = "stable"
        
        # 冷却期处理
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return RiskState(
                date=str(dt),
                vix_level=vix,
                vix_sma5=vix_sma5,
                vix_sma20=vix_sma20,
                vix_trend=vix_trend,
                spy_above_sma50=spy_above_sma50,
                spy_momentum=spy_mom,
                risk_mode=self._last_risk_mode,
                max_exposure=self._get_exposure_for_mode(self._last_risk_mode),
                min_positions=self._get_min_positions_for_mode(self._last_risk_mode),
                cooldown_days=self._cooldown_remaining,
                trigger_reason=f"冷却期({self._cooldown_remaining}天)"
            )
        
        # 危险模式
        if vix > 28 or (vix > 22 and vix_mom5 > 0.3):
            new_mode = "danger"
            cooldown = 10
            reason = f"VIX高位({vix:.1f})" if vix > 28 else f"VIX急升({vix_mom5:.1%})"
        
        # 警戒模式
        elif vix > 22 and vix_trend == "rising" and not spy_above_sma50:
            new_mode = "caution"
            cooldown = 5
            reason = f"VIX上升({vix:.1f}) + SPY破位"
        
        # 观察模式
        elif vix > 20 and vix_trend == "rising":
            new_mode = "watch"
            cooldown = 3
            reason = f"VIX上升趋势({vix:.1f})"
        
        # 正常模式
        else:
            new_mode = "normal"
            cooldown = 0
            reason = "正常运行"
        
        # 状态恶化立即响应
        if self._mode_severity(new_mode) > self._mode_severity(self._last_risk_mode):
            self._cooldown_remaining = cooldown
            self._last_risk_mode = new_mode
            self._recovery_days = 0  # 重置恢复计数
        elif self._mode_severity(new_mode) < self._mode_severity(self._last_risk_mode):
            if self._cooldown_remaining == 0:
                self._last_risk_mode = new_mode
                self._recovery_days = 0  # 开始恢复计数
        
        return RiskState(
            date=str(dt),
            vix_level=vix,
            vix_sma5=vix_sma5,
            vix_sma20=vix_sma20,
            vix_trend=vix_trend,
            spy_above_sma50=spy_above_sma50,
            spy_momentum=spy_mom,
            risk_mode=self._last_risk_mode,
            max_exposure=self._get_exposure_for_mode(self._last_risk_mode),
            min_positions=self._get_min_positions_for_mode(self._last_risk_mode),
            cooldown_days=self._cooldown_remaining,
            trigger_reason=reason
        )
    
    def _mode_severity(self, mode: str) -> int:
        return {"normal": 0, "watch": 1, "caution": 2, "danger": 3}.get(mode, 0)
    
    def _get_exposure_for_mode(self, mode: str) -> float:
        return {"normal": 0.95, "watch": 0.70, "caution": 0.50, "danger": 0.30}.get(mode, 0.95)
    
    def _get_min_positions_for_mode(self, mode: str) -> int:
        """各模式最少保留持仓数"""
        return {"normal": 0, "watch": 2, "caution": 1, "danger": 1}.get(mode, 0)
    
    # ================================================================
    # 交易逻辑
    # ================================================================
    
    def _portfolio_value(self, dt: date) -> float:
        pos_val = sum(
            p.shares * (self._get(s, dt, 'close') or p.avg_cost)
            for s, p in self.positions.items()
        )
        return self.cash + pos_val
    
    def _rank_stocks(self, dt: date) -> List[Tuple[str, float]]:
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
    
    def _check_stops(self, dt: date, v3_pass: bool):
        """止损逻辑 - 新增硬性止损 12%"""
        for sym in list(self.positions.keys()):
            p = self.positions[sym]
            price = self._get(sym, dt, 'close')
            if not price:
                continue
            
            p.highest_price = max(p.highest_price, price)
            
            # 跟踪止损 15%
            drawdown = (p.highest_price - price) / p.highest_price
            if drawdown > 0.15:
                self._sell(sym, dt, f"跟踪止损({drawdown:.1%})")
                continue
            
            # 硬性止损 12% (新增)
            loss_from_cost = (p.avg_cost - price) / p.avg_cost
            if loss_from_cost > 0.12:
                self._sell(sym, dt, f"硬性止损({loss_from_cost:.1%})")
                continue
            
            # 熊市保护
            if not v3_pass and price < p.avg_cost * 0.92:
                self._sell(sym, dt, "熊市保护")
    
    def _smart_reduce_exposure(self, dt: date, target_exposure: float, min_positions: int, reason: str):
        """
        智能减仓 - 保留核心持仓
        """
        pv = self._portfolio_value(dt)
        current_exposure = (pv - self.cash) / pv if pv > 0 else 0
        
        if current_exposure <= target_exposure:
            return
        
        if len(self.positions) <= min_positions:
            return
        
        # 按收益率排序
        holdings = []
        for sym, pos in self.positions.items():
            price = self._get(sym, dt, 'close') or pos.avg_cost
            pnl_pct = (price - pos.avg_cost) / pos.avg_cost
            # 核心持仓加分
            is_core = sym in self.CORE_HOLDINGS
            score = pnl_pct + (0.1 if is_core else 0)
            holdings.append((sym, score, pnl_pct))
        
        holdings.sort(key=lambda x: x[1])  # 分数最低的先卖
        
        # 每次只卖 1 只 (渐进减仓)
        for sym, _, pnl_pct in holdings:
            if len(self.positions) <= min_positions:
                break
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
        print("V5d 策略: V3趋势跟踪 + 预防性风控 + V3过滤器 + 硬性止损")
        print("=" * 70)
        print("  核心改进:")
        print("    1. V3 过滤器 - SPY > SMA50 才允许建仓")
        print("    2. 硬性止损 12% - 防止单笔亏损过大")
        print("    3. 保留核心持仓 - 风控期间不完全清仓")
        print("    4. 渐进建仓 - 风控解除后分批进入")
        
        self._load_data(start - timedelta(days=100), end)
        
        trading_days = sorted(self._prices['SPY'].index.tolist())
        trading_days = [d for d in trading_days if start <= d <= end]
        
        print(f"\n  回测区间: {start} ~ {end} ({len(trading_days)} 天)")
        print(f"  初始资金: ${self.initial_capital:,.0f}")
        
        rebal_count = 0
        peak = self.initial_capital
        last_logged_mode = "normal"
        
        for i, dt in enumerate(trading_days):
            pv = self._portfolio_value(dt)
            peak = max(peak, pv)
            
            spy_price = self._get('SPY', dt, 'close') or 0
            spy_base = self._get('SPY', start, 'close') or 1
            spy_val = self.initial_capital * spy_price / spy_base
            
            self.equity_curve.append((str(dt), pv, spy_val))
            
            if i % 150 == 0:
                print(f"  [{i+1}/{len(trading_days)}] {dt}: ${pv:,.0f} (SPY: ${spy_val:,.0f})")
            
            # V3 过滤器检查
            v3_pass = self._v3_filter_pass(dt)
            
            # 预防性风控检查
            self._current_risk = self._check_risk_preventive(dt)
            
            # 风控状态变化
            if self._current_risk.risk_mode != last_logged_mode:
                self.risk_history.append(self._current_risk)
                if self._current_risk.risk_mode != "normal":
                    print(f"\n  ⚠️ [{dt}] 风控: {self._current_risk.risk_mode.upper()} "
                          f"- {self._current_risk.trigger_reason} "
                          f"(max: {self._current_risk.max_exposure:.0%}, "
                          f"保留: {self._current_risk.min_positions}只)")
                else:
                    print(f"\n  ✅ [{dt}] 风控解除 (V3: {'通过' if v3_pass else '未通过'})")
                last_logged_mode = self._current_risk.risk_mode
            
            # 智能减仓
            if self._current_risk.risk_mode != "normal":
                self._smart_reduce_exposure(dt, self._current_risk.max_exposure,
                                           self._current_risk.min_positions,
                                           self._current_risk.trigger_reason)
            
            # 止损检查
            self._check_stops(dt, v3_pass)
            
            # 再平衡 (需要 V3 过滤器通过)
            rebal_count += 1
            if rebal_count >= 10:
                rebal_count = 0
                
                # 必须通过 V3 过滤器才能建仓
                if not v3_pass:
                    continue
                
                # 恢复期计数
                if self._current_risk.risk_mode == "normal" and self._recovery_days < 20:
                    self._recovery_days += 1
                
                # 正常模式
                if self._current_risk.risk_mode == "normal":
                    candidates = self._rank_stocks(dt)
                    
                    # 恢复期前 10 天: 渐进建仓 (最多 3 只)
                    if self._recovery_days <= 10:
                        target_positions = 3
                        position_pct = 0.15
                    else:
                        target_positions = 5
                        position_pct = 0.19
                    
                    for sym, _ in candidates:
                        if len(self.positions) >= target_positions:
                            break
                        if sym in self.positions:
                            continue
                        
                        budget = pv * position_pct
                        self._buy(sym, dt, min(budget, self.cash * 0.95))
                
                # watch/caution 模式: 可以小仓位
                elif self._current_risk.risk_mode in ["watch", "caution"]:
                    candidates = self._rank_stocks(dt)
                    target_positions = 2 if self._current_risk.risk_mode == "watch" else 1
                    position_pct = 0.12
                    
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
        max_dd_date = ""
        for i, v in enumerate(values):
            peak = max(peak, v)
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
                max_dd_date = self.equity_curve[i][0]
        
        rets = pd.Series(values).pct_change().dropna()
        sharpe = np.sqrt(252) * rets.mean() / rets.std() if rets.std() > 0 else 0
        
        sells = [t for t in self.trades if t.action == "SELL"]
        wins = [t for t in sells if t.pnl > 0]
        win_rate = len(wins) / len(sells) if sells else 0
        
        total_win = sum(t.pnl for t in wins)
        total_loss = abs(sum(t.pnl for t in sells if t.pnl < 0))
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        
        risk_triggers = [r for r in self.risk_history if r.risk_mode != "normal"]
        
        return {
            "final_value": final,
            "total_return": total_ret,
            "annualized_return": ann_ret,
            "spy_return": spy_ret,
            "alpha": total_ret - spy_ret,
            "max_drawdown": max_dd,
            "max_drawdown_date": max_dd_date,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(self.trades),
            "avg_win": np.mean([t.pnl for t in wins]) if wins else 0,
            "avg_loss": np.mean([t.pnl for t in sells if t.pnl < 0]) if any(t.pnl < 0 for t in sells) else 0,
            "risk_triggers": len(risk_triggers),
        }


def main():
    bt = TrendFollowingWithV3Filter(100000.0)
    result = bt.run(date(2023, 1, 3), date(2026, 1, 16))
    
    print("\n" + "=" * 70)
    print("V5d 回测结果")
    print("=" * 70)
    print(f"\n  最终价值: ${result['final_value']:,.0f}")
    print(f"  总收益率: {result['total_return']:+.2%}")
    print(f"  年化收益: {result['annualized_return']:+.2%}")
    print(f"  SPY收益:  {result['spy_return']:+.2%}")
    print(f"  超额收益: {result['alpha']:+.2%}")
    print(f"\n  最大回撤: {result['max_drawdown']:.2%} ({result['max_drawdown_date']})")
    print(f"  夏普比率: {result['sharpe']:.2f}")
    print(f"  胜率: {result['win_rate']:.1%}")
    print(f"  盈亏比: {result['profit_factor']:.2f}")
    print(f"  总交易: {result['total_trades']} 笔")
    print(f"  平均盈利: ${result['avg_win']:,.0f}")
    print(f"  平均亏损: ${result['avg_loss']:,.0f}")
    print(f"\n  风控触发次数: {result['risk_triggers']} 次")
    
    # 保存结果
    output = Path("storage/backtest_3y_v5d")
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
         "vix_trend": r.vix_trend, "spy_above_sma50": r.spy_above_sma50,
         "max_exposure": r.max_exposure, "min_positions": r.min_positions,
         "cooldown": r.cooldown_days, "reason": r.trigger_reason}
        for r in bt.risk_history
    ]
    with open(output / "risk_history.json", "w") as f:
        json.dump(risk_data, f, indent=2)
    
    equity_df = pd.DataFrame(bt.equity_curve, columns=['date', 'portfolio', 'spy'])
    equity_df.to_csv(output / "equity_curve.csv", index=False)
    
    print(f"\n📁 保存到: {output}")
    
    print("\n【最大盈利交易】")
    top = sorted([t for t in bt.trades if t.action == "SELL"], key=lambda x: -x.pnl)[:5]
    for t in top:
        print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
    
    print("\n【最大亏损交易】")
    bottom = sorted([t for t in bt.trades if t.action == "SELL"], key=lambda x: x.pnl)[:5]
    for t in bottom:
        print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
    
    print("\n" + "=" * 70)
    print("策略对比")
    print("=" * 70)
    print("""
    | 指标       | V3 趋势 | V5 融合 | V5c 预防 | V5d 过滤 |
    |------------|---------|---------|----------|----------|
    | 总收益率   | +117%   | +90%    | +114%    | 待确认   |
    | Alpha      | +35%    | +9%     | +33%     | 待确认   |
    | 夏普比率   | 1.32    | 1.43    | 1.34     | 待确认   |
    | 最大回撤   | 16.1%   | 12.6%   | 20.3%    | 待确认   |
    
    V5d 目标:
    - 收益接近 V5c (>100%)
    - 回撤降到 <15%
    - V3 过滤器 + 硬性止损 = 更稳健
    """)


if __name__ == "__main__":
    main()
