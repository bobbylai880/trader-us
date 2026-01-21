#!/usr/bin/env python3
"""
3年回测 V5c - V3趋势跟踪 + 预防性风控

核心改进 (相比 V5b):
┌─────────────────────────────────────────────────────────────┐
│  1. 提前预警 - VIX 趋势上升时就开始减仓                       │
│     - VIX 5日均线 > 20日均线 = 警戒信号                       │
│     - VIX > 20 且上升趋势 = 开始减仓                          │
│                                                              │
│  2. 延迟恢复 - 风控触发后等待冷却期                           │
│     - danger 模式: 冷却 10 个交易日                           │
│     - caution 模式: 冷却 5 个交易日                           │
│                                                              │
│  3. 渐进减仓 - 分批减仓而非一次性清仓                         │
│     - 每次最多减仓 1 只股票                                   │
│     - 优先减仓亏损最大的                                      │
└─────────────────────────────────────────────────────────────┘

目标: 保留 V3 的高收益 (+117%), 同时降低回撤到 <15%
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
    """预防性风控状态"""
    date: str
    vix_level: float
    vix_sma5: float
    vix_sma20: float
    vix_trend: str  # "rising", "falling", "stable"
    spy_below_sma50: bool
    spy_momentum: float
    risk_mode: str  # "normal", "watch", "caution", "danger"
    max_exposure: float
    cooldown_days: int  # 剩余冷却天数
    trigger_reason: str


class TrendFollowingWithPreventiveRisk:
    """V5c: V3趋势跟踪 + 预防性风控"""
    
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
        self._cooldown_remaining: int = 0  # 冷却期剩余天数
        self._last_risk_mode: str = "normal"
    
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
    # 预防性风控 (V5c 核心改进)
    # ================================================================
    
    def _check_risk_preventive(self, dt: date) -> RiskState:
        """
        预防性风控逻辑 - 提前预警，延迟恢复
        
        关键改进:
        1. VIX 趋势检测 (5日均线 vs 20日均线)
        2. 冷却期机制 (防止快速切换)
        3. 分级响应 (watch -> caution -> danger)
        """
        vix = self._get('VIX', dt, 'close') or 20
        vix_sma5 = self._get('VIX', dt, 'sma5') or vix
        vix_sma20 = self._get('VIX', dt, 'sma20') or vix
        vix_mom5 = self._get('VIX', dt, 'mom5') or 0
        
        spy_close = self._get('SPY', dt, 'close') or 0
        spy_sma50 = self._get('SPY', dt, 'sma50') or spy_close
        spy_mom = self._get('SPY', dt, 'mom20') or 0
        
        spy_below_sma50 = spy_close < spy_sma50
        
        # VIX 趋势判断
        if vix_sma5 > vix_sma20 * 1.1:
            vix_trend = "rising"
        elif vix_sma5 < vix_sma20 * 0.9:
            vix_trend = "falling"
        else:
            vix_trend = "stable"
        
        # 冷却期处理 - 如果在冷却期内，维持当前风控状态
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return RiskState(
                date=str(dt),
                vix_level=vix,
                vix_sma5=vix_sma5,
                vix_sma20=vix_sma20,
                vix_trend=vix_trend,
                spy_below_sma50=spy_below_sma50,
                spy_momentum=spy_mom,
                risk_mode=self._last_risk_mode,
                max_exposure=self._get_exposure_for_mode(self._last_risk_mode),
                cooldown_days=self._cooldown_remaining,
                trigger_reason=f"冷却期({self._cooldown_remaining}天)"
            )
        
        # 危险模式: VIX > 28 或 VIX 快速上升
        if vix > 28 or (vix > 22 and vix_mom5 > 0.3):
            new_mode = "danger"
            cooldown = 10  # 危险模式需要 10 天冷却
            max_exp = 0.30
            reason = f"VIX高位({vix:.1f}) 或快速上升" if vix > 28 else f"VIX急升({vix_mom5:.1%})"
        
        # 警戒模式: VIX > 22 且上升趋势 且 SPY 破位
        elif vix > 22 and vix_trend == "rising" and spy_below_sma50:
            new_mode = "caution"
            cooldown = 5
            max_exp = 0.50
            reason = f"VIX上升趋势({vix:.1f}) + SPY破位"
        
        # 观察模式: VIX > 20 且上升趋势
        elif vix > 20 and vix_trend == "rising":
            new_mode = "watch"
            cooldown = 3
            max_exp = 0.70
            reason = f"VIX上升趋势({vix:.1f})"
        
        # 轻度警戒: SPY 明显下跌
        elif spy_mom < -0.08:
            new_mode = "watch"
            cooldown = 3
            max_exp = 0.70
            reason = f"市场回调({spy_mom:.1%})"
        
        # 正常模式
        else:
            new_mode = "normal"
            cooldown = 0
            max_exp = 0.95
            reason = "正常运行"
        
        # 状态恶化时立即响应，状态好转时需要冷却
        if self._mode_severity(new_mode) > self._mode_severity(self._last_risk_mode):
            # 恶化 - 立即响应
            self._cooldown_remaining = cooldown
            self._last_risk_mode = new_mode
        elif self._mode_severity(new_mode) < self._mode_severity(self._last_risk_mode):
            # 好转 - 需要冷却
            if self._cooldown_remaining == 0:
                self._last_risk_mode = new_mode
        
        return RiskState(
            date=str(dt),
            vix_level=vix,
            vix_sma5=vix_sma5,
            vix_sma20=vix_sma20,
            vix_trend=vix_trend,
            spy_below_sma50=spy_below_sma50,
            spy_momentum=spy_mom,
            risk_mode=self._last_risk_mode,
            max_exposure=self._get_exposure_for_mode(self._last_risk_mode),
            cooldown_days=self._cooldown_remaining,
            trigger_reason=reason
        )
    
    def _mode_severity(self, mode: str) -> int:
        """风控模式严重程度"""
        return {"normal": 0, "watch": 1, "caution": 2, "danger": 3}.get(mode, 0)
    
    def _get_exposure_for_mode(self, mode: str) -> float:
        """各模式对应的最大仓位"""
        return {"normal": 0.95, "watch": 0.70, "caution": 0.50, "danger": 0.30}.get(mode, 0.95)
    
    # ================================================================
    # V3 趋势跟踪核心逻辑 (完整保留)
    # ================================================================
    
    def _is_bull_market(self, dt: date) -> bool:
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
        """V3 止损逻辑"""
        for sym in list(self.positions.keys()):
            p = self.positions[sym]
            price = self._get(sym, dt, 'close')
            if not price:
                continue
            
            p.highest_price = max(p.highest_price, price)
            
            drawdown = (p.highest_price - price) / p.highest_price
            if drawdown > 0.15:
                self._sell(sym, dt, f"跟踪止损({drawdown:.1%})")
                continue
            
            if not is_bull and price < p.avg_cost * 0.92:
                self._sell(sym, dt, "熊市保护")
    
    def _gradual_reduce_exposure(self, dt: date, target_exposure: float, reason: str):
        """
        渐进式减仓 - V5c 核心改进
        每次只减仓 1 只股票，优先减仓亏损最大的
        """
        pv = self._portfolio_value(dt)
        current_exposure = (pv - self.cash) / pv if pv > 0 else 0
        
        if current_exposure <= target_exposure:
            return
        
        # 按收益率排序，卖出表现最差的 (只卖 1 只)
        holdings = []
        for sym, pos in self.positions.items():
            price = self._get(sym, dt, 'close') or pos.avg_cost
            pnl_pct = (price - pos.avg_cost) / pos.avg_cost
            holdings.append((sym, pnl_pct))
        
        if not holdings:
            return
        
        holdings.sort(key=lambda x: x[1])
        
        # 只卖出 1 只最差的
        sym, _ = holdings[0]
        self._sell(sym, dt, f"预防性减仓: {reason}")
    
    # ================================================================
    # 主运行循环
    # ================================================================
    
    def run(self, start: date, end: date) -> dict:
        print("\n" + "=" * 70)
        print("V5c 策略: V3趋势跟踪 + 预防性风控")
        print("=" * 70)
        print("  核心改进:")
        print("    1. 提前预警 - VIX 趋势上升时开始减仓")
        print("    2. 延迟恢复 - 冷却期机制 (danger=10天, caution=5天)")
        print("    3. 渐进减仓 - 每次只减 1 只，优先减亏损最大的")
        
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
            
            # ============ 预防性风控检查 ============
            self._current_risk = self._check_risk_preventive(dt)
            
            # 风控状态变化时记录
            if self._current_risk.risk_mode != last_logged_mode:
                self.risk_history.append(self._current_risk)
                if self._current_risk.risk_mode != "normal":
                    print(f"\n  ⚠️ [{dt}] 风控: {self._current_risk.risk_mode.upper()} "
                          f"- {self._current_risk.trigger_reason} "
                          f"(max: {self._current_risk.max_exposure:.0%}, "
                          f"冷却: {self._current_risk.cooldown_days}天)")
                else:
                    print(f"\n  ✅ [{dt}] 风控解除: 恢复正常运行")
                last_logged_mode = self._current_risk.risk_mode
            
            # 渐进式减仓 (每天最多减 1 只)
            if self._current_risk.risk_mode != "normal":
                self._gradual_reduce_exposure(dt, self._current_risk.max_exposure,
                                              self._current_risk.trigger_reason)
            
            # ============ V3 趋势跟踪主逻辑 ============
            is_bull = self._is_bull_market(dt)
            
            self._check_stops(dt, is_bull)
            
            rebal_count += 1
            if rebal_count >= 10:
                rebal_count = 0
                
                # 正常模式: 全力做多
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
                
                # 观察/警戒模式: 可以小仓位操作
                elif is_bull and self._current_risk.risk_mode in ["watch", "caution"]:
                    candidates = self._rank_stocks(dt)
                    target_positions = 3 if self._current_risk.risk_mode == "watch" else 2
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
    bt = TrendFollowingWithPreventiveRisk(100000.0)
    result = bt.run(date(2023, 1, 3), date(2026, 1, 16))
    
    print("\n" + "=" * 70)
    print("V5c 回测结果")
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
    output = Path("storage/backtest_3y_v5c")
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
         "vix_trend": r.vix_trend, "max_exposure": r.max_exposure,
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
    
    if bt.risk_history:
        print("\n【风控触发记录】")
        for r in bt.risk_history[:10]:
            print(f"  {r.date}: {r.risk_mode} - {r.trigger_reason} "
                  f"(max: {r.max_exposure:.0%}, 冷却: {r.cooldown_days}天)")
    
    print("\n" + "=" * 70)
    print("策略对比总结")
    print("=" * 70)
    print("""
    | 指标       | V3 趋势跟踪 | V5 融合 | V5b 反应式 | V5c 预防式 |
    |------------|-------------|---------|------------|------------|
    | 总收益率   | +117.02%    | +90.43% | +97.06%    | 待确认     |
    | 年化收益   | +29.05%     | +23.61% | +25.01%    | 待确认     |
    | Alpha      | +35.40%     | +8.80%  | +15.43%    | 待确认     |
    | 夏普比率   | 1.32        | 1.43    | 1.15       | 待确认     |
    | 最大回撤   | 16.10%      | 12.56%  | 22.60%     | 待确认     |
    
    V5c 设计目标:
    - 收益 > V5b (>97%)
    - 回撤 < V3 (<16%)
    - 提前预警 + 延迟恢复 = 更平滑的风控切换
    """)


if __name__ == "__main__":
    main()
