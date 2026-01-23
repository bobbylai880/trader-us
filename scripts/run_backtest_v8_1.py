#!/usr/bin/env python3
"""
V8.1 策略: V5c预防式风控 + 事前主题生成 (Forward-Looking Theme)

与 V8.0 的区别:
┌─────────────────────────────────────────────────────────────┐
│  V8.0 (后视镜)                                               │
│  - 人工编写季度主题配置                                      │
│  - 使用事后信息判断板块/个股                                 │
│  - Alpha: +84% (但不可实盘)                                  │
├─────────────────────────────────────────────────────────────┤
│  V8.1 (事前)                                                 │
│  - ForwardThemeGenerator 自动生成主题                        │
│  - 只使用 as_of 日期之前的数据                               │
│  - 融合: 动量(30%) + Insider(20%) + Analyst(15%) +          │
│         Options(10%) + Social(10%) + Policy(10%) + Fed(5%)  │
│  - Alpha: 预期 +30-50% (可实盘)                              │
└─────────────────────────────────────────────────────────────┘
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

from ai_trader_assist.alternative_data.db_theme_generator import DatabaseThemeGenerator, ThemeConfig

SYMBOL_TO_SECTOR = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "AMD": "XLK",
    "ADBE": "XLK", "CRM": "XLK", "ORCL": "XLK",
    "META": "XLC", "GOOGL": "XLC", "NFLX": "XLC",
    "AMZN": "XLY", "TSLA": "XLY",
    "JPM": "XLF", "GS": "XLF",
    "UNH": "XLV", "LLY": "XLV",
    "XOM": "XLE", "CVX": "XLE",
}

BASE_UNIVERSE = ["NVDA", "META", "GOOGL", "AMZN", "MSFT", "AAPL", "AMD", "AVGO", "NFLX", "TSLA"]


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
    spy_below_sma50: bool
    spy_momentum: float
    risk_mode: str
    max_exposure: float
    cooldown_days: int
    trigger_reason: str


class V81BacktestEngine:
    
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
        self.theme_history: List[Tuple[str, ThemeConfig]] = []
        
        self._current_risk: Optional[RiskState] = None
        self._cooldown_remaining: int = 0
        self._last_risk_mode: str = "normal"
        self._current_theme: Optional[ThemeConfig] = None
        self._dynamic_universe: List[str] = []
        
        self._theme_generator = DatabaseThemeGenerator(universe=BASE_UNIVERSE)
    
    def _load_data(self, start: date, end: date):
        all_symbols = set(BASE_UNIVERSE)
        all_symbols.update(['SPY', 'VIX'])
        all_symbols.update(["JPM", "GS", "UNH", "LLY", "XOM", "CVX", "ADBE", "CRM", "ORCL"])
        
        query = """
            SELECT symbol, trade_date, close
            FROM daily_prices
            WHERE trade_date BETWEEN %s AND %s
              AND symbol IN %s
            ORDER BY symbol, trade_date
        """
        df = pd.read_sql(query, self.conn, params=(start - timedelta(days=100), end, tuple(all_symbols)))
        
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
    
    def _compute_momentum_scores(self, dt: date) -> Dict[str, float]:
        scores = {}
        for sym in self._dynamic_universe:
            mom = self._get(sym, dt, 'mom20')
            if mom is not None:
                scores[sym] = max(-1, min(1, mom * 5))
            else:
                scores[sym] = 0.0
        return scores
    
    def _generate_forward_theme(self, dt: date) -> ThemeConfig:
        momentum_scores = self._compute_momentum_scores(dt)
        return self._theme_generator.generate_theme(dt, momentum_scores)
    
    def _build_dynamic_universe(self, theme: ThemeConfig) -> List[str]:
        universe = set(BASE_UNIVERSE)
        universe.update(theme.focus_stocks)
        for sym in theme.avoid_stocks:
            universe.discard(sym)
        return list(universe)
    
    def _check_risk_preventive(self, dt: date) -> RiskState:
        vix = self._get('VIX', dt, 'close') or 20
        vix_sma5 = self._get('VIX', dt, 'sma5') or vix
        vix_sma20 = self._get('VIX', dt, 'sma20') or vix
        vix_mom5 = self._get('VIX', dt, 'mom5') or 0
        
        spy_close = self._get('SPY', dt, 'close') or 0
        spy_sma50 = self._get('SPY', dt, 'sma50') or spy_close
        spy_mom = self._get('SPY', dt, 'mom20') or 0
        
        spy_below_sma50 = spy_close < spy_sma50
        
        if vix_sma5 > vix_sma20 * 1.1:
            vix_trend = "rising"
        elif vix_sma5 < vix_sma20 * 0.9:
            vix_trend = "falling"
        else:
            vix_trend = "stable"
        
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return RiskState(
                date=str(dt), vix_level=vix, vix_sma5=vix_sma5, vix_sma20=vix_sma20,
                vix_trend=vix_trend, spy_below_sma50=spy_below_sma50, spy_momentum=spy_mom,
                risk_mode=self._last_risk_mode,
                max_exposure=self._get_exposure_for_mode(self._last_risk_mode),
                cooldown_days=self._cooldown_remaining,
                trigger_reason=f"冷却期({self._cooldown_remaining}天)"
            )
        
        if vix > 28 or (vix > 22 and vix_mom5 > 0.3):
            new_mode, cooldown, max_exp = "danger", 10, 0.30
            reason = f"VIX高位({vix:.1f})" if vix > 28 else f"VIX急升({vix_mom5:.1%})"
        elif vix > 22 and vix_trend == "rising" and spy_below_sma50:
            new_mode, cooldown, max_exp = "caution", 5, 0.50
            reason = f"VIX上升({vix:.1f}) + SPY破位"
        elif vix > 20 and vix_trend == "rising":
            new_mode, cooldown, max_exp = "watch", 3, 0.70
            reason = f"VIX上升趋势({vix:.1f})"
        elif spy_mom < -0.08:
            new_mode, cooldown, max_exp = "watch", 3, 0.70
            reason = f"市场回调({spy_mom:.1%})"
        else:
            new_mode, cooldown, max_exp = "normal", 0, 0.95
            reason = "正常运行"
        
        if self._mode_severity(new_mode) > self._mode_severity(self._last_risk_mode):
            self._cooldown_remaining = cooldown
            self._last_risk_mode = new_mode
        elif self._mode_severity(new_mode) < self._mode_severity(self._last_risk_mode):
            if self._cooldown_remaining == 0:
                self._last_risk_mode = new_mode
        
        return RiskState(
            date=str(dt), vix_level=vix, vix_sma5=vix_sma5, vix_sma20=vix_sma20,
            vix_trend=vix_trend, spy_below_sma50=spy_below_sma50, spy_momentum=spy_mom,
            risk_mode=self._last_risk_mode,
            max_exposure=self._get_exposure_for_mode(self._last_risk_mode),
            cooldown_days=self._cooldown_remaining, trigger_reason=reason
        )
    
    def _mode_severity(self, mode: str) -> int:
        return {"normal": 0, "watch": 1, "caution": 2, "danger": 3}.get(mode, 0)
    
    def _get_exposure_for_mode(self, mode: str) -> float:
        return {"normal": 0.95, "watch": 0.70, "caution": 0.50, "danger": 0.30}.get(mode, 0.95)
    
    def _is_bull_market(self, dt: date) -> bool:
        spy_close = self._get('SPY', dt, 'close')
        spy_sma50 = self._get('SPY', dt, 'sma50')
        spy_mom = self._get('SPY', dt, 'mom20')
        if spy_close is None or spy_sma50 is None:
            return False
        return spy_close > spy_sma50 and (spy_mom is None or spy_mom > -0.05)
    
    def _portfolio_value(self, dt: date) -> float:
        pos_val = sum(p.shares * (self._get(s, dt, 'close') or p.avg_cost) for s, p in self.positions.items())
        return self.cash + pos_val
    
    def _rank_stocks_with_theme(self, dt: date) -> List[Tuple[str, float]]:
        if self._current_theme is None:
            return []
        
        theme = self._current_theme
        focus_stocks = set(theme.focus_stocks)
        sector_bonus = theme.sector_bonus
        stock_bonus = theme.stock_bonus
        avoid_sectors = set(theme.avoid_sectors)
        
        ranked = []
        for sym in self._dynamic_universe:
            if sym not in self._prices:
                continue
            
            sector = SYMBOL_TO_SECTOR.get(sym, "XLK")
            if sector in avoid_sectors:
                continue
            
            mom = self._get(sym, dt, 'mom20')
            close = self._get(sym, dt, 'close')
            sma20 = self._get(sym, dt, 'sma20')
            
            if mom is None or close is None or sma20 is None:
                continue
            if close <= sma20 or mom <= 0:
                continue
            
            score = mom * 100
            score += sector_bonus.get(sector, 0) * 30
            score += stock_bonus.get(sym, 0) * 30
            if sym in focus_stocks:
                score += 3
            
            ranked.append((sym, score))
        
        ranked.sort(key=lambda x: -x[1])
        return ranked[:8]
    
    def _buy(self, sym: str, dt: date, budget: float, reason: str = "买入") -> bool:
        price = self._get(sym, dt, 'close')
        if not price or budget < 1000:
            return False
        shares = int(budget / price)
        if shares <= 0 or shares * price > self.cash:
            return False
        
        self.cash -= shares * price
        if sym in self.positions:
            p = self.positions[sym]
            total = p.shares + shares
            p.avg_cost = (p.avg_cost * p.shares + price * shares) / total
            p.shares = total
            p.highest_price = max(p.highest_price, price)
        else:
            self.positions[sym] = Position(sym, shares, price, str(dt), price)
        
        self.trades.append(Trade(str(dt), sym, "BUY", price, shares, reason=reason))
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
        pv = self._portfolio_value(dt)
        current_exposure = (pv - self.cash) / pv if pv > 0 else 0
        if current_exposure <= target_exposure:
            return
        
        holdings = []
        for sym, pos in self.positions.items():
            price = self._get(sym, dt, 'close') or pos.avg_cost
            pnl_pct = (price - pos.avg_cost) / pos.avg_cost
            holdings.append((sym, pnl_pct))
        
        if not holdings:
            return
        holdings.sort(key=lambda x: x[1])
        sym, _ = holdings[0]
        self._sell(sym, dt, f"预防性减仓: {reason}")
    
    def run(self, start: date, end: date) -> dict:
        print("\n" + "=" * 70)
        print("V8.1 策略: V5c预防式风控 + 事前主题生成 (Forward-Looking)")
        print("=" * 70)
        print("  融合数据源:")
        print("    - 动量 (30%) + Insider (20%) + Analyst (15%)")
        print("    - Options PCR (10%) + Social (10%) + Policy (10%) + Fed (5%)")
        print("  关键区别: 只使用 as_of 日期之前可获取的数据")
        
        self._load_data(start, end)
        
        trading_days = sorted(self._prices['SPY'].index.tolist())
        trading_days = [d for d in trading_days if start <= d <= end]
        
        print(f"\n  回测区间: {start} ~ {end} ({len(trading_days)} 天)")
        print(f"  初始资金: ${self.initial_capital:,.0f}")
        
        rebal_count = 0
        last_logged_mode = "normal"
        last_theme_update = None
        theme_update_interval = 20
        
        for i, dt in enumerate(trading_days):
            if last_theme_update is None or (dt - last_theme_update).days >= theme_update_interval:
                self._current_theme = self._generate_forward_theme(dt)
                self._dynamic_universe = self._build_dynamic_universe(self._current_theme)
                self.theme_history.append((str(dt), self._current_theme))
                
                if i == 0 or (dt - last_theme_update).days >= 60 if last_theme_update else True:
                    print(f"\n  📊 [{dt}] 事前主题更新:")
                    print(f"      焦点板块: {self._current_theme.focus_sectors}")
                    print(f"      焦点个股: {self._current_theme.focus_stocks[:5]}")
                    print(f"      风险等级: {self._current_theme.risk_level}")
                    if self._current_theme.theme_drivers:
                        print(f"      驱动因素: {self._current_theme.theme_drivers[:2]}")
                
                last_theme_update = dt
            
            pv = self._portfolio_value(dt)
            spy_price = self._get('SPY', dt, 'close') or 0
            spy_base = self._get('SPY', start, 'close') or 1
            spy_val = self.initial_capital * spy_price / spy_base
            self.equity_curve.append((str(dt), pv, spy_val))
            
            if i % 150 == 0:
                print(f"  [{i+1}/{len(trading_days)}] {dt}: ${pv:,.0f} (SPY: ${spy_val:,.0f})")
            
            self._current_risk = self._check_risk_preventive(dt)
            
            if self._current_risk.risk_mode != last_logged_mode:
                self.risk_history.append(self._current_risk)
                if self._current_risk.risk_mode != "normal":
                    print(f"\n  ⚠️ [{dt}] 风控: {self._current_risk.risk_mode.upper()} - "
                          f"{self._current_risk.trigger_reason} (max: {self._current_risk.max_exposure:.0%})")
                else:
                    print(f"\n  ✅ [{dt}] 风控解除: 恢复正常运行")
                last_logged_mode = self._current_risk.risk_mode
            
            if self._current_risk.risk_mode != "normal":
                self._gradual_reduce_exposure(dt, self._current_risk.max_exposure,
                                              self._current_risk.trigger_reason)
            
            is_bull = self._is_bull_market(dt)
            self._check_stops(dt, is_bull)
            
            rebal_count += 1
            if rebal_count >= 10:
                rebal_count = 0
                
                if is_bull and self._current_risk.risk_mode == "normal":
                    candidates = self._rank_stocks_with_theme(dt)
                    target_positions = 5
                    position_pct = 0.19
                    
                    for sym, _ in candidates:
                        if len(self.positions) >= target_positions:
                            break
                        if sym in self.positions:
                            continue
                        budget = pv * position_pct
                        self._buy(sym, dt, min(budget, self.cash * 0.95), "事前主题+趋势买入")
                
                elif is_bull and self._current_risk.risk_mode in ["watch", "caution"]:
                    candidates = self._rank_stocks_with_theme(dt)
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
                        self._buy(sym, dt, min(budget, self.cash * 0.95), "风控下买入")
        
        return self._calc_results(start, end)
    
    def _calc_results(self, start: date, end: date) -> dict:
        final = self.equity_curve[-1][1]
        spy_final = self.equity_curve[-1][2]
        total_ret = final / self.initial_capital - 1
        spy_ret = spy_final / self.initial_capital - 1
        years = (end - start).days / 365
        ann_ret = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
        
        values = [e[1] for e in self.equity_curve]
        peak, max_dd = self.initial_capital, 0
        for v in values:
            peak = max(peak, v)
            max_dd = max(max_dd, (peak - v) / peak)
        
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
            "final_value": final, "total_return": total_ret, "annualized_return": ann_ret,
            "spy_return": spy_ret, "alpha": total_ret - spy_ret, "max_drawdown": max_dd,
            "sharpe": sharpe, "win_rate": win_rate, "profit_factor": profit_factor,
            "total_trades": len(self.trades),
            "avg_win": np.mean([t.pnl for t in wins]) if wins else 0,
            "avg_loss": np.mean([t.pnl for t in sells if t.pnl < 0]) if any(t.pnl < 0 for t in sells) else 0,
            "risk_triggers": len(risk_triggers),
            "theme_updates": len(self.theme_history),
        }


def main():
    bt = V81BacktestEngine(100000.0)
    result = bt.run(date(2020, 1, 2), date(2026, 1, 16))
    
    print("\n" + "=" * 70)
    print("V8.1 回测结果 (3年) - 事前主题生成")
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
    print(f"  风控触发: {result['risk_triggers']} 次")
    print(f"  主题更新: {result['theme_updates']} 次")
    
    output = Path("storage/backtest_v8_1")
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    trades_data = [{"date": t.date, "symbol": t.symbol, "action": t.action,
                    "price": t.price, "shares": t.shares, "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct, "reason": t.reason} for t in bt.trades]
    with open(output / "trades.json", "w") as f:
        json.dump(trades_data, f, indent=2)
    
    pd.DataFrame(bt.equity_curve, columns=['date', 'portfolio', 'spy']).to_csv(output / "equity_curve.csv", index=False)
    
    theme_data = [{"date": d, "focus_sectors": t.focus_sectors, "focus_stocks": t.focus_stocks,
                   "avoid_sectors": t.avoid_sectors, "risk_level": t.risk_level,
                   "theme_drivers": t.theme_drivers} for d, t in bt.theme_history]
    with open(output / "theme_history.json", "w") as f:
        json.dump(theme_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 保存到: {output}")
    
    sells = [t for t in bt.trades if t.action == "SELL"]
    print("\n【最大盈利】")
    for t in sorted(sells, key=lambda x: -x.pnl)[:5]:
        print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
    print("\n【最大亏损】")
    for t in sorted(sells, key=lambda x: x.pnl)[:5]:
        print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
    
    print("\n" + "=" * 70)
    print("【V8.0 (后视镜) vs V8.1 (事前) 策略对比】")
    print("=" * 70)
    
    v80_file = Path("storage/backtest_v8_0/result.json")
    v5c_file = Path("storage/backtest_3y_v5c/result.json")
    
    comparisons = {"V8.1": result}
    if v80_file.exists():
        with open(v80_file) as f:
            comparisons["V8.0"] = json.load(f)
    if v5c_file.exists():
        with open(v5c_file) as f:
            comparisons["V5c"] = json.load(f)
    
    print(f"\n  {'指标':<12} ", end="")
    for name in ["V5c", "V8.0", "V8.1"]:
        print(f"{name:<15}", end="")
    print()
    print(f"  {'-'*55}")
    
    for metric, fmt in [("total_return", "+.1%"), ("alpha", "+.1%"), ("max_drawdown", ".1%"), ("sharpe", ".2f")]:
        print(f"  {metric:<12} ", end="")
        for name in ["V5c", "V8.0", "V8.1"]:
            if name in comparisons:
                val = comparisons[name].get(metric, 0)
                print(f"{val:{fmt}:<15}", end="")
            else:
                print(f"{'N/A':<15}", end="")
        print()
    
    print("\n  📊 结论:")
    if "V8.0" in comparisons:
        alpha_diff = (result['alpha'] - comparisons["V8.0"]['alpha']) * 100
        print(f"     V8.1 vs V8.0: Alpha差异 {alpha_diff:+.1f}pp")
        if alpha_diff > -40:
            print(f"     ✅ 事前数据源有效 (损失 < 40pp, 可实盘)")
        else:
            print(f"     ⚠️ 事前数据源效果有限 (损失 > 40pp)")
    
    if "V5c" in comparisons:
        alpha_vs_v5c = (result['alpha'] - comparisons["V5c"]['alpha']) * 100
        if alpha_vs_v5c > 0:
            print(f"     ✅ V8.1 优于 V5c 基线 (Alpha +{alpha_vs_v5c:.1f}pp)")
        else:
            print(f"     ❌ V8.1 不如 V5c 基线 (Alpha {alpha_vs_v5c:.1f}pp)")


if __name__ == "__main__":
    main()
