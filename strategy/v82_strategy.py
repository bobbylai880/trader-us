"""V8.2 主策略引擎 - 客观宏观 + 预防式风控 + 趋势跟踪"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_loader import DataLoader
from .risk_control import RiskControl, RiskState
from .macro_theme import MacroTheme, ThemeConfig

# 股票-板块映射
SYMBOL_TO_SECTOR = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "AMD": "XLK",
    "META": "XLC", "GOOGL": "XLC", "NFLX": "XLC",
    "AMZN": "XLY", "TSLA": "XLY",
}


# 股票-板块映射
SYMBOL_TO_SECTOR = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "AMD": "XLK",
    "META": "XLC", "GOOGL": "XLC", "NFLX": "XLC",
    "AMZN": "XLY", "TSLA": "XLY",
}

# 基础股票池
BASE_UNIVERSE = ["NVDA", "META", "GOOGL", "AMZN", "MSFT", "AAPL", "AMD", "AVGO", "NFLX", "TSLA"]

# 宏观数据符号
MACRO_SYMBOLS = ["SPY", "VIX", "^VIX", "HYG", "TLT", "^TNX", "XLY", "XLP", "UUP", "^IRX"]


@dataclass
class Position:
    """持仓"""
    symbol: str
    shares: int
    avg_cost: float
    entry_date: str
    highest_price: float


@dataclass
class Trade:
    """交易记录"""
    date: str
    symbol: str
    action: str  # BUY/SELL
    price: float
    shares: int
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""


@dataclass
class BacktestResult:
    """回测结果"""
    final_value: float
    total_return: float
    annualized_return: float
    spy_return: float
    alpha: float
    max_drawdown: float
    sharpe: float
    win_rate: float
    profit_factor: float
    total_trades: int
    risk_triggers: int
    theme_updates: int


class V82Strategy:
    """V8.2 策略引擎
    
    核心逻辑：
    1. 客观宏观主题生成 (HYG/TLT, XLY/XLP, VIX, TNX, UUP, IRX)
    2. 预防式4级风控 (Normal/Watch/Caution/Danger)
    3. 趋势跟踪执行 (SPY>SMA50 + 15%跟踪止损)
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[str, float, float]] = []
        self.risk_history: List[RiskState] = []
        self.theme_history: List[Tuple[str, ThemeConfig]] = []
        
        self._loader = DataLoader()
        self._risk = RiskControl()
        self._theme: Optional[MacroTheme] = None
        self._current_theme: Optional[ThemeConfig] = None
        self._dynamic_universe: List[str] = []
    
    def run(self, start: date, end: date) -> BacktestResult:
        """运行回测"""
        print("\n" + "=" * 70)
        print("V8.2 策略: 客观宏观 + 预防式风控 + 趋势跟踪")
        print("=" * 70)
        print(f"  回测区间: {start} ~ {end}")
        print(f"  初始资金: ${self.initial_capital:,.0f}")
        
        # 加载数据
        all_symbols = set(BASE_UNIVERSE) | set(MACRO_SYMBOLS)
        self._loader.load_prices(list(all_symbols), start, end, lookback=100)
        self._theme = MacroTheme(BASE_UNIVERSE)
        
        trading_days = self._loader.get_trading_days(start, end)
        print(f"  交易日数: {len(trading_days)}")
        
        # 回测主循环
        rebal_count = 0
        last_theme_update: Optional[date] = None
        last_risk_mode = "normal"
        
        for i, dt in enumerate(trading_days):
            # 更新主题 (每20天)
            if last_theme_update is None or (dt - last_theme_update).days >= 20:
                momentum_scores = self._compute_momentum(dt)
                self._current_theme = self._theme.generate_theme(dt, momentum_scores)
                self._dynamic_universe = self._build_universe(self._current_theme)
                self.theme_history.append((str(dt), self._current_theme))
                last_theme_update = dt
                
                if i == 0 or i % 60 == 0:
                    print(f"\n  📊 [{dt}] 主题更新: {self._current_theme.focus_sectors}")
            
            # 计算净值
            pv = self._portfolio_value(dt)
            spy_price = self._loader.get("SPY", dt, "close") or 0
            spy_base = self._loader.get("SPY", start, "close") or 1
            spy_val = self.initial_capital * spy_price / spy_base
            self.equity_curve.append((str(dt), pv, spy_val))
            
            if i % 150 == 0:
                print(f"  [{i+1}/{len(trading_days)}] {dt}: ${pv:,.0f} (SPY: ${spy_val:,.0f})")
            
            # 风控检查
            risk_state = self._risk.check(dt, self._loader.get)
            
            if risk_state.mode != last_risk_mode:
                self.risk_history.append(risk_state)
                if risk_state.mode != "normal":
                    print(f"\n  ⚠️ [{dt}] 风控: {risk_state.mode.upper()} - {risk_state.reason}")
                else:
                    print(f"\n  ✅ [{dt}] 风控解除")
                last_risk_mode = risk_state.mode
            
            # 预防性减仓
            if risk_state.mode != "normal":
                self._reduce_exposure(dt, risk_state.max_exposure, risk_state.reason)
            
            # 止损检查
            is_bull = self._is_bull_market(dt)
            self._check_stops(dt, is_bull)
            
            # 再平衡 (每10天)
            rebal_count += 1
            if rebal_count >= 10:
                rebal_count = 0
                self._rebalance(dt, is_bull, risk_state, pv)
        
        return self._calc_results(start, end)
    
    def _compute_momentum(self, dt: date) -> Dict[str, float]:
        scores = {}
        universe = self._dynamic_universe if self._dynamic_universe else BASE_UNIVERSE
        for sym in universe:
            mom = self._loader.get(sym, dt, "mom20")
            if mom is not None:
                scores[sym] = max(-1, min(1, mom * 5))
            else:
                scores[sym] = 0.0
        return scores
    
    def _build_universe(self, theme: ThemeConfig) -> List[str]:
        """构建动态股票池"""
        universe = set(BASE_UNIVERSE)
        universe.update(theme.focus_stocks)
        for sym in theme.avoid_stocks:
            universe.discard(sym)
        return list(universe)
    
    def _is_bull_market(self, dt: date) -> bool:
        """判断牛市环境: SPY > SMA50 且动量 > -5%"""
        spy_close = self._loader.get("SPY", dt, "close")
        spy_sma50 = self._loader.get("SPY", dt, "sma50")
        spy_mom = self._loader.get("SPY", dt, "mom20")
        if spy_close is None or spy_sma50 is None:
            return False
        return spy_close > spy_sma50 and (spy_mom is None or spy_mom > -0.05)
    
    def _portfolio_value(self, dt: date) -> float:
        """计算组合总价值"""
        pos_val = sum(
            p.shares * (self._loader.get(s, dt, "close") or p.avg_cost)
            for s, p in self.positions.items()
        )
        return self.cash + pos_val
    
    def _rank_stocks(self, dt: date) -> List[Tuple[str, float]]:
        """主题+动量排序选股"""
        if self._current_theme is None:
            return []
        
        theme = self._current_theme
        ranked = []
        
        for sym in self._dynamic_universe:
            sector = SYMBOL_TO_SECTOR.get(sym, "XLK")
            if sector in theme.focus_sectors or True:  # 简化：不排除板块
                mom = self._loader.get(sym, dt, "mom20")
                close = self._loader.get(sym, dt, "close")
                sma20 = self._loader.get(sym, dt, "sma20")
                
                if mom is None or close is None or sma20 is None:
                    continue
                if close <= sma20 or mom <= 0:
                    continue
                
                # 评分: 动量 + 板块加成 + 个股加成 + 焦点加成
                score = mom * 100
                score += theme.sector_bonus.get(sector, 0) * 30
                score += theme.stock_bonus.get(sym, 0) * 30
                if sym in theme.focus_stocks:
                    score += 3
                
                ranked.append((sym, score))
        
        ranked.sort(key=lambda x: -x[1])
        return ranked[:8]
    
    def _buy(self, sym: str, dt: date, budget: float, reason: str) -> bool:
        """买入"""
        price = self._loader.get(sym, dt, "close")
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
        """卖出"""
        if sym not in self.positions:
            return 0
        p = self.positions[sym]
        price = self._loader.get(sym, dt, "close") or p.avg_cost
        proceeds = p.shares * price
        pnl = proceeds - p.shares * p.avg_cost
        pnl_pct = pnl / (p.shares * p.avg_cost) if p.shares * p.avg_cost > 0 else 0
        self.cash += proceeds
        self.trades.append(Trade(str(dt), sym, "SELL", price, p.shares, pnl, pnl_pct, reason))
        del self.positions[sym]
        return pnl
    
    def _check_stops(self, dt: date, is_bull: bool):
        """检查止损"""
        for sym in list(self.positions.keys()):
            p = self.positions[sym]
            price = self._loader.get(sym, dt, "close")
            if not price:
                continue
            p.highest_price = max(p.highest_price, price)
            drawdown = (p.highest_price - price) / p.highest_price
            
            # 15% 跟踪止损
            if drawdown > 0.15:
                self._sell(sym, dt, f"跟踪止损({drawdown:.1%})")
                continue
            
            # 熊市保护: 8%止损
            if not is_bull and price < p.avg_cost * 0.92:
                self._sell(sym, dt, "熊市保护")
    
    def _reduce_exposure(self, dt: date, target_exp: float, reason: str):
        """预防性减仓"""
        pv = self._portfolio_value(dt)
        current_exp = (pv - self.cash) / pv if pv > 0 else 0
        if current_exp <= target_exp:
            return
        
        # 卖出盈亏最差的持仓
        holdings = []
        for sym, pos in self.positions.items():
            price = self._loader.get(sym, dt, "close") or pos.avg_cost
            pnl_pct = (price - pos.avg_cost) / pos.avg_cost
            holdings.append((sym, pnl_pct))
        
        if holdings:
            holdings.sort(key=lambda x: x[1])
            sym, _ = holdings[0]
            self._sell(sym, dt, f"预防性减仓: {reason}")
    
    def _rebalance(self, dt: date, is_bull: bool, risk: RiskState, pv: float):
        """再平衡"""
        if not is_bull:
            return
        
        if risk.mode == "normal":
            candidates = self._rank_stocks(dt)
            target_pos = 5
            pos_pct = 0.19
        elif risk.mode in ["watch", "caution"]:
            candidates = self._rank_stocks(dt)
            target_pos = 3 if risk.mode == "watch" else 2
            pos_pct = 0.15
        else:
            return  # danger 模式不开新仓
        
        for sym, _ in candidates:
            if len(self.positions) >= target_pos:
                break
            if sym in self.positions:
                continue
            
            current_exp = (pv - self.cash) / pv if pv > 0 else 0
            if current_exp >= risk.max_exposure:
                break
            
            budget = pv * pos_pct
            self._buy(sym, dt, min(budget, self.cash * 0.95), "趋势买入")
    
    def _calc_results(self, start: date, end: date) -> BacktestResult:
        """计算回测结果"""
        final = self.equity_curve[-1][1]
        spy_final = self.equity_curve[-1][2]
        total_ret = final / self.initial_capital - 1
        spy_ret = spy_final / self.initial_capital - 1
        years = (end - start).days / 365
        ann_ret = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
        
        # 最大回撤
        values = [e[1] for e in self.equity_curve]
        peak, max_dd = self.initial_capital, 0.0
        for v in values:
            peak = max(peak, v)
            max_dd = max(max_dd, (peak - v) / peak)
        
        # 夏普比率
        rets = pd.Series(values).pct_change().dropna()
        sharpe = float(np.sqrt(252) * rets.mean() / rets.std()) if rets.std() > 0 else 0
        
        # 胜率和盈亏比
        sells = [t for t in self.trades if t.action == "SELL"]
        wins = [t for t in sells if t.pnl > 0]
        win_rate = len(wins) / len(sells) if sells else 0
        total_win = sum(t.pnl for t in wins)
        total_loss = abs(sum(t.pnl for t in sells if t.pnl < 0))
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        
        risk_triggers = len([r for r in self.risk_history if r.mode != "normal"])
        
        return BacktestResult(
            final_value=final,
            total_return=total_ret,
            annualized_return=ann_ret,
            spy_return=spy_ret,
            alpha=total_ret - spy_ret,
            max_drawdown=max_dd,
            sharpe=sharpe,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            risk_triggers=risk_triggers,
            theme_updates=len(self.theme_history),
        )
    
    def save_results(self, output_dir: Path):
        """保存结果"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 交易记录
        trades_data = [
            {"date": t.date, "symbol": t.symbol, "action": t.action,
             "price": t.price, "shares": t.shares, "pnl": t.pnl,
             "pnl_pct": t.pnl_pct, "reason": t.reason}
            for t in self.trades
        ]
        with open(output_dir / "trades.json", "w") as f:
            json.dump(trades_data, f, indent=2)
        
        # 净值曲线
        pd.DataFrame(
            self.equity_curve, columns=["date", "portfolio", "spy"]
        ).to_csv(output_dir / "equity_curve.csv", index=False)
        
        print(f"\n📁 结果保存到: {output_dir}")
    
    def print_summary(self, result: BacktestResult):
        """打印摘要"""
        print("\n" + "=" * 70)
        print("V8.2 回测结果")
        print("=" * 70)
        print(f"  最终价值: ${result.final_value:,.0f}")
        print(f"  总收益率: {result.total_return:+.2%}")
        print(f"  年化收益: {result.annualized_return:+.2%}")
        print(f"  SPY收益:  {result.spy_return:+.2%}")
        print(f"  超额收益: {result.alpha:+.2%}")
        print(f"\n  最大回撤: {result.max_drawdown:.2%}")
        print(f"  夏普比率: {result.sharpe:.2f}")
        print(f"  胜率: {result.win_rate:.1%}")
        print(f"  盈亏比: {result.profit_factor:.2f}")
        print(f"  总交易: {result.total_trades} 笔")
        print(f"  风控触发: {result.risk_triggers} 次")
        
        # 最大盈亏
        sells = [t for t in self.trades if t.action == "SELL"]
        if sells:
            print("\n【最大盈利】")
            for t in sorted(sells, key=lambda x: -x.pnl)[:5]:
                print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
            print("\n【最大亏损】")
            for t in sorted(sells, key=lambda x: x.pnl)[:5]:
                print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
