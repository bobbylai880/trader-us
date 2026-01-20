"""
三套市场策略 - 根据市场状态自动切换

牛市策略: 买入持有 + 仅止损保护 (最大化趋势收益)
震荡市策略: 区间交易 + 均值回归 (高抛低吸)
熊市策略: 保守防御 + 严格风控 (保护本金)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from ai_trader_assist.risk_engine.market_regime import MarketRegime


class StrategyMode(Enum):
    BULL_TREND_FOLLOW = "bull_trend_follow"
    RANGE_MEAN_REVERT = "range_mean_revert"
    BEAR_DEFENSIVE = "bear_defensive"


@dataclass
class StrategyConfig:
    mode: StrategyMode
    max_exposure: float
    max_single_weight: float
    buy_threshold: float
    stop_loss_pct: float
    trailing_stop_pct: float
    take_profit_enabled: bool
    take_profit_pct: float
    partial_profit_enabled: bool
    score_exit_enabled: bool
    min_hold_days: int
    reentry_wait_days: int


BULL_STRATEGY = StrategyConfig(
    mode=StrategyMode.BULL_TREND_FOLLOW,
    max_exposure=0.95,
    max_single_weight=0.40,
    buy_threshold=0.40,
    stop_loss_pct=0.15,
    trailing_stop_pct=0.20,
    take_profit_enabled=False,
    take_profit_pct=0.0,
    partial_profit_enabled=False,
    score_exit_enabled=False,
    min_hold_days=20,
    reentry_wait_days=3,
)

RANGE_STRATEGY = StrategyConfig(
    mode=StrategyMode.RANGE_MEAN_REVERT,
    max_exposure=0.60,
    max_single_weight=0.20,
    buy_threshold=0.55,
    stop_loss_pct=0.08,
    trailing_stop_pct=0.0,
    take_profit_enabled=True,
    take_profit_pct=0.15,
    partial_profit_enabled=True,
    score_exit_enabled=True,
    min_hold_days=5,
    reentry_wait_days=2,
)

BEAR_STRATEGY = StrategyConfig(
    mode=StrategyMode.BEAR_DEFENSIVE,
    max_exposure=0.30,
    max_single_weight=0.10,
    buy_threshold=0.75,
    stop_loss_pct=0.05,
    trailing_stop_pct=0.0,
    take_profit_enabled=True,
    take_profit_pct=0.10,
    partial_profit_enabled=False,
    score_exit_enabled=True,
    min_hold_days=3,
    reentry_wait_days=5,
)


def get_strategy_for_regime(regime: MarketRegime) -> StrategyConfig:
    if regime in [MarketRegime.BULL_TREND, MarketRegime.BULL_PULLBACK]:
        return BULL_STRATEGY
    elif regime in [MarketRegime.BEAR_TREND, MarketRegime.BEAR_RALLY]:
        return BEAR_STRATEGY
    else:
        return RANGE_STRATEGY


class AdaptiveStrategyEngine:
    def __init__(self):
        self.current_strategy: Optional[StrategyConfig] = None
        self.strategy_history: List[Tuple[str, StrategyMode]] = []
        self.position_entry_day: int = 0
        self.last_exit_day: int = 0
        
    def update_strategy(self, regime: MarketRegime, current_day: int) -> StrategyConfig:
        new_strategy = get_strategy_for_regime(regime)
        
        if self.current_strategy is None or self.current_strategy.mode != new_strategy.mode:
            self.strategy_history.append((str(current_day), new_strategy.mode))
            self.current_strategy = new_strategy
            
        return self.current_strategy
    
    def should_buy(
        self,
        score: float,
        current_day: int,
        current_exposure: float,
    ) -> Tuple[bool, str]:
        if self.current_strategy is None:
            return False, "no_strategy"
        
        cfg = self.current_strategy
        
        wait_days = current_day - self.last_exit_day
        if wait_days < cfg.reentry_wait_days:
            return False, f"wait_{cfg.reentry_wait_days - wait_days}d"
        
        if current_exposure >= cfg.max_exposure:
            return False, "max_exposure"
        
        if score >= cfg.buy_threshold:
            return True, f"score_{score:.2f}"
        
        return False, "score_low"
    
    def should_sell(
        self,
        score: float,
        entry_price: float,
        current_price: float,
        peak_price: float,
        current_day: int,
        bb_score: float = 0.5,
    ) -> Tuple[bool, str]:
        if self.current_strategy is None:
            return False, ""
        
        cfg = self.current_strategy
        
        hold_days = current_day - self.position_entry_day
        if hold_days < cfg.min_hold_days:
            stop_price = entry_price * (1 - cfg.stop_loss_pct)
            if current_price <= stop_price:
                return True, "stop_loss_override"
            return False, f"min_hold_{cfg.min_hold_days - hold_days}d"
        
        stop_price = entry_price * (1 - cfg.stop_loss_pct)
        if current_price <= stop_price:
            return True, "stop_loss"
        
        if cfg.mode == StrategyMode.BULL_TREND_FOLLOW:
            if cfg.trailing_stop_pct > 0 and peak_price > 0:
                trailing_stop = peak_price * (1 - cfg.trailing_stop_pct)
                if current_price <= trailing_stop and current_price > entry_price:
                    return True, f"trailing_stop_{cfg.trailing_stop_pct:.0%}"
            return False, ""
        
        elif cfg.mode == StrategyMode.RANGE_MEAN_REVERT:
            if cfg.take_profit_enabled:
                profit_pct = (current_price - entry_price) / entry_price
                if profit_pct >= cfg.take_profit_pct:
                    return True, f"take_profit_{profit_pct:.0%}"
            
            if bb_score < 0.2:
                return True, "overbought_bb"
            
            if cfg.score_exit_enabled and score < 0.40:
                return True, "score_exit"
            
            return False, ""
        
        elif cfg.mode == StrategyMode.BEAR_DEFENSIVE:
            if cfg.take_profit_enabled:
                profit_pct = (current_price - entry_price) / entry_price
                if profit_pct >= cfg.take_profit_pct:
                    return True, f"take_profit_{profit_pct:.0%}"
            
            if cfg.score_exit_enabled and score < 0.50:
                return True, "score_exit"
            
            return False, ""
        
        return False, ""
    
    def record_entry(self, current_day: int):
        self.position_entry_day = current_day
    
    def record_exit(self, current_day: int):
        self.last_exit_day = current_day
        self.position_entry_day = 0
    
    def get_position_size(self, available_cash: float, price: float) -> int:
        if self.current_strategy is None:
            return 0
        
        max_position = available_cash * self.current_strategy.max_single_weight
        return int(max_position / price)
