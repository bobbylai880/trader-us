"""Sector and stock scoring with adaptive parameters.

重构版本：支持市场状态感知和自适应权重/阈值。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..risk_engine.market_regime import MarketRegime
from ..risk_engine.adaptive_params import (
    AdaptiveParameterManager,
    RegimeParameters,
    ScoringWeights,
)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _finite(value: float, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


@dataclass
class StockScoringConfig:
    buy_threshold: float = 0.60
    hold_threshold: float = 0.45
    reduce_threshold: float = 0.40
    premarket_penalty_weight: float = 0.25
    volatility_penalty_factor: float = 0.10
    news_bonus_factor: float = 0.08
    sector_linkage_factor: float = 0.08
    earnings_penalty_enabled: bool = True
    concentration_limit: float = 0.30


STOCK_SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK",
    "AMZN": "XLY", "META": "XLC", "GOOGL": "XLC", "GOOG": "XLC",
    "TSLA": "XLY", "JPM": "XLF", "BAC": "XLF", "WFC": "XLF",
    "JNJ": "XLV", "UNH": "XLV", "PFE": "XLV",
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE",
}


@dataclass
class StockDecisionEngine:
    config: StockScoringConfig = field(default_factory=StockScoringConfig)
    param_manager: AdaptiveParameterManager = field(default_factory=AdaptiveParameterManager)
    sector_scores: Dict[str, float] = field(default_factory=dict)
    buy_threshold: float = 0.60
    reduce_threshold: float = 0.45
    
    def set_regime(self, regime: MarketRegime) -> None:
        self.param_manager.set_regime(regime)
    
    def set_sector_scores(self, sector_features: Dict[str, Dict]) -> None:
        for symbol, feats in sector_features.items():
            momentum = _finite(feats.get("momentum_20d", 0.0))
            rs = _finite(feats.get("rs", 0.0))
            self.sector_scores[symbol] = 0.5 + (momentum + rs) / 2
    
    def get_current_params(self) -> RegimeParameters:
        return self.param_manager.current_params
    
    def get_sector_boost(self, symbol: str) -> float:
        sector = STOCK_SECTOR_MAP.get(symbol, "")
        if not sector:
            return 0.0
        sector_score = self.sector_scores.get(sector, 0.5)
        return self.config.sector_linkage_factor * (sector_score - 0.5)
    
    def get_earnings_penalty(self, days_to_earnings: Optional[int]) -> float:
        if not self.config.earnings_penalty_enabled or days_to_earnings is None:
            return 0.0
        if days_to_earnings <= 5:
            return 0.25
        elif days_to_earnings <= 10:
            return 0.12
        elif days_to_earnings <= 14:
            return 0.05
        return 0.0
    
    def get_thresholds(self) -> Dict[str, float]:
        params = self.param_manager.current_params
        return {
            "buy": params.buy_threshold,
            "hold": params.hold_threshold,
            "reduce": params.reduce_threshold,
        }
    
    def get_weights(self) -> ScoringWeights:
        return self.param_manager.current_params.scoring_weights

    def score_sectors(self, sector_features: Dict[str, Dict]) -> List[Dict]:
        self.set_sector_scores(sector_features)
        weights = self.get_weights()
        
        results: List[Dict] = []
        for symbol, feats in sector_features.items():
            momentum5 = float(feats.get("momentum_5d", 0.0))
            momentum20 = float(feats.get("momentum_20d", 0.0))
            relative_strength = float(feats.get("rs", 0.0))
            volume_trend = float(feats.get("volume_trend", 0.0))
            news = float(feats.get("news_score", 0.0))
            trend_strength = float(feats.get("trend_strength", 0.0))
            
            # 使用自适应权重
            score = (
                weights.momentum * (momentum5 + momentum20) / 2
                + weights.relative_strength * relative_strength
                + weights.volume * volume_trend
                + weights.news * news
                + weights.trend * trend_strength
            )
            
            results.append({
                "symbol": symbol,
                "score": score,
                "features": {
                    "momentum_5d": momentum5,
                    "momentum_20d": momentum20,
                    "rs": relative_strength,
                    "volume_trend": volume_trend,
                    "news_score": news,
                    "news": feats.get("news", []),
                    "trend_strength": trend_strength,
                    "trend_state": feats.get("trend_state", "flat"),
                },
            })
        
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def score_stocks(
        self,
        stock_features: Dict[str, Dict],
        premarket_flags: Dict[str, Dict],
    ) -> List[Dict]:
        params = self.param_manager.current_params
        weights = params.scoring_weights
        thresholds = self.get_thresholds()
        
        scored: List[Dict] = []
        for symbol, feats in stock_features.items():
            rsi_norm = _finite(feats.get("rsi_norm", 0.5), default=0.5)
            macd_signal = _finite(feats.get("macd_signal", 0.0))
            trend_slope = _finite(feats.get("trend_slope", 0.0))
            volume_score = _finite(feats.get("volume_score", 0.0))
            structure_score = _finite(feats.get("structure_score", 0.0))
            risk_modifier = _finite(feats.get("risk_modifier", 0.0))
            news_score = _finite(feats.get("news_score", 0.0))
            trend_strength = _finite(feats.get("trend_strength", 0.0))
            trend_state = feats.get("trend_state", "flat")
            momentum_10d = _finite(feats.get("momentum_10d", 0.0))
            volatility_trend = _finite(feats.get("volatility_trend", 1.0), default=1.0)
            position_shares = _finite(feats.get("position_shares", 0.0))
            relative_strength = _finite(feats.get("rs", 0.0))
            days_to_earnings = feats.get("days_to_earnings")

            trend_component = _clip01(0.5 + 0.5 * trend_strength)
            if trend_state == "uptrend":
                trend_component = _clip01(trend_component + 0.1)
            elif trend_state == "downtrend":
                trend_component = _clip01(trend_component - 0.1)

            momentum_component = _clip01(0.5 + momentum_10d / 0.2)
            macd_component = _clip01(0.5 + macd_signal * 4)
            slope_component = _clip01(0.5 + trend_slope * 10)
            volume_component = _clip01(0.5 + volume_score / 2)
            structure_component = _clip01(0.5 + structure_score / 2)
            rs_component = _clip01(0.5 + relative_strength / 2)
            
            mean_reversion_component = 0.5
            if rsi_norm < 0.3:
                mean_reversion_component = 0.7
            elif rsi_norm > 0.7:
                mean_reversion_component = 0.3

            base_score = (
                weights.trend * trend_component
                + weights.momentum * momentum_component
                + weights.relative_strength * rs_component
                + weights.volume * volume_component
                + weights.structure * structure_component
                + weights.mean_reversion * mean_reversion_component
            )
            
            base_score += risk_modifier
            base_score += self.config.news_bonus_factor * news_score
            
            sector_boost = self.get_sector_boost(symbol)
            base_score += sector_boost
            
            earnings_penalty = self.get_earnings_penalty(days_to_earnings)
            base_score -= earnings_penalty
            
            vol_penalty = max(0.0, volatility_trend - 1.0) * self.config.volatility_penalty_factor
            base_score -= vol_penalty
            
            base_score = _clip01(base_score)

            premarket = premarket_flags.get(symbol, {})
            pre_penalty = _finite(premarket.get("score", 0.0)) * self.config.premarket_penalty_weight
            adjusted_score = max(0.0, min(1.0, base_score * (1 - pre_penalty)))

            if adjusted_score >= thresholds["buy"]:
                action = "buy"
            elif adjusted_score < thresholds["reduce"]:
                action = "reduce" if position_shares > 0 else "avoid"
            elif adjusted_score < thresholds["hold"]:
                action = "hold_weak"
            else:
                action = "hold"

            scored.append({
                "symbol": symbol,
                "score": adjusted_score,
                "action": action,
                "confidence": adjusted_score,
                "atr_pct": _finite(feats.get("atr_pct", 0.02), default=0.02),
                "price": _finite(feats.get("price", 0.0)),
                "premarket": _finite(premarket.get("score", 0.0)),
                "trend_score": trend_component,
                "position_shares": position_shares,
                "sector_boost": sector_boost,
                "earnings_penalty": earnings_penalty,
                "thresholds_used": thresholds,
                "regime": self.param_manager.current_regime.value,
                "features": {
                    "rsi_norm": rsi_norm,
                    "macd_signal": macd_signal,
                    "trend_slope": trend_slope,
                    "volume_score": volume_score,
                    "structure_score": structure_score,
                    "risk_modifier": risk_modifier,
                    "news_score": news_score,
                    "recent_news": feats.get("recent_news", []),
                    "trend_slope_5d": float(feats.get("trend_slope_5d", 0.0)),
                    "trend_slope_20d": float(feats.get("trend_slope_20d", 0.0)),
                    "momentum_10d": momentum_10d,
                    "volatility_trend": volatility_trend,
                    "moving_avg_cross": int(feats.get("moving_avg_cross", 0)),
                    "trend_strength": trend_strength,
                    "trend_state": trend_state,
                    "momentum_state": feats.get("momentum_state", "stable"),
                    "trend_score": trend_component,
                    "position_shares": position_shares,
                    "position_value": _finite(feats.get("position_value", 0.0)),
                    "days_to_earnings": days_to_earnings,
                },
            })

        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored
    
    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "StockDecisionEngine":
        """从配置字典创建引擎"""
        scoring_config = StockScoringConfig()
        
        if "scoring" in config_dict:
            cfg = config_dict["scoring"]
            scoring_config.buy_threshold = cfg.get("buy_threshold", 0.60)
            scoring_config.hold_threshold = cfg.get("hold_threshold", 0.45)
            scoring_config.reduce_threshold = cfg.get("reduce_threshold", 0.40)
            scoring_config.premarket_penalty_weight = cfg.get("premarket_penalty_weight", 0.25)
            scoring_config.volatility_penalty_factor = cfg.get("volatility_penalty_factor", 0.10)
            scoring_config.news_bonus_factor = cfg.get("news_bonus_factor", 0.08)
        
        param_manager = AdaptiveParameterManager.from_config(config_dict)
        
        return cls(
            config=scoring_config,
            param_manager=param_manager,
        )
