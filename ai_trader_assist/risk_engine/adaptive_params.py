"""Adaptive Parameters System - 自适应参数系统.

根据市场状态动态调整所有交易参数，包括：
- 买卖阈值
- 评分权重
- 止损止盈系数
- 最大仓位限制
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .market_regime import MarketRegime


@dataclass
class ScoringWeights:
    """评分权重配置"""
    trend: float = 0.20
    momentum: float = 0.20
    relative_strength: float = 0.15
    volume: float = 0.15
    structure: float = 0.10
    news: float = 0.10
    mean_reversion: float = 0.10
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "trend": self.trend,
            "momentum": self.momentum,
            "relative_strength": self.relative_strength,
            "volume": self.volume,
            "structure": self.structure,
            "news": self.news,
            "mean_reversion": self.mean_reversion,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "ScoringWeights":
        return cls(
            trend=d.get("trend", 0.20),
            momentum=d.get("momentum", 0.20),
            relative_strength=d.get("relative_strength", 0.15),
            volume=d.get("volume", 0.15),
            structure=d.get("structure", 0.10),
            news=d.get("news", 0.10),
            mean_reversion=d.get("mean_reversion", 0.10),
        )


@dataclass
class RegimeParameters:
    """单个市场状态下的参数集"""
    # 仓位控制
    max_exposure: float = 0.70
    max_single_weight: float = 0.20
    
    # 买卖阈值
    buy_threshold: float = 0.60
    hold_threshold: float = 0.45
    reduce_threshold: float = 0.40
    
    # 止损止盈
    stop_atr_mult: float = 1.5
    target_atr_mult: float = 2.5
    trailing_stop_pct: float = 0.08
    
    # 评分权重
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    
    # 风险调整
    position_size_factor: float = 1.0  # 仓位缩放因子
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_exposure": self.max_exposure,
            "max_single_weight": self.max_single_weight,
            "buy_threshold": self.buy_threshold,
            "hold_threshold": self.hold_threshold,
            "reduce_threshold": self.reduce_threshold,
            "stop_atr_mult": self.stop_atr_mult,
            "target_atr_mult": self.target_atr_mult,
            "trailing_stop_pct": self.trailing_stop_pct,
            "scoring_weights": self.scoring_weights.to_dict(),
            "position_size_factor": self.position_size_factor,
        }


# 预定义的各市场状态参数
DEFAULT_REGIME_PARAMETERS: Dict[MarketRegime, RegimeParameters] = {
    MarketRegime.BULL_TREND: RegimeParameters(
        max_exposure=0.95,           # 牛市满仓
        max_single_weight=0.30,      # 允许更大单只仓位
        buy_threshold=0.45,          # 更容易买入
        hold_threshold=0.35,
        reduce_threshold=0.30,       # 更难触发减仓
        stop_atr_mult=2.5,           # 更宽止损，给趋势空间
        target_atr_mult=6.0,         # 让利润奔跑
        trailing_stop_pct=0.15,      # 更宽追踪止损
        scoring_weights=ScoringWeights(
            trend=0.35,              # 牛市重趋势
            momentum=0.25,
            relative_strength=0.20,
            volume=0.08,
            structure=0.02,
            news=0.10,
            mean_reversion=0.00,     # 不做均值回归
        ),
        position_size_factor=1.3,    # 放大仓位
    ),
    
    MarketRegime.BULL_PULLBACK: RegimeParameters(
        max_exposure=0.75,
        max_single_weight=0.20,
        buy_threshold=0.50,          # 回调时更积极买入
        hold_threshold=0.40,
        reduce_threshold=0.35,
        stop_atr_mult=1.5,
        target_atr_mult=2.5,
        trailing_stop_pct=0.08,
        scoring_weights=ScoringWeights(
            trend=0.20,
            momentum=0.15,
            relative_strength=0.25,  # 重相对强度，找强势股
            volume=0.15,
            structure=0.10,
            news=0.10,
            mean_reversion=0.05,
        ),
        position_size_factor=1.0,
    ),
    
    MarketRegime.RANGE_BOUND: RegimeParameters(
        max_exposure=0.60,
        max_single_weight=0.15,
        buy_threshold=0.65,          # 震荡市要求更高
        hold_threshold=0.50,
        reduce_threshold=0.45,
        stop_atr_mult=1.2,           # 更紧止损
        target_atr_mult=2.0,
        trailing_stop_pct=0.06,
        scoring_weights=ScoringWeights(
            trend=0.10,
            momentum=0.15,
            relative_strength=0.15,
            volume=0.15,
            structure=0.15,
            news=0.10,
            mean_reversion=0.20,     # 震荡市重均值回归
        ),
        position_size_factor=0.8,
    ),
    
    MarketRegime.BEAR_RALLY: RegimeParameters(
        max_exposure=0.40,
        max_single_weight=0.12,
        buy_threshold=0.70,          # 熊市反弹要非常谨慎
        hold_threshold=0.55,
        reduce_threshold=0.50,
        stop_atr_mult=1.0,           # 紧止损
        target_atr_mult=1.5,         # 较低目标，快进快出
        trailing_stop_pct=0.05,
        scoring_weights=ScoringWeights(
            trend=0.10,
            momentum=0.20,
            relative_strength=0.20,
            volume=0.15,
            structure=0.10,
            news=0.15,
            mean_reversion=0.10,
        ),
        position_size_factor=0.6,
    ),
    
    MarketRegime.BEAR_TREND: RegimeParameters(
        max_exposure=0.30,
        max_single_weight=0.10,
        buy_threshold=0.80,          # 几乎不买
        hold_threshold=0.60,
        reduce_threshold=0.55,       # 更容易减仓
        stop_atr_mult=0.8,           # 非常紧止损
        target_atr_mult=1.2,
        trailing_stop_pct=0.04,
        scoring_weights=ScoringWeights(
            trend=0.05,
            momentum=0.10,
            relative_strength=0.15,
            volume=0.10,
            structure=0.10,
            news=0.15,
            mean_reversion=0.35,     # 熊市重均值回归/超卖反弹
        ),
        position_size_factor=0.5,
    ),
    
    MarketRegime.UNKNOWN: RegimeParameters(
        max_exposure=0.50,
        max_single_weight=0.15,
        buy_threshold=0.65,
        hold_threshold=0.50,
        reduce_threshold=0.45,
        stop_atr_mult=1.5,
        target_atr_mult=2.0,
        trailing_stop_pct=0.06,
        scoring_weights=ScoringWeights(),  # 默认权重
        position_size_factor=0.7,
    ),
}


class AdaptiveParameterManager:
    """自适应参数管理器
    
    根据当前市场状态提供相应的参数集，
    支持从配置文件加载自定义参数。
    """
    
    def __init__(
        self,
        regime_params: Optional[Dict[MarketRegime, RegimeParameters]] = None,
    ):
        self.regime_params = regime_params or DEFAULT_REGIME_PARAMETERS.copy()
        self._current_regime: MarketRegime = MarketRegime.UNKNOWN
        self._override_params: Optional[RegimeParameters] = None
    
    def set_regime(self, regime: MarketRegime) -> None:
        """设置当前市场状态"""
        self._current_regime = regime
    
    def get_params(self, regime: Optional[MarketRegime] = None) -> RegimeParameters:
        """获取指定市场状态的参数，默认返回当前状态参数"""
        if self._override_params is not None:
            return self._override_params
        
        target_regime = regime or self._current_regime
        return self.regime_params.get(target_regime, self.regime_params[MarketRegime.UNKNOWN])
    
    def set_override(self, params: RegimeParameters) -> None:
        """设置覆盖参数（用于手动干预）"""
        self._override_params = params
    
    def clear_override(self) -> None:
        """清除覆盖参数"""
        self._override_params = None
    
    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime
    
    @property
    def current_params(self) -> RegimeParameters:
        return self.get_params()
    
    # 便捷属性访问
    @property
    def max_exposure(self) -> float:
        return self.current_params.max_exposure
    
    @property
    def max_single_weight(self) -> float:
        return self.current_params.max_single_weight
    
    @property
    def buy_threshold(self) -> float:
        return self.current_params.buy_threshold
    
    @property
    def reduce_threshold(self) -> float:
        return self.current_params.reduce_threshold
    
    @property
    def stop_atr_mult(self) -> float:
        return self.current_params.stop_atr_mult
    
    @property
    def target_atr_mult(self) -> float:
        return self.current_params.target_atr_mult
    
    @property
    def scoring_weights(self) -> ScoringWeights:
        return self.current_params.scoring_weights
    
    def get_summary(self) -> Dict[str, Any]:
        """获取当前参数摘要"""
        params = self.current_params
        return {
            "regime": self._current_regime.value,
            "max_exposure": params.max_exposure,
            "max_single_weight": params.max_single_weight,
            "buy_threshold": params.buy_threshold,
            "reduce_threshold": params.reduce_threshold,
            "stop_atr_mult": params.stop_atr_mult,
            "target_atr_mult": params.target_atr_mult,
            "position_size_factor": params.position_size_factor,
            "has_override": self._override_params is not None,
        }
    
    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "AdaptiveParameterManager":
        """从配置字典创建管理器"""
        regime_config = config_dict.get("market_regimes", {})
        
        regime_params = DEFAULT_REGIME_PARAMETERS.copy()
        
        # 遍历配置中的各状态参数
        regime_mapping = {
            "bull_trend": MarketRegime.BULL_TREND,
            "bull_pullback": MarketRegime.BULL_PULLBACK,
            "range_bound": MarketRegime.RANGE_BOUND,
            "bear_rally": MarketRegime.BEAR_RALLY,
            "bear_trend": MarketRegime.BEAR_TREND,
        }
        
        for key, regime in regime_mapping.items():
            if key in regime_config:
                cfg = regime_config[key]
                base_params = regime_params[regime]
                
                # 更新参数
                updated = RegimeParameters(
                    max_exposure=cfg.get("max_exposure", base_params.max_exposure),
                    max_single_weight=cfg.get("max_single_weight", base_params.max_single_weight),
                    buy_threshold=cfg.get("buy_threshold", base_params.buy_threshold),
                    hold_threshold=cfg.get("hold_threshold", base_params.hold_threshold),
                    reduce_threshold=cfg.get("reduce_threshold", base_params.reduce_threshold),
                    stop_atr_mult=cfg.get("stop_atr_mult", base_params.stop_atr_mult),
                    target_atr_mult=cfg.get("target_atr_mult", base_params.target_atr_mult),
                    trailing_stop_pct=cfg.get("trailing_stop_pct", base_params.trailing_stop_pct),
                    position_size_factor=cfg.get("position_size_factor", base_params.position_size_factor),
                )
                
                # 更新评分权重
                if "scoring_weights" in cfg:
                    updated.scoring_weights = ScoringWeights.from_dict(cfg["scoring_weights"])
                else:
                    updated.scoring_weights = base_params.scoring_weights
                
                regime_params[regime] = updated
        
        return cls(regime_params)
