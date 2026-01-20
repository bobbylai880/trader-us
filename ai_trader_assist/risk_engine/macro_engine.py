"""Macro risk engine producing daily exposure targets.

重构版本：支持动态参数和市场状态感知。
Phase 2: 添加状态转换平滑逻辑。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..agent_tools.tool_math import clip
from .market_regime import MarketRegime, MarketRegimeDetector, RegimeSignals, RegimeResult
from .adaptive_params import AdaptiveParameterManager, RegimeParameters


@dataclass
class RegimeTransitionSmoother:
    """状态转换平滑器 - 避免频繁切换
    
    需要连续 N 天确认相同状态才会切换，减少噪音导致的频繁交易。
    支持动态平滑窗口：高波动时减少平滑周期以快速响应。
    """
    confirmation_periods: int = 3
    dynamic_smoothing: bool = True
    min_periods: int = 2
    max_periods: int = 5
    _history: List[MarketRegime] = field(default_factory=list)
    _confirmed_regime: MarketRegime = MarketRegime.UNKNOWN
    _current_periods: int = 3
    
    def update(self, new_regime: MarketRegime, vix_level: float = 20.0) -> MarketRegime:
        if self.dynamic_smoothing:
            self._current_periods = self._calc_dynamic_periods(vix_level)
        else:
            self._current_periods = self.confirmation_periods
        
        self._history.append(new_regime)
        if len(self._history) > self._current_periods:
            self._history.pop(0)
        
        if len(self._history) < self._current_periods:
            return self._confirmed_regime
        
        if all(r == new_regime for r in self._history[-self._current_periods:]):
            self._confirmed_regime = new_regime
        
        return self._confirmed_regime
    
    def _calc_dynamic_periods(self, vix_level: float) -> int:
        if vix_level >= 30:
            return self.min_periods
        elif vix_level >= 25:
            return self.min_periods + 1
        elif vix_level <= 15:
            return self.max_periods
        else:
            return self.confirmation_periods
    
    def reset(self) -> None:
        self._history = []
        self._confirmed_regime = MarketRegime.UNKNOWN
        self._current_periods = self.confirmation_periods
    
    @property
    def raw_regime(self) -> MarketRegime:
        return self._history[-1] if self._history else MarketRegime.UNKNOWN
    
    @property
    def smoothed_regime(self) -> MarketRegime:
        return self._confirmed_regime
    
    @property
    def current_confirmation_periods(self) -> int:
        return self._current_periods


@dataclass
class MacroRiskConfig:
    """宏观风险引擎配置"""
    # 信号权重（可从配置文件加载）
    weights: Dict[str, float] = field(default_factory=lambda: {
        "RS_SPY": 1.2,
        "RS_QQQ": 1.0,
        "VIX_Z": -1.1,
        "PUTCALL_Z": -0.9,
        "BREADTH": 0.8,
        "MOMENTUM": 0.6,
        "SMA_POSITION": 0.8,
    })
    
    # 风险等级阈值（可配置）
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "high_risk": 0.33,
        "medium_risk": 0.66,
    })
    
    # VIX 相关阈值
    vix_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "normal": 18.0,
        "elevated": 22.0,
        "high": 28.0,
        "extreme": 35.0,
    })


@dataclass
class MacroRiskEngine:
    config: MacroRiskConfig = field(default_factory=MacroRiskConfig)
    regime_detector: MarketRegimeDetector = field(default_factory=MarketRegimeDetector)
    param_manager: AdaptiveParameterManager = field(default_factory=AdaptiveParameterManager)
    smoother: RegimeTransitionSmoother = field(default_factory=lambda: RegimeTransitionSmoother(confirmation_periods=3))
    use_smoothing: bool = True
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid 函数，将任意值映射到 0-1"""
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
    
    def detect_regime(self, features: Dict[str, float]) -> RegimeResult:
        """识别当前市场状态
        
        Args:
            features: 市场特征字典
            
        Returns:
            RegimeResult: 市场状态识别结果
        """
        # 构建信号对象
        signals = RegimeSignals(
            spy_vs_sma200=features.get("spy_vs_sma200", features.get("RS_SPY", 0) * 10),
            spy_vs_sma50=features.get("spy_vs_sma50", 0),
            qqq_vs_sma200=features.get("qqq_vs_sma200", features.get("RS_QQQ", 0) * 10),
            sma50_slope=features.get("sma50_slope", 0),
            sma200_slope=features.get("sma200_slope", 0),
            breadth=features.get("BREADTH", features.get("breadth", 0.5)),
            advance_decline_ratio=features.get("advance_decline_ratio", 1.0),
            new_high_count=int(features.get("new_high_count", 0)),
            new_low_count=int(features.get("new_low_count", 0)),
            nh_nl_ratio=features.get("nh_nl_ratio", 1.0),
            vix_value=features.get("vix_value", features.get("VIX", 20.0)),
            vix_zscore=features.get("VIX_Z", features.get("vix_zscore", 0.0)),
            vix_term_contango=bool(features.get("vix_term_contango", True)),
            spy_momentum_20d=features.get("spy_momentum_20d", 0.0),
            qqq_momentum_20d=features.get("qqq_momentum_20d", 0.0),
        )
        
        return self.regime_detector.detect(signals)
    
    def evaluate(self, features: Dict[str, float]) -> Dict[str, Any]:
        """计算风险得分和目标仓位
        
        Args:
            features: 市场特征字典
            
        Returns:
            包含风险评估结果的字典
        """
        # 1. 先识别市场状态
        regime_result = self.detect_regime(features)
        
        # 2. 应用平滑逻辑（如果启用）
        vix_level = features.get("vix_value", features.get("VIX", 20.0))
        if self.use_smoothing:
            smoothed_regime = self.smoother.update(regime_result.regime, vix_level)
            self.param_manager.set_regime(smoothed_regime)
        else:
            self.param_manager.set_regime(regime_result.regime)
        
        # 2. 获取当前状态对应的参数
        params = self.param_manager.current_params
        
        # 3. 计算加权得分
        drivers = {}
        weighted_sum = 0.0
        for key, weight in self.config.weights.items():
            value = float(features.get(key, 0.0))
            contribution = weight * value
            weighted_sum += contribution
            drivers[key] = {
                "value": value,
                "weight": weight,
                "contribution": contribution,
            }
        
        # 4. 计算宏观得分
        macro_score = self._sigmoid(weighted_sum)
        
        # 5. 根据市场状态和得分计算目标仓位
        # 使用自适应参数中的 max_exposure 作为上限
        regime_max_exposure = regime_result.recommended_exposure
        param_max_exposure = params.max_exposure
        
        # 取两者中更保守的
        effective_max_exposure = min(regime_max_exposure, param_max_exposure)
        
        # 基于得分在 30% ~ max_exposure 之间线性插值
        base_exposure = 0.30
        target_exposure = clip(
            base_exposure + (effective_max_exposure - base_exposure) * macro_score,
            base_exposure,
            effective_max_exposure,
        )
        
        # 6. 判定风险等级（使用可配置阈值）
        thresholds = self.config.risk_thresholds
        if macro_score < thresholds["high_risk"]:
            risk_level = "high"
            bias = "bearish"
        elif macro_score < thresholds["medium_risk"]:
            risk_level = "medium"
            bias = "neutral"
        else:
            risk_level = "low"
            bias = "bullish"
        
        # 7. VIX 极端情况覆盖
        vix_value = features.get("vix_value", features.get("VIX", 20.0))
        vix_thresholds = self.config.vix_thresholds
        
        vix_warning = None
        if vix_value > vix_thresholds["extreme"]:
            risk_level = "extreme"
            bias = "bearish"
            target_exposure = min(target_exposure, 0.20)
            vix_warning = f"VIX 极端高位 ({vix_value:.1f})，强制降低仓位"
        elif vix_value > vix_thresholds["high"]:
            if risk_level != "high":
                risk_level = "high"
            target_exposure = min(target_exposure, 0.40)
            vix_warning = f"VIX 高位 ({vix_value:.1f})，限制仓位"
        
        return {
            "score": macro_score,
            "target_exposure": round(target_exposure, 3),
            "risk_level": risk_level,
            "bias": bias,
            "drivers": drivers,
            "regime": regime_result.regime.value,
            "regime_confidence": regime_result.confidence,
            "regime_description": regime_result.description,
            "regime_recommended_exposure": regime_result.recommended_exposure,
            "adaptive_params": self.param_manager.get_summary(),
            "vix_warning": vix_warning,
            "smoothing_enabled": self.use_smoothing,
            "smoothing_periods": self.smoother.current_confirmation_periods if self.use_smoothing else 0,
            "raw_regime": regime_result.regime.value,
            "smoothed_regime": self.smoother.smoothed_regime.value if self.use_smoothing else regime_result.regime.value,
        }
    
    def get_adaptive_params(self) -> RegimeParameters:
        """获取当前自适应参数"""
        return self.param_manager.current_params
    
    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "MacroRiskEngine":
        """从配置字典创建引擎"""
        # 加载权重配置
        risk_config = MacroRiskConfig()
        
        if "macro_risk" in config_dict:
            macro_cfg = config_dict["macro_risk"]
            if "weights" in macro_cfg:
                risk_config.weights.update(macro_cfg["weights"])
            if "risk_thresholds" in macro_cfg:
                risk_config.risk_thresholds.update(macro_cfg["risk_thresholds"])
            if "vix_thresholds" in macro_cfg:
                risk_config.vix_thresholds.update(macro_cfg["vix_thresholds"])
        
        # 创建市场状态识别器和参数管理器
        regime_detector = MarketRegimeDetector.from_config(config_dict)
        param_manager = AdaptiveParameterManager.from_config(config_dict)
        
        return cls(
            config=risk_config,
            regime_detector=regime_detector,
            param_manager=param_manager,
            smoother=RegimeTransitionSmoother(
                confirmation_periods=config_dict.get("market_regimes", {}).get("smoothing_periods", 3)
            ),
            use_smoothing=config_dict.get("market_regimes", {}).get("use_smoothing", True),
        )


# 保持向后兼容：旧版本的简化接口
def evaluate_macro_risk(features: Dict[str, float]) -> Dict[str, Any]:
    """便捷函数：使用默认配置评估宏观风险"""
    engine = MacroRiskEngine()
    return engine.evaluate(features)
