"""Market Regime Detection - 市场状态识别器.

识别当前市场处于以下状态之一：
- BULL_TREND: 牛市趋势
- BEAR_TREND: 熊市趋势
- BULL_PULLBACK: 牛市回调
- BEAR_RALLY: 熊市反弹
- RANGE_BOUND: 区间震荡
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class MarketRegime(Enum):
    """市场状态枚举"""
    BULL_TREND = "bull_trend"           # 牛市趋势
    BEAR_TREND = "bear_trend"           # 熊市趋势
    BULL_PULLBACK = "bull_pullback"     # 牛市回调
    BEAR_RALLY = "bear_rally"           # 熊市反弹
    RANGE_BOUND = "range_bound"         # 区间震荡
    UNKNOWN = "unknown"                 # 未知


@dataclass
class RegimeSignals:
    """市场状态识别的输入信号"""
    # 价格位置信号
    spy_vs_sma200: float = 0.0      # SPY 相对 200日均线位置 (%)
    spy_vs_sma50: float = 0.0       # SPY 相对 50日均线位置 (%)
    qqq_vs_sma200: float = 0.0      # QQQ 相对 200日均线位置 (%)
    
    # 均线斜率信号
    sma50_slope: float = 0.0        # 50日均线斜率 (日化)
    sma200_slope: float = 0.0       # 200日均线斜率 (日化)
    
    # 市场宽度信号
    breadth: float = 0.5            # 上涨股票占比 (0-1)
    advance_decline_ratio: float = 1.0  # 涨跌比
    
    # 新高新低信号
    new_high_count: int = 0         # 52周新高数量
    new_low_count: int = 0          # 52周新低数量
    nh_nl_ratio: float = 1.0        # 新高/新低比率
    
    # 波动率信号
    vix_value: float = 20.0         # VIX 绝对值
    vix_zscore: float = 0.0         # VIX Z-score
    vix_term_contango: bool = True  # VIX 期限结构是否正向 (正向=利多)
    
    # 动量信号
    spy_momentum_20d: float = 0.0   # SPY 20日动量
    qqq_momentum_20d: float = 0.0   # QQQ 20日动量


@dataclass
class RegimeConfig:
    """市场状态识别器配置"""
    # 各信号权重
    weights: Dict[str, float] = field(default_factory=lambda: {
        "sma200_position": 1.0,     # 价格在200日均线上方/下方
        "sma50_position": 0.8,      # 价格在50日均线上方/下方
        "sma50_slope": 1.0,         # 50日均线斜率
        "breadth": 1.0,             # 市场宽度
        "nh_nl_ratio": 0.8,         # 新高新低比
        "vix_structure": 0.8,       # VIX期限结构
        "momentum": 0.6,            # 动量
    })
    
    # 状态判定阈值
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "bull_score": 2.5,          # >= 此分数判定为牛市 (降低门槛)
        "bear_score": 1.5,          # <= 此分数判定为熊市
        "strong_trend": 4.0,        # >= 此分数判定为强趋势 (降低门槛)
        "sma_slope_threshold": 0.0005,  # 均线斜率阈值 (更敏感)
        "breadth_bullish": 0.55,    # 宽度看多阈值 (更容易满足)
        "breadth_bearish": 0.45,    # 宽度看空阈值
        "nh_nl_bullish": 1.5,       # 新高新低比看多阈值 (降低)
        "nh_nl_bearish": 0.7,       # 新高新低比看空阈值
        "vix_high": 25.0,           # VIX 高位阈值
        "vix_extreme": 35.0,        # VIX 极端阈值
    })


@dataclass
class RegimeResult:
    """市场状态识别结果"""
    regime: MarketRegime
    confidence: float               # 置信度 0-1
    bull_score: float               # 多头得分 (原始)
    signals_used: Dict[str, float]  # 各信号贡献
    description: str                # 状态描述
    recommended_exposure: float     # 建议最大仓位
    

class MarketRegimeDetector:
    """市场状态识别器
    
    基于多维度信号判断当前市场处于什么状态，
    并为后续的评分和仓位计算提供参数调整依据。
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
    
    def detect(self, signals: RegimeSignals) -> RegimeResult:
        """识别当前市场状态
        
        Args:
            signals: 市场信号数据
            
        Returns:
            RegimeResult: 识别结果
        """
        weights = self.config.weights
        thresholds = self.config.thresholds
        
        # 计算各维度得分 (每个维度贡献 0 或 1)
        scores: Dict[str, float] = {}
        
        # 1. SMA200 位置
        if signals.spy_vs_sma200 > 0:
            scores["sma200_position"] = weights["sma200_position"]
        elif signals.spy_vs_sma200 < -5:  # 明显低于200日均线
            scores["sma200_position"] = -weights["sma200_position"]
        else:
            scores["sma200_position"] = 0
            
        # 2. SMA50 位置
        if signals.spy_vs_sma50 > 0:
            scores["sma50_position"] = weights["sma50_position"]
        elif signals.spy_vs_sma50 < -3:
            scores["sma50_position"] = -weights["sma50_position"]
        else:
            scores["sma50_position"] = 0
            
        # 3. SMA50 斜率
        if signals.sma50_slope > thresholds["sma_slope_threshold"]:
            scores["sma50_slope"] = weights["sma50_slope"]
        elif signals.sma50_slope < -thresholds["sma_slope_threshold"]:
            scores["sma50_slope"] = -weights["sma50_slope"]
        else:
            scores["sma50_slope"] = 0
            
        # 4. 市场宽度
        if signals.breadth > thresholds["breadth_bullish"]:
            scores["breadth"] = weights["breadth"]
        elif signals.breadth < thresholds["breadth_bearish"]:
            scores["breadth"] = -weights["breadth"]
        else:
            scores["breadth"] = 0
            
        # 5. 新高新低比
        if signals.nh_nl_ratio > thresholds["nh_nl_bullish"]:
            scores["nh_nl_ratio"] = weights["nh_nl_ratio"]
        elif signals.nh_nl_ratio < thresholds["nh_nl_bearish"]:
            scores["nh_nl_ratio"] = -weights["nh_nl_ratio"]
        else:
            scores["nh_nl_ratio"] = 0
            
        # 6. VIX 期限结构
        if signals.vix_term_contango:  # 正向结构，市场正常
            scores["vix_structure"] = weights["vix_structure"]
        else:  # 倒挂，恐慌信号
            scores["vix_structure"] = -weights["vix_structure"]
            
        # 7. 动量
        avg_momentum = (signals.spy_momentum_20d + signals.qqq_momentum_20d) / 2
        if avg_momentum > 0.03:  # 3% 以上正动量
            scores["momentum"] = weights["momentum"]
        elif avg_momentum < -0.03:
            scores["momentum"] = -weights["momentum"]
        else:
            scores["momentum"] = 0
            
        # 计算总分
        bull_score = sum(scores.values())
        max_possible = sum(weights.values())
        
        # 判定市场状态
        regime, confidence, description = self._classify_regime(
            bull_score, max_possible, signals, thresholds
        )
        
        # 根据状态给出建议仓位上限
        recommended_exposure = self._get_recommended_exposure(regime, signals)
        
        return RegimeResult(
            regime=regime,
            confidence=confidence,
            bull_score=bull_score,
            signals_used=scores,
            description=description,
            recommended_exposure=recommended_exposure,
        )
    
    def _classify_regime(
        self,
        bull_score: float,
        max_possible: float,
        signals: RegimeSignals,
        thresholds: Dict[str, float],
    ) -> tuple[MarketRegime, float, str]:
        """根据得分判定市场状态"""
        
        # 归一化置信度
        confidence = abs(bull_score) / max_possible if max_possible > 0 else 0
        confidence = min(1.0, confidence)
        
        # VIX 极端情况优先判断
        if signals.vix_value > thresholds["vix_extreme"]:
            return (
                MarketRegime.BEAR_TREND,
                0.9,
                f"VIX 极端高位 ({signals.vix_value:.1f})，市场恐慌"
            )
        
        # 根据得分判定
        if bull_score >= thresholds["strong_trend"]:
            return (
                MarketRegime.BULL_TREND,
                confidence,
                "强势牛市：价格在均线上方，宽度健康，动量向上"
            )
        elif bull_score >= thresholds["bull_score"]:
            # 检查是否是牛市回调
            if signals.spy_vs_sma50 < 0 and signals.spy_vs_sma200 > 0:
                return (
                    MarketRegime.BULL_PULLBACK,
                    confidence,
                    "牛市回调：价格回踩50日均线，但仍在200日均线上方"
                )
            return (
                MarketRegime.BULL_TREND,
                confidence,
                "牛市趋势：多数信号看多"
            )
        elif bull_score <= -thresholds["bull_score"]:
            return (
                MarketRegime.BEAR_TREND,
                confidence,
                "熊市趋势：价格在均线下方，宽度恶化"
            )
        elif bull_score <= thresholds["bear_score"]:
            # 检查是否是熊市反弹
            if signals.spy_vs_sma50 > 0 and signals.spy_vs_sma200 < 0:
                return (
                    MarketRegime.BEAR_RALLY,
                    confidence,
                    "熊市反弹：短期反弹但长期趋势仍向下"
                )
            if bull_score < 0:
                return (
                    MarketRegime.BEAR_TREND,
                    confidence,
                    "弱势市场：多数信号偏空"
                )
        
        # 默认区间震荡
        return (
            MarketRegime.RANGE_BOUND,
            confidence,
            "区间震荡：多空信号混合，缺乏明确方向"
        )
    
    def _get_recommended_exposure(
        self,
        regime: MarketRegime,
        signals: RegimeSignals,
    ) -> float:
        """根据市场状态给出建议最大仓位"""
        
        base_exposures = {
            MarketRegime.BULL_TREND: 0.90,
            MarketRegime.BULL_PULLBACK: 0.70,
            MarketRegime.RANGE_BOUND: 0.60,
            MarketRegime.BEAR_RALLY: 0.40,
            MarketRegime.BEAR_TREND: 0.30,
            MarketRegime.UNKNOWN: 0.50,
        }
        
        exposure = base_exposures.get(regime, 0.50)
        
        # VIX 高位进一步降低仓位
        if signals.vix_value > 30:
            exposure *= 0.7
        elif signals.vix_value > 25:
            exposure *= 0.85
            
        # VIX Z-score 调整
        if signals.vix_zscore > 2.0:
            exposure *= 0.5
        elif signals.vix_zscore > 1.5:
            exposure *= 0.75
            
        return round(min(0.95, max(0.20, exposure)), 2)
    
    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "MarketRegimeDetector":
        """从配置字典创建识别器"""
        regime_config = config_dict.get("market_regimes", {}).get("detection", {})
        
        config = RegimeConfig()
        if "weights" in regime_config:
            config.weights.update(regime_config["weights"])
        if "thresholds" in regime_config:
            config.thresholds.update(regime_config["thresholds"])
            
        return cls(config)
