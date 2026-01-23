"""策略基类和数据结构定义."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional


@dataclass
class StrategyMetrics:
    """策略回测指标."""
    
    total_return: float
    annualized_return: float
    spy_return: float
    alpha: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    total_trades: Optional[int] = None
    risk_triggers: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyMetrics":
        """从字典创建指标对象."""
        return cls(
            total_return=data.get("total_return") or 0,
            annualized_return=data.get("annualized_return") or 0,
            spy_return=data.get("spy_return") or 0,
            alpha=data.get("alpha") or 0,
            max_drawdown=data.get("max_drawdown") or 0,
            sharpe_ratio=data.get("sharpe_ratio") or 0,
            win_rate=data.get("win_rate"),
            profit_factor=data.get("profit_factor"),
            total_trades=data.get("total_trades"),
            risk_triggers=data.get("risk_triggers"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "spy_return": self.spy_return,
            "alpha": self.alpha,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "risk_triggers": self.risk_triggers,
        }


@dataclass
class StrategyPrinciple:
    """策略原理说明."""
    
    core_idea: str
    entry_rules: List[str] = field(default_factory=list)
    exit_rules: List[str] = field(default_factory=list)
    key_improvements: List[str] = field(default_factory=list)
    risk_control: Optional[Dict[str, Any]] = None
    position_sizing: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyPrinciple":
        """从字典创建."""
        return cls(
            core_idea=data.get("core_idea", ""),
            entry_rules=data.get("entry_rules", []),
            exit_rules=data.get("exit_rules", []),
            key_improvements=data.get("key_improvements", []),
            risk_control=data.get("risk_control"),
            position_sizing=data.get("position_sizing"),
        )


@dataclass
class StrategyInfo:
    """策略元信息."""
    
    id: str
    name: str
    version: str
    description: str
    principle: StrategyPrinciple
    risk_level: str  # low, medium, medium-high, high
    suitable_for: List[str]
    recommended_regimes: List[str]
    enabled: bool = True
    module: Optional[str] = None
    class_name: Optional[str] = None
    result_path: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyInfo":
        """从字典创建策略信息."""
        principle_data = data.get("principle", {})
        backtest_data = data.get("backtest", {})
        
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            version=data.get("version", ""),
            description=data.get("description", ""),
            principle=StrategyPrinciple.from_dict(principle_data),
            risk_level=data.get("risk_level", "medium"),
            suitable_for=data.get("suitable_for", []),
            recommended_regimes=data.get("recommended_regimes", []),
            enabled=data.get("enabled", True),
            module=data.get("module"),
            class_name=data.get("class"),
            result_path=backtest_data.get("result_path"),
        )


class BaseStrategy(ABC):
    """策略基类 - 所有策略必须继承此类."""
    
    @classmethod
    @abstractmethod
    def get_info(cls) -> StrategyInfo:
        """返回策略元信息.
        
        Returns:
            StrategyInfo: 策略的元信息，包括名称、描述、原理等
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_backtest_metrics(cls) -> StrategyMetrics:
        """返回最新回测指标.
        
        Returns:
            StrategyMetrics: 策略的回测性能指标
        """
        pass
    
    @abstractmethod
    def run(self, start: date, end: date) -> Dict[str, Any]:
        """运行策略回测.
        
        Args:
            start: 回测开始日期
            end: 回测结束日期
            
        Returns:
            Dict: 回测结果，包含收益、回撤、交易记录等
        """
        pass
    
    @abstractmethod
    def generate_signals(self, dt: date) -> Dict[str, str]:
        """生成交易信号 (用于实时运行).
        
        Args:
            dt: 当前日期
            
        Returns:
            Dict: 交易信号，键为股票代码，值为操作建议
        """
        pass
