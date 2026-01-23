"""策略选择器 - 管理、对比、选择和运行策略."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import StrategyInfo, StrategyMetrics


class SelectionMode(Enum):
    """选择模式."""
    
    MANUAL = "manual"                # 手动指定策略ID
    BY_REGIME = "by_regime"          # 根据市场状态自动选择
    BY_PREFERENCE = "by_preference"  # 根据用户偏好选择
    BY_RISK = "by_risk"              # 根据风险承受能力选择


@dataclass
class StrategyComparison:
    """策略对比结果."""
    
    strategies: List[StrategyInfo]
    metrics: Dict[str, StrategyMetrics]
    recommendation: str
    reason: str


@dataclass
class StrategyRecord:
    """策略记录 (用于内部管理)."""
    
    info: StrategyInfo
    metrics: StrategyMetrics
    raw_config: Dict[str, Any]


class StrategySelector:
    """策略选择器 - 管理、对比、选择策略.
    
    主要功能:
    1. 从配置文件加载所有策略的元信息和回测指标
    2. 提供策略对比功能
    3. 根据不同模式(手动/市场状态/用户偏好)选择策略
    4. 自动推荐最适合当前情况的策略
    
    使用示例:
        selector = StrategySelector()
        
        # 列出所有策略
        for info in selector.list_strategies():
            print(f"{info.name}: {info.description}")
        
        # 对比策略
        comparison = selector.compare_strategies(sort_by="sharpe_ratio")
        print(f"推荐: {comparison.recommendation}")
        
        # 根据市场状态选择
        strategy_id = selector.select(
            mode=SelectionMode.BY_REGIME,
            market_regime="bull_trend"
        )
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """初始化策略选择器.
        
        Args:
            config_path: 策略配置文件路径，默认为 configs/strategies.json
        """
        if config_path is None:
            # 从项目根目录查找配置文件
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "strategies.json"
        
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._strategies: Dict[str, StrategyRecord] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """加载策略配置."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"策略配置文件不存在: {self.config_path}")
        
        with open(self.config_path, encoding="utf-8") as f:
            self._config = json.load(f)
        
        # 解析每个策略
        strategies_data = self._config.get("strategies", {})
        for strategy_id, strategy_data in strategies_data.items():
            if not strategy_data.get("enabled", True):
                continue
            
            info = StrategyInfo.from_dict(strategy_data)
            
            # 解析回测指标
            backtest_data = strategy_data.get("backtest", {})
            metrics_data = backtest_data.get("metrics", {})
            metrics = StrategyMetrics.from_dict(metrics_data)
            
            self._strategies[strategy_id] = StrategyRecord(
                info=info,
                metrics=metrics,
                raw_config=strategy_data,
            )
    
    def reload_config(self) -> None:
        """重新加载配置文件."""
        self._strategies.clear()
        self._load_config()
    
    @property
    def default_strategy(self) -> str:
        """获取默认策略ID."""
        return self._config.get("default_strategy", "v5c_preventive_risk")
    
    def list_strategies(self, enabled_only: bool = True) -> List[StrategyInfo]:
        """列出所有已注册策略.
        
        Args:
            enabled_only: 是否只返回已启用的策略
            
        Returns:
            策略信息列表
        """
        result = []
        for record in self._strategies.values():
            if enabled_only and not record.info.enabled:
                continue
            result.append(record.info)
        return result
    
    def get_strategy_info(self, strategy_id: str) -> Optional[StrategyInfo]:
        """获取指定策略的信息.
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            策略信息，不存在则返回 None
        """
        record = self._strategies.get(strategy_id)
        return record.info if record else None
    
    def get_strategy_metrics(self, strategy_id: str) -> Optional[StrategyMetrics]:
        """获取指定策略的回测指标.
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            回测指标，不存在则返回 None
        """
        record = self._strategies.get(strategy_id)
        return record.metrics if record else None
    
    def compare_strategies(
        self,
        strategy_ids: Optional[List[str]] = None,
        sort_by: str = "sharpe_ratio",
        descending: bool = True,
    ) -> StrategyComparison:
        """对比策略表现.
        
        Args:
            strategy_ids: 要对比的策略ID列表，为空则对比所有策略
            sort_by: 排序指标 (total_return, alpha, max_drawdown, sharpe_ratio, win_rate)
            descending: 是否降序排序 (对于 max_drawdown 会自动反转)
            
        Returns:
            StrategyComparison: 对比结果
        """
        ids = strategy_ids or list(self._strategies.keys())
        
        infos = []
        metrics = {}
        
        for sid in ids:
            if sid in self._strategies:
                record = self._strategies[sid]
                infos.append(record.info)
                metrics[sid] = record.metrics
        
        # 排序逻辑
        def sort_key(sid: str) -> float:
            m = metrics[sid]
            value = getattr(m, sort_by, 0) or 0
            # max_drawdown 越小越好，所以取负值
            if sort_by == "max_drawdown":
                return -value if descending else value
            return value
        
        sorted_ids = sorted(metrics.keys(), key=sort_key, reverse=descending)
        
        best = sorted_ids[0] if sorted_ids else None
        best_name = self._strategies[best].info.name if best else "无"
        
        return StrategyComparison(
            strategies=infos,
            metrics=metrics,
            recommendation=best or "",
            reason=f"基于 {sort_by} 指标排名第一 ({best_name})"
        )
    
    def select(
        self,
        mode: SelectionMode = SelectionMode.MANUAL,
        strategy_id: Optional[str] = None,
        market_regime: Optional[str] = None,
        preference: Optional[str] = None,
        risk_tolerance: Optional[str] = None,
    ) -> str:
        """选择策略.
        
        Args:
            mode: 选择模式
            strategy_id: 手动模式下的策略ID
            market_regime: 市场状态 (bull_trend, bull_pullback, range_bound, bear_rally, bear_trend)
            preference: 用户偏好 (max_return, min_drawdown, best_balance, high_win_rate, sector_rotation, all_weather)
            risk_tolerance: 风险承受能力 (aggressive, moderate, conservative)
            
        Returns:
            str: 选中的策略ID
        """
        selected_id = None
        
        if mode == SelectionMode.MANUAL:
            selected_id = strategy_id or self.default_strategy
            
        elif mode == SelectionMode.BY_REGIME:
            rules = self._config.get("selection_rules", {}).get("by_market_regime", {})
            selected_id = rules.get(market_regime, self.default_strategy)
            
        elif mode == SelectionMode.BY_PREFERENCE:
            rules = self._config.get("selection_rules", {}).get("by_user_preference", {})
            selected_id = rules.get(preference, self.default_strategy)
            
        elif mode == SelectionMode.BY_RISK:
            rules = self._config.get("selection_rules", {}).get("by_risk_tolerance", {})
            selected_id = rules.get(risk_tolerance, self.default_strategy)
        
        # 验证策略存在
        if selected_id not in self._strategies:
            raise ValueError(f"策略 '{selected_id}' 不存在或未启用")
        
        return selected_id
    
    def recommend(self, market_regime: Optional[str] = None) -> str:
        """根据当前市场状态推荐策略.
        
        Args:
            market_regime: 当前市场状态，如果为 None 则返回默认策略
            
        Returns:
            str: 推荐的策略ID
        """
        if market_regime:
            return self.select(
                mode=SelectionMode.BY_REGIME,
                market_regime=market_regime
            )
        return self.default_strategy
    
    def print_comparison_table(self, sort_by: str = "sharpe_ratio") -> None:
        """打印策略对比表.
        
        Args:
            sort_by: 排序指标
        """
        comparison = self.compare_strategies(sort_by=sort_by)
        
        print("\n" + "=" * 90)
        print("📊 策略对比总览")
        print("=" * 90)
        
        # 表头
        headers = ["策略", "收益率", "Alpha", "回撤", "夏普", "胜率", "盈亏比", "风险"]
        col_widths = [16, 10, 10, 8, 8, 8, 8, 10]
        
        header_line = "| " + " | ".join(
            h.center(w) for h, w in zip(headers, col_widths)
        ) + " |"
        separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
        
        print(header_line)
        print(separator)
        
        # 按推荐排序的策略ID
        sorted_ids = sorted(
            comparison.metrics.keys(),
            key=lambda x: getattr(comparison.metrics[x], sort_by, 0) or 0,
            reverse=(sort_by != "max_drawdown")
        )
        
        # 数据行
        for sid in sorted_ids:
            info = self._strategies[sid].info
            m = comparison.metrics[sid]
            
            row = [
                info.name[:16],
                f"{m.total_return:+.1%}" if m.total_return else "N/A",
                f"{m.alpha:+.1%}" if m.alpha else "N/A",
                f"{m.max_drawdown:.1%}" if m.max_drawdown else "N/A",
                f"{m.sharpe_ratio:.2f}" if m.sharpe_ratio else "N/A",
                f"{m.win_rate:.1%}" if m.win_rate else "N/A",
                f"{m.profit_factor:.2f}" if m.profit_factor else "N/A",
                info.risk_level,
            ]
            
            row_line = "| " + " | ".join(
                str(r).center(w) for r, w in zip(row, col_widths)
            ) + " |"
            print(row_line)
        
        print(separator)
        print(f"\n📌 推荐策略: {comparison.recommendation}")
        print(f"   原因: {comparison.reason}")
    
    def print_strategy_detail(self, strategy_id: str) -> None:
        """打印策略详细信息.
        
        Args:
            strategy_id: 策略ID
        """
        record = self._strategies.get(strategy_id)
        if not record:
            print(f"❌ 策略 '{strategy_id}' 不存在")
            return
        
        info = record.info
        metrics = record.metrics
        
        print("\n" + "=" * 70)
        print(f"📋 {info.name} (v{info.version})")
        print("=" * 70)
        
        print(f"\n📝 描述: {info.description}")
        print(f"⚠️  风险等级: {info.risk_level}")
        print(f"🎯 适用场景: {', '.join(info.suitable_for)}")
        print(f"📈 推荐市场状态: {', '.join(info.recommended_regimes)}")
        
        print("\n💡 核心原理:")
        print(f"   {info.principle.core_idea}")
        
        if info.principle.entry_rules:
            print("\n📥 入场规则:")
            for rule in info.principle.entry_rules:
                print(f"   • {rule}")
        
        if info.principle.exit_rules:
            print("\n📤 出场规则:")
            for rule in info.principle.exit_rules:
                print(f"   • {rule}")
        
        if info.principle.key_improvements:
            print("\n🔧 核心改进:")
            for improvement in info.principle.key_improvements:
                print(f"   • {improvement}")
        
        if info.principle.risk_control:
            print("\n🛡️  风控机制:")
            for mode, config in info.principle.risk_control.items():
                if isinstance(config, dict):
                    trigger = config.get("trigger", "")
                    max_exp = config.get("max_exposure", 1.0)
                    cooldown = config.get("cooldown_days", 0)
                    print(f"   • {mode}: {trigger} → 最大仓位 {max_exp:.0%}, 冷却 {cooldown}天")
        
        print("\n📊 回测指标:")
        print(f"   总收益率: {metrics.total_return:+.1%}" if metrics.total_return else "   总收益率: N/A")
        print(f"   年化收益: {metrics.annualized_return:+.1%}" if metrics.annualized_return else "   年化收益: N/A")
        print(f"   Alpha: {metrics.alpha:+.1%}" if metrics.alpha else "   Alpha: N/A")
        print(f"   最大回撤: {metrics.max_drawdown:.1%}" if metrics.max_drawdown else "   最大回撤: N/A")
        print(f"   夏普比率: {metrics.sharpe_ratio:.2f}" if metrics.sharpe_ratio else "   夏普比率: N/A")
        if metrics.win_rate:
            print(f"   胜率: {metrics.win_rate:.1%}")
        if metrics.profit_factor:
            print(f"   盈亏比: {metrics.profit_factor:.2f}")
        if metrics.risk_triggers:
            print(f"   风控触发: {metrics.risk_triggers} 次")
        
        print()
    
    def get_selection_rules(self) -> Dict[str, Dict[str, str]]:
        """获取所有选择规则.
        
        Returns:
            Dict: 选择规则配置
        """
        return self._config.get("selection_rules", {})
