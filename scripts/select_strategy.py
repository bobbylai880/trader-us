#!/usr/bin/env python3
"""策略选择器命令行工具.

使用示例:
    # 列出所有策略
    python scripts/select_strategy.py list
    
    # 对比策略 (按夏普比率排序)
    python scripts/select_strategy.py compare --sort-by sharpe_ratio
    
    # 查看策略详情
    python scripts/select_strategy.py info v5c_preventive_risk
    
    # 根据偏好选择策略
    python scripts/select_strategy.py select --mode by_preference --preference max_return
    
    # 根据市场状态推荐
    python scripts/select_strategy.py recommend --regime bull_trend
    
    # 运行指定策略
    python scripts/select_strategy.py run --strategy v3_trend_following
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_trader_assist.strategies.selector import StrategySelector, SelectionMode


def cmd_list(args: argparse.Namespace, selector: StrategySelector) -> None:
    """列出所有策略."""
    strategies = selector.list_strategies(enabled_only=not args.all)
    
    print("\n📋 可用策略列表:")
    print("-" * 70)
    
    for info in strategies:
        status = "✅" if info.enabled else "❌"
        print(f"\n{status} {info.id}")
        print(f"   名称: {info.name} (v{info.version})")
        print(f"   描述: {info.description}")
        print(f"   风险: {info.risk_level}")
        print(f"   适用: {', '.join(info.suitable_for[:3])}")
    
    print(f"\n共 {len(strategies)} 个策略")
    print(f"默认策略: {selector.default_strategy}")


def cmd_compare(args: argparse.Namespace, selector: StrategySelector) -> None:
    """对比策略."""
    selector.print_comparison_table(sort_by=args.sort_by)


def cmd_info(args: argparse.Namespace, selector: StrategySelector) -> None:
    """查看策略详情."""
    selector.print_strategy_detail(args.strategy)


def cmd_select(args: argparse.Namespace, selector: StrategySelector) -> None:
    """选择策略."""
    mode = SelectionMode(args.mode)
    
    try:
        strategy_id = selector.select(
            mode=mode,
            strategy_id=args.strategy,
            market_regime=args.regime,
            preference=args.preference,
            risk_tolerance=args.risk,
        )
        
        info = selector.get_strategy_info(strategy_id)
        metrics = selector.get_strategy_metrics(strategy_id)
        
        print(f"\n✅ 选中策略: {strategy_id}")
        print(f"   名称: {info.name}")
        print(f"   描述: {info.description}")
        if metrics:
            print(f"   收益: {metrics.total_return:+.1%}" if metrics.total_return else "")
            print(f"   回撤: {metrics.max_drawdown:.1%}" if metrics.max_drawdown else "")
            print(f"   夏普: {metrics.sharpe_ratio:.2f}" if metrics.sharpe_ratio else "")
        
    except ValueError as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


def cmd_recommend(args: argparse.Namespace, selector: StrategySelector) -> None:
    """推荐策略."""
    regime = args.regime
    
    print(f"\n📈 市场状态: {regime or '未指定'}")
    
    strategy_id = selector.recommend(market_regime=regime)
    info = selector.get_strategy_info(strategy_id)
    metrics = selector.get_strategy_metrics(strategy_id)
    
    print(f"\n🎯 推荐策略: {strategy_id}")
    print(f"   名称: {info.name}")
    print(f"   描述: {info.description}")
    print(f"   风险: {info.risk_level}")
    
    if metrics:
        print(f"\n📊 回测表现:")
        print(f"   总收益: {metrics.total_return:+.1%}" if metrics.total_return else "")
        print(f"   Alpha: {metrics.alpha:+.1%}" if metrics.alpha else "")
        print(f"   回撤: {metrics.max_drawdown:.1%}" if metrics.max_drawdown else "")
        print(f"   夏普: {metrics.sharpe_ratio:.2f}" if metrics.sharpe_ratio else "")
    
    # 显示其他市场状态的推荐
    print("\n📋 各市场状态推荐:")
    rules = selector.get_selection_rules().get("by_market_regime", {})
    for r, s in rules.items():
        if r.startswith("_"):
            continue
        marker = "👉" if r == regime else "  "
        print(f"   {marker} {r}: {s}")


def cmd_run(args: argparse.Namespace, selector: StrategySelector) -> None:
    """运行策略."""
    strategy_id = args.strategy or selector.default_strategy
    
    info = selector.get_strategy_info(strategy_id)
    if not info:
        print(f"❌ 策略 '{strategy_id}' 不存在")
        sys.exit(1)
    
    print(f"\n🚀 准备运行策略: {info.name}")
    print(f"   模块: {info.module}")
    print(f"   类名: {info.class_name}")
    
    if args.dry_run:
        print("\n⚠️  试运行模式，不实际执行")
        return
    
    # 动态导入并运行策略
    try:
        import importlib
        
        module_path = info.module.replace("scripts.", "")
        script_path = Path(__file__).parent / f"{module_path}.py"
        
        if not script_path.exists():
            print(f"❌ 脚本文件不存在: {script_path}")
            sys.exit(1)
        
        print(f"\n📂 执行脚本: {script_path}")
        print("-" * 70)
        
        # 使用 exec 运行脚本
        import runpy
        runpy.run_path(str(script_path), run_name="__main__")
        
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        sys.exit(1)


def cmd_rules(args: argparse.Namespace, selector: StrategySelector) -> None:
    """显示选择规则."""
    rules = selector.get_selection_rules()
    
    print("\n📋 策略选择规则")
    print("=" * 70)
    
    print("\n🌍 按市场状态 (by_market_regime):")
    for regime, strategy in rules.get("by_market_regime", {}).items():
        if regime.startswith("_"):
            continue
        info = selector.get_strategy_info(strategy)
        name = info.name if info else strategy
        print(f"   {regime:20} → {name}")
    
    print("\n🎯 按用户偏好 (by_user_preference):")
    for pref, strategy in rules.get("by_user_preference", {}).items():
        if pref.startswith("_"):
            continue
        info = selector.get_strategy_info(strategy)
        name = info.name if info else strategy
        print(f"   {pref:20} → {name}")
    
    print("\n⚖️  按风险承受能力 (by_risk_tolerance):")
    for risk, strategy in rules.get("by_risk_tolerance", {}).items():
        if risk.startswith("_"):
            continue
        info = selector.get_strategy_info(strategy)
        name = info.name if info else strategy
        print(f"   {risk:20} → {name}")


def main():
    """主函数."""
    parser = argparse.ArgumentParser(
        description="策略选择器 - 管理、对比、选择交易策略",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s list                              列出所有策略
  %(prog)s compare --sort-by sharpe_ratio    按夏普比率对比
  %(prog)s info v5c_preventive_risk          查看策略详情
  %(prog)s recommend --regime bull_trend     根据市场状态推荐
  %(prog)s select --mode by_preference --preference max_return
  %(prog)s run --strategy v3_trend_following 运行指定策略
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # list 命令
    list_parser = subparsers.add_parser("list", help="列出所有可用策略")
    list_parser.add_argument("--all", "-a", action="store_true", help="包含已禁用的策略")
    
    # compare 命令
    compare_parser = subparsers.add_parser("compare", help="对比策略表现")
    compare_parser.add_argument(
        "--sort-by", "-s",
        default="sharpe_ratio",
        choices=["total_return", "alpha", "max_drawdown", "sharpe_ratio", "win_rate"],
        help="排序指标 (默认: sharpe_ratio)"
    )
    
    # info 命令
    info_parser = subparsers.add_parser("info", help="查看策略详情")
    info_parser.add_argument("strategy", help="策略ID")
    
    # select 命令
    select_parser = subparsers.add_parser("select", help="选择策略")
    select_parser.add_argument(
        "--mode", "-m",
        default="manual",
        choices=["manual", "by_regime", "by_preference", "by_risk"],
        help="选择模式"
    )
    select_parser.add_argument("--strategy", "-s", help="手动指定策略ID")
    select_parser.add_argument(
        "--regime", "-r",
        choices=["bull_trend", "bull_pullback", "range_bound", "bear_rally", "bear_trend"],
        help="市场状态"
    )
    select_parser.add_argument(
        "--preference", "-p",
        choices=["max_return", "min_drawdown", "best_balance", "high_win_rate", "sector_rotation", "all_weather"],
        help="用户偏好"
    )
    select_parser.add_argument(
        "--risk",
        choices=["aggressive", "moderate", "conservative"],
        help="风险承受能力"
    )
    
    # recommend 命令
    recommend_parser = subparsers.add_parser("recommend", help="根据市场状态推荐策略")
    recommend_parser.add_argument(
        "--regime", "-r",
        choices=["bull_trend", "bull_pullback", "range_bound", "bear_rally", "bear_trend"],
        help="当前市场状态"
    )
    
    # run 命令
    run_parser = subparsers.add_parser("run", help="运行策略回测")
    run_parser.add_argument("--strategy", "-s", help="策略ID (默认使用推荐策略)")
    run_parser.add_argument("--dry-run", "-n", action="store_true", help="试运行，不实际执行")
    
    # rules 命令
    subparsers.add_parser("rules", help="显示所有选择规则")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # 初始化选择器
    try:
        selector = StrategySelector()
    except FileNotFoundError as e:
        print(f"❌ 配置文件错误: {e}")
        sys.exit(1)
    
    # 执行命令
    commands = {
        "list": cmd_list,
        "compare": cmd_compare,
        "info": cmd_info,
        "select": cmd_select,
        "recommend": cmd_recommend,
        "run": cmd_run,
        "rules": cmd_rules,
    }
    
    handler = commands.get(args.command)
    if handler:
        handler(args, selector)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
