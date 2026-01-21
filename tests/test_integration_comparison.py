"""Integration test: Compare old vs new system outputs - 集成测试对比报告"""
import json
from datetime import datetime
from pathlib import Path

from ai_trader_assist.risk_engine.macro_engine import MacroRiskEngine
from ai_trader_assist.risk_engine.market_regime import MarketRegime, RegimeSignals
from ai_trader_assist.risk_engine.adaptive_params import (
    AdaptiveParameterManager,
    DEFAULT_REGIME_PARAMETERS,
)


def generate_comparison_report():
    """Generate comparison report between old hardcoded and new adaptive system"""
    
    report_lines = [
        "# AI Trader Assist - Phase 1 进化迭代对比报告",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. 市场状态参数对比",
        "",
        "### 1.1 旧系统 (硬编码参数)",
        "",
        "| 参数 | 值 | 说明 |",
        "|------|-----|------|",
        "| buy_threshold | 0.60 | 固定，不随市场变化 |",
        "| max_exposure | 0.85 | 固定上限 |",
        "| stop_atr_mult | 1.5 | 固定止损系数 |",
        "| scoring_weights | 固定 | 趋势/动量/相对强度等权重不变 |",
        "",
        "### 1.2 新系统 (自适应参数)",
        "",
    ]
    
    # Generate table for each regime
    report_lines.append("| 市场状态 | buy_threshold | max_exposure | stop_atr | mean_reversion权重 |")
    report_lines.append("|----------|---------------|--------------|----------|-------------------|")
    
    for regime in [
        MarketRegime.BULL_TREND,
        MarketRegime.BULL_PULLBACK,
        MarketRegime.RANGE_BOUND,
        MarketRegime.BEAR_RALLY,
        MarketRegime.BEAR_TREND,
    ]:
        params = DEFAULT_REGIME_PARAMETERS[regime]
        mr_weight = params.scoring_weights.mean_reversion
        report_lines.append(
            f"| {regime.value} | {params.buy_threshold:.2f} | {params.max_exposure:.2f} | "
            f"{params.stop_atr_mult:.1f} | {mr_weight:.2f} |"
        )
    
    report_lines.extend([
        "",
        "**关键差异：**",
        "- 牛市时 buy_threshold 降至 0.55（更容易买入），熊市升至 0.80（几乎不买）",
        "- 牛市 max_exposure 可达 90%，熊市仅 30%",
        "- 熊市 mean_reversion 权重 0.35，利于捕捉超卖反弹",
        "",
        "---",
        "",
        "## 2. 场景模拟对比",
        "",
    ])
    
    # Define test scenarios
    scenarios = [
        {
            "name": "牛市趋势",
            "features": {
                "RS_SPY": 0.8, "RS_QQQ": 0.7, "VIX_Z": -1.0, "PUTCALL_Z": -0.5,
                "BREADTH": 0.75, "MOMENTUM": 0.5, "SMA_POSITION": 0.8,
                "spy_vs_sma200": 8.0, "spy_vs_sma50": 3.0, "sma50_slope": 0.003,
                "vix_value": 12.0, "vix_term_contango": True,
                "spy_momentum_20d": 0.08, "qqq_momentum_20d": 0.07,
                "nh_nl_ratio": 3.0,
            },
        },
        {
            "name": "牛市回调",
            "features": {
                "RS_SPY": 0.3, "RS_QQQ": 0.2, "VIX_Z": 0.5, "PUTCALL_Z": 0.3,
                "BREADTH": 0.45, "MOMENTUM": -0.1, "SMA_POSITION": 0.4,
                "spy_vs_sma200": 4.0, "spy_vs_sma50": -2.0, "sma50_slope": 0.001,
                "vix_value": 22.0, "vix_term_contango": True,
                "spy_momentum_20d": -0.03, "qqq_momentum_20d": -0.02,
                "nh_nl_ratio": 1.2,
            },
        },
        {
            "name": "区间震荡",
            "features": {
                "RS_SPY": 0.1, "RS_QQQ": 0.0, "VIX_Z": 0.0, "PUTCALL_Z": 0.0,
                "BREADTH": 0.50, "MOMENTUM": 0.0, "SMA_POSITION": 0.0,
                "spy_vs_sma200": 1.0, "spy_vs_sma50": -0.5, "sma50_slope": 0.0002,
                "vix_value": 18.0, "vix_term_contango": True,
                "spy_momentum_20d": 0.01, "qqq_momentum_20d": 0.00,
                "nh_nl_ratio": 1.0,
            },
        },
        {
            "name": "熊市反弹",
            "features": {
                "RS_SPY": 0.2, "RS_QQQ": 0.3, "VIX_Z": 0.8, "PUTCALL_Z": 0.5,
                "BREADTH": 0.40, "MOMENTUM": 0.1, "SMA_POSITION": -0.3,
                "spy_vs_sma200": -5.0, "spy_vs_sma50": 1.0, "sma50_slope": -0.001,
                "vix_value": 26.0, "vix_term_contango": False,
                "spy_momentum_20d": 0.04, "qqq_momentum_20d": 0.05,
                "nh_nl_ratio": 0.8,
            },
        },
        {
            "name": "熊市趋势",
            "features": {
                "RS_SPY": -0.5, "RS_QQQ": -0.6, "VIX_Z": 2.0, "PUTCALL_Z": 1.5,
                "BREADTH": 0.25, "MOMENTUM": -0.5, "SMA_POSITION": -0.8,
                "spy_vs_sma200": -12.0, "spy_vs_sma50": -6.0, "sma50_slope": -0.004,
                "vix_value": 35.0, "vix_term_contango": False,
                "spy_momentum_20d": -0.12, "qqq_momentum_20d": -0.10,
                "nh_nl_ratio": 0.3,
            },
        },
        {
            "name": "极端恐慌",
            "features": {
                "RS_SPY": -0.8, "RS_QQQ": -0.9, "VIX_Z": 3.0, "PUTCALL_Z": 2.0,
                "BREADTH": 0.15, "MOMENTUM": -0.8, "SMA_POSITION": -1.0,
                "spy_vs_sma200": -18.0, "spy_vs_sma50": -10.0, "sma50_slope": -0.006,
                "vix_value": 45.0, "vix_term_contango": False,
                "spy_momentum_20d": -0.20, "qqq_momentum_20d": -0.18,
                "nh_nl_ratio": 0.1,
            },
        },
    ]
    
    # Run engine for each scenario
    engine = MacroRiskEngine()
    
    report_lines.append("| 场景 | 识别状态 | 风险等级 | 目标仓位 | buy_threshold | 置信度 |")
    report_lines.append("|------|----------|----------|----------|---------------|--------|")
    
    scenario_results = []
    for scenario in scenarios:
        result = engine.evaluate(scenario["features"])
        params = engine.get_adaptive_params()
        
        scenario_results.append({
            "name": scenario["name"],
            "regime": result["regime"],
            "risk_level": result["risk_level"],
            "target_exposure": result["target_exposure"],
            "buy_threshold": params.buy_threshold,
            "confidence": result["regime_confidence"],
        })
        
        report_lines.append(
            f"| {scenario['name']} | {result['regime']} | {result['risk_level']} | "
            f"{result['target_exposure']:.1%} | {params.buy_threshold:.2f} | "
            f"{result['regime_confidence']:.2f} |"
        )
    
    report_lines.extend([
        "",
        "---",
        "",
        "## 3. 新旧系统行为对比",
        "",
        "### 3.1 买入决策对比（假设某股票综合得分 0.62）",
        "",
        "| 市场状态 | 旧系统 (threshold=0.60) | 新系统 | 差异 |",
        "|----------|-------------------------|--------|------|",
        "| 牛市趋势 | ✅ 买入 | ✅ 买入 (threshold=0.55) | 新系统更积极 |",
        "| 区间震荡 | ✅ 买入 | ❌ 观望 (threshold=0.65) | 新系统更谨慎 |",
        "| 熊市趋势 | ✅ 买入 | ❌ 观望 (threshold=0.80) | 新系统保护资金 |",
        "",
        "### 3.2 仓位控制对比",
        "",
        "| 市场状态 | 旧系统 max_exposure | 新系统 max_exposure | 差异 |",
        "|----------|---------------------|---------------------|------|",
        "| 牛市趋势 | 85% | 90% | 新系统允许更高仓位 |",
        "| 区间震荡 | 85% | 60% | 新系统自动降仓 |",
        "| 熊市趋势 | 85% | 30% | 新系统大幅降仓保护 |",
        "",
        "### 3.3 止损系数对比",
        "",
        "| 市场状态 | 旧系统 stop_atr | 新系统 stop_atr | 说明 |",
        "|----------|-----------------|-----------------|------|",
        "| 牛市趋势 | 1.5 | 2.0 | 给趋势更多空间 |",
        "| 区间震荡 | 1.5 | 1.2 | 更紧止损避免假突破 |",
        "| 熊市趋势 | 1.5 | 0.8 | 严格止损保护本金 |",
        "",
        "---",
        "",
        "## 4. 评分权重对比",
        "",
        "### 4.1 牛市趋势权重",
        "```",
        "trend: 0.30        # 重趋势追随",
        "momentum: 0.25     # 重动量",
        "mean_reversion: 0  # 不做均值回归",
        "```",
        "",
        "### 4.2 熊市趋势权重",
        "```",
        "trend: 0.05        # 弱化趋势",
        "momentum: 0.10",
        "mean_reversion: 0.35  # 重均值回归，捕捉超卖反弹",
        "```",
        "",
        "### 4.3 区间震荡权重",
        "```",
        "trend: 0.10",
        "structure: 0.15    # 重结构（支撑阻力）",
        "mean_reversion: 0.20  # 适度均值回归",
        "```",
        "",
        "---",
        "",
        "## 5. 结论",
        "",
        "### 5.1 主要改进",
        "",
        "1. **市场状态感知**：系统现在能识别 5 种市场状态，而非一刀切",
        "2. **动态参数调整**：买卖阈值、仓位上限、止损系数随市场变化",
        "3. **权重自适应**：牛市重趋势追随，熊市重均值回归",
        "4. **风险保护**：熊市自动降低仓位至 30%，极端恐慌降至 20%",
        "",
        "### 5.2 预期效果",
        "",
        "| 指标 | 旧系统 | 新系统 | 改进 |",
        "|------|--------|--------|------|",
        "| 熊市回撤 | 高（满仓被套）| 低（自动降仓）| ⬆️ 显著改善 |",
        "| 牛市收益 | 中等 | 高（允许更高仓位）| ⬆️ 提升 |",
        "| 震荡市交易频率 | 高（频繁进出）| 低（提高阈值）| ⬆️ 减少无效交易 |",
        "| 止损效率 | 固定 | 自适应 | ⬆️ 更合理 |",
        "",
        "### 5.3 下一步",
        "",
        "- [ ] 使用历史数据回测验证",
        "- [ ] 调优各状态参数边界",
        "- [ ] 添加状态转换平滑逻辑",
        "",
        "---",
        "",
        "*报告由 AI Trader Assist 自动生成*",
    ])
    
    return "\n".join(report_lines), scenario_results


def test_integration_comparison():
    """Integration test: verify system outputs expected results for all scenarios"""
    _, results = generate_comparison_report()
    
    # Verify bull trend detection
    bull = next(r for r in results if r["name"] == "牛市趋势")
    assert bull["regime"] == "bull_trend"
    assert bull["target_exposure"] >= 0.45
    assert bull["buy_threshold"] <= 0.70
    
    # Verify bear trend detection
    bear = next(r for r in results if r["name"] == "熊市趋势")
    assert bear["regime"] == "bear_trend"
    assert bear["target_exposure"] <= 0.40
    assert bear["buy_threshold"] >= 0.60
    
    # Verify extreme panic
    panic = next(r for r in results if r["name"] == "极端恐慌")
    assert panic["regime"] == "bear_trend"
    assert panic["target_exposure"] <= 0.25
    
    # Verify range bound
    rng = next(r for r in results if r["name"] == "区间震荡")
    assert rng["regime"] in ["range_bound", "bull_pullback"]
    
    # Verify exposure ordering: bull > range > bear
    assert bull["target_exposure"] > bear["target_exposure"]


def test_regime_transitions():
    """Test that regime transitions produce different parameters"""
    manager = AdaptiveParameterManager()
    
    # Bull trend params
    manager.set_regime(MarketRegime.BULL_TREND)
    bull_buy = manager.buy_threshold
    bull_exp = manager.max_exposure
    
    # Bear trend params
    manager.set_regime(MarketRegime.BEAR_TREND)
    bear_buy = manager.buy_threshold
    bear_exp = manager.max_exposure
    
    # Assertions
    assert bull_buy < bear_buy, "Bull should have lower buy threshold"
    assert bull_exp > bear_exp, "Bull should have higher max exposure"
    assert bear_buy >= 0.75, "Bear buy threshold should be high"
    assert bear_exp <= 0.35, "Bear max exposure should be low"


if __name__ == "__main__":
    report, _ = generate_comparison_report()
    
    # Save report
    output_dir = Path("storage")
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / "phase1_comparison_report.md"
    report_path.write_text(report, encoding="utf-8")
    
    print(f"Report saved to: {report_path}")
    print("\n" + "=" * 60 + "\n")
    print(report)
