"""Multi-stock batch backtest framework - 多股票批量回测框架"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .amd_backtest import AMDBacktester, OldSystemBacktester, BacktestResult


@dataclass
class BatchBacktestResult:
    start_date: str
    end_date: str
    initial_capital: float
    symbols: List[str]
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


def run_batch_backtest(
    symbols: List[str],
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    initial_capital: float = 100000.0,
    market_symbol: str = "SPY",
) -> BatchBacktestResult:
    """Run backtest for multiple symbols and compare results"""
    
    batch_result = BatchBacktestResult(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        symbols=symbols,
    )
    
    total_old_return = 0.0
    total_new_return = 0.0
    total_new_smooth_return = 0.0
    
    print(f"\n{'='*70}")
    print(f"BATCH BACKTEST: {len(symbols)} symbols from {start_date} to {end_date}")
    print(f"{'='*70}\n")
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Processing {symbol}...")
        
        try:
            new_system = AMDBacktester(initial_capital, use_smoothing=True, smoothing_periods=3)
            new_result = new_system.run_backtest(symbol, market_symbol, start_date, end_date)
            
            new_no_smooth = AMDBacktester(initial_capital, use_smoothing=False)
            new_no_smooth_result = new_no_smooth.run_backtest(symbol, market_symbol, start_date, end_date)
            
            old_system = OldSystemBacktester(initial_capital)
            old_result = old_system.run_backtest(symbol, market_symbol, start_date, end_date)
            
            batch_result.results[symbol] = {
                "old_system": {
                    "total_return": old_result.total_return,
                    "max_drawdown": old_result.max_drawdown,
                    "sharpe_ratio": old_result.sharpe_ratio,
                    "win_rate": old_result.win_rate,
                    "total_trades": old_result.total_trades,
                    "final_value": old_result.final_value,
                },
                "new_no_smoothing": {
                    "total_return": new_no_smooth_result.total_return,
                    "max_drawdown": new_no_smooth_result.max_drawdown,
                    "sharpe_ratio": new_no_smooth_result.sharpe_ratio,
                    "win_rate": new_no_smooth_result.win_rate,
                    "total_trades": new_no_smooth_result.total_trades,
                    "final_value": new_no_smooth_result.final_value,
                },
                "new_with_smoothing": {
                    "total_return": new_result.total_return,
                    "max_drawdown": new_result.max_drawdown,
                    "sharpe_ratio": new_result.sharpe_ratio,
                    "win_rate": new_result.win_rate,
                    "total_trades": new_result.total_trades,
                    "final_value": new_result.final_value,
                    "regime_distribution": new_result.regime_distribution,
                },
                "improvement": {
                    "return_vs_old": new_result.total_return - old_result.total_return,
                    "drawdown_reduction": old_result.max_drawdown - new_result.max_drawdown,
                    "trade_reduction": old_result.total_trades - new_result.total_trades,
                },
            }
            
            total_old_return += old_result.total_return
            total_new_return += new_no_smooth_result.total_return
            total_new_smooth_return += new_result.total_return
            
            print(f"    Old: {old_result.total_return:+.2%} | New: {new_no_smooth_result.total_return:+.2%} | Smooth: {new_result.total_return:+.2%}")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            batch_result.results[symbol] = {"error": str(e)}
    
    n = len([s for s in symbols if "error" not in batch_result.results.get(s, {})])
    if n > 0:
        batch_result.summary = {
            "avg_old_return": total_old_return / n,
            "avg_new_return": total_new_return / n,
            "avg_new_smooth_return": total_new_smooth_return / n,
            "avg_improvement": (total_new_smooth_return - total_old_return) / n,
            "symbols_processed": n,
            "symbols_failed": len(symbols) - n,
        }
    
    print(f"\n{'='*70}")
    print("BATCH SUMMARY")
    print(f"{'='*70}")
    print(f"Symbols processed: {n}/{len(symbols)}")
    if n > 0:
        print(f"Avg Old System Return:    {batch_result.summary['avg_old_return']:+.2%}")
        print(f"Avg New (no smooth):      {batch_result.summary['avg_new_return']:+.2%}")
        print(f"Avg New (with smooth):    {batch_result.summary['avg_new_smooth_return']:+.2%}")
        print(f"Avg Improvement:          {batch_result.summary['avg_improvement']:+.2%}")
    
    return batch_result


def generate_batch_report(batch_result: BatchBacktestResult) -> str:
    """Generate markdown report for batch backtest results"""
    
    lines = [
        f"# 批量回测报告",
        f"",
        f"**回测期间**: {batch_result.start_date} ~ {batch_result.end_date}",
        f"**初始资金**: ${batch_result.initial_capital:,.0f}",
        f"**股票数量**: {len(batch_result.symbols)}",
        f"",
        f"---",
        f"",
        f"## 汇总结果",
        f"",
        f"| 指标 | 旧系统 | 新系统(无平滑) | 新系统(平滑) |",
        f"|------|--------|----------------|--------------|",
    ]
    
    if batch_result.summary:
        s = batch_result.summary
        lines.append(f"| 平均收益 | {s['avg_old_return']:+.2%} | {s['avg_new_return']:+.2%} | {s['avg_new_smooth_return']:+.2%} |")
        lines.append(f"| 平均改进 | - | - | {s['avg_improvement']:+.2%} |")
    
    lines.extend([
        f"",
        f"---",
        f"",
        f"## 个股详情",
        f"",
        f"| 股票 | 旧系统收益 | 新系统(平滑)收益 | 改进 | 最大回撤 | 交易次数 |",
        f"|------|-----------|-----------------|------|---------|---------|",
    ])
    
    for symbol in batch_result.symbols:
        r = batch_result.results.get(symbol, {})
        if "error" in r:
            lines.append(f"| {symbol} | ERROR | - | - | - | - |")
            continue
        
        old = r.get("old_system", {})
        new = r.get("new_with_smoothing", {})
        imp = r.get("improvement", {})
        
        lines.append(
            f"| {symbol} | {old.get('total_return', 0):+.2%} | "
            f"{new.get('total_return', 0):+.2%} | "
            f"{imp.get('return_vs_old', 0):+.2%} | "
            f"{new.get('max_drawdown', 0):.2%} | "
            f"{new.get('total_trades', 0)} |"
        )
    
    lines.extend([
        f"",
        f"---",
        f"",
        f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "AMD"]
    
    result = run_batch_backtest(
        symbols=symbols,
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_capital=100000.0,
    )
    
    output_dir = Path("storage")
    output_dir.mkdir(exist_ok=True)
    
    json_path = output_dir / "batch_backtest_2024.json"
    json_path.write_text(json.dumps({
        "start_date": result.start_date,
        "end_date": result.end_date,
        "initial_capital": result.initial_capital,
        "symbols": result.symbols,
        "results": result.results,
        "summary": result.summary,
    }, indent=2, default=str))
    
    report = generate_batch_report(result)
    report_path = output_dir / "batch_backtest_2024_report.md"
    report_path.write_text(report)
    
    print(f"\nResults saved to: {json_path}")
    print(f"Report saved to: {report_path}")
