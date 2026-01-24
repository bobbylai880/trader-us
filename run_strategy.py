#!/usr/bin/env python3
"""V8.2 策略运行入口"""
from datetime import date
from pathlib import Path

from strategy import V82Strategy


def main():
    # 初始化策略
    strategy = V82Strategy(initial_capital=100000.0)
    
    # 运行回测
    result = strategy.run(
        start=date(2020, 1, 2),
        end=date(2026, 1, 16),
    )
    
    # 打印结果
    strategy.print_summary(result)
    
    # 保存结果
    strategy.save_results(Path("storage/v82_results"))


if __name__ == "__main__":
    main()
