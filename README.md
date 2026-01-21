# AI Trader Assist

> **自适应美股量化交易系统** — 基于市场状态识别的多策略切换引擎 + 预防性风控

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PostgreSQL 18](https://img.shields.io/badge/PostgreSQL-18-336791.svg)](https://www.postgresql.org/)
[![3Y Backtest](https://img.shields.io/badge/3Y%20Return-+114.52%25-brightgreen.svg)](#回测验证)
[![Sharpe Ratio](https://img.shields.io/badge/Sharpe-1.34-blue.svg)](#回测验证)
[![Alpha](https://img.shields.io/badge/Alpha-+32.90%25-green.svg)](#回测验证)

**3年回测成果 (2023-2026)**: 收益率 **+114.52%** (vs SPY +81.62%)，Alpha **+32.90%**，胜率 **61.1%**

---

## 目录

- [系统特性](#系统特性)
- [快速开始](#快速开始)
- [策略演进](#策略演进)
- [回测验证](#回测验证)
- [配置说明](#配置说明)
- [开发指南](#开发指南)

---

## 系统特性

### 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│  V3 趋势跟踪 = 主引擎 (100% 时间运行)                         │
│  - 科技龙头聚焦 (NVDA/META/GOOGL/AMZN/MSFT...)               │
│  - 动量选股 + 跟踪止损 15%                                    │
│  - SPY > SMA50 才开仓                                        │
├─────────────────────────────────────────────────────────────┤
│  V4 风控开关 = 刹车系统 (只在危险时介入)                      │
│  - 提前预警: VIX 趋势上升时开始减仓                           │
│  - 延迟恢复: 冷却期机制 (danger=10天, caution=5天)            │
│  - 渐进减仓: 每次只减 1 只，优先减亏损最大的                   │
└─────────────────────────────────────────────────────────────┘
```

### 风控状态机

| 状态 | 触发条件 | 最大仓位 | 冷却期 |
|------|----------|----------|--------|
| **Normal** | 默认状态 | 95% | 0 |
| **Watch** | VIX > 20 且上升趋势 | 70% | 3天 |
| **Caution** | VIX > 22 + 上升趋势 + SPY破位 | 50% | 5天 |
| **Danger** | VIX > 28 或 VIX 急升 >30% | 30% | 10天 |

---

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/your-repo/trader.git
cd trader

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入 PostgreSQL 和 API 配置
```

### 3. 运行回测

```bash
# V5c 预防性风控策略 (推荐)
PYTHONPATH=. python scripts/run_backtest_v5c.py

# V3 趋势跟踪 (最高收益)
PYTHONPATH=. python scripts/run_backtest_v3.py

# V5 融合策略 (最低回撤)
PYTHONPATH=. python scripts/run_backtest_v5.py
```

---

## 策略演进

### 从 V1 到 V5c 的迭代历程

| 版本 | 策略名称 | 核心思路 | 收益 | 回撤 | 问题/改进 |
|------|----------|----------|------|------|-----------|
| V1 | 均值回归 | RSI 超卖买入 | +30.87% | 21.64% | 牛市频繁止盈 |
| V2 | 动量策略 | 追涨强势股 | +28.21% | 26.33% | 震荡市表现差 |
| **V3** | **趋势跟踪** | SPY>SMA50 + 跟踪止损 | **+117.02%** | 16.10% | 收益最高 |
| V4 | 分层决策 | 宏观→板块→个股 | +40.65% | 13.69% | 过于保守 |
| V5 | 融合策略 | V3+V4 权重控制 | +90.43% | **12.56%** | 回撤最低 |
| V5b | 反应式风控 | V3 + VIX>30 熔断 | +97.06% | 22.60% | 割在最低点 |
| **V5c** | **预防式风控** | V3 + 提前预警 + 延迟恢复 | **+114.52%** | 20.26% | **最优平衡** |

### V5c 核心改进

1. **提前预警**: VIX 趋势上升时就开始减仓 (不是等 VIX>30)
2. **延迟恢复**: 冷却期机制防止频繁切换 (danger=10天)
3. **渐进减仓**: 每次只减 1 只最差的 (不是一次清仓)

---

## 回测验证

### 策略对比 (2023-01 ~ 2026-01, 3年)

| 指标 | V3 趋势跟踪 | V5 融合策略 | **V5c 预防式** |
|------|-------------|-------------|----------------|
| **总收益率** | +117.02% | +90.43% | **+114.52%** |
| **年化收益** | +29.05% | +23.61% | **+28.56%** |
| **Alpha** | +35.40% | +8.80% | **+32.90%** |
| **夏普比率** | 1.32 | **1.43** | 1.34 |
| **最大回撤** | 16.10% | **12.56%** | 20.26% |
| **胜率** | - | 34.5% | **61.1%** |
| **盈亏比** | - | 2.47 | **3.67** |
| **风控触发** | 0 | - | 12次 |

### 策略选择指南

| 目标 | 推荐策略 | 理由 |
|------|----------|------|
| **追求最高收益** | V3 趋势跟踪 | +117.02% 收益, Alpha +35.40% |
| **追求最低回撤** | V5 融合策略 | 12.56% 回撤, 夏普 1.43 |
| **追求最优平衡** | V5c 预防式风控 | +114.52% 收益, 61.1% 胜率, 3.67 盈亏比 |

### V5c 最大盈利交易

| 日期 | 股票 | 盈亏 | 收益率 | 原因 |
|------|------|------|--------|------|
| 2025-03-07 | NFLX | **+$23,390** | +92.9% | 跟踪止损 |
| 2024-04-25 | META | +$21,715 | +114.9% | 跟踪止损 |
| 2025-12-15 | AVGO | +$20,531 | +63.2% | 跟踪止损 |
| 2025-09-05 | AMD | +$15,311 | +47.0% | 跟踪止损 |
| 2024-04-19 | NVDA | +$14,066 | +55.9% | 跟踪止损 |

---

## 配置说明

### 环境变量 (.env)

```bash
# PostgreSQL 数据库
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=trader
PG_USER=trader
PG_PASSWORD=your_password_here

# DeepSeek LLM
DEEPSEEK_API_KEY=your_api_key_here

# FRED 宏观数据 (可选)
FRED_API_KEY=your_fred_api_key_here
```

### 风控参数调优

```python
# scripts/run_backtest_v5c.py

# VIX 阈值
DANGER_VIX = 28       # 危险模式触发
CAUTION_VIX = 22      # 警戒模式触发
WATCH_VIX = 20        # 观察模式触发

# 冷却期
DANGER_COOLDOWN = 10  # 危险模式冷却天数
CAUTION_COOLDOWN = 5  # 警戒模式冷却天数
WATCH_COOLDOWN = 3    # 观察模式冷却天数

# 科技龙头股票池
UNIVERSE = ["NVDA", "META", "GOOGL", "AMZN", "MSFT", 
            "AAPL", "AMD", "AVGO", "NFLX", "TSLA"]
```

---

## 开发指南

### 运行测试

```bash
# 运行所有测试
pytest tests -q

# 运行特定策略回测
PYTHONPATH=. python scripts/run_backtest_v5c.py
```

### 添加新策略

1. 复制 `scripts/run_backtest_v5c.py` 作为模板
2. 修改风控逻辑或选股逻辑
3. 运行回测对比结果

---

## 免责声明

⚠️ **本项目仅用于研究与教学，不连接任何券商系统，不构成投资建议。**

- 回测结果不代表未来收益
- 实盘交易需考虑滑点、手续费、流动性等因素
- 请在人工复核后执行所有交易
- 投资有风险，入市需谨慎

---

## 参考资料

- [HKUDS/AI-Trader](https://github.com/HKUDS/AI-Trader) - 架构参考
- [yfinance](https://github.com/ranaroussi/yfinance) - 行情数据
- [PostgreSQL 18](https://www.postgresql.org/docs/18/) - 数据库

---

*Built with Python. Powered by Data. Guided by Discipline.*
