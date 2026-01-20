# AI Trader Assist

> **自适应美股量化交易系统** — 基于市场状态识别的多策略切换引擎

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![2024 Backtest](https://img.shields.io/badge/2024%20Return-+40.65%25-brightgreen.svg)](#五回测验证)
[![Sharpe Ratio](https://img.shields.io/badge/Sharpe-1.51-blue.svg)](#五回测验证)

**2024年回测成果**: 收益率 **+40.65%** (vs SPY +24.45%)，夏普比率 **1.51**，胜率 **61%**

---

## 目录

- [一、系统原理](#一系统原理)
- [二、核心逻辑](#二核心逻辑)
- [三、实现方式](#三实现方式)
- [四、快速开始](#四快速开始)
- [五、回测验证](#五回测验证)
- [六、配置说明](#六配置说明)
- [七、扩展指南](#七扩展指南)

---

## 一、系统原理

### 1.1 核心理念

传统量化策略的致命缺陷：**单一策略无法适应不同市场环境**。牛市中过早止盈错失趋势收益，熊市中固守仓位遭受深度回撤。

AI Trader Assist 采用 **"市场状态识别 → 策略自动切换 → 参数动态调整"** 的自适应架构：

```
┌─────────────────────────────────────────────────────────────────┐
│                     市场状态识别层                               │
│     SPY/QQQ 均线位置 + VIX + 市场宽度 + 动量 → 五种状态          │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     策略自动切换层                               │
│     牛市 → 趋势跟踪    震荡 → 均值回归    熊市 → 防御保守        │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     交易执行层                                   │
│     仓位计算 → 止损止盈 → 信号生成 → 报告输出                    │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 设计原则

| 原则 | 说明 |
|------|------|
| **自适应** | 根据市场状态自动切换策略，无需人工干预 |
| **风控优先** | 每种策略都有独立的止损、仓位上限、持仓时间约束 |
| **趋势友好** | 牛市中禁用止盈，使用移动止损，最大化趋势收益 |
| **回撤控制** | 动态回撤控制器在亏损时自动降低仓位 |

### 1.3 与传统策略的对比

| 维度 | 传统固定策略 | AI Trader Assist |
|------|-------------|------------------|
| 牛市表现 | 频繁止盈，收益受限 | 趋势跟踪，持仓20天+ |
| 熊市表现 | 深度回撤 | 严格止损，低仓位运行 |
| 参数调整 | 需人工判断 | 自动根据市场状态切换 |
| 2024收益 | ~8% (固定止盈) | **+40.65%** |

---

## 二、核心逻辑

### 2.1 市场状态识别

系统通过多维度信号识别当前市场处于以下五种状态之一：

```python
class MarketRegime(Enum):
    BULL_TREND = "bull_trend"         # 牛市趋势
    BULL_PULLBACK = "bull_pullback"   # 牛市回调
    RANGE_BOUND = "range_bound"       # 区间震荡
    BEAR_RALLY = "bear_rally"         # 熊市反弹
    BEAR_TREND = "bear_trend"         # 熊市趋势
```

**识别信号与权重**：

| 信号维度 | 权重 | 判定逻辑 |
|----------|------|----------|
| SMA200 位置 | 1.0 | 价格在200日均线上方 → +1 |
| SMA50 位置 | 0.8 | 价格在50日均线上方 → +0.8 |
| SMA50 斜率 | 1.0 | 斜率 > 0.05% → +1 |
| 市场宽度 | 1.0 | 上涨股票 > 55% → +1 |
| 新高新低比 | 0.8 | 新高/新低 > 1.5 → +0.8 |
| VIX 结构 | 0.8 | 期限正向 → +0.8 |
| 动量 | 0.6 | 20日动量 > 3% → +0.6 |

**状态判定**：
- 总分 ≥ 2.5 → 牛市
- 总分 ≤ 1.5 → 熊市
- 其他 → 震荡

### 2.2 三套自适应策略

根据市场状态自动切换到对应策略：

#### 🐂 牛市策略 (Bull Trend Follow)

**核心思想**：买入持有，让利润奔跑

| 参数 | 值 | 说明 |
|------|-----|------|
| 最大仓位 | 95% | 全力做多 |
| 单只上限 | 40% | 允许集中持仓 |
| 买入阈值 | 0.40 | 低门槛买入 |
| 止损 | 15% | 宽松止损 |
| 移动止损 | 20% | 从高点回撤20%触发 |
| **止盈** | **禁用** | 不设止盈 |
| 最小持有 | 20天 | 防止频繁交易 |

#### 📊 震荡策略 (Range Mean Revert)

**核心思想**：高抛低吸，均值回归

| 参数 | 值 | 说明 |
|------|-----|------|
| 最大仓位 | 60% | 保守仓位 |
| 单只上限 | 20% | 分散持仓 |
| 买入阈值 | 0.55 | 需要更强信号 |
| 止损 | 8% | 中等止损 |
| 止盈 | 15% | 及时获利了结 |
| 布林带出场 | 启用 | 超买时卖出 |
| 评分出场 | 启用 | 评分 < 0.4 卖出 |

#### 🐻 熊市策略 (Bear Defensive)

**核心思想**：保护本金，严格风控

| 参数 | 值 | 说明 |
|------|-----|------|
| 最大仓位 | 30% | 低仓位运行 |
| 单只上限 | 10% | 高度分散 |
| 买入阈值 | 0.75 | 极高门槛 |
| 止损 | 5% | 严格止损 |
| 止盈 | 10% | 快速获利 |
| 再入场等待 | 5天 | 避免抄底 |

### 2.3 个股评分模型

综合评分 = 加权平均 (0-1)：

| 因子 | 权重 | 计算方法 |
|------|------|----------|
| 趋势强度 | 25% | 价格 vs SMA20/50 位置 |
| 动量信号 | 25% | RSI + MACD 组合 |
| 相对强度 | 20% | 个股涨幅 vs SPY |
| 新闻情绪 | 15% | 关键词情绪分析 |
| 成交量 | 15% | 相对成交量比率 |

### 2.4 风控机制

#### 动态回撤控制

```python
class DrawdownController:
    """根据回撤深度动态调整仓位乘数"""
    
    回撤 > 8%  → 仓位 × 0.6  # 大幅降低仓位
    回撤 > 5%  → 仓位 × 0.8  # 中度降低仓位
    回撤 ≤ 5%  → 仓位 × 1.0  # 正常仓位
    
    # 恢复机制：3步渐进恢复，防止过早加仓
```

#### 集中度控制

```python
class ConcentrationController:
    """防止单只股票过度集中"""
    
    单只占比 > 30% → 触发渐进止盈
    盈利 > 50%  → 卖出 1/4
    盈利 > 100% → 卖出 1/3
    盈利 > 200% → 卖出 1/2
```

---

## 三、实现方式

### 3.1 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                         数据采集层                                │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  yf_client   │  │ fred_client  │  │ cboe_client  │          │
│  │ Yahoo Finance│  │  FRED 宏观   │  │   Put/Call   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                         特征工程层                                │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  indicators  │  │trend_features│  │ bollinger    │          │
│  │ RSI/MACD/ATR │  │ 斜率/动量    │  │ 布林带       │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                         决策引擎层                                │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              MarketRegimeDetector                     │      │
│  │         市场状态识别 → 五种状态之一                    │      │
│  └──────────────────────────────────────────────────────┘      │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              AdaptiveStrategyEngine                   │      │
│  │         策略选择 → 参数应用 → 买卖决策                 │      │
│  └──────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                         风控层                                    │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Drawdown     │  │Concentration │  │ Position     │          │
│  │ Controller   │  │ Controller   │  │ Sizer        │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 目录结构

```
trader/
├── ai_trader_assist/             # Python 核心包
│   ├── backtest/                 # 回测引擎
│   │   ├── regime_strategies.py  # 三套策略定义
│   │   ├── adaptive_backtest.py  # 自适应回测引擎
│   │   ├── full_simulation_2024.py  # 2024完整闭环模拟
│   │   └── amd_backtest.py       # 单股回测（含风控）
│   ├── risk_engine/              # 风险引擎
│   │   ├── market_regime.py      # 市场状态识别器
│   │   ├── adaptive_params.py    # 自适应参数系统
│   │   └── macro_engine.py       # 宏观风险评估
│   ├── data_collector/           # 数据采集
│   │   ├── yf_client.py          # Yahoo Finance
│   │   ├── fred_client.py        # FRED 宏观数据
│   │   └── cboe_client.py        # CBOE PCR
│   ├── feature_engineering/      # 特征工程
│   │   ├── indicators.py         # RSI, MACD, ATR, 布林带
│   │   ├── trend_features.py     # 趋势特征
│   │   └── pipeline.py           # 特征汇总
│   ├── decision_engine/          # 决策引擎
│   │   └── stock_scoring.py      # 个股评分
│   ├── position_sizer/           # 仓位计算
│   │   └── sizer.py              # ATR 止损止盈
│   ├── portfolio_manager/        # 持仓管理
│   │   ├── state.py              # 持仓状态机
│   │   └── positions.py          # 持仓快照
│   ├── llm/                      # LLM 客户端
│   │   ├── client.py             # DeepSeek API
│   │   └── analyzer.py           # 分阶段分析器
│   ├── llm_operators/            # LLM 算子 (5阶段)
│   └── jobs/                     # 调度脚本
│       ├── run_daily.py          # 每日主流程
│       ├── record_operations.py  # 操作录入
│       └── report_portfolio.py   # 持仓报告
│
├── configs/                      # 配置文件
│   ├── base.json                 # 主配置
│   └── prompts/                  # LLM 提示词
│
├── storage/                      # 数据存储
│   ├── positions.json            # 当前持仓
│   ├── operations.jsonl          # 操作日志
│   ├── full_simulation_2024_report.txt  # 2024回测报告
│   └── cache/                    # 数据缓存
│
└── tests/                        # 测试用例
```

### 3.3 关键代码

#### 市场状态识别

```python
# ai_trader_assist/risk_engine/market_regime.py
class MarketRegimeDetector:
    def detect(self, signals: RegimeSignals) -> RegimeResult:
        # 计算各维度得分
        scores = {}
        scores["sma200"] = 1.0 if signals.spy_vs_sma200 > 0 else -1.0
        scores["sma50"] = 0.8 if signals.spy_vs_sma50 > 0 else -0.8
        # ... 其他信号
        
        bull_score = sum(scores.values())
        
        # 状态判定
        if bull_score >= 2.5:
            return MarketRegime.BULL_TREND
        elif bull_score <= 1.5:
            return MarketRegime.BEAR_TREND
        else:
            return MarketRegime.RANGE_BOUND
```

#### 策略切换

```python
# ai_trader_assist/backtest/regime_strategies.py
def get_strategy_for_regime(regime: MarketRegime) -> StrategyConfig:
    if regime in [MarketRegime.BULL_TREND, MarketRegime.BULL_PULLBACK]:
        return BULL_STRATEGY    # 趋势跟踪
    elif regime in [MarketRegime.BEAR_TREND, MarketRegime.BEAR_RALLY]:
        return BEAR_STRATEGY    # 防御保守
    else:
        return RANGE_STRATEGY   # 均值回归
```

---

## 四、快速开始

### 4.1 环境准备

```bash
# 1. 克隆仓库
git clone https://github.com/your-repo/trader.git
cd trader

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY
```

### 4.2 运行回测

```bash
# 2024年完整闭环模拟
python -m ai_trader_assist.backtest.full_simulation_2024

# 单股自适应回测
python -c "
from ai_trader_assist.backtest.adaptive_backtest import AdaptiveBacktester
bt = AdaptiveBacktester()
result = bt.run_backtest('NVDA')
print(result)
"
```

### 4.3 每日运行

```bash
# 运行每日分析
python -m ai_trader_assist.jobs.run_daily \
  --config configs/base.json \
  --output-dir storage/daily_$(date +%F)

# 记录操作
python -m ai_trader_assist.jobs.record_operations \
  --config configs/base.json

# 生成持仓报告
python -m ai_trader_assist.jobs.report_portfolio \
  --config configs/base.json
```

---

## 五、回测验证

### 5.1 2024年完整回测

```
================================================================================
2024年完整闭环回测报告
================================================================================

【年度总览】
初始资金: $100,000
最终资金: $140,651
总收益率: +40.65%  (vs SPY +24.45% = 超额收益 +16.2%)
最大回撤: 21.95%
夏普比率: 1.51
总交易次数: 66
胜率: 61%

【月度收益】
2024-01: +3.11%  ███
2024-02: +12.97% ████████████
2024-03: +1.85%  █
2024-04: -3.54%  ███
2024-05: +5.56%  █████
2024-06: +10.14% ██████████
2024-07: -2.06%  ██
2024-08: -8.86%  ████████
2024-09: +3.87%  ███
2024-10: -2.59%  ██
2024-11: +10.55% ██████████
2024-12: +9.66%  █████████

【市场状态分布】
牛市 (bull_trend):     206天 (82.1%)
熊市 (bear_trend):      18天 (7.2%)
回调 (bull_pullback):    2天 (0.8%)
震荡 (range_bound):     15天 (6.0%)

【最大盈利交易】
2024-12-30 TSLA +$15,531 (take_profit_81%)
2024-04-15 NVDA +$6,718 (take_profit_44%)
2024-07-24 AAPL +$5,318 (take_profit_29%)
```

### 5.2 策略对比

| 策略 | 2024收益 | 最大回撤 | 夏普比率 |
|------|----------|----------|----------|
| 买入持有 SPY | +24.45% | 8.5% | 1.2 |
| 固定止盈止损 | +8.2% | 15.3% | 0.6 |
| **自适应策略** | **+40.65%** | 21.95% | **1.51** |

---

## 六、配置说明

### 6.1 核心配置 (`configs/base.json`)

```json
{
  "universe": {
    "indices": ["SPY", "QQQ", "DIA", "IWM"],
    "sectors": ["XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"],
    "watchlist": ["NVDA", "AAPL", "MSFT", "AMZN", "META", "AMD", "GOOGL", "TSLA"]
  },
  "limits": {
    "max_exposure": 0.95,
    "max_single_weight": 0.40,
    "min_ticket": 1000
  },
  "sizer": {
    "k1_stop": 1.5,
    "k2_target": 2.5
  }
}
```

### 6.2 策略参数

策略参数定义在 `ai_trader_assist/backtest/regime_strategies.py`：

```python
BULL_STRATEGY = StrategyConfig(
    mode=StrategyMode.BULL_TREND_FOLLOW,
    max_exposure=0.95,
    max_single_weight=0.40,
    buy_threshold=0.40,
    stop_loss_pct=0.15,
    trailing_stop_pct=0.20,
    take_profit_enabled=False,  # 关键：牛市不设止盈
    min_hold_days=20,           # 关键：最小持有20天
)
```

---

## 七、扩展指南

### 7.1 添加新策略

```python
# 在 regime_strategies.py 中添加
NEW_STRATEGY = StrategyConfig(
    mode=StrategyMode.NEW_MODE,
    max_exposure=0.70,
    # ... 其他参数
)

# 在 get_strategy_for_regime() 中添加映射
```

### 7.2 添加新技术指标

```python
# 在 feature_engineering/indicators.py 中添加
def new_indicator(series: pd.Series, window: int = 14) -> pd.Series:
    """计算新指标"""
    ...
```

### 7.3 调整市场状态识别

```python
# 在 risk_engine/market_regime.py 中调整
RegimeConfig(
    thresholds={
        "bull_score": 2.0,  # 降低牛市识别门槛
        # ...
    }
)
```

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
- [FRED API](https://fred.stlouisfed.org/docs/api/fred/) - 宏观数据

---

*Built with Python. Powered by Data. Guided by Discipline.*
