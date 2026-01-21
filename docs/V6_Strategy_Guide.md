# V6 "Neuro-Adaptive" 策略指南

> **AI 增强型趋势跟踪 + 宏观事件驱动**
>
> ⚠️ **实验性策略** - 回测结果不理想，需进一步优化

---

## 目录

1. [策略概览](#策略概览)
2. [核心进化](#核心进化)
3. [系统架构](#系统架构)
4. [Phase 1: 风控补丁](#phase-1-风控补丁)
5. [Phase 2: 动态宇宙](#phase-2-动态宇宙)
6. [回测结果](#回测结果)
7. [问题诊断](#问题诊断)
8. [后续优化方向](#后续优化方向)

---

## 策略概览

### 设计目标

V6 策略旨在解决 V5 的几个核心问题：

| 模块 | V5 瓶颈 | V6 进化方案 |
|------|---------|-------------|
| **股票池** | 静态硬编码 (NVDA, META...) | 动态构建 (Quant筛选) |
| **宏观择时** | 月度轮动，滞后严重 | 事件驱动 (每日熔断检查) |
| **防御资产** | 防御板块股票 (XLP, XLU) | 真·无风险资产 (SGOV, BIL) |
| **止损机制** | 固定 18% / 10% | ATR 波动率自适应 + 利润锁定 |

### 预期指标

| 指标 | 目标 |
|------|------|
| 年化收益 | > 25% |
| 最大回撤 | < 15% |
| 2022熊市 | 接近持平 |

---

## 核心进化

### 1. 每日熔断检查 (Circuit Breaker)

从 T+30 响应提升至 T+0：

```python
CIRCUIT_BREAKER = {
    "vix_spike": 30,           # VIX > 30 触发
    "market_crash_pct": 0.025, # 单日跌幅 > 2.5%
    "cooldown_days": 3,        # 熔断后冷却天数
    "recovery_vix": 25,        # VIX < 25 可恢复
}
```

**触发条件 (任一满足)**：
- VIX 单日收盘 > 30
- SPY 或 QQQ 单日跌幅 > 2.5%
- SPY 收盘跌破 SMA200

### 2. ATR 自适应止损

根据市场状态动态调整止损宽度：

```python
ATR_MULTIPLIER = {
    "offensive": 3.0,   # 宽止损，允许高波动
    "neutral": 2.0,     # 中等止损
    "defensive": 1.5,   # 极窄止损
}

stop_price = highest_price - (multiplier * ATR_14)
```

### 3. 利润锁定机制 (Profit Locker)

分层保护利润：

| 浮盈 | 止损调整 | 说明 |
|------|----------|------|
| > 15% | cost × 1.02 | 保本微利 |
| > 30% | highest × 0.90 | 锁定90%最高价 |

### 4. 真·避险资产

```python
SAFE_HAVEN_ASSETS = ["SGOV", "BIL", "SHY"]
# SGOV: 0-3个月美债 ETF, ~4.5%年化, 几乎无回撤
```

| 市场状态 | 股票仓位 | 避险仓位 |
|----------|----------|----------|
| Offensive | 100% | 0% |
| Neutral | 60% | 40% |
| Defensive | 0-10% | 90-100% |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    V6 Neuro-Adaptive 系统                    │
├─────────────────────────────────────────────────────────────┤
│  每日熔断检查                                                │
│  ├── VIX > 30 ?                                             │
│  ├── SPY 单日跌 > 2.5% ?                                    │
│  └── SPY < SMA200 ?                                         │
│       ↓ 触发则强制 Defensive                                 │
├─────────────────────────────────────────────────────────────┤
│  宏观状态评分 (月度)                                         │
│  ├── VIX 评分 (-2 ~ +2)                                     │
│  ├── SPY 趋势评分 (-2 ~ +2)                                 │
│  └── VIX 趋势 (-1 ~ +1)                                     │
│       ↓                                                      │
│  score >= 1 → Offensive (100%)                              │
│  score >= -1 → Neutral (70%)                                │
│  score < -1 → Defensive (30%)                               │
├─────────────────────────────────────────────────────────────┤
│  动态龙头池 (季度)                                           │
│  ├── Quant 初筛: 价格>SMA200, RSI>45, 动量>0                │
│  ├── 相对强度: vs SPY 60日动量                              │
│  └── 输出: Top 10 动态龙头                                  │
├─────────────────────────────────────────────────────────────┤
│  交易执行 (每5天)                                            │
│  ├── ATR 自适应止损                                         │
│  ├── 利润锁定机制                                           │
│  └── 再平衡组合                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: 风控补丁

### 1.1 熔断检查器

```python
def _check_circuit_breaker(self, dt: date) -> Optional[str]:
    vix = self._get('VIX', dt, 'close') or 20
    spy_change = self._get('SPY', dt, 'change_1d') or 0
    spy_close = self._get('SPY', dt, 'close') or 0
    spy_sma200 = self._get('SPY', dt, 'sma200') or spy_close
    
    if vix > 30:
        return f"VIX恐慌({vix:.1f})"
    if spy_change < -0.025:
        return f"SPY暴跌({spy_change*100:.1f}%)"
    if spy_close < spy_sma200 * 0.98:
        return f"SPY跌破SMA200"
    
    return None
```

### 1.2 ATR 止损计算

```python
def _calc_stop_price(self, pos: Position, dt: date, regime: str) -> float:
    current_atr = self._get(pos.symbol, dt, 'atr')
    multiplier = ATR_MULTIPLIER.get(regime, 2.0)
    
    atr_stop = pos.highest_price - (multiplier * current_atr)
    
    # 利润锁定
    pnl_pct = (current_price - pos.avg_cost) / pos.avg_cost
    if pnl_pct >= 0.30:
        profit_stop = pos.highest_price * 0.90
    elif pnl_pct >= 0.15:
        profit_stop = pos.avg_cost * 1.02
    else:
        profit_stop = 0
    
    return max(atr_stop, profit_stop)
```

---

## Phase 2: 动态宇宙

### 2.1 Quant 初筛

```python
# 筛选条件
- 价格 > SMA200 (处于上升趋势)
- RSI > 45 (非超卖)
- 20日动量 > 0 (正向动量)
- 相对强度 vs SPY (60日动量差)
```

### 2.2 评分公式

```python
quant_score = 0

if close > sma200:
    quant_score += 2
if rsi > 45:
    quant_score += 1
if mom20 > 0.05:
    quant_score += 2
elif mom20 > 0:
    quant_score += 1

rs = mom60 - spy_mom60
if rs > 0.1:
    quant_score += 2
elif rs > 0:
    quant_score += 1
```

### 2.3 科技龙头优先

```python
TECH_LEADERS = [
    "NVDA", "META", "GOOGL", "AMZN", "MSFT",
    "AAPL", "AMD", "AVGO", "NFLX", "TSLA"
]
```

---

## 回测结果

### 4年回测 (2022-01-03 ~ 2026-01-16)

| 指标 | V6 结果 | V5 对比 | V3 对比 |
|------|---------|---------|---------|
| **最终价值** | $89,403 | $190,429 | $217,020 |
| **总收益率** | **-10.60%** | +90.43% | +117.02% |
| **年化收益** | -2.74% | +23.61% | +29.05% |
| **Alpha** | -92.27% | +8.80% | +35.40% |
| **最大回撤** | 32.68% | 12.56% | 16.10% |
| **夏普比率** | -0.20 | 1.43 | 1.32 |
| **胜率** | 38.9% | 34.5% | - |

### 年度收益分解

| 年份 | V6 策略 | SPY | Alpha |
|------|---------|-----|-------|
| 2022 | +0.0% | +0.4% | -0.4% |
| 2023 | +0.0% | +24.8% | **-24.8%** |
| 2024 | +5.2% | +24.0% | -18.8% |
| 2025 | -14.0% | +16.6% | **-30.6%** |
| 2026 | -2.2% | +1.2% | -3.5% |

### 宏观状态分布

| 状态 | 月数 | 占比 |
|------|------|------|
| Offensive | 25 | 66% |
| Neutral | 9 | 24% |
| Defensive | 4 | 10% |

### Top 盈利交易

| 日期 | 股票 | 盈亏 | 收益率 | 原因 |
|------|------|------|--------|------|
| 2024-06-24 | NVDA | +$6,741 | +42.9% | 止损触发 |
| 2024-07-01 | ORCL | +$3,713 | +24.5% | 轮出龙头池 |
| 2024-03-20 | AMD | +$3,623 | +23.0% | 止损触发 |
| 2025-09-29 | AVGO | +$3,474 | +23.9% | 止损触发 |
| 2024-04-04 | WFC | +$3,435 | +17.5% | 轮出龙头池 |

### Top 亏损交易

| 日期 | 股票 | 盈亏 | 收益率 | 原因 |
|------|------|------|--------|------|
| 2024-07-30 | NVDA | -$2,961 | -15.4% | 止损触发 |
| 2025-01-27 | AVGO | -$2,400 | -14.5% | 止损触发 |
| 2024-09-06 | AVGO | -$2,338 | -12.3% | 止损触发 |
| 2025-02-07 | TSLA | -$1,977 | -12.0% | 止损触发 |
| 2024-04-19 | NVDA | -$1,931 | -11.3% | 止损触发 |

---

## 问题诊断

### 核心缺陷

| 问题 | 表现 | 根因 |
|------|------|------|
| **2022-2023 空仓** | 0% 收益 (错过牛市) | 龙头池首次更新在 2024-01-02 |
| **龙头池质量差** | 选出 T, CVX, VZ 等弱势股 | Quant 评分未区分成长/价值 |
| **止损过频** | 127笔交易，胜率仅39% | ATR 止损在震荡市频繁触发 |
| **无避险资产** | 防御模式=现金 | SGOV/BIL 不在数据库 |
| **龙头池重复** | AAPL, GOOGL 出现两次 | 去重逻辑缺失 |

### 与 V5 对比分析

| 维度 | V5 | V6 | 结论 |
|------|-----|-----|------|
| 股票池 | 静态科技龙头 | 动态Quant筛选 | V5 简单但有效 |
| 宏观判定 | 月度评分 | 月度+每日熔断 | 熔断未触发 |
| 止损 | 固定18%跟踪 | ATR自适应 | ATR过于敏感 |
| 避险 | 防御板块 | SGOV (无数据) | V5 更实用 |

---

## 后续优化方向

### 1. 修复龙头池逻辑 (高优先级)

```python
# 问题: 季度更新太慢，2023全年无持仓
# 方案: 改为月度更新，首月即建仓

if current_month != last_universe_month:  # 月度更新
    leaders = self._build_dynamic_universe(dt)
```

### 2. 优化 Quant 评分 (高优先级)

```python
# 问题: 评分未区分成长股/价值股
# 方案: 增加科技龙头加权

if sym in TECH_LEADERS:
    quant_score += 3  # 科技龙头额外加分
```

### 3. 导入避险资产数据 (中优先级)

```sql
-- 添加 SGOV/BIL 到数据库
INSERT INTO daily_prices (symbol, trade_date, ...)
SELECT * FROM yfinance WHERE symbol IN ('SGOV', 'BIL', 'SHY');
```

### 4. 放宽 ATR 止损 (中优先级)

```python
ATR_MULTIPLIER = {
    "offensive": 4.0,   # 从 3.0 放宽到 4.0
    "neutral": 3.0,     # 从 2.0 放宽到 3.0
    "defensive": 2.0,   # 从 1.5 放宽到 2.0
}
```

### 5. 增加熔断敏感度 (低优先级)

```python
# 当前熔断未触发，说明阈值过高
CIRCUIT_BREAKER = {
    "vix_spike": 25,    # 从 30 降到 25
    "market_crash_pct": 0.02,  # 从 2.5% 降到 2%
}
```

---

## 建议

### 当前状态

⚠️ **V6 策略处于实验阶段，回测结果不理想，不建议实盘使用。**

### 推荐策略

| 目标 | 推荐策略 | 理由 |
|------|----------|------|
| **追求最高收益** | V3 趋势跟踪 | +117.02% 收益 |
| **追求稳健** | V5 融合策略 | 12.56% 回撤，夏普 1.43 |
| **实验性** | V6 (待优化) | 需要 2-3 周额外开发 |

### V6 后续开发计划

| 阶段 | 任务 | 预计耗时 |
|------|------|----------|
| **Phase 4a** | 修复龙头池月度更新 | 1天 |
| **Phase 4b** | 导入 SGOV/BIL 数据 | 1天 |
| **Phase 4c** | 优化 Quant 评分 | 2天 |
| **Phase 4d** | 重新回测验证 | 1天 |

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `scripts/run_backtest_v6.py` | V6 回测脚本 |
| `storage/backtest_v6/result.json` | 回测结果 |
| `storage/backtest_v6/trades.json` | 交易记录 |
| `storage/backtest_v6/equity_curve.csv` | 净值曲线 |
| `storage/backtest_v6/macro_history.json` | 宏观状态历史 |
| `storage/backtest_v6/leader_history.json` | 龙头池更新历史 |

---

## 运行命令

```bash
cd /Users/bobbylai/Programs/trader
source .venv/bin/activate
set -a && source .env && set +a

# 运行 V6 回测
PYTHONPATH=. python scripts/run_backtest_v6.py

# 对比 V3 回测
PYTHONPATH=. python scripts/run_backtest_v3.py

# 对比 V5 回测
PYTHONPATH=. python scripts/run_backtest_v5.py
```

---

*文档生成日期: 2026-01-21*

*AI Trader Assist Project*
