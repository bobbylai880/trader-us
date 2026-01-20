---
name: position-sizing
description: 基于 ATR 计算仓位大小、止损止盈价位。触发词：计算仓位、买多少股、position sizing、止损价、仓位规划
license: MIT
compatibility: opencode
metadata:
  category: trading
  workflow: order-execution
---

# Position Sizing Skill

基于 ATR（平均真实波幅）计算仓位大小和止损止盈价位。

## 触发条件

- 用户询问买入数量
- 用户请求计算止损止盈
- 准备执行交易前的仓位规划

## 核心公式

### 仓位计算

```
股数 = 风险预算 / (ATR × 止损系数)
```

### 止损止盈

```
止损价 = 入场价 - (ATR × k1_stop)
止盈价 = 入场价 + (ATR × k2_target)
```

默认参数（可在 configs/base.json 调整）：
- `k1_stop`: 1.5 (止损 = 1.5 倍 ATR)
- `k2_target`: 2.5 (止盈 = 2.5 倍 ATR)
- 风险收益比: 1:1.67

## 执行步骤

### 1. 获取必要数据

```
1. get_portfolio() - 获取当前现金和持仓
2. get_price("{SYMBOL}") - 获取当前价格
3. calc_indicators("{SYMBOL}", ["atr"]) - 获取 ATR
```

### 2. 读取风控参数

从配置获取：
- `max_exposure`: 最大总仓位（默认 0.8 = 80%）
- `max_single_weight`: 单只最大权重（默认 0.15 = 15%）
- `min_ticket`: 最小交易金额（默认 1000）

### 3. 计算可用预算

```python
总权益 = 现金 + 持仓市值
当前仓位 = 持仓市值 / 总权益
可加仓空间 = max_exposure - 当前仓位
单只上限 = 总权益 × max_single_weight

# 如果已持有该股
已有权重 = 现有持仓市值 / 总权益
可加权重 = min(可加仓空间, max_single_weight - 已有权重)
可用预算 = 总权益 × 可加权重
```

### 4. 生成订单建议

使用 MCP 工具：

```
generate_orders("{SYMBOL}", action="BUY", budget=可用预算)
```

### 5. 输出格式

```markdown
## 仓位计算结果

**股票**: {SYMBOL}
**当前价格**: $XXX.XX
**ATR(14)**: $X.XX (X.X%)

### 账户状态

| 项目 | 金额 |
|------|------|
| 现金 | $XX,XXX |
| 持仓市值 | $XX,XXX |
| 总权益 | $XX,XXX |
| 当前仓位 | XX% |
| 可加仓空间 | XX% |

### 订单建议

| 项目 | 数值 |
|------|------|
| 建议股数 | XXX 股 |
| 交易金额 | $XX,XXX |
| 新增权重 | X.X% |

### 止损止盈

| 价位 | 金额 | 幅度 |
|------|------|------|
| 入场价 | $XXX.XX | - |
| 止损价 | $XXX.XX | -X.X% |
| 止盈价 | $XXX.XX | +X.X% |

### 风险评估

| 项目 | 金额 |
|------|------|
| 最大亏损 | $X,XXX |
| 预期收益 | $X,XXX |
| 风险收益比 | 1:X.X |
| 账户风险 | X.X% |
```

## 特殊情况处理

### 已有持仓加仓

```
原有成本 × 原有股数 + 新买价格 × 新买股数
新均价 = ─────────────────────────────────────
                原有股数 + 新买股数
```

### 减仓计算

```
generate_orders("{SYMBOL}", action="SELL", shares=减仓股数)
```

### 超出限制

如果计算结果超出限制：
- 可加仓空间 ≤ 0：提示"已达最大仓位"
- 预算 < min_ticket：提示"金额过小，不建议交易"
- 单只超限：按 max_single_weight 截断

## 后续步骤

确认仓位后：
1. 在券商端执行交易
2. `/record-operation` - 记录实际成交
