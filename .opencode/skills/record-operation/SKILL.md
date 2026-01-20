---
name: record-operation
description: 记录交易操作到操作日志，更新持仓状态。触发词：记录操作、记录交易、我买了、我卖了、成交了
license: MIT
compatibility: opencode
metadata:
  category: trading
  workflow: portfolio-management
---

# Record Operation Skill

记录交易操作到操作日志，并自动更新持仓状态。

## 触发条件

- 用户告知已执行交易
- 用户请求记录操作
- 收盘后录入当日操作

## 支持的操作类型

| 操作 | 说明 | 示例 |
|------|------|------|
| BUY | 买入 | "买了 100 股 NVDA" |
| SELL | 全部卖出 | "卖掉 AAPL" |
| REDUCE | 部分减仓 | "减仓 50 股 MSFT" |
| HOLD | 继续持有 | 仅记录，不改变持仓 |

## 执行步骤

### 1. 解析用户输入

从用户消息中提取：
- **股票代码**: 必需
- **操作类型**: 必需 (BUY/SELL/REDUCE)
- **股数**: 必需
- **成交价格**: 必需（如未提供，尝试获取当前价格）
- **原因**: 可选

### 2. 验证信息

```
# 获取当前价格验证
price_data = get_price("{SYMBOL}")

# 获取当前持仓
portfolio = get_portfolio()
```

验证项：
- 股票代码有效
- 股数 > 0
- 价格合理（与市价偏差 < 5%）
- SELL/REDUCE 时检查持仓是否足够

### 3. 记录操作

```
save_operation(
    symbol="{SYMBOL}",
    action="{ACTION}",
    shares={SHARES},
    price={PRICE},
    reason="{REASON}"
)
```

### 4. 更新持仓

```
update_positions()
```

### 5. 确认信息

```markdown
## 操作已记录 ✅

| 项目 | 内容 |
|------|------|
| 日期 | YYYY-MM-DD |
| 股票 | {SYMBOL} |
| 操作 | {ACTION} |
| 股数 | {SHARES} |
| 价格 | ${PRICE} |
| 金额 | ${NOTIONAL} |
| 原因 | {REASON} |

### 持仓变化

| 项目 | 变化前 | 变化后 |
|------|--------|--------|
| {SYMBOL} 股数 | XX | XX |
| {SYMBOL} 均价 | $XX | $XX |
| 现金 | $XX,XXX | $XX,XXX |
| 总仓位 | XX% | XX% |
```

## 批量录入

如需录入多笔操作，逐一调用：

```
save_operation("NVDA", "BUY", 50, 145.20, "趋势突破")
save_operation("AAPL", "REDUCE", 20, 228.50, "止盈减仓")
update_positions()  # 最后统一更新
```

## 操作日志格式

每条操作记录保存到 `storage/operations.jsonl`：

```json
{
  "date": "2025-10-28",
  "symbol": "NVDA",
  "action": "BUY",
  "shares": 50,
  "price": 145.20,
  "reason": "趋势突破",
  "source": "mcp_tool",
  "timestamp": "2025-10-28T14:30:00+00:00"
}
```

## 常见场景

### 场景 1: 买入

用户: "今天买了 100 股英伟达，成交价 145.50"

```
save_operation("NVDA", "BUY", 100, 145.50)
update_positions()
```

### 场景 2: 止盈减仓

用户: "AAPL 涨到目标价了，卖了一半，50 股 @ 235"

```
save_operation("AAPL", "REDUCE", 50, 235.00, "止盈减仓")
update_positions()
```

### 场景 3: 止损清仓

用户: "AMD 跌破止损，全部卖掉了，80 股 @ 118"

```
save_operation("AMD", "SELL", 80, 118.00, "止损")
update_positions()
```

## 错误处理

| 错误 | 处理 |
|------|------|
| 股票代码无效 | 提示确认代码 |
| 卖出股数超过持仓 | 拒绝并提示当前持仓 |
| 价格异常 | 警告并请求确认 |
| 重复记录 | 警告可能重复 |

## 后续步骤

记录完成后：
1. `/show-portfolio` - 查看最新持仓
2. 次日盘前 `/daily-report` - 生成新报告
