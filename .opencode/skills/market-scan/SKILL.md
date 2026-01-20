---
name: market-scan
description: 扫描市场整体状况，评估风险等级和市场倾向。触发词：市场怎么样、今天市场情况、market scan、扫描市场、风险评估
license: MIT
compatibility: opencode
metadata:
  category: trading
  workflow: daily-analysis
---

# Market Scan Skill

扫描市场整体状况，评估当前风险等级和市场倾向。

## 触发条件

- 用户询问市场整体状况
- 用户请求风险评估
- 每日盘前分析流程的第一步

## 执行步骤

### 1. 获取市场数据

使用 MCP 工具获取以下数据：

```
1. get_quotes(["SPY", "QQQ", "DIA", "IWM"]) - 获取主要指数报价
2. get_macro() - 获取宏观指标（VIX、利率、CPI 等）
3. get_pcr() - 获取 Put/Call Ratio
4. get_news("SPY", max_items=3) - 获取市场新闻
```

### 2. 分析风险等级

基于以下指标评估风险：

| 指标 | 低风险 | 中等风险 | 高风险 |
|------|--------|----------|--------|
| VIX Z-score | < 0 | 0-1.5 | > 1.5 |
| PCR Total | 0.7-1.0 | 1.0-1.2 或 < 0.7 | > 1.2 |
| 指数涨跌 | 全部上涨 | 涨跌互现 | 全部下跌 |

### 3. 判断市场倾向

- **Risk-On（风险偏好）**: VIX 低、PCR 低、指数上涨
- **Risk-Off（避险）**: VIX 高、PCR 高、指数下跌
- **Neutral（中性）**: 指标混合

### 4. 输出格式

```markdown
## 市场扫描结果

**日期**: YYYY-MM-DD HH:MM PT
**风险等级**: 低/中/高
**市场倾向**: Risk-On / Risk-Off / Neutral

### 关键指标

| 指标 | 数值 | 状态 |
|------|------|------|
| VIX | XX.X | 正常/偏高/偏低 |
| VIX Z-score | X.XX | |
| PCR | X.XX | 偏多/中性/偏空 |
| SPY 涨跌 | +X.XX% | |

### 主要驱动因素

1. [因素1]
2. [因素2]

### 新闻摘要

- [新闻1]
- [新闻2]
```

## 注意事项

- 盘前时段（PT 04:00-06:30）数据可能不完整
- VIX 异常波动时需特别关注
- PCR 极端值（>1.3 或 <0.6）通常是反转信号

## 后续步骤

完成市场扫描后，通常继续：
1. `/sector-analysis` - 板块分析
2. `/stock-analysis` - 个股分析
