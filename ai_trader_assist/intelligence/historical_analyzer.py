#!/usr/bin/env python3
"""
历史回测版情报分析器

基于历史事件时间线，为每个季度生成投资主题配置。
支持两种模式:
1. 规则回退模式: 基于关键词匹配快速生成
2. LLM分析模式: 使用 Claude/DeepSeek 深度分析

用于 V7.1 回测验证 LLM 分析的有效性。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .historical_events import HISTORICAL_EVENTS, get_quarter_events, get_all_quarters


@dataclass
class QuarterTheme:
    """季度投资主题配置"""
    quarter: str
    theme: str
    focus_sectors: List[str]
    focus_stocks: List[str]
    avoid_sectors: List[str]
    sector_bonus: Dict[str, int]
    confidence: str  # high/medium/low
    reasoning: str
    source: str  # "rule" or "llm"


# 板块龙头股映射
SECTOR_LEADERS = {
    "XLK": ["NVDA", "AAPL", "MSFT", "AVGO", "AMD", "ADBE", "CRM", "ORCL"],
    "XLC": ["META", "GOOGL", "NFLX", "DIS", "TMUS"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX"],
    "XLF": ["JPM", "BAC", "WFC", "GS", "MS", "BLK"],
    "XLV": ["UNH", "LLY", "JNJ", "PFE", "MRK", "ABBV"],
    "XLE": ["XOM", "CVX", "COP", "SLB"],
    "XLI": ["CAT", "DE", "UNP", "HON", "GE", "RTX"],
    "XLP": ["PG", "KO", "PEP", "COST", "WMT"],
    "XLU": ["NEE", "DUK", "SO"],
    "XLB": ["LIN", "APD", "ECL"],
    "XLRE": ["AMT", "PLD", "CCI"],
}

# 主题关键词到板块映射
THEME_SECTOR_MAP = {
    "ai": ["XLK", "XLC"],
    "nvidia": ["XLK"],
    "semiconductor": ["XLK"],
    "banking_crisis": ["XLP", "XLV", "XLU"],  # 避险到防御
    "rate_cuts": ["XLF", "XLRE", "XLY"],  # 降息利好
    "energy": ["XLE"],
    "oil": ["XLE"],
    "trump": ["XLF", "XLE", "XLI"],  # Trump交易
    "tariff": ["XLP", "XLV"],  # 关税避险到防御
    "china": ["XLP", "XLV"],  # 中美紧张避险
    "crypto": ["XLF"],
    "deregulation": ["XLF"],
    "japan_carry": ["XLP", "XLV", "XLU"],  # 套息平仓避险
    "china_stimulus": ["XLB", "XLI"],  # 中国刺激利好
    "soft_landing": ["XLY", "XLF"],
    "data_center": ["XLK", "XLI"],
    "defense": ["XLI"],
    "healthcare": ["XLV"],
}

# 应回避的板块映射
AVOID_SECTOR_MAP = {
    "banking_crisis": ["XLF"],
    "tariff": ["XLI", "XLY"],  # 关税影响工业和消费
    "china": ["XLK"],  # 科技脱钩风险
    "rates": ["XLRE", "XLU"],  # 高利率影响地产和公用事业
    "japan_carry": ["XLY", "XLF"],  # 风险资产回避
}


class HistoricalThemeAnalyzer:
    """历史回测版主题分析器"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("storage/intelligence_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._llm_cache: Dict[str, QuarterTheme] = {}
    
    def analyze_quarter_rule_based(self, quarter: str) -> QuarterTheme:
        """基于规则的季度主题分析"""
        events = get_quarter_events(quarter)
        if not events:
            return self._default_theme(quarter)
        
        hot_topics = events.get("hot_topics", [])
        narratives = events.get("leading_narratives", [])
        sentiment = events.get("market_sentiment", "")
        fed_policy = events.get("fed_policy", "")
        
        # 收集焦点板块
        focus_sectors = []
        avoid_sectors = []
        
        for topic in hot_topics:
            if topic in THEME_SECTOR_MAP:
                focus_sectors.extend(THEME_SECTOR_MAP[topic])
            if topic in AVOID_SECTOR_MAP:
                avoid_sectors.extend(AVOID_SECTOR_MAP[topic])
        
        # 根据Fed政策调整
        if "降息" in fed_policy or "rate_cuts" in hot_topics:
            focus_sectors.extend(["XLF", "XLRE"])
        if "加息" in fed_policy or "高利率" in sentiment:
            avoid_sectors.extend(["XLRE", "XLU"])
        
        # 根据情绪调整
        if "谨慎" in sentiment or "避险" in sentiment:
            focus_sectors.extend(["XLP", "XLV", "XLU"])
            avoid_sectors.extend(["XLY"])
        if "乐观" in sentiment:
            focus_sectors.extend(["XLK", "XLY"])
        
        # 去重并排序
        focus_sectors = list(dict.fromkeys(focus_sectors))[:4]
        avoid_sectors = list(dict.fromkeys([s for s in avoid_sectors if s not in focus_sectors]))[:3]
        
        # 生成焦点股票
        focus_stocks = []
        for sector in focus_sectors:
            focus_stocks.extend(SECTOR_LEADERS.get(sector, [])[:2])
        
        # AI主题特殊处理
        if "ai" in hot_topics or "nvidia" in hot_topics:
            if "NVDA" not in focus_stocks:
                focus_stocks.insert(0, "NVDA")
            for s in ["AMD", "AVGO", "MSFT"]:
                if s not in focus_stocks:
                    focus_stocks.append(s)
        
        focus_stocks = list(dict.fromkeys(focus_stocks))[:8]
        
        # 生成板块加成
        sector_bonus = {s: 3 - i for i, s in enumerate(focus_sectors)}
        
        # 生成主题名称
        theme = " + ".join(narratives[:2]) if narratives else "市场观望"
        
        reasoning = f"Fed: {fed_policy} | 情绪: {sentiment} | 热点: {', '.join(hot_topics[:3])}"
        
        return QuarterTheme(
            quarter=quarter,
            theme=theme,
            focus_sectors=focus_sectors,
            focus_stocks=focus_stocks,
            avoid_sectors=avoid_sectors,
            sector_bonus=sector_bonus,
            confidence="medium",
            reasoning=reasoning,
            source="rule",
        )
    
    def _default_theme(self, quarter: str) -> QuarterTheme:
        """默认主题配置"""
        return QuarterTheme(
            quarter=quarter,
            theme="均衡配置",
            focus_sectors=["XLK", "XLV"],
            focus_stocks=["NVDA", "AAPL", "MSFT", "UNH"],
            avoid_sectors=[],
            sector_bonus={"XLK": 2, "XLV": 1},
            confidence="low",
            reasoning="无历史数据，使用默认配置",
            source="rule",
        )
    
    def generate_llm_prompt(self, quarter: str) -> str:
        """生成LLM分析提示"""
        events = get_quarter_events(quarter)
        if not events:
            return ""
        
        major_events = "\n".join(f"- {e}" for e in events.get("major_events", []))
        hot_topics = ", ".join(events.get("hot_topics", []))
        narratives = ", ".join(events.get("leading_narratives", []))
        sentiment = events.get("market_sentiment", "")
        fed_policy = events.get("fed_policy", "")
        
        prompt = f"""你是一位资深的美股投资策略分析师。请基于以下 {quarter} 的市场情报，进行深度分析并给出投资主题建议。

## {quarter} 市场情报

### 重大事件
{major_events}

### 市场情绪
{sentiment}

### Fed 政策
{fed_policy}

### 热点话题
{hot_topics}

### 主导叙事
{narratives}

## 分析要求

1. **市场周期判断**: 当前处于什么样的市场周期?
2. **主题趋势**: 主要投资主题是什么?
3. **板块配置**: 哪些板块应该重点配置? 哪些应该回避?
4. **个股选择**: 给出8只重点关注股票

## 输出格式 (严格按此JSON格式)

```json
{{
  "market_cycle": "市场周期判断",
  "theme": "投资主题(简洁)",
  "focus_sectors": ["XLK", "XLE"],
  "focus_stocks": ["NVDA", "XOM", "CVX", "AAPL", "MSFT", "UNH", "JPM", "META"],
  "avoid_sectors": ["XLF"],
  "confidence": "high/medium/low",
  "reasoning": "分析理由(2-3句话)"
}}
```

请直接输出JSON，不要有其他内容。
"""
        return prompt
    
    def parse_llm_response(self, quarter: str, response: str) -> Optional[QuarterTheme]:
        """解析LLM响应"""
        try:
            # 提取JSON部分
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return None
            
            data = json.loads(json_match.group())
            
            focus_sectors = data.get("focus_sectors", [])[:4]
            sector_bonus = {s: 3 - i for i, s in enumerate(focus_sectors)}
            
            return QuarterTheme(
                quarter=quarter,
                theme=data.get("theme", ""),
                focus_sectors=focus_sectors,
                focus_stocks=data.get("focus_stocks", [])[:8],
                avoid_sectors=data.get("avoid_sectors", [])[:3],
                sector_bonus=sector_bonus,
                confidence=data.get("confidence", "medium"),
                reasoning=data.get("reasoning", ""),
                source="llm",
            )
        except Exception as e:
            print(f"解析LLM响应失败: {e}")
            return None
    
    def analyze_all_quarters_rule_based(self) -> Dict[str, QuarterTheme]:
        """分析所有季度(规则模式)"""
        results = {}
        for quarter in get_all_quarters():
            results[quarter] = self.analyze_quarter_rule_based(quarter)
        return results
    
    def export_themes_for_backtest(
        self, 
        themes: Dict[str, QuarterTheme]
    ) -> Dict[str, Dict]:
        """导出主题配置供回测使用"""
        export = {}
        for quarter, theme in themes.items():
            export[quarter] = {
                "theme": theme.theme,
                "focus_sectors": theme.focus_sectors,
                "focus_stocks": theme.focus_stocks,
                "avoid_sectors": theme.avoid_sectors,
                "sector_bonus": theme.sector_bonus,
            }
        return export
    
    def save_analysis(self, themes: Dict[str, QuarterTheme], output_path: Path):
        """保存分析结果"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存完整分析
        full_data = {
            quarter: {
                "quarter": t.quarter,
                "theme": t.theme,
                "focus_sectors": t.focus_sectors,
                "focus_stocks": t.focus_stocks,
                "avoid_sectors": t.avoid_sectors,
                "sector_bonus": t.sector_bonus,
                "confidence": t.confidence,
                "reasoning": t.reasoning,
                "source": t.source,
            }
            for quarter, t in themes.items()
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)
        
        print(f"分析结果已保存到: {output_path}")
    
    def load_analysis(self, input_path: Path) -> Dict[str, QuarterTheme]:
        """加载已保存的分析结果"""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        themes = {}
        for quarter, t in data.items():
            themes[quarter] = QuarterTheme(
                quarter=t["quarter"],
                theme=t["theme"],
                focus_sectors=t["focus_sectors"],
                focus_stocks=t["focus_stocks"],
                avoid_sectors=t["avoid_sectors"],
                sector_bonus=t["sector_bonus"],
                confidence=t.get("confidence", "medium"),
                reasoning=t.get("reasoning", ""),
                source=t.get("source", "rule"),
            )
        return themes


def generate_llm_prompts_batch(output_dir: Path = None):
    """批量生成所有季度的LLM分析提示"""
    output_dir = output_dir or Path("storage/intelligence_cache/llm_prompts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer = HistoricalThemeAnalyzer()
    
    print("=" * 60)
    print("批量生成 LLM 分析提示")
    print("=" * 60)
    
    all_prompts = []
    
    for quarter in get_all_quarters():
        prompt = analyzer.generate_llm_prompt(quarter)
        if prompt:
            # 保存单个提示
            prompt_file = output_dir / f"prompt_{quarter.replace('-', '_')}.txt"
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt)
            
            all_prompts.append(f"### {quarter}\n\n{prompt}\n")
            print(f"  ✅ {quarter} 提示已生成")
    
    # 保存合并的提示
    combined_file = output_dir / "all_prompts.md"
    with open(combined_file, "w", encoding="utf-8") as f:
        f.write("# 所有季度 LLM 分析提示\n\n")
        f.write("\n---\n\n".join(all_prompts))
    
    print(f"\n📁 所有提示已保存到: {output_dir}")
    print(f"📄 合并文件: {combined_file}")


def analyze_all_rule_based():
    """使用规则模式分析所有季度"""
    analyzer = HistoricalThemeAnalyzer()
    
    print("=" * 60)
    print("规则模式分析所有季度")
    print("=" * 60)
    
    themes = analyzer.analyze_all_quarters_rule_based()
    
    for quarter, theme in themes.items():
        print(f"\n{quarter}:")
        print(f"  主题: {theme.theme}")
        print(f"  焦点板块: {', '.join(theme.focus_sectors)}")
        print(f"  焦点股票: {', '.join(theme.focus_stocks[:5])}...")
        print(f"  回避板块: {', '.join(theme.avoid_sectors) if theme.avoid_sectors else '无'}")
    
    # 保存结果
    output_path = Path("storage/intelligence_cache/rule_based_themes.json")
    analyzer.save_analysis(themes, output_path)
    
    # 导出回测配置
    backtest_config = analyzer.export_themes_for_backtest(themes)
    config_path = Path("storage/intelligence_cache/backtest_themes.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(backtest_config, f, indent=2, ensure_ascii=False)
    print(f"\n回测配置已保存到: {config_path}")
    
    return themes


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--prompts":
        generate_llm_prompts_batch()
    else:
        analyze_all_rule_based()
