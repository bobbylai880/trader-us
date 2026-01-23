"""
Alternative Data Sources for Forward-Looking Theme Generation

免费数据源模块，用于事前捕获市场信号，增强主题生成的有效性。

数据源分类:
1. insider/  - SEC Form 4 内部人交易
2. options/  - 期权 Put/Call Ratio 和异常活动
3. social/   - Reddit, Twitter/X, Truth Social 社交媒体情绪
4. fed/      - Fed 讲话和会议纪要鹰鸽分析
5. analyst/  - 分析师评级变化追踪

所有数据源遵循"事前"原则：只使用查询日期之前可获得的信息。
"""

from .theme_generator import ForwardThemeGenerator

__all__ = ["ForwardThemeGenerator"]
