#!/usr/bin/env python3
"""
V5 融合策略培训文档生成器
生成包含策略原理、逻辑说明、回测报告和图表的 PDF 文件
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 颜色方案
COLORS = {
    'primary': '#2E86AB',      # 蓝色
    'secondary': '#A23B72',    # 紫红
    'success': '#28A745',      # 绿色
    'danger': '#DC3545',       # 红色
    'warning': '#FFC107',      # 黄色
    'info': '#17A2B8',         # 青色
    'light': '#F8F9FA',        # 浅灰
    'dark': '#343A40',         # 深灰
    'offensive': '#28A745',    # 进攻模式
    'neutral': '#FFC107',      # 中性模式
    'defensive': '#DC3545',    # 防御模式
}


def load_data():
    """加载回测数据"""
    base_path = Path("storage/backtest_3y_v5")
    
    with open(base_path / "result.json") as f:
        result = json.load(f)
    
    equity_df = pd.read_csv(base_path / "equity_curve.csv")
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    
    with open(base_path / "trades.json") as f:
        trades = json.load(f)
    
    with open(base_path / "macro_history.json") as f:
        macro_history = json.load(f)
    
    return result, equity_df, trades, macro_history


def create_title_page(pdf):
    """创建标题页"""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    # 标题
    ax.text(0.5, 0.7, 'V5 融合策略', fontsize=48, fontweight='bold',
            ha='center', va='center', color=COLORS['primary'])
    ax.text(0.5, 0.58, '培训手册', fontsize=36,
            ha='center', va='center', color=COLORS['dark'])
    
    # 副标题
    ax.text(0.5, 0.42, 'V3 趋势跟踪 + V4 分层决策', fontsize=20,
            ha='center', va='center', color=COLORS['secondary'])
    
    # 核心指标
    metrics_text = """
    3年回测收益: +90.43%  |  Alpha: +8.80%  |  夏普比率: 1.43  |  最大回撤: 12.56%
    """
    ax.text(0.5, 0.28, metrics_text, fontsize=14,
            ha='center', va='center', color=COLORS['info'],
            bbox=dict(boxstyle='round', facecolor=COLORS['light'], edgecolor=COLORS['info']))
    
    # 日期
    ax.text(0.5, 0.12, f'生成日期: {datetime.now().strftime("%Y-%m-%d")}', fontsize=12,
            ha='center', va='center', color=COLORS['dark'])
    
    ax.text(0.5, 0.06, 'AI Trader Assist Project', fontsize=10,
            ha='center', va='center', color='gray')
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close()


def create_strategy_overview(pdf):
    """创建策略概览页"""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    ax.text(0.5, 0.95, '策略概览', fontsize=28, fontweight='bold',
            ha='center', va='top', color=COLORS['primary'])
    
    # 策略定位
    content = """
    【策略定位】
    
    V5 融合策略是 V3 趋势跟踪与 V4 分层决策的融合版本。
    
    • 继承 V3 优点: 科技龙头聚焦、宽松止损(18%)、禁用止盈
    • 融合 V4 框架: 宏观→板块→个股 三层决策体系
    • 核心目标: 收益接近 V3 (+100%+)，回撤控制在 15% 以内
    
    
    【核心改进】
    
    1. 放宽进攻阈值: score ≥ 2 即进入 offensive 模式 (原 V4 需要 ≥ 3)
    2. 延长持有周期: 板块轮动从周度改为双周
    3. 仓位动态调整: 根据宏观状态在 30%-95% 之间动态调整
    4. 科技龙头优先: offensive 模式下优先选择 TECH_LEADERS
    
    
    【科技龙头池】
    
    NVDA  |  META  |  GOOGL  |  AMZN  |  MSFT  |  AAPL  |  AMD  |  AVGO  |  NFLX  |  TSLA
    
    
    【适用场景】
    
    • 适合保守型投资者
    • 适合震荡市场环境
    • 追求稳健的风险调整收益
    """
    
    ax.text(0.05, 0.88, content, fontsize=11, fontfamily='monospace',
            ha='left', va='top', color=COLORS['dark'],
            linespacing=1.4)
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close()


def create_three_layer_framework(pdf):
    """创建三层决策框架图"""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    ax.text(0.5, 0.96, '三层决策框架', fontsize=28, fontweight='bold',
            ha='center', va='top', color=COLORS['primary'])
    
    # 绘制三个层级的方框
    layers = [
        {'y': 0.78, 'title': '第一层: 宏观趋势分析', 'freq': '月度更新', 'color': COLORS['primary'],
         'desc': '• 输入: VIX、SPY动量、新闻情绪\n• 输出: offensive / neutral / defensive\n• 决定: 目标仓位 (95% / 70% / 30%)'},
        {'y': 0.52, 'title': '第二层: 板块轮动', 'freq': '双周更新', 'color': COLORS['secondary'],
         'desc': '• 输入: 11个板块ETF动量\n• 输出: Top 2-4 强势板块\n• 公式: 0.5×mom20 + 0.25×RS + 0.15×mom5 + 0.1×sentiment'},
        {'y': 0.26, 'title': '第三层: 个股选择', 'freq': '每5天再平衡', 'color': COLORS['info'],
         'desc': '• offensive: 优先科技龙头 (Top 6)\n• neutral/defensive: 板块内选股 (每板块 Top 2)\n• 评分: 0.6×mom20 + 0.3×mom5 + 成交量加分'},
    ]
    
    for layer in layers:
        # 主框
        rect = mpatches.FancyBboxPatch((0.08, layer['y']-0.15), 0.84, 0.18,
                                        boxstyle="round,pad=0.02",
                                        facecolor='white', edgecolor=layer['color'],
                                        linewidth=3)
        ax.add_patch(rect)
        
        # 标题
        ax.text(0.5, layer['y']+0.01, layer['title'], fontsize=16, fontweight='bold',
                ha='center', va='center', color=layer['color'])
        
        # 频率标签
        ax.text(0.88, layer['y']+0.01, layer['freq'], fontsize=10,
                ha='right', va='center', color='gray',
                bbox=dict(boxstyle='round', facecolor=COLORS['light'], edgecolor='gray', alpha=0.5))
        
        # 描述
        ax.text(0.12, layer['y']-0.08, layer['desc'], fontsize=10,
                ha='left', va='center', color=COLORS['dark'], linespacing=1.5)
    
    # 箭头连接
    for i in range(2):
        y_start = layers[i]['y'] - 0.15
        y_end = layers[i+1]['y'] + 0.03
        ax.annotate('', xy=(0.5, y_end), xytext=(0.5, y_start),
                    arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close()


def create_macro_scoring_page(pdf):
    """创建宏观评分规则页"""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    ax.text(0.5, 0.96, '第一层: 宏观评分规则', fontsize=28, fontweight='bold',
            ha='center', va='top', color=COLORS['primary'])
    
    # VIX 评分表格
    ax.text(0.05, 0.88, '【VIX 评分】', fontsize=14, fontweight='bold', color=COLORS['dark'])
    
    vix_data = [
        ['VIX < 18', '+2', '低位(贪婪)'],
        ['18 ≤ VIX < 22', '+1', '正常'],
        ['22 ≤ VIX < 30', '-1', '偏高(谨慎)'],
        ['VIX ≥ 30', '-2', '恐慌'],
    ]
    
    table1 = ax.table(cellText=vix_data,
                      colLabels=['条件', '分数', '状态'],
                      loc='upper left',
                      bbox=[0.05, 0.68, 0.4, 0.18],
                      cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 1.5)
    
    # SPY 动量评分
    ax.text(0.55, 0.88, '【SPY 动量评分】', fontsize=14, fontweight='bold', color=COLORS['dark'])
    
    spy_data = [
        ['SPY>SMA50 且 mom20>2%', '+2', '强势上涨'],
        ['SPY > SMA50', '+1', '均线上方'],
        ['SPY<SMA50 且 mom20<-5%', '-2', '弱势下跌'],
        ['其他', '0', '中性'],
    ]
    
    table2 = ax.table(cellText=spy_data,
                      colLabels=['条件', '分数', '状态'],
                      loc='upper right',
                      bbox=[0.55, 0.68, 0.4, 0.18],
                      cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 1.5)
    
    # 新闻情绪
    ax.text(0.05, 0.62, '【新闻情绪】 sentiment > 0.2: +1  |  sentiment < -0.3: -1', 
            fontsize=11, color=COLORS['dark'])
    
    # 市场状态判定
    ax.text(0.5, 0.52, '市场状态判定', fontsize=18, fontweight='bold',
            ha='center', color=COLORS['secondary'])
    
    regime_data = [
        ['score ≥ 2', 'offensive (进攻)', '95%', COLORS['offensive']],
        ['0 ≤ score < 2', 'neutral (中性)', '70%', COLORS['neutral']],
        ['score < 0', 'defensive (防御)', '30%', COLORS['defensive']],
    ]
    
    # 绘制状态卡片
    for i, (condition, regime, exposure, color) in enumerate(regime_data):
        x = 0.17 + i * 0.28
        rect = mpatches.FancyBboxPatch((x-0.1, 0.32), 0.22, 0.15,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor=color,
                                        alpha=0.2, linewidth=2)
        ax.add_patch(rect)
        ax.text(x+0.01, 0.43, regime, fontsize=12, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x+0.01, 0.38, condition, fontsize=10,
                ha='center', va='center', color=COLORS['dark'])
        ax.text(x+0.01, 0.34, f'目标仓位: {exposure}', fontsize=10,
                ha='center', va='center', color=COLORS['dark'])
    
    # 止损规则
    ax.text(0.5, 0.24, '止损规则', fontsize=18, fontweight='bold',
            ha='center', color=COLORS['danger'])
    
    stop_rules = """
    • 跟踪止损: 从最高点回撤 > 18% → 卖出
    • 防御模式硬止损: 价格 < 成本 × 0.90 → 卖出
    • 趋势破坏: 价格 < SMA50 × 0.92 且 mom20 < -10% → 卖出
    • 止盈: ❌ 禁用 (让利润奔跑)
    """
    ax.text(0.5, 0.08, stop_rules, fontsize=11, ha='center', va='center',
            color=COLORS['dark'], linespacing=1.6,
            bbox=dict(boxstyle='round', facecolor=COLORS['light'], edgecolor=COLORS['danger'], alpha=0.3))
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close()


def create_equity_curve_page(pdf, equity_df, result):
    """创建净值曲线页"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), height_ratios=[2, 1])
    
    fig.suptitle('回测净值曲线与回撤分析', fontsize=20, fontweight='bold', color=COLORS['primary'])
    
    # 上图: 净值曲线
    ax1 = axes[0]
    ax1.plot(equity_df['date'], equity_df['portfolio'], 
             label=f'V5 策略 (+{result["total_return"]*100:.1f}%)', 
             color=COLORS['primary'], linewidth=2)
    ax1.plot(equity_df['date'], equity_df['spy'], 
             label=f'SPY 基准 (+{result["spy_return"]*100:.1f}%)', 
             color=COLORS['secondary'], linewidth=1.5, linestyle='--')
    
    ax1.fill_between(equity_df['date'], equity_df['portfolio'], equity_df['spy'],
                     where=(equity_df['portfolio'] > equity_df['spy']),
                     alpha=0.2, color=COLORS['success'], label='超额收益')
    ax1.fill_between(equity_df['date'], equity_df['portfolio'], equity_df['spy'],
                     where=(equity_df['portfolio'] <= equity_df['spy']),
                     alpha=0.2, color=COLORS['danger'])
    
    ax1.set_ylabel('组合价值 ($)', fontsize=12)
    ax1.set_title('净值曲线对比', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(equity_df['date'].min(), equity_df['date'].max())
    
    # 标注最高点和最低点
    max_idx = equity_df['portfolio'].idxmax()
    min_idx = equity_df['portfolio'].idxmin()
    ax1.annotate(f"最高: ${equity_df.loc[max_idx, 'portfolio']:,.0f}",
                 xy=(equity_df.loc[max_idx, 'date'], equity_df.loc[max_idx, 'portfolio']),
                 xytext=(10, 10), textcoords='offset points', fontsize=9,
                 arrowprops=dict(arrowstyle='->', color=COLORS['success']))
    
    # 下图: 回撤曲线
    ax2 = axes[1]
    portfolio = equity_df['portfolio'].values
    peak = np.maximum.accumulate(portfolio)
    drawdown = (peak - portfolio) / peak * 100
    
    ax2.fill_between(equity_df['date'], 0, -drawdown, 
                     color=COLORS['danger'], alpha=0.5)
    ax2.axhline(y=-result['max_drawdown']*100, color=COLORS['danger'], 
                linestyle='--', linewidth=1, label=f'最大回撤: {result["max_drawdown"]*100:.2f}%')
    
    ax2.set_ylabel('回撤 (%)', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_title('回撤曲线', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(equity_df['date'].min(), equity_df['date'].max())
    ax2.set_ylim(-20, 2)
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close()


def create_performance_metrics_page(pdf, result, trades):
    """创建绩效指标页"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    
    fig.suptitle('回测绩效分析', fontsize=20, fontweight='bold', color=COLORS['primary'])
    
    # 左上: 核心指标卡片
    ax1 = axes[0, 0]
    ax1.axis('off')
    
    metrics = [
        ('总收益率', f"+{result['total_return']*100:.2f}%", COLORS['success']),
        ('年化收益', f"+{result['annualized_return']*100:.2f}%", COLORS['success']),
        ('Alpha', f"+{result['alpha']*100:.2f}%", COLORS['info']),
        ('夏普比率', f"{result['sharpe']:.2f}", COLORS['primary']),
        ('最大回撤', f"{result['max_drawdown']*100:.2f}%", COLORS['danger']),
        ('胜率', f"{result['win_rate']*100:.1f}%", COLORS['warning']),
        ('盈亏比', f"{result['profit_factor']:.2f}", COLORS['secondary']),
        ('总交易', f"{result['total_trades']}笔", COLORS['dark']),
    ]
    
    for i, (name, value, color) in enumerate(metrics):
        row, col = i // 2, i % 2
        x = 0.1 + col * 0.45
        y = 0.75 - row * 0.22
        
        rect = mpatches.FancyBboxPatch((x, y-0.08), 0.38, 0.16,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor=color,
                                        alpha=0.15, linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x+0.19, y+0.04, value, fontsize=16, fontweight='bold',
                ha='center', va='center', color=color)
        ax1.text(x+0.19, y-0.04, name, fontsize=11,
                ha='center', va='center', color=COLORS['dark'])
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('核心指标', fontsize=14, fontweight='bold', pad=10)
    
    # 右上: 宏观状态分布饼图
    ax2 = axes[0, 1]
    regime_dist = result['regime_distribution']
    labels = list(regime_dist.keys())
    sizes = list(regime_dist.values())
    colors = [COLORS.get(label, COLORS['dark']) for label in labels]
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors,
                                        autopct='%1.0f%%', startangle=90,
                                        explode=[0.02]*len(sizes))
    ax2.set_title('宏观状态分布 (月度)', fontsize=14, fontweight='bold')
    
    # 左下: 交易来源分布
    ax3 = axes[1, 0]
    source_dist = result['source_distribution']
    bars = ax3.bar(source_dist.keys(), source_dist.values(),
                   color=[COLORS['primary'], COLORS['secondary']])
    ax3.set_ylabel('交易笔数')
    ax3.set_title('交易来源分布', fontsize=14, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 右下: 板块交易分布
    ax4 = axes[1, 1]
    sector_dist = result['sector_distribution']
    sorted_sectors = sorted(sector_dist.items(), key=lambda x: -x[1])
    sectors = [s[0] for s in sorted_sectors]
    counts = [s[1] for s in sorted_sectors]
    
    colors_list = plt.cm.Blues(np.linspace(0.4, 0.9, len(sectors)))
    bars = ax4.barh(sectors, counts, color=colors_list)
    ax4.set_xlabel('交易笔数')
    ax4.set_title('板块交易分布', fontsize=14, fontweight='bold')
    ax4.invert_yaxis()
    
    for bar in bars:
        width = bar.get_width()
        ax4.annotate(f'{int(width)}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close()


def create_trade_analysis_page(pdf, trades):
    """创建交易分析页"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    
    fig.suptitle('交易分析详情', fontsize=20, fontweight='bold', color=COLORS['primary'])
    
    # 过滤卖出交易
    sells = [t for t in trades if t['action'] == 'SELL']
    
    # 左上: Top 5 盈利交易
    ax1 = axes[0, 0]
    ax1.axis('off')
    
    top_wins = sorted(sells, key=lambda x: -x['pnl'])[:5]
    ax1.set_title('Top 5 盈利交易', fontsize=14, fontweight='bold', color=COLORS['success'])
    
    win_data = [[t['date'], t['symbol'], f"${t['pnl']:+,.0f}", f"{t['pnl_pct']*100:+.1f}%"] 
                for t in top_wins]
    table1 = ax1.table(cellText=win_data,
                       colLabels=['日期', '股票', '盈亏', '收益率'],
                       loc='center',
                       cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 2)
    
    # 设置表头颜色
    for i in range(4):
        table1[(0, i)].set_facecolor(COLORS['success'])
        table1[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # 右上: Top 5 亏损交易
    ax2 = axes[0, 1]
    ax2.axis('off')
    
    top_losses = sorted(sells, key=lambda x: x['pnl'])[:5]
    ax2.set_title('Top 5 亏损交易', fontsize=14, fontweight='bold', color=COLORS['danger'])
    
    loss_data = [[t['date'], t['symbol'], f"${t['pnl']:+,.0f}", f"{t['pnl_pct']*100:+.1f}%"] 
                 for t in top_losses]
    table2 = ax2.table(cellText=loss_data,
                       colLabels=['日期', '股票', '盈亏', '收益率'],
                       loc='center',
                       cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 2)
    
    for i in range(4):
        table2[(0, i)].set_facecolor(COLORS['danger'])
        table2[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # 左下: 盈亏分布直方图
    ax3 = axes[1, 0]
    pnls = [t['pnl'] for t in sells]
    
    bins = np.linspace(min(pnls), max(pnls), 20)
    n, bins, patches = ax3.hist(pnls, bins=bins, edgecolor='white')
    
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor(COLORS['danger'])
        else:
            patch.set_facecolor(COLORS['success'])
    
    ax3.axvline(x=0, color=COLORS['dark'], linestyle='--', linewidth=1)
    ax3.set_xlabel('盈亏金额 ($)')
    ax3.set_ylabel('交易次数')
    ax3.set_title('盈亏分布', fontsize=14, fontweight='bold')
    
    # 右下: 按卖出原因统计
    ax4 = axes[1, 1]
    
    reason_counts = {}
    for t in sells:
        reason = t['reason'].split('(')[0]  # 去掉括号内容
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    reasons = list(reason_counts.keys())
    counts = list(reason_counts.values())
    
    colors_list = [COLORS['danger'] if '止损' in r else COLORS['info'] for r in reasons]
    bars = ax4.barh(reasons, counts, color=colors_list)
    ax4.set_xlabel('次数')
    ax4.set_title('卖出原因统计', fontsize=14, fontweight='bold')
    
    for bar in bars:
        width = bar.get_width()
        ax4.annotate(f'{int(width)}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close()


def create_macro_timeline_page(pdf, macro_history):
    """创建宏观状态时间线页"""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    ax.set_title('宏观状态时间线', fontsize=20, fontweight='bold', color=COLORS['primary'])
    
    # 准备数据
    dates = [m['date'] for m in macro_history]
    regimes = [m['regime'] for m in macro_history]
    scores = [m['score'] for m in macro_history]
    vix_levels = [m['vix'] for m in macro_history]
    
    # 创建颜色映射
    regime_colors = {'offensive': COLORS['offensive'], 
                     'neutral': COLORS['neutral'], 
                     'defensive': COLORS['defensive']}
    colors = [regime_colors[r] for r in regimes]
    
    # 绘制条形图
    x = range(len(dates))
    bars = ax.bar(x, scores, color=colors, edgecolor='white', width=0.8)
    
    # 添加 VIX 线
    ax2 = ax.twinx()
    ax2.plot(x, vix_levels, color=COLORS['dark'], linewidth=2, marker='o', 
             markersize=4, label='VIX')
    ax2.set_ylabel('VIX', fontsize=12, color=COLORS['dark'])
    ax2.axhline(y=22, color=COLORS['warning'], linestyle='--', linewidth=1, alpha=0.7)
    ax2.axhline(y=30, color=COLORS['danger'], linestyle='--', linewidth=1, alpha=0.7)
    
    # 设置 x 轴标签
    ax.set_xticks(x[::3])  # 每3个月显示一个标签
    ax.set_xticklabels([dates[i][:7] for i in range(0, len(dates), 3)], rotation=45, ha='right')
    
    ax.set_xlabel('月份', fontsize=12)
    ax.set_ylabel('宏观评分', fontsize=12)
    ax.axhline(y=2, color=COLORS['offensive'], linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0, color=COLORS['dark'], linestyle='-', linewidth=0.5, alpha=0.3)
    
    # 图例
    legend_patches = [
        mpatches.Patch(color=COLORS['offensive'], label='Offensive (进攻)'),
        mpatches.Patch(color=COLORS['neutral'], label='Neutral (中性)'),
        mpatches.Patch(color=COLORS['defensive'], label='Defensive (防御)'),
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close()


def create_comparison_page(pdf):
    """创建策略对比页"""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'V3 vs V5 策略对比', fontsize=28, fontweight='bold',
            ha='center', va='top', color=COLORS['primary'])
    
    # 对比表格数据
    comparison_data = [
        ['选股范围', '只有科技龙头', '科技龙头 + 板块轮动'],
        ['仓位控制', '固定 95%', '动态 30%-95%'],
        ['止损', '15% 跟踪止损', '18% 跟踪止损'],
        ['决策频率', '每周', '月度+双周+每5天'],
        ['总收益', '+117.02%', '+90.43%'],
        ['最大回撤', '16.10%', '12.56%'],
        ['夏普比率', '1.32', '1.43'],
        ['Alpha', '+35.40%', '+8.80%'],
    ]
    
    table = ax.table(cellText=comparison_data,
                     colLabels=['维度', 'V3 趋势跟踪', 'V5 融合策略'],
                     loc='center',
                     cellLoc='center',
                     bbox=[0.1, 0.35, 0.8, 0.45])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.2)
    
    # 设置表头样式
    for i in range(3):
        table[(0, i)].set_facecolor(COLORS['primary'])
        table[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=13)
    
    # 高亮差异行
    highlight_rows = [5, 6, 7, 8]  # 收益、回撤、夏普、Alpha
    for row in highlight_rows:
        for col in range(3):
            table[(row, col)].set_facecolor(COLORS['light'])
    
    # 结论
    conclusion = """
    【选择建议】
    
    • V3 趋势跟踪: 适合激进型投资者，牛市环境，追求最高收益
    • V5 融合策略: 适合保守型投资者，震荡市场，追求稳健的风险调整收益
    
    【V5 核心优势】
    
    ✓ 最低回撤 (12.56%) - 心理压力更小
    ✓ 最高夏普比率 (1.43) - 风险调整后收益最优
    ✓ 分层决策框架 - 逻辑清晰，易于理解和执行
    """
    
    ax.text(0.5, 0.22, conclusion, fontsize=11, ha='center', va='top',
            color=COLORS['dark'], linespacing=1.5,
            bbox=dict(boxstyle='round', facecolor=COLORS['light'], 
                     edgecolor=COLORS['info'], alpha=0.5))
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close()


def create_summary_page(pdf):
    """创建总结页"""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    ax.text(0.5, 0.92, '总结与执行要点', fontsize=28, fontweight='bold',
            ha='center', va='top', color=COLORS['primary'])
    
    summary = """
    【执行频率】
    
    ┌─────────────────────────────────────────────────────────────┐
    │  月初:   检查宏观状态 (VIX、SPY动量、新闻情绪)               │
    │          → 确定 offensive / neutral / defensive            │
    │          → 设定目标仓位 (95% / 70% / 30%)                   │
    ├─────────────────────────────────────────────────────────────┤
    │  双周:   评估板块强度                                        │
    │          → 选出 Top 2-4 强势板块                            │
    ├─────────────────────────────────────────────────────────────┤
    │  每日:   检查止损条件                                        │
    │          → 跟踪止损 18%、趋势破坏                            │
    ├─────────────────────────────────────────────────────────────┤
    │  每5天:  再平衡组合                                          │
    │          → 根据评分买入候选股                                │
    └─────────────────────────────────────────────────────────────┘
    
    
    【核心原则】
    
    1. 趋势为王: 只在 SPY > SMA50 时大举进攻
    2. 科技龙头优先: offensive 模式聚焦 TECH_LEADERS
    3. 让利润奔跑: 禁用止盈，使用宽松的 18% 跟踪止损
    4. 动态仓位: 根据宏观状态灵活调整 (30%-95%)
    5. 分散风险: 板块轮动 + 科技龙头双轨制
    
    
    【风险提示】
    
    ⚠️ 本策略仅供研究参考，不构成投资建议
    ⚠️ 回测结果不代表未来收益
    ⚠️ 实盘需考虑滑点、手续费、流动性等因素
    ⚠️ 请在人工复核后执行所有交易
    """
    
    ax.text(0.5, 0.85, summary, fontsize=11, ha='center', va='top',
            color=COLORS['dark'], linespacing=1.4, fontfamily='monospace')
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close()


def main():
    """主函数"""
    print("加载回测数据...")
    result, equity_df, trades, macro_history = load_data()
    
    output_path = Path("storage/V5_Strategy_Training.pdf")
    
    print("生成 PDF 文档...")
    with PdfPages(output_path) as pdf:
        print("  - 标题页")
        create_title_page(pdf)
        
        print("  - 策略概览")
        create_strategy_overview(pdf)
        
        print("  - 三层决策框架")
        create_three_layer_framework(pdf)
        
        print("  - 宏观评分规则")
        create_macro_scoring_page(pdf)
        
        print("  - 净值曲线")
        create_equity_curve_page(pdf, equity_df, result)
        
        print("  - 绩效指标")
        create_performance_metrics_page(pdf, result, trades)
        
        print("  - 交易分析")
        create_trade_analysis_page(pdf, trades)
        
        print("  - 宏观时间线")
        create_macro_timeline_page(pdf, macro_history)
        
        print("  - 策略对比")
        create_comparison_page(pdf)
        
        print("  - 总结页")
        create_summary_page(pdf)
    
    print(f"\n✅ PDF 生成完成: {output_path}")
    print(f"   文件大小: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
