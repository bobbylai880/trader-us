"""Formatting helpers for portfolio reports."""
from __future__ import annotations

from datetime import date
from typing import Dict, Iterable, List


def _format_currency(value: float) -> str:
    return f"{value:,.2f}"


def _format_signed_currency(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):,.2f}"


def _format_percentage(value: float) -> str:
    return f"{value * 100:+.2f}%"


def build_markdown_report(
    *,
    as_of: date,
    pnl_payload: Dict[str, object],
    history: Iterable[Dict[str, object]],
) -> str:
    lines: List[str] = []
    lines.append(f"# 📊 Portfolio Report — {as_of.isoformat()}")
    lines.append("")

    positions = pnl_payload.get("positions", []) if pnl_payload else []
    if positions:
        lines.append("## 当前仓位")
        lines.append("| Symbol | Shares | Avg Cost | Last Price | P/L ($) | P/L (%) |")
        lines.append("|---------|--------:|---------:|-----------:|--------:|--------:|")
        for entry in positions:
            lines.append(
                "| {symbol} | {shares:.0f} | {avg_cost} | {last_price} | {pnl} | {pnl_pct} |".format(
                    symbol=entry["symbol"],
                    shares=entry["shares"],
                    avg_cost=_format_currency(entry["avg_cost"]),
                    last_price=_format_currency(entry["last_price"]),
                    pnl=_format_signed_currency(entry["unrealized_pnl"]),
                    pnl_pct=_format_percentage(entry["pnl_pct"]),
                )
            )
    else:
        lines.append("## 当前仓位")
        lines.append("(无持仓)")
    lines.append("")

    total_unrealized = pnl_payload.get("total_unrealized_pnl", 0.0) if pnl_payload else 0.0
    cash = pnl_payload.get("cash", 0.0) if pnl_payload else 0.0
    exposure = pnl_payload.get("total_exposure", 0.0) if pnl_payload else 0.0
    lines.append(f"**Total Unrealized P/L:** {_format_signed_currency(total_unrealized)} USD  ")
    lines.append(f"**Cash:** {_format_currency(cash)} USD  ")
    lines.append(f"**Exposure:** {_format_percentage(exposure)}")
    lines.append("\n---\n")

    history_list = list(history)
    if history_list:
        lines.append("## 历史趋势")
        first = history_list[0]
        last = history_list[-1]
        value_delta = last["total_value"] - first["total_value"]
        value_pct = (value_delta / first["total_value"]) if first["total_value"] else 0.0
        lines.append(
            f"- 组合价值：从 {_format_currency(first['total_value'])} → {_format_currency(last['total_value'])} ({_format_percentage(value_pct)})"
        )
        for symbol in sorted({s for snap in history_list for s in snap["holdings"].keys()}):
            start = first["holdings"].get(symbol, 0)
            end = last["holdings"].get(symbol, 0)
            lines.append(f"- {symbol} 持仓：{start} → {end} 股")
        exposures = [snap.get("exposure", 0.0) for snap in history_list]
        if exposures:
            avg_exposure = sum(exposures) / len(exposures)
            lines.append(f"- 平均敞口：{_format_percentage(avg_exposure)}")
    else:
        lines.append("## 历史趋势")
        lines.append("暂无历史操作记录。")

    lines.append("\n---\n")
    lines.append("## 可视化建议")
    lines.append("- 📈 组合权益曲线")
    lines.append("- 🧱 仓位堆叠图")
    lines.append("- 💰 累计盈亏曲线")

    return "\n".join(lines) + "\n"

