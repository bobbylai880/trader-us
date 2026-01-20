"""Portfolio management tools for MCP Server."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP


def register_portfolio_tools(mcp: FastMCP, config: Dict[str, Any], project_root: Path) -> None:
    """Register portfolio management MCP tools."""

    storage_path = project_root / "storage"
    positions_path = storage_path / config.get("logging", {}).get("positions_path", "positions.json").replace("storage/", "")
    operations_path = storage_path / config.get("logging", {}).get("operations_path", "operations.jsonl").replace("storage/", "")

    # Ensure paths are absolute
    if not positions_path.is_absolute():
        positions_path = storage_path / positions_path.name
    if not operations_path.is_absolute():
        operations_path = storage_path / operations_path.name

    @mcp.tool()
    def get_portfolio() -> Dict[str, Any]:
        """获取当前持仓状态。

        Returns:
            包含持仓信息的字典：cash, positions, equity_value, exposure
        """
        if not positions_path.exists():
            return {
                "cash": 0.0,
                "positions": [],
                "equity_value": 0.0,
                "exposure": 0.0,
                "last_updated": None,
                "message": "尚无持仓记录",
            }

        try:
            data = json.loads(positions_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return {
                "error": "无法读取持仓文件",
                "path": str(positions_path),
            }

        positions = data.get("positions", [])
        cash = data.get("cash", 0.0)
        equity_value = data.get("equity_value", cash)
        exposure = data.get("exposure", 0.0)

        # 计算各持仓的市值和权重
        position_details = []
        for pos in positions:
            shares = pos.get("shares", 0)
            avg_cost = pos.get("avg_cost", 0)
            market_value = shares * avg_cost  # 使用成本作为估算
            weight = market_value / equity_value if equity_value > 0 else 0
            position_details.append({
                "symbol": pos.get("symbol"),
                "shares": shares,
                "avg_cost": avg_cost,
                "market_value": round(market_value, 2),
                "weight": round(weight * 100, 1),
            })

        return {
            "cash": cash,
            "positions": position_details,
            "position_count": len(positions),
            "equity_value": equity_value,
            "exposure": round(exposure * 100, 1),
            "last_updated": data.get("last_updated"),
        }

    @mcp.tool()
    def save_operation(
        symbol: str,
        action: str,
        shares: int,
        price: float,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """记录一笔交易操作。

        Args:
            symbol: 股票代码（如 NVDA, AAPL）
            action: 操作类型（BUY, SELL, REDUCE）
            shares: 股数
            price: 成交价格
            reason: 操作原因（可选）

        Returns:
            确认信息
        """
        # 验证输入
        action = action.upper()
        if action not in ("BUY", "SELL", "REDUCE", "HOLD"):
            return {
                "error": f"无效的操作类型: {action}",
                "valid_actions": ["BUY", "SELL", "REDUCE", "HOLD"],
            }

        if shares <= 0:
            return {"error": "股数必须大于 0"}

        if price <= 0:
            return {"error": "价格必须大于 0"}

        # 构建操作记录
        timestamp = datetime.now(timezone.utc)
        operation = {
            "date": timestamp.strftime("%Y-%m-%d"),
            "symbol": symbol.upper(),
            "action": action,
            "shares": shares,
            "price": price,
            "source": "mcp_tool",
            "timestamp": timestamp.isoformat(),
        }
        if reason:
            operation["reason"] = reason

        # 备份并追加
        operations_path.parent.mkdir(parents=True, exist_ok=True)
        if operations_path.exists():
            backup_path = operations_path.with_suffix(".jsonl.bak")
            backup_path.write_text(operations_path.read_text(encoding="utf-8"), encoding="utf-8")

        with operations_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(operation, ensure_ascii=False) + "\n")

        # 计算交易金额
        notional = shares * price

        return {
            "success": True,
            "operation": operation,
            "notional": round(notional, 2),
            "message": f"已记录: {action} {shares} {symbol.upper()} @ ${price:.2f} (${notional:,.2f})",
        }

    @mcp.tool()
    def update_positions() -> Dict[str, Any]:
        """根据操作日志更新持仓快照。

        读取 operations.jsonl 中的所有操作记录，计算最新持仓状态，
        并更新 positions.json。

        Returns:
            更新后的持仓摘要
        """
        from ai_trader_assist.portfolio_manager.positions import (
            load_positions_snapshot,
            read_operations_log,
            apply_daily_operations,
            save_positions_snapshot,
        )
        from ai_trader_assist.portfolio_manager.state import PortfolioState

        # 读取当前持仓
        if positions_path.exists():
            state = load_positions_snapshot(positions_path)
        else:
            state = PortfolioState()

        # 读取操作日志
        if operations_path.exists():
            operations = read_operations_log(operations_path)
        else:
            operations = []

        # 应用操作
        state = apply_daily_operations(state, operations)

        # 保存更新后的持仓
        positions_path.parent.mkdir(parents=True, exist_ok=True)
        if positions_path.exists():
            backup_path = positions_path.with_suffix(".json.bak")
            backup_path.write_text(positions_path.read_text(encoding="utf-8"), encoding="utf-8")

        save_positions_snapshot(positions_path, state)

        return {
            "success": True,
            "cash": state.cash,
            "position_count": len(state.positions),
            "positions": [
                {"symbol": p.symbol, "shares": p.shares, "avg_cost": p.avg_cost}
                for p in state.positions
            ],
            "total_equity": state.total_equity,
            "exposure": round(state.current_exposure * 100, 1),
            "last_updated": state.last_updated,
            "message": "持仓已更新",
        }

    @mcp.tool()
    def get_operations_history(days: int = 30) -> Dict[str, Any]:
        """获取历史操作记录。

        Args:
            days: 回溯天数，默认 30 天

        Returns:
            操作记录列表
        """
        if not operations_path.exists():
            return {
                "operations": [],
                "count": 0,
                "message": "尚无操作记录",
            }

        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        operations = []
        for line in operations_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                op = json.loads(line)
                op_date = op.get("date", "")
                if op_date >= cutoff_str:
                    operations.append(op)
            except json.JSONDecodeError:
                continue

        # 按日期降序排列
        operations.sort(key=lambda x: x.get("timestamp", x.get("date", "")), reverse=True)

        return {
            "operations": operations,
            "count": len(operations),
            "period": f"最近 {days} 天",
        }
