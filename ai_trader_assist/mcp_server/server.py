"""FastMCP Server for AI Trader Assist.

Provides tools for:
- Price data retrieval (get_price, get_history, get_quotes)
- News and macro data (get_news, get_macro, get_pcr)
- Portfolio management (get_portfolio, save_operation, update_positions)
- Analysis tools (calc_indicators, score_stocks, generate_orders)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "FastMCP not installed. Install with: pip install mcp[cli]"
    )

from .tools.price_tools import register_price_tools
from .tools.news_tools import register_news_tools
from .tools.portfolio_tools import register_portfolio_tools
from .tools.analysis_tools import register_analysis_tools

# Initialize FastMCP server
mcp = FastMCP(
    "AI Trader Assist",
    instructions="美股投资分析助手 MCP Server - 提供股票数据、分析和持仓管理工具",
)

# Configuration
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.json"
STORAGE_PATH = PROJECT_ROOT / "storage"


def load_config() -> Dict[str, Any]:
    """Load the base configuration."""
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


# Register all tools
config = load_config()
register_price_tools(mcp, config, PROJECT_ROOT)
register_news_tools(mcp, config, PROJECT_ROOT)
register_portfolio_tools(mcp, config, PROJECT_ROOT)
register_analysis_tools(mcp, config, PROJECT_ROOT)


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
