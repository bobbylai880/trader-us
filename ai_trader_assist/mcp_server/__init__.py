"""AI Trader Assist MCP Server.

This module provides MCP (Model Context Protocol) tools for the AI Trader Assist system.
Run with: python -m ai_trader_assist.mcp_server
"""

from .server import mcp, main

__all__ = ["mcp", "main"]
