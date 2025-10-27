"""Daily orchestration job used by the MVP system."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from ..agent.base_agent import BaseAgent
from ..data_collector.fred_client import FredClient
from ..data_collector.yf_client import YahooFinanceClient
from ..decision_engine.stock_scoring import StockDecisionEngine
from ..feature_engineering.pipeline import prepare_feature_sets
from ..portfolio_manager.positions import (
    apply_daily_operations,
    load_positions_snapshot,
    read_operations_log,
    save_positions_snapshot,
)
from ..llm.analyzer import DeepSeekAnalyzer
from ..position_sizer.sizer import PositionSizer
from ..report_builder.builder import DailyReportBuilder
from ..risk_engine.macro_engine import MacroRiskEngine


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pre-market daily job")
    parser.add_argument(
        "--config",
        default="configs/base.json",
        help="Path to the configuration JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where daily artefacts will be written.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Optional trading day override in YYYY-MM-DD format.",
    )
    return parser.parse_args()


def resolve_path(root: Path, maybe_relative: str | Path) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else root / path


def main() -> None:
    load_dotenv()
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    config_path = resolve_path(project_root, args.config)
    config = load_config(config_path)

    tz_name = os.getenv("TZ", config.get("schedule", {}).get("tz", "America/Los_Angeles"))
    tzinfo = ZoneInfo(tz_name)

    if args.date:
        trading_day = datetime.strptime(args.date, "%Y-%m-%d").date()
        now = datetime.combine(trading_day, datetime.min.time(), tzinfo)
    else:
        now = datetime.now(tzinfo)
        trading_day = now.date()

    operations_path = project_root / "storage" / "operations.jsonl"
    positions_path = project_root / "storage" / "positions.json"

    operations = read_operations_log(operations_path)
    state = load_positions_snapshot(positions_path)
    apply_daily_operations(state, operations)

    fred_key = os.getenv("FRED_API_KEY")
    yf_client = YahooFinanceClient(cache_dir=project_root / "storage" / "cache" / "yf")
    fred_client = FredClient(api_key=fred_key, cache_dir=project_root / "storage" / "cache" / "fred")

    market, sectors, stocks, premarket = prepare_feature_sets(
        config=config,
        state=state,
        yf_client=yf_client,
        fred_client=fred_client,
        trading_day=now,
    )

    latest_prices = {
        symbol: data.get("price", 0.0)
        for symbol, data in stocks.items()
        if data.get("price")
    }
    if latest_prices:
        state.update_prices(latest_prices)

    llm_config = config.get("llm", {})
    prompt_files = {
        key: resolve_path(project_root, path)
        for key, path in llm_config.get("prompt_files", {}).items()
    }
    analyzer = DeepSeekAnalyzer(prompt_files=prompt_files) if prompt_files else None

    agent = BaseAgent(
        config=config,
        macro_engine=MacroRiskEngine(),
        stock_engine=StockDecisionEngine(),
        sizer=PositionSizer(config["limits"], config["sizer"]),
        portfolio_state=state,
        report_builder=DailyReportBuilder(config["sizer"]),
        analyzer=analyzer,
    )

    if args.output_dir:
        output_dir = resolve_path(project_root, args.output_dir)
    else:
        output_dir = project_root / "storage" / f"daily_{trading_day.isoformat()}"

    agent.run(
        trading_day=trading_day,
        market_features=market,
        sector_features=sectors,
        stock_features=stocks,
        premarket_flags=premarket,
        output_dir=output_dir,
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "market_features.json").write_text(
            json.dumps(market, indent=2), encoding="utf-8"
        )
        (output_dir / "sector_features.json").write_text(
            json.dumps(sectors, indent=2), encoding="utf-8"
        )
        (output_dir / "stock_features.json").write_text(
            json.dumps(stocks, indent=2), encoding="utf-8"
        )
        (output_dir / "premarket_flags.json").write_text(
            json.dumps(premarket, indent=2), encoding="utf-8"
        )

    save_positions_snapshot(positions_path, state, trading_day)


if __name__ == "__main__":
    main()
