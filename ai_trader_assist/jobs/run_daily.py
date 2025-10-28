"""Daily orchestration job used by the MVP system."""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
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
from ..utils import setup_logger


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
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory where log files should be written (default: storage/logs).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only emit errors to the console while keeping full log files.",
    )
    return parser.parse_args()


def resolve_path(root: Path, maybe_relative: str | Path) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else root / path


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.verbose and args.quiet:
        raise SystemExit("--verbose and --quiet cannot be used together.")
    project_root = Path(__file__).resolve().parents[2]

    config_path = resolve_path(project_root, args.config)
    config = load_config(config_path)

    logging_cfg = config.get("logging", {})
    default_log_dir = logging_cfg.get("log_dir", "storage/logs")
    log_dir_setting = args.log_dir or default_log_dir
    log_dir = resolve_path(project_root, log_dir_setting)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    console_level = logging.ERROR if args.quiet else log_level
    logger, log_path = setup_logger(
        name="run_daily",
        log_dir=log_dir,
        level=log_level,
        console_level=console_level,
    )

    start_time = perf_counter()
    logger.info("启动 run_daily 流程 (配置文件: %s)", config_path)

    try:
        tz_name = os.getenv("TZ", config.get("schedule", {}).get("tz", "America/Los_Angeles"))
        tzinfo = ZoneInfo(tz_name)

        if args.date:
            trading_day = datetime.strptime(args.date, "%Y-%m-%d").date()
            now = datetime.combine(trading_day, datetime.min.time(), tzinfo)
        else:
            now = datetime.now(tzinfo)
            trading_day = now.date()

        operations_path = resolve_path(
            project_root, logging_cfg.get("operations_path", "storage/operations.jsonl")
        )
        positions_path = resolve_path(
            project_root, logging_cfg.get("positions_path", "storage/positions.json")
        )

        fred_detected = "已检测" if os.getenv("FRED_API_KEY") else "缺失"
        deepseek_detected = "已检测" if os.getenv("DEEPSEEK_API_KEY") else "缺失"
        logger.info(
            "环境变量加载完毕 (FRED_API_KEY: %s, DEEPSEEK_API_KEY: %s)",
            fred_detected,
            deepseek_detected,
        )
        logger.info("运行时区: %s, 目标交易日: %s", tz_name, trading_day.isoformat())

        logger.info("读取操作日志: %s", operations_path)
        operations = read_operations_log(operations_path)
        logger.debug("共加载 %d 条历史操作记录", len(operations))

        logger.info("载入持仓快照: %s", positions_path)
        state = load_positions_snapshot(positions_path)
        apply_daily_operations(state, operations)
        logger.debug("持仓快照更新完成 (现金=%.2f, 持仓数=%d)", state.cash, len(state.positions))

        fred_key = os.getenv("FRED_API_KEY")
        yf_client = YahooFinanceClient(cache_dir=project_root / "storage" / "cache" / "yf")
        fred_client = FredClient(
            api_key=fred_key, cache_dir=project_root / "storage" / "cache" / "fred"
        )

        logger.info("开始采集行情与特征数据")
        (
            market,
            sectors,
            stocks,
            premarket,
            news_bundle,
            trend_bundle,
        ) = prepare_feature_sets(
            config=config,
            state=state,
            yf_client=yf_client,
            fred_client=fred_client,
            trading_day=now,
        )
        logger.info(
            "完成特征准备: 市场=%d 字段, 板块=%d, 个股=%d",
            len(market),
            len(sectors),
            len(stocks),
        )

        latest_prices = {
            symbol: data.get("price", 0.0)
            for symbol, data in stocks.items()
            if data.get("price")
        }
        if latest_prices:
            state.update_prices(latest_prices)
            logger.debug("已更新 %d 个个股的最新价格", len(latest_prices))

        llm_config = config.get("llm", {})
        prompt_files = {
            key: resolve_path(project_root, path)
            for key, path in llm_config.get("prompt_files", {}).items()
        }
        analyzer = DeepSeekAnalyzer(prompt_files=prompt_files) if prompt_files else None
        if analyzer:
            logger.info("加载 DeepSeek 提示词 (%d 个阶段)", len(prompt_files))
        else:
            logger.info("未配置 DeepSeek 提示词，跳过 LLM 分析阶段")

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

        logger.info("开始执行日常流程，输出目录: %s", output_dir)

        agent.run(
            trading_day=trading_day,
            market_features=market,
            sector_features=sectors,
            stock_features=stocks,
            premarket_flags=premarket,
            news=news_bundle,
            output_dir=output_dir,
        )

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("写入特征与分析文件至 %s", output_dir)
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
            (output_dir / "news_bundle.json").write_text(
                json.dumps(news_bundle, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (output_dir / "trend_features.json").write_text(
                json.dumps(trend_bundle, indent=2), encoding="utf-8"
            )

        save_positions_snapshot(positions_path, state, trading_day)
        logger.info("已更新持仓文件: %s", positions_path)

    except Exception:
        logger.exception("执行 run_daily 发生异常")
        logger.error("❌ 任务中断")
        raise
    else:
        duration = perf_counter() - start_time
        logger.info("✅ 任务执行完成，用时 %.1f 秒 (日志文件: %s)", duration, log_path)


if __name__ == "__main__":
    main()
