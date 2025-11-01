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
from hashlib import sha256
from uuid import uuid4

from dotenv import load_dotenv

from typing import Any, List, Optional, Mapping, Sequence

from ..agent.base_agent import BaseAgent
from ..agent.orchestrator import LLMOrchestrator
from ..agent.safe_mode import SafeModeConfig
from ..data_collector.cboe_client import CboeClient
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
from ..llm.client import DeepSeekClient
from ..position_sizer.sizer import PositionSizer
from ..report_builder.hybrid_builder import HybridReportBuilder
from ..report_builder.markdown_renderer import MarkdownRenderConfig
from ..risk_engine.macro_engine import MacroRiskEngine
from ..utils import log_ok, log_result, log_step, setup_logger


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

    def to_relative(path: Path) -> str:
        try:
            return str(path.relative_to(project_root))
        except ValueError:
            return str(path)
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

        snapshot_id = uuid4().hex
        config_profile = config.get("profile") or config.get("name")
        config_payload = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
        input_hash = f"sha256:{sha256(config_payload).hexdigest()}"
        snapshot_meta = {
            "snapshot_id": snapshot_id,
            "as_of": now.isoformat(),
            "input_hash": input_hash,
            "config_profile": config_profile,
        }

        md_cfg = config.get("md", {})
        renderer_kwargs: dict[str, Any] = {}
        if "locale" in md_cfg:
            renderer_kwargs["locale"] = md_cfg["locale"]
        decimals_cfg = md_cfg.get("decimals", {}) if isinstance(md_cfg, dict) else {}
        if isinstance(decimals_cfg, dict):
            if "money" in decimals_cfg:
                renderer_kwargs["decimals_money"] = int(decimals_cfg["money"])
            if "percent" in decimals_cfg:
                renderer_kwargs["decimals_percent"] = int(decimals_cfg["percent"])
        if "max_rows_per_section" in md_cfg:
            renderer_kwargs["max_rows_per_section"] = int(md_cfg["max_rows_per_section"])
        if "hide_below_weight" in md_cfg:
            renderer_kwargs["hide_below_weight"] = md_cfg["hide_below_weight"]
        if "show_raw_json_appendix" in md_cfg:
            renderer_kwargs["show_raw_json_appendix"] = bool(md_cfg["show_raw_json_appendix"])
        if "raw_json_preview_lines" in md_cfg:
            renderer_kwargs["raw_json_preview_lines"] = int(md_cfg["raw_json_preview_lines"])
        if "news_highlights_visible" in md_cfg:
            renderer_kwargs["news_highlights_visible"] = int(md_cfg["news_highlights_visible"])
        renderer_config = MarkdownRenderConfig(**renderer_kwargs)

        operations_path = resolve_path(
            project_root, logging_cfg.get("operations_path", "storage/operations.jsonl")
        )
        positions_path = resolve_path(
            project_root, logging_cfg.get("positions_path", "storage/positions.json")
        )

        fred_detected = "已检测" if os.getenv("FRED_API_KEY") else "缺失"
        deepseek_detected = "已检测" if os.getenv("DEEPSEEK_API_KEY") else "缺失"
        log_result(
            logger,
            "run_daily",
            f"环境变量检测: FRED_API_KEY={fred_detected}, DEEPSEEK_API_KEY={deepseek_detected}",
        )
        logger.info("运行时区: %s, 目标交易日: %s", tz_name, trading_day.isoformat())

        phases_executed: List[str] = []

        portfolio_start = perf_counter()
        log_step(
            logger,
            "portfolio_manager",
            f"同步操作日志与持仓 (operations={operations_path.name}, positions={positions_path.name})",
        )
        operations = read_operations_log(operations_path)
        state = load_positions_snapshot(positions_path)
        apply_daily_operations(state, operations)
        log_result(
            logger,
            "portfolio_manager",
            f"loaded_operations={len(operations)}, positions={len(state.positions)}, cash={state.cash:.2f}",
        )
        log_ok(
            logger,
            "portfolio_manager",
            f"Completed in {perf_counter() - portfolio_start:.2f}s",
        )
        phases_executed.append("portfolio_manager")

        fred_key = os.getenv("FRED_API_KEY")
        yf_client = YahooFinanceClient(cache_dir=project_root / "storage" / "cache" / "yf")
        fred_client = FredClient(
            api_key=fred_key,
            cache_dir=project_root / "storage" / "cache" / "fred",
            logger=logger,
        )
        cboe_client = CboeClient(logger=logger)

        (
            market,
            sectors,
            stocks,
            premarket,
            news_bundle,
            trend_bundle,
            macro_flags,
            feature_metrics,
        ) = prepare_feature_sets(
            config=config,
            state=state,
            yf_client=yf_client,
            fred_client=fred_client,
            cboe_client=cboe_client,
            trading_day=now,
            logger=logger,
        )
        phases_executed.extend(feature_metrics.keys())
        data_metrics = feature_metrics.get("data_collector", {})
        feature_summary = feature_metrics.get("feature_engineering", {})
        log_result(
            logger,
            "run_daily",
            "数据摘要: history_rows=%s, news_articles=%s, fred_series=%s"
            % (
                data_metrics.get("history_rows", 0),
                data_metrics.get("news_articles", 0),
                data_metrics.get("fred_series", 0),
            ),
        )
        log_result(
            logger,
            "run_daily",
            "特征摘要: stocks=%s, uptrend=%s, downtrend=%s"
            % (
                feature_summary.get("stock_symbols", 0),
                feature_summary.get("trend_states", {}).get("uptrend", 0),
                feature_summary.get("trend_states", {}).get("downtrend", 0),
            ),
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
        base_prompt_path = llm_config.get("base_prompt")
        if args.output_dir:
            output_dir = resolve_path(project_root, args.output_dir)
        else:
            output_dir = project_root / "storage" / f"daily_{trading_day.isoformat()}"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_json_path = output_dir / "report.json"

        reproduction_bits = [
            "python -m ai_trader_assist.jobs.run_daily",
            f"--config {to_relative(config_path)}",
            f"--output-dir {to_relative(output_dir)}",
            f"--date {trading_day.isoformat()}",
        ]
        reproduction_command = " ".join(reproduction_bits)
        renderer_config.report_json_path = to_relative(report_json_path)
        renderer_config.reproduction_command = reproduction_command

        def count_news_articles(bundle: Mapping[str, Any]) -> int:
            total = 0
            if not isinstance(bundle, Mapping):
                return total
            market_news = bundle.get("market")
            if isinstance(market_news, Mapping):
                headlines = market_news.get("headlines")
                if isinstance(headlines, Sequence):
                    total += len(headlines)
            for section in ("stocks", "sectors"):
                section_payload = bundle.get(section)
                if isinstance(section_payload, Mapping):
                    for item in section_payload.values():
                        if isinstance(item, Mapping):
                            headlines = item.get("headlines")
                            if isinstance(headlines, Sequence):
                                total += len(headlines)
            return total

        artefact_summary = [
            {
                "name": operations_path.name,
                "path": to_relative(operations_path),
                "entries": len(operations),
            },
            {
                "name": positions_path.name,
                "path": to_relative(positions_path),
                "entries": len(state.positions),
            },
            {
                "name": "market_features.json",
                "path": to_relative(output_dir / "market_features.json"),
                "entries": len(market) if isinstance(market, Mapping) else 0,
            },
            {
                "name": "sector_features.json",
                "path": to_relative(output_dir / "sector_features.json"),
                "entries": len(sectors) if isinstance(sectors, Sequence) else 0,
            },
            {
                "name": "stock_features.json",
                "path": to_relative(output_dir / "stock_features.json"),
                "entries": len(stocks) if isinstance(stocks, Mapping) else 0,
            },
            {
                "name": "premarket_flags.json",
                "path": to_relative(output_dir / "premarket_flags.json"),
                "entries": len(premarket) if isinstance(premarket, Mapping) else 0,
            },
            {
                "name": "news_bundle.json",
                "path": to_relative(output_dir / "news_bundle.json"),
                "entries": count_news_articles(news_bundle),
            },
            {
                "name": "trend_features.json",
                "path": to_relative(output_dir / "trend_features.json"),
                "entries": len(trend_bundle) if isinstance(trend_bundle, Mapping) else 0,
            },
            {
                "name": "macro_flags.json",
                "path": to_relative(output_dir / "macro_flags.json"),
                "entries": len(macro_flags) if isinstance(macro_flags, Mapping) else 0,
            },
        ]

        analyzer: Optional[DeepSeekAnalyzer] = None
        deepseek_client: Optional[DeepSeekClient] = None

        if prompt_files:
            try:
                deepseek_client = DeepSeekClient.from_env()
                analyzer = DeepSeekAnalyzer(
                    prompt_files=prompt_files,
                    client=deepseek_client,
                    base_prompt=resolve_path(project_root, base_prompt_path)
                    if base_prompt_path
                    else None,
                    logger=logger,
                )
            except RuntimeError as exc:
                log_result(logger, "llm", f"Analyzer disabled: {exc}")

        if analyzer:
            log_result(
                logger,
                "llm",
                f"DeepSeek prompts configured: {len(prompt_files)} stages",
            )
        elif not prompt_files:
            log_result(logger, "llm", "Skipped (no prompt files configured)")

        llm_orchestrator: Optional[LLMOrchestrator] = None
        operator_settings = llm_config.get("operators", {})
        if operator_settings:
            try:
                if deepseek_client is None:
                    deepseek_client = DeepSeekClient.from_env()
                resolved_ops = {
                    stage: {
                        **settings,
                        "prompt_file": resolve_path(
                            project_root, settings.get("prompt_file", "")
                        ),
                    }
                    for stage, settings in operator_settings.items()
                }
                prepared_llm_config = dict(llm_config)
                prepared_llm_config["operators"] = resolved_ops
                guardrails = llm_config.get("guardrails", {})
                safe_dict = llm_config.get("safe_mode", {})
                safe_config = SafeModeConfig(
                    on_llm_failure=safe_dict.get("on_llm_failure", "no_new_risk"),
                    max_exposure_cap=float(safe_dict.get("max_exposure_cap", 0.4)),
                )
                base_prompt_full = (
                    resolve_path(project_root, llm_config.get("base_prompt"))
                    if llm_config.get("base_prompt")
                    else None
                )
                llm_orchestrator = LLMOrchestrator(
                    client=deepseek_client,
                    operator_configs=prepared_llm_config,
                    base_prompt_path=base_prompt_full,
                    storage_dir=output_dir / "llm",
                    guardrails=guardrails,
                    safe_mode_config=safe_config,
                    logger=logger,
                )
                log_result(logger, "llm", "LLM orchestrator ready")
            except RuntimeError as exc:
                log_result(logger, "llm", f"Orchestrator disabled: {exc}")

        agent = BaseAgent(
            config=config,
            macro_engine=MacroRiskEngine(),
            stock_engine=StockDecisionEngine(),
            sizer=PositionSizer(config["limits"], config["sizer"]),
            portfolio_state=state,
            report_builder=HybridReportBuilder(config["sizer"]),
            analyzer=analyzer,
            llm_orchestrator=llm_orchestrator,
            logger=logger,
        )

        logger.info("开始执行日常流程，输出目录: %s", output_dir)

        report_meta = {
            "appendix": {
                "report_json_path": to_relative(report_json_path),
                "reproduction_command": reproduction_command,
            },
            "artefact_summary": artefact_summary,
        }

        context = agent.run(
            trading_day=trading_day,
            market_features=market,
            sector_features=sectors,
            stock_features=stocks,
            premarket_flags=premarket,
            trend_features=trend_bundle,
            news=news_bundle,
            macro_flags=macro_flags,
            output_dir=output_dir,
            snapshot_meta=snapshot_meta,
            renderer_config=renderer_config,
            report_meta=report_meta,
        )
        phases_executed.extend(context.stage_metrics.keys())

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            log_step(logger, "run_daily", f"写入特征文件到 {output_dir}")
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
            (output_dir / "macro_flags.json").write_text(
                json.dumps(macro_flags, indent=2), encoding="utf-8"
            )
            (output_dir / "snapshot_meta.json").write_text(
                json.dumps(snapshot_meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        save_positions_snapshot(positions_path, state, trading_day)
        log_result(
            logger,
            "portfolio_manager",
            f"Positions snapshot updated ({positions_path.name})",
        )

        summary_duration = perf_counter() - start_time
        unique_phases: List[str] = []
        seen = set()
        for phase in phases_executed:
            if phase not in seen:
                unique_phases.append(phase)
                seen.add(phase)

        stock_actions = {"buy": 0, "hold": 0, "reduce": 0, "avoid": 0}
        for item in context.stock_scores:
            key = item.get("action", "hold")
            if key in stock_actions:
                stock_actions[key] += 1

        buy_notional = sum(order.get("notional", 0.0) for order in context.orders.get("buy", []))
        report_path = output_dir / "report.md" if output_dir else None

        logger.info("==================== SUMMARY ====================")
        logger.info("Total time: %.2fs", summary_duration)
        logger.info(
            "Phases executed: %s",
            " → ".join(unique_phases) if unique_phases else "n/a",
        )
        logger.info(
            "Stocks analyzed: %d (Buy=%d, Hold=%d, Reduce=%d, Avoid=%d)",
            len(context.stock_scores),
            stock_actions["buy"],
            stock_actions["hold"],
            stock_actions["reduce"],
            stock_actions["avoid"],
        )
        logger.info(
            "Orders: buy=%d (notional=%.2f), sell=%d",
            len(context.orders.get("buy", [])),
            buy_notional,
            len(context.orders.get("sell", [])),
        )
        if report_path:
            logger.info("Report saved: %s", report_path)
        logger.info("=================================================")

    except Exception:
        logger.exception("执行 run_daily 发生异常")
        logger.error("❌ 任务中断")
        raise
    else:
        duration = perf_counter() - start_time
        logger.info("✅ 任务执行完成，用时 %.1f 秒 (日志文件: %s)", duration, log_path)


if __name__ == "__main__":
    main()
