"""CLI entrypoint to generate portfolio history and PnL reports."""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

from ..report_tools import PortfolioReporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate portfolio reports")
    parser.add_argument("--config", default="configs/base.json", help="Configuration file path")
    parser.add_argument(
        "--as-of",
        help="Target date for the report (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to write report artefacts. Defaults to storage/reports/<date>.",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Do not fetch remote prices; rely solely on cached feature files.",
    )
    return parser.parse_args()


def resolve_path(root: Path, maybe_relative: str | None) -> Path:
    if not maybe_relative:
        return root
    path = Path(maybe_relative)
    return path if path.is_absolute() else root / path


def load_config(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config_path = resolve_path(project_root, args.config)
    config = load_config(config_path)

    logging_cfg = config.get("logging", {})
    operations_path = resolve_path(project_root, logging_cfg.get("operations_path", "storage/operations.jsonl"))
    positions_path = resolve_path(project_root, logging_cfg.get("positions_path", "storage/positions.json"))

    if args.as_of:
        as_of = datetime.fromisoformat(args.as_of).date()
    else:
        as_of = date.today()

    if args.output_dir:
        output_dir = resolve_path(project_root, args.output_dir)
    else:
        output_dir = project_root / "storage" / "reports" / as_of.isoformat()

    reporter = PortfolioReporter(
        project_root=project_root,
        operations_path=operations_path,
        positions_path=positions_path,
        allow_fetch=not args.no_fetch,
    )
    result = reporter.generate(as_of=as_of, output_dir=output_dir)

    print("✅ current_pnl.json written to", output_dir / "current_pnl.json")
    print("✅ history_report.json written to", output_dir / "history_report.json")
    print("✅ portfolio_report.md written to", output_dir / "portfolio_report.md")
    print("\nPreview:\n")
    print(result["markdown"])


if __name__ == "__main__":
    main()

