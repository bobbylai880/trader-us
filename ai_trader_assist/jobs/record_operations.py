"""Interactive CLI to record manual trade operations and update positions."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List

from ..portfolio_manager.positions import (
    load_positions_snapshot,
    save_positions_snapshot,
)
from ..portfolio_manager.state import PortfolioState


ALLOWED_ACTIONS = {"BUY", "SELL", "REDUCE", "HOLD"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record manual trade operations")
    parser.add_argument(
        "--config",
        default="configs/base.json",
        help="Optional configuration file containing logging paths.",
    )
    return parser.parse_args()


def resolve_path(root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else root / path


def load_logging_paths(project_root: Path, config_path: Path) -> tuple[Path, Path]:
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        config = {}
    logging_cfg = config.get("logging", {})
    operations_path = resolve_path(
        project_root, logging_cfg.get("operations_path", "storage/operations.jsonl")
    )
    positions_path = resolve_path(
        project_root, logging_cfg.get("positions_path", "storage/positions.json")
    )
    return operations_path, positions_path


def prompt_with_default(prompt: str, default: str) -> str:
    response = input(f"{prompt} (default: {default}): ").strip()
    return response or default


def prompt_symbol() -> str:
    while True:
        value = input("Symbol (e.g., NVDA): ").strip().upper()
        if value:
            return value
        print("Symbol cannot be empty. Please try again.")


def prompt_action() -> str:
    while True:
        value = input("Action [BUY/SELL/REDUCE/HOLD]: ").strip().upper()
        if value in ALLOWED_ACTIONS:
            return value
        print("Invalid action. Choose from BUY, SELL, REDUCE, HOLD.")


def prompt_float(prompt: str) -> float:
    while True:
        value = input(f"{prompt}: ").strip()
        if not value:
            print("Value is required. Please try again.")
            continue
        try:
            return float(value)
        except ValueError:
            print("Invalid number. Please try again.")


def backup_file(path: Path) -> None:
    if not path.exists():
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = path.with_suffix(path.suffix + f".{timestamp}.bak")
    backup_path.write_bytes(path.read_bytes())


def append_operations(path: Path, entries: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_file(path)
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def update_positions(path: Path, state: PortfolioState, operations: List[dict], timestamp: datetime) -> None:
    state.apply_operations(operations)
    state.update_last_updated(timestamp)
    backup_file(path)
    save_positions_snapshot(path, state, snapshot_date=timestamp.date(), updated_at=timestamp)


def main() -> None:
    try:
        args = parse_args()
        project_root = Path(__file__).resolve().parents[2]
        config_path = resolve_path(project_root, args.config)
        operations_path, positions_path = load_logging_paths(project_root, config_path)

        today = date.today().isoformat()
        print("=== Record Today's Operations ===")
        entry_date = prompt_with_default("Date", today)

        recorded: List[dict] = []
        while True:
            symbol = prompt_symbol()
            action = prompt_action()
            quantity = 0.0
            price = 0.0
            if action != "HOLD":
                quantity = prompt_float("Quantity")
                price = prompt_float("Price")
            reason = input("Reason (optional): ").strip()
            timestamp = datetime.now(timezone.utc)
            record = {
                "date": entry_date,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": price,
                "reason": reason or None,
                "source": "manual",
                "timestamp": timestamp.isoformat(),
            }
            recorded.append(record)
            cont = input("Add another? (y/n): ").strip().lower()
            if cont != "y":
                break

        if not recorded:
            print("No operations recorded. Exiting.")
            return

        confirm = input("Save and update positions? (y/n): ").strip().lower()
        if confirm != "y":
            print("Operations discarded.")
            return

        append_operations(operations_path, recorded)
        state = load_positions_snapshot(positions_path)
        operations_for_state = [
            {
                "symbol": entry["symbol"],
                "action": entry["action"].lower(),
                "shares": entry.get("quantity", 0.0),
                "price": entry.get("price", 0.0),
            }
            for entry in recorded
        ]
        update_positions(positions_path, state, operations_for_state, datetime.now(timezone.utc))

        print(f"✅ {len(recorded)} operations appended to {operations_path.name}")
        print("✅ positions.json updated successfully")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
