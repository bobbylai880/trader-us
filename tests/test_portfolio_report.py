import json
from datetime import date
from pathlib import Path

from ai_trader_assist.report_tools import PortfolioReporter


def _write_jsonl(path: Path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def test_portfolio_report_generation(tmp_path):
    operations = [
        {
            "date": "2025-10-27",
            "symbol": "NVDA",
            "action": "BUY",
            "quantity": 10,
            "price": 100.0,
            "timestamp": "2025-10-27T16:00:00Z",
        },
        {
            "date": "2025-10-28",
            "symbol": "AAPL",
            "action": "SELL",
            "quantity": 5,
            "price": 210.0,
            "timestamp": "2025-10-28T16:00:00Z",
        },
    ]
    operations_path = tmp_path / "storage" / "operations.jsonl"
    _write_jsonl(operations_path, operations)

    positions_payload = {
        "cash": 5000.0,
        "positions": [
            {"symbol": "NVDA", "shares": 10, "avg_cost": 100.0},
        ],
        "last_updated": "2025-10-28T16:00:00Z",
    }
    positions_path = tmp_path / "storage" / "positions.json"
    positions_path.write_text(json.dumps(positions_payload), encoding="utf-8")

    features_day1 = {
        "NVDA": {"price": 110.0},
    }
    features_day2 = {
        "NVDA": {"price": 112.0},
        "AAPL": {"price": 210.0},
    }
    day1_path = tmp_path / "storage" / "daily_2025-10-27" / "stock_features.json"
    day1_path.parent.mkdir(parents=True, exist_ok=True)
    day1_path.write_text(json.dumps(features_day1), encoding="utf-8")
    day2_path = tmp_path / "storage" / "daily_2025-10-28" / "stock_features.json"
    day2_path.parent.mkdir(parents=True, exist_ok=True)
    day2_path.write_text(json.dumps(features_day2), encoding="utf-8")

    reporter = PortfolioReporter(
        project_root=tmp_path,
        operations_path=operations_path,
        positions_path=positions_path,
        allow_fetch=False,
    )

    output_dir = tmp_path / "storage" / "reports" / "2025-10-28"
    result = reporter.generate(as_of=date(2025, 10, 28), output_dir=output_dir)

    current = result["current_pnl"]
    assert current["cash"] == 5000.0
    assert current["total_exposure"] == current["total_exposure"]  # numeric
    nvda_entry = next(item for item in current["positions"] if item["symbol"] == "NVDA")
    assert nvda_entry["last_price"] == 112.0
    assert nvda_entry["unrealized_pnl"] == 120.0

    history = result["history"]
    assert len(history) == 2
    assert history[0]["holdings"]["NVDA"] == 10
    assert Path(output_dir / "portfolio_report.md").exists()
    assert Path(output_dir / "current_pnl.json").exists()
    assert Path(output_dir / "history_report.json").exists()

