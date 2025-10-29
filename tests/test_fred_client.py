import json
import logging
from pathlib import Path

import pytest
import requests

from ai_trader_assist.data_collector.fred_client import FredClient


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - simple stub
        return None

    def json(self):
        return self._payload


def test_fetch_macro_indicators_returns_latest_and_change(tmp_path, monkeypatch):
    payload = {
        "observations": [
            {"date": "2025-09-01", "value": "100.0"},
            {"date": "2025-10-01", "value": "104.2"},
        ]
    }

    def fake_get(url, params, timeout):
        assert params["series_id"] == "TEST"
        return _DummyResponse(payload)

    monkeypatch.setattr("ai_trader_assist.data_collector.fred_client.requests.get", fake_get)

    client = FredClient(api_key="demo", cache_dir=tmp_path)
    result = client.fetch_macro_indicators(["TEST"], start_date="2025-01-01")

    assert "TEST" in result
    assert result["TEST"]["value"] == pytest.approx(104.2)
    assert result["TEST"]["change"] == pytest.approx(4.2)
    assert result["TEST"]["as_of"] == "2025-10-01"


def test_fetch_macro_indicators_uses_cache_on_failure(tmp_path, monkeypatch, caplog):
    cache_path = Path(tmp_path) / "TEST.json"
    cache_payload = [
        {"date": "2025-09-01", "value": "98.0"},
        {"date": "2025-10-01", "value": "101.0"},
    ]
    cache_path.write_text(json.dumps(cache_payload))

    def failing_get(url, params, timeout):
        raise requests.RequestException("boom")

    monkeypatch.setattr(
        "ai_trader_assist.data_collector.fred_client.requests.get", failing_get
    )

    logger = logging.getLogger("fred-test")
    caplog.set_level(logging.WARNING, logger="fred-test")

    client = FredClient(api_key="demo", cache_dir=tmp_path, logger=logger)
    result = client.fetch_macro_indicators(["TEST"], start_date="2025-01-01")

    assert result["TEST"]["value"] == pytest.approx(101.0)
    assert any("[WARN] FRED fetch failed, using last cache" in record.message for record in caplog.records)
