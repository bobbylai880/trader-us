"""Unit tests for the Yahoo Finance client news handling."""
from datetime import datetime, timezone

from ai_trader_assist.data_collector import yf_client
from ai_trader_assist.data_collector.yf_client import YahooFinanceClient


class _StubTicker:
    """Lightweight stub that mimics ``yfinance.Ticker`` for news payloads."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    @property
    def news(self):  # type: ignore[override]
        return [
            {
                "content": {
                    "title": "Sample headline",
                    "description": "Concise summary for testing.",
                    "pubDate": "2025-10-25T12:00:00Z",
                    "canonicalUrl": {"url": "https://example.com/article"},
                },
                "provider": {"displayName": "Yahoo Finance"},
            }
        ]


class _StubYF:
    def Ticker(self, symbol: str) -> _StubTicker:  # noqa: N802 - mimic yfinance API
        return _StubTicker(symbol)


def test_fetch_news_parses_nested_content(tmp_path, monkeypatch):
    """Ensure modern yfinance news payloads are normalised correctly."""

    monkeypatch.setattr(yf_client, "yf", _StubYF())

    client = YahooFinanceClient(cache_dir=tmp_path)
    articles = client.fetch_news("AAPL", lookback_days=7, max_items=5, force=True)

    assert len(articles) == 1
    article = articles[0]

    assert article["title"] == "Sample headline"
    assert article["summary"] == "Concise summary for testing."
    assert article["publisher"] == "Yahoo Finance"
    assert article["link"] == "https://example.com/article"

    published = datetime.fromisoformat(article["published"])
    assert published.replace(tzinfo=published.tzinfo or timezone.utc) >= datetime(
        2025, 10, 25, tzinfo=timezone.utc
    )
