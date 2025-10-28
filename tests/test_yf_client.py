"""Unit tests for the Yahoo Finance client news handling."""
from datetime import datetime, timezone

from ai_trader_assist.data_collector import yf_client
from ai_trader_assist.data_collector.yf_client import YahooFinanceClient


class _StubTicker:
    """Lightweight stub that mimics ``yfinance.Ticker`` for news payloads."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._info = {"longName": "Apple Inc."}

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

    def get_info(self):
        return self._info

    @property
    def info(self):  # type: ignore[override]
        return self._info


class _StubYF:
    def Ticker(self, symbol: str) -> _StubTicker:  # noqa: N802 - mimic yfinance API
        return _StubTicker(symbol)


class _StubSearch:
    def __init__(self, query: str) -> None:
        self.query = query
        self.quotes = [
            {"symbol": "AAPL", "longname": "Apple Inc."},
        ]
        self.news = [
            {
                "title": f"{query} hits the wires",
                "publisher": "Example News",
                "link": "https://example.com/company",
                "providerPublishTime": int(
                    datetime(2025, 10, 26, 12, 0, tzinfo=timezone.utc).timestamp()
                ),
            }
        ]


def test_fetch_news_parses_nested_content(tmp_path, monkeypatch):
    """Ensure modern yfinance news payloads are normalised correctly."""

    monkeypatch.setattr(yf_client, "yf", _StubYF())
    monkeypatch.setattr(yf_client, "YFSearch", _StubSearch)

    client = YahooFinanceClient(cache_dir=tmp_path)
    articles = client.fetch_news("AAPL", lookback_days=7, max_items=5, force=True)

    assert len(articles) == 2
    titles = {article["title"] for article in articles}
    assert "Sample headline" in titles
    assert "Apple Inc. hits the wires" in titles

    sample = next(article for article in articles if article["title"] == "Sample headline")

    assert sample["summary"] == "Concise summary for testing."
    assert sample["publisher"] == "Yahoo Finance"
    assert sample["link"] == "https://example.com/article"

    published = datetime.fromisoformat(sample["published"])
    assert published.replace(tzinfo=published.tzinfo or timezone.utc) >= datetime(
        2025, 10, 25, tzinfo=timezone.utc
    )
