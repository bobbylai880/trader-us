"""Thin wrapper around yfinance with basic caching and offline fallbacks."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

try:  # pragma: no cover - optional dependency during unit tests
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

try:  # pragma: no cover - optional dependency during unit tests
    from yfinance.search import Search as YFSearch
except Exception:  # pragma: no cover
    YFSearch = None


class _ArticleTextExtractor(HTMLParser):
    """Minimal HTML -> text extractor for article bodies."""

    _BLOCK_TAGS = {"p", "div", "br", "li", "section", "article", "h1", "h2", "h3", "h4"}
    _SKIP_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._skip_level = 0

    def handle_starttag(self, tag: str, attrs):  # noqa: D401 - part of HTMLParser API
        tag_lower = tag.lower()
        if tag_lower in self._SKIP_TAGS:
            self._skip_level += 1
            return
        if tag_lower in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str):  # noqa: D401 - part of HTMLParser API
        tag_lower = tag.lower()
        if tag_lower in self._SKIP_TAGS and self._skip_level > 0:
            self._skip_level -= 1
            return
        if tag_lower in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str):  # noqa: D401 - part of HTMLParser API
        if self._skip_level:
            return
        text = data.strip()
        if not text:
            return
        self._parts.append(unescape(text))

    def get_text(self) -> str:
        content = " ".join(segment.strip() for segment in self._parts if segment.strip())
        content = re.sub(r"\s+", " ", content)
        return content.strip()


class YahooFinanceClient:
    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or Path("storage/cache/yf")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._company_name_cache: Dict[str, Optional[str]] = {}
        self._news_content_dir = self.cache_dir / "news_content"
        self._news_content_dir.mkdir(parents=True, exist_ok=True)
        # Cap the number of network fetches for article bodies in a single
        # session so offline environments do not spend minutes waiting for
        # repeated timeouts when traversing a large universe.
        self._content_fetch_budget = 8

    # ------------------------------------------------------------------
    # Price history helpers
    # ------------------------------------------------------------------

    def _cache_path(self, symbol: str, start: datetime, end: datetime, interval: str) -> Path:
        key = f"{symbol}_{start.date()}_{end.date()}_{interval}.parquet"
        return self.cache_dir / key

    def _synthetic_history(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        """Generate a deterministic synthetic price series as a fallback."""
        if interval != "1d":
            return pd.DataFrame()

        # yfinance treats ``end`` as exclusive; mirror that behaviour.
        end_exclusive = end - timedelta(days=1)
        index = pd.date_range(start=start, end=end_exclusive, freq="B")
        if index.empty:
            index = pd.date_range(end=end_exclusive, periods=30, freq="B")

        if index.empty:
            return pd.DataFrame()

        # Create a stable pseudo-random seed per symbol so results are repeatable.
        seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(seed)

        base_price = rng.uniform(40, 320)
        drift = rng.normal(0.05, 0.01)
        shocks = rng.normal(0, 1.5, size=len(index))
        close = np.maximum(1.0, base_price + np.cumsum(drift + shocks))

        open_price = close * (1 + rng.normal(0, 0.01, len(index)))
        high = np.maximum(open_price, close) * (1 + np.abs(rng.normal(0.01, 0.005, len(index))))
        low = np.minimum(open_price, close) * (1 - np.abs(rng.normal(0.01, 0.005, len(index))))
        volume = rng.integers(1_000_000, 5_000_000, len(index))

        data = pd.DataFrame(
            {
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Adj Close": close,
                "Volume": volume,
            },
            index=index,
        )
        return data

    def fetch_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        force: bool = False,
    ) -> pd.DataFrame:
        """Fetch price history, caching results locally or using a fallback."""
        cache_path = self._cache_path(symbol, start, end, interval)
        if cache_path.exists() and not force:
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                cache_path.unlink(missing_ok=True)

        if yf is None:
            data = pd.DataFrame()
        else:
            try:
                data = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                )
            except Exception:
                data = pd.DataFrame()

        if data.empty and cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                cache_path.unlink(missing_ok=True)

        if data.empty:
            data = self._synthetic_history(symbol, start, end, interval)

        if data.empty:
            return data

        try:
            data.to_parquet(cache_path)
        except Exception:
            pass
        return data

    def latest_price(self, symbol: str) -> Optional[float]:
        end = datetime.utcnow()
        start = end - timedelta(days=10)
        history = self.fetch_history(symbol, start=start, end=end, interval="1d")
        if history.empty or "Close" not in history or history["Close"].empty:
            return None
        return float(history["Close"].iloc[-1])

    # ------------------------------------------------------------------
    # News helpers
    # ------------------------------------------------------------------

    def _news_cache_path(self, symbol: str, as_of: Optional[datetime] = None) -> Path:
        news_dir = self.cache_dir / "news"
        news_dir.mkdir(parents=True, exist_ok=True)
        suffix = "latest"
        if as_of is not None:
            if as_of.tzinfo is None:
                suffix = as_of.date().isoformat()
            else:
                suffix = as_of.astimezone(timezone.utc).date().isoformat()
        return news_dir / f"{symbol}_{suffix}.json"

    def _article_cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self._news_content_dir / f"{digest}.json"

    @staticmethod
    def _extract_text_from_html(html: str) -> str:
        if not html:
            return ""
        parser = _ArticleTextExtractor()
        try:
            parser.feed(html)
            parser.close()
        except Exception:
            pass
        text = parser.get_text()
        if not text:
            return ""
        # Limit to a reasonable size for prompt payloads.
        return text[:8000]

    def _fetch_article_content(self, url: str, *, force: bool = False) -> str:
        if not url:
            return ""

        if self._content_fetch_budget <= 0 and not force:
            return ""

        cache_path = self._article_cache_path(url)
        if cache_path.exists() and not force:
            try:
                payload = json.loads(cache_path.read_text())
                cached = payload.get("content", "")
                if cached:
                    return cached
            except Exception:
                cache_path.unlink(missing_ok=True)

        self._content_fetch_budget -= 1

        try:
            response = requests.get(
                url,
                timeout=(1, 2),
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; AI-Trader/1.0; +https://example.com)"
                },
            )
        except Exception:
            response = None

        text = ""
        if response is not None and getattr(response, "status_code", 0) == 200:
            try:
                text = self._extract_text_from_html(response.text)
            except Exception:
                text = ""

        if not text:
            return ""

        try:
            cache_path.write_text(
                json.dumps({"content": text, "_cached_at": datetime.utcnow().isoformat()}, ensure_ascii=False)
            )
        except Exception:
            pass

        return text

    # ------------------------------------------------------------------
    # Company metadata helpers
    # ------------------------------------------------------------------

    def resolve_company_name(self, symbol: str) -> Optional[str]:
        """Return the long company name for ``symbol`` when available."""

        key = symbol.upper()
        if key in self._company_name_cache:
            cached = self._company_name_cache[key]
            return cached

        long_name: Optional[str] = None

        if YFSearch is not None:
            try:
                search_client = YFSearch(symbol)
                raw_quotes = getattr(search_client, "quotes", []) or []
                for quote in raw_quotes:
                    if not isinstance(quote, dict):
                        continue
                    quote_symbol = str(quote.get("symbol") or "").upper()
                    if quote_symbol and quote_symbol != key:
                        continue
                    for field in ("longname", "shortname", "name"):
                        candidate = quote.get(field)
                        if isinstance(candidate, str) and candidate.strip():
                            long_name = candidate.strip()
                            break
                    if long_name:
                        break
                if not long_name:
                    query_value = getattr(search_client, "query", None)
                    if isinstance(query_value, str) and query_value.strip():
                        long_name = query_value.strip()
            except Exception:
                long_name = None

        if long_name is None and yf is not None:
            try:
                ticker = yf.Ticker(symbol)
                raw_info = getattr(ticker, "info", {})
                if isinstance(raw_info, dict):
                    for field in ("longName", "shortName"):
                        candidate = raw_info.get(field)
                        if isinstance(candidate, str) and candidate.strip():
                            long_name = candidate.strip()
                            break
            except Exception:
                long_name = None

        self._company_name_cache[key] = long_name
        return long_name

    @staticmethod
    def _filter_articles(
        articles: List[dict],
        *,
        cutoff: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[dict]:
        """Normalise cached articles and drop stale/empty entries."""

        filtered: List[dict] = []
        for article in articles:
            title = (article.get("title") or "").strip()
            summary = (article.get("summary") or "").strip()
            content = (article.get("content") or "").strip()
            if not (title or summary or content):
                continue

            if not summary and title:
                summary = title

            if not content:
                content = summary or title

            if not content:
                fallback_bits = [title]
                fallback_bits.append("未能获取新闻正文，保留标题作为参考。")
                content = " ".join(bit for bit in fallback_bits if bit).strip()

            published_raw = article.get("published")
            published_dt: Optional[datetime]
            if isinstance(published_raw, str):
                try:
                    published_dt = datetime.fromisoformat(published_raw)
                except ValueError:
                    published_dt = None
            else:
                published_dt = None

            if cutoff is not None and published_dt is not None:
                published_norm = published_dt.replace(
                    tzinfo=published_dt.tzinfo or timezone.utc
                )
                if published_norm < cutoff:
                    continue
            if end is not None and published_dt is not None:
                published_norm = published_dt.replace(
                    tzinfo=published_dt.tzinfo or timezone.utc
                )
                if published_norm > end:
                    continue

            normalised = {
                "title": title,
                "summary": summary,
                "publisher": (article.get("publisher") or "").strip(),
                "link": (article.get("link") or "").strip(),
                "published": (
                    published_dt.isoformat()
                    if published_dt is not None
                    else article.get("published")
                ),
                "content": content,
            }
            filtered.append(normalised)
        return filtered

    def fetch_news(
        self,
        symbol: str,
        max_items: int = 20,
        lookback_days: int = 7,
        *,
        force: bool = False,
        as_of: Optional[datetime] = None,
    ) -> List[dict]:
        """Return recent news articles for ``symbol``.

        Results are cached locally for a few hours to limit upstream calls and to
        keep offline runs deterministic.  Each entry mirrors the structure
        produced by ``yfinance.Ticker.news`` with only the relevant fields.
        """

        if as_of is None:
            now = datetime.utcnow().replace(tzinfo=timezone.utc)
        else:
            if as_of.tzinfo is None:
                now = as_of.replace(tzinfo=timezone.utc)
            else:
                now = as_of.astimezone(timezone.utc)

        cache_path = self._news_cache_path(symbol, now)
        ttl = timedelta(hours=3)

        cutoff = now - timedelta(days=max(lookback_days, 1))

        cached_articles: List[dict] = []
        cached_at: Optional[datetime] = None
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text())
                cached_at_raw = payload.get("_cached_at")
                cached_at = datetime.fromisoformat(cached_at_raw) if cached_at_raw else None
                if cached_at is not None:
                    if cached_at.tzinfo is None:
                        cached_at = cached_at.replace(tzinfo=timezone.utc)
                    else:
                        cached_at = cached_at.astimezone(timezone.utc)
                cached_articles = self._filter_articles(
                    payload.get("articles", []), cutoff=cutoff, end=now
                )
                if (
                    cached_articles
                    and cached_at is not None
                    and not force
                    and timedelta(0) <= now - cached_at <= ttl
                ):
                    return cached_articles[:max_items]
            except Exception:
                cache_path.unlink(missing_ok=True)

        def _parse_datetime(value: object) -> Optional[datetime]:
            if isinstance(value, (int, float)):
                try:
                    return datetime.fromtimestamp(value, tz=timezone.utc)
                except (OverflowError, OSError, ValueError):
                    return None
            if isinstance(value, str):
                candidate = value.strip()
                if not candidate:
                    return None
                for normalised in (candidate, candidate.replace("Z", "+00:00")):
                    try:
                        return datetime.fromisoformat(normalised)
                    except ValueError:
                        continue
            return None

        def _extract_url(container: object) -> str:
            if isinstance(container, str):
                return container.strip()
            if isinstance(container, dict):
                for key in ("url", "href", "link"):
                    value = container.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
            return ""

        def _normalise_article(item: dict) -> Optional[dict]:
            content = item.get("content") if isinstance(item.get("content"), dict) else {}

            title_candidates = [
                item.get("title"),
                content.get("title"),
                content.get("headline"),
            ]
            summary_candidates = [
                item.get("summary"),
                content.get("summary"),
                content.get("description"),
            ]

            title = next((str(c).strip() for c in title_candidates if c), "")
            summary = next((str(c).strip() for c in summary_candidates if c), "")
            if not summary and title:
                summary = title

            publisher_candidates = [
                item.get("publisher"),
                item.get("provider", {}).get("displayName")
                if isinstance(item.get("provider"), dict)
                else None,
                content.get("publisher"),
                content.get("provider", {}).get("displayName")
                if isinstance(content.get("provider"), dict)
                else None,
            ]
            publisher = next(
                (str(c).strip() for c in publisher_candidates if isinstance(c, str) and c.strip()),
                "",
            )

            link_candidates = [
                item.get("link"),
                _extract_url(item.get("canonicalUrl")),
                _extract_url(item.get("clickThroughUrl")),
                content.get("link"),
                _extract_url(content.get("canonicalUrl")),
            ]
            link = next((c for c in link_candidates if isinstance(c, str) and c.strip()), "")

            published_dt = _parse_datetime(item.get("providerPublishTime"))
            if published_dt is None:
                published_dt = _parse_datetime(content.get("pubDate"))
            if published_dt is None:
                published_dt = _parse_datetime(item.get("pubDate"))
            if published_dt is None:
                published_dt = _parse_datetime(content.get("displayTime"))
            if published_dt is None:
                published_dt = _parse_datetime(item.get("displayTime"))
            if published_dt is None:
                published_dt = now.replace(tzinfo=timezone.utc)

            if not (title or summary):
                return None

            return {
                "title": title,
                "summary": summary,
                "publisher": publisher,
                "link": link,
                "published": published_dt.isoformat(),
                "content": summary,
            }

        articles: List[dict] = []
        if yf is not None:
            try:
                raw_items = getattr(yf.Ticker(symbol), "news", []) or []
                for item in raw_items[: max_items * 3]:
                    normalised = _normalise_article(item)
                    if not normalised:
                        continue

                    published_dt = _parse_datetime(normalised.get("published"))
                    if published_dt is None:
                        published_dt = now
                    if published_dt.tzinfo is None:
                        published_dt = published_dt.replace(tzinfo=timezone.utc)
                    else:
                        published_dt = published_dt.astimezone(timezone.utc)
                    if published_dt >= cutoff and published_dt <= now:
                        articles.append(normalised)

                    if len(articles) >= max_items:
                        break
            except Exception:
                articles = []

        company_articles: List[dict] = []
        company_name = self.resolve_company_name(symbol)
        if company_name and YFSearch is not None:
            try:
                search_client = YFSearch(company_name)
                raw_items = getattr(search_client, "news", []) or []
                for item in raw_items[: max_items * 3]:
                    normalised = _normalise_article(item)
                    if not normalised:
                        continue

                    published_dt = _parse_datetime(normalised.get("published"))
                    if published_dt is None:
                        published_dt = now
                    if published_dt.tzinfo is None:
                        published_dt = published_dt.replace(tzinfo=timezone.utc)
                    else:
                        published_dt = published_dt.astimezone(timezone.utc)
                    normalised["published"] = published_dt.isoformat()

                    if published_dt >= cutoff and published_dt <= now:
                        company_articles.append(normalised)

                    if len(company_articles) >= max_items:
                        break
            except Exception:
                company_articles = []

        combined: List[dict] = []
        if cached_articles:
            combined.extend(cached_articles)
        if articles:
            combined.extend(articles)
        if company_articles:
            combined.extend(company_articles)

        if combined:
            # Deduplicate by (title, published) and keep the most recent entries.
            dedup_map = {}
            for entry in combined:
                key = (entry.get("title"), entry.get("published"))
                dedup_map[key] = entry

            def _sort_key(item: dict) -> datetime:
                published = item.get("published")
                if isinstance(published, str):
                    try:
                        return datetime.fromisoformat(published)
                    except ValueError:
                        pass
                return datetime.min.replace(tzinfo=timezone.utc)

            combined = sorted(dedup_map.values(), key=_sort_key, reverse=True)
            combined = combined[: max(max_items, 10)]

            for idx, entry in enumerate(combined):
                existing_content = (entry.get("content") or "").strip()
                summary_text = (entry.get("summary") or "").strip()
                if idx >= 1 and not force:
                    fallback_text = existing_content or summary_text
                    if not fallback_text:
                        title_text = (entry.get("title") or "").strip()
                        fallback_bits = [title_text]
                        fallback_bits.append("未能获取新闻正文，保留标题作为参考。")
                        fallback_text = " ".join(bit for bit in fallback_bits if bit)
                    entry["content"] = fallback_text.strip()
                    continue
                should_fetch = force or not existing_content or existing_content == summary_text
                content_text = existing_content
                if should_fetch:
                    link = (entry.get("link") or "").strip()
                    if link:
                        fetched = self._fetch_article_content(link, force=force)
                        if fetched:
                            content_text = fetched.strip()
                if not content_text:
                    content_text = summary_text
                if not content_text:
                    fallback_bits = [entry.get("title", "").strip()]
                    fallback_bits.append("未能获取新闻正文，保留标题作为参考。")
                    content_text = " ".join(bit for bit in fallback_bits if bit)
                entry["content"] = content_text.strip()

        if not combined:
            # Provide a deterministic synthetic article so downstream logic has
            # context even when running offline.
            seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:4], "big")
            pseudo_sentiment = (seed % 200 - 100) / 100.0
            tone = "积极" if pseudo_sentiment >= 0 else "谨慎"
            combined = [
                {
                    "title": f"{symbol} {tone} 动态（离线合成）",
                    "summary": "无实时新闻数据，生成合成概览以支持新闻因子分析。",
                    "publisher": "synthetic",
                    "link": "",
                    "published": now.astimezone(timezone.utc).isoformat(),
                    "content": "无实时新闻数据，生成合成概览以支持新闻因子分析。",
                }
            ]

        try:
            cache_path.write_text(
                json.dumps({
                    "_cached_at": now.isoformat(),
                    "articles": combined,
                }, ensure_ascii=False)
            )
        except Exception:
            pass

        return combined[:max_items]
