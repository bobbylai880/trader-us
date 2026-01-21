"""动态选股模块 - UniverseBuilder.

实现分层股票池结构：
- Core Pool: 龙头股常驻 (8只)
- Rotation Pool: 板块轮动 (15-20只)
- Candidate Pool: 新兴强势股 (10-15只)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from ai_trader_assist.data_collector.pg_client import PostgresMarketDB, get_db
from ai_trader_assist.risk_engine.market_regime import MarketRegime


class PoolType(Enum):
    CORE = "core"
    ROTATION = "rotation"
    CANDIDATE = "candidate"


@dataclass
class StockCandidate:
    symbol: str
    pool: PoolType
    sector_etf: Optional[str] = None
    score: float = 0.0
    momentum_10d: float = 0.0
    rsi_14: float = 50.0
    volume_ratio: float = 1.0
    atr_pct: float = 0.02
    close: float = 0.0


@dataclass
class UniverseConfig:
    core_symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "UNH", "JNJ"
    ])
    sector_etfs: List[str] = field(default_factory=lambda: [
        "XLK", "XLC", "XLY", "XLF", "XLV", "XLE", "XLI", "XLP", "XLU", "XLB", "XLRE"
    ])
    top_k_sectors: int = 3
    top_n_per_sector: int = 6
    max_candidates: int = 12
    min_volume_ratio: float = 0.8
    min_dollar_volume: float = 20_000_000
    max_atr_pct: Dict[str, float] = field(default_factory=lambda: {
        "bull_trend": 0.06,
        "bull_pullback": 0.06,
        "range_bound": 0.04,
        "bear_trend": 0.03,
        "bear_rally": 0.03,
    })
    max_sector_exposure: Dict[str, float] = field(default_factory=lambda: {
        "bull_trend": 0.40,
        "bull_pullback": 0.40,
        "range_bound": 0.28,
        "bear_trend": 0.18,
        "bear_rally": 0.18,
    })


@dataclass 
class SectorScore:
    etf: str
    ret_5d: float = 0.0
    ret_20d: float = 0.0
    ret_60d: float = 0.0
    rs_vs_spy: float = 0.0
    composite_score: float = 0.0


class UniverseBuilder:
    
    def __init__(
        self,
        db: Optional[PostgresMarketDB] = None,
        config: Optional[UniverseConfig] = None,
    ):
        self.db = db or get_db()
        self.config = config or UniverseConfig()
        self._sector_history: List[Tuple[date, List[str]]] = []
        self._last_rotation_date: Optional[date] = None
    
    def build_universe(
        self,
        as_of_date: date,
        regime: MarketRegime = MarketRegime.RANGE_BOUND,
        force_refresh: bool = False,
    ) -> List[StockCandidate]:
        core_candidates = self._build_core_pool(as_of_date)
        
        should_rotate = force_refresh or self._should_rotate(as_of_date)
        if should_rotate:
            rotation_candidates = self._build_rotation_pool(as_of_date, regime)
            self._last_rotation_date = as_of_date
        else:
            rotation_candidates = self._get_cached_rotation_pool(as_of_date)
        
        candidate_pool = self._build_candidate_pool(as_of_date, regime)
        
        all_candidates = self._merge_and_filter(
            core_candidates,
            rotation_candidates,
            candidate_pool,
            regime,
            as_of_date,
        )
        
        return all_candidates
    
    def _should_rotate(self, as_of_date: date) -> bool:
        if self._last_rotation_date is None:
            return True
        days_since = (as_of_date - self._last_rotation_date).days
        return days_since >= 5
    
    def _build_core_pool(self, as_of_date: date) -> List[StockCandidate]:
        candidates = []
        start_date = as_of_date - timedelta(days=30)
        
        for symbol in self.config.core_symbols:
            prices = self.db.get_daily_prices_single(symbol, start_date, as_of_date)
            indicators = self.db.get_indicators([symbol], start_date, as_of_date)
            
            if prices.empty:
                continue
            
            latest_price = prices.iloc[-1]
            latest_ind = indicators.iloc[-1] if not indicators.empty else {}
            
            sector = self._get_sector_for_symbol(symbol)
            
            candidates.append(StockCandidate(
                symbol=symbol,
                pool=PoolType.CORE,
                sector_etf=sector,
                score=0.8,
                momentum_10d=float(latest_ind.get("momentum_10d", 0) or 0),
                rsi_14=float(latest_ind.get("rsi_14", 50) or 50),
                volume_ratio=float(latest_ind.get("volume_ratio", 1) or 1),
                atr_pct=self._calc_atr_pct(latest_ind, latest_price),
                close=float(latest_price.get("close", 0) or 0),
            ))
        
        return candidates
    
    def _build_rotation_pool(
        self,
        as_of_date: date,
        regime: MarketRegime,
    ) -> List[StockCandidate]:
        sector_scores = self._score_sectors(as_of_date)
        
        k = self._get_top_k_for_regime(regime)
        top_sectors = [s.etf for s in sector_scores[:k] if s.composite_score > 0]
        
        if regime in [MarketRegime.BEAR_TREND, MarketRegime.BEAR_RALLY]:
            defensive = ["XLP", "XLV", "XLU"]
            for d in defensive:
                if d not in top_sectors:
                    top_sectors.append(d)
            top_sectors = top_sectors[:k + 1]
        
        self._sector_history.append((as_of_date, top_sectors))
        if len(self._sector_history) > 10:
            self._sector_history = self._sector_history[-10:]
        
        candidates = []
        start_date = as_of_date - timedelta(days=30)
        
        for sector_etf in top_sectors:
            sector_stocks = self._get_sector_holdings(sector_etf)
            
            stock_scores: List[Tuple[str, float, Dict]] = []
            for symbol in sector_stocks:
                if symbol in self.config.core_symbols:
                    continue
                
                indicators = self.db.get_indicators([symbol], start_date, as_of_date)
                if indicators.empty:
                    continue
                
                latest = indicators.iloc[-1]
                momentum = float(latest.get("momentum_10d", 0) or 0)
                volume_ratio = float(latest.get("volume_ratio", 1) or 1)
                
                score = momentum * 0.6 + (volume_ratio - 1) * 0.4
                stock_scores.append((symbol, score, latest.to_dict()))
            
            stock_scores.sort(key=lambda x: x[1], reverse=True)
            
            for symbol, score, ind_data in stock_scores[:self.config.top_n_per_sector]:
                prices = self.db.get_daily_prices_single(symbol, start_date, as_of_date)
                latest_price = prices.iloc[-1] if not prices.empty else {}
                
                candidates.append(StockCandidate(
                    symbol=symbol,
                    pool=PoolType.ROTATION,
                    sector_etf=sector_etf,
                    score=score,
                    momentum_10d=float(ind_data.get("momentum_10d", 0) or 0),
                    rsi_14=float(ind_data.get("rsi_14", 50) or 50),
                    volume_ratio=float(ind_data.get("volume_ratio", 1) or 1),
                    atr_pct=self._calc_atr_pct(ind_data, latest_price),
                    close=float(latest_price.get("close", 0) or 0),
                ))
        
        return candidates
    
    def _build_candidate_pool(
        self,
        as_of_date: date,
        regime: MarketRegime,
    ) -> List[StockCandidate]:
        candidates = []
        start_date = as_of_date - timedelta(days=30)
        
        all_symbols = self._get_all_tradeable_symbols()
        existing = set(self.config.core_symbols)
        
        stock_scores: List[Tuple[str, float, Dict, Dict]] = []
        
        for symbol in all_symbols:
            if symbol in existing:
                continue
            
            indicators = self.db.get_indicators([symbol], start_date, as_of_date)
            if indicators.empty:
                continue
            
            latest = indicators.iloc[-1]
            
            if not self._passes_regime_filter(latest, regime):
                continue
            
            momentum = float(latest.get("momentum_10d", 0) or 0)
            volume_ratio = float(latest.get("volume_ratio", 1) or 1)
            rsi = float(latest.get("rsi_14", 50) or 50)
            
            if regime in [MarketRegime.BULL_TREND, MarketRegime.BULL_PULLBACK]:
                score = momentum * 0.5 + (volume_ratio - 1) * 0.3 + (rsi - 50) / 100 * 0.2
            elif regime == MarketRegime.RANGE_BOUND:
                score = (50 - abs(rsi - 40)) / 50 * 0.4 + volume_ratio * 0.3 + momentum * 0.3
            else:
                score = (1 - abs(momentum)) * 0.4 + volume_ratio * 0.3 + (70 - rsi) / 100 * 0.3
            
            prices = self.db.get_daily_prices_single(symbol, start_date, as_of_date)
            latest_price = prices.iloc[-1] if not prices.empty else {}
            
            stock_scores.append((symbol, score, latest.to_dict(), latest_price.to_dict() if hasattr(latest_price, 'to_dict') else {}))
        
        stock_scores.sort(key=lambda x: x[1], reverse=True)
        
        for symbol, score, ind_data, price_data in stock_scores[:self.config.max_candidates]:
            sector = self._get_sector_for_symbol(symbol)
            
            candidates.append(StockCandidate(
                symbol=symbol,
                pool=PoolType.CANDIDATE,
                sector_etf=sector,
                score=score,
                momentum_10d=float(ind_data.get("momentum_10d", 0) or 0),
                rsi_14=float(ind_data.get("rsi_14", 50) or 50),
                volume_ratio=float(ind_data.get("volume_ratio", 1) or 1),
                atr_pct=self._calc_atr_pct(ind_data, price_data),
                close=float(price_data.get("close", 0) or 0),
            ))
        
        return candidates
    
    def _score_sectors(self, as_of_date: date) -> List[SectorScore]:
        scores = []
        start_date = as_of_date - timedelta(days=90)
        
        spy_prices = self.db.get_daily_prices_single("SPY", start_date, as_of_date)
        if spy_prices.empty:
            return scores
        
        spy_returns = self._calc_returns(spy_prices)
        
        for etf in self.config.sector_etfs:
            prices = self.db.get_daily_prices_single(etf, start_date, as_of_date)
            if prices.empty or len(prices) < 20:
                continue
            
            returns = self._calc_returns(prices)
            
            score = SectorScore(
                etf=etf,
                ret_5d=returns.get("ret_5d", 0),
                ret_20d=returns.get("ret_20d", 0),
                ret_60d=returns.get("ret_60d", 0),
                rs_vs_spy=returns.get("ret_20d", 0) - spy_returns.get("ret_20d", 0),
            )
            
            score.composite_score = (
                0.4 * self._zscore(score.ret_20d, 0, 0.05) +
                0.3 * self._zscore(score.rs_vs_spy, 0, 0.03) +
                0.3 * self._zscore(score.ret_60d, 0, 0.10)
            )
            
            scores.append(score)
        
        scores.sort(key=lambda x: x.composite_score, reverse=True)
        return scores
    
    def _calc_returns(self, prices: pd.DataFrame) -> Dict[str, float]:
        if prices.empty:
            return {}
        
        close = prices["close"]
        current = float(close.iloc[-1])
        
        ret_5d = (current / float(close.iloc[-5]) - 1) if len(close) >= 5 else 0
        ret_20d = (current / float(close.iloc[-20]) - 1) if len(close) >= 20 else 0
        ret_60d = (current / float(close.iloc[-60]) - 1) if len(close) >= 60 else 0
        
        return {"ret_5d": ret_5d, "ret_20d": ret_20d, "ret_60d": ret_60d}
    
    def _zscore(self, value: float, mean: float, std: float) -> float:
        if std == 0:
            return 0
        return (value - mean) / std
    
    def _get_top_k_for_regime(self, regime: MarketRegime) -> int:
        mapping = {
            MarketRegime.BULL_TREND: 4,
            MarketRegime.BULL_PULLBACK: 3,
            MarketRegime.RANGE_BOUND: 2,
            MarketRegime.BEAR_RALLY: 1,
            MarketRegime.BEAR_TREND: 1,
        }
        return mapping.get(regime, 3)
    
    def _passes_regime_filter(self, indicators: pd.Series, regime: MarketRegime) -> bool:
        momentum = float(indicators.get("momentum_10d", 0) or 0)
        rsi = float(indicators.get("rsi_14", 50) or 50)
        volume_ratio = float(indicators.get("volume_ratio", 1) or 1)
        
        if regime in [MarketRegime.BULL_TREND, MarketRegime.BULL_PULLBACK]:
            return momentum > 0 and volume_ratio >= 0.8
        elif regime == MarketRegime.RANGE_BOUND:
            return 30 <= rsi <= 70 and volume_ratio >= 0.6
        else:
            return rsi < 60 and volume_ratio >= 0.5
    
    def _merge_and_filter(
        self,
        core: List[StockCandidate],
        rotation: List[StockCandidate],
        candidates: List[StockCandidate],
        regime: MarketRegime,
        as_of_date: date,
    ) -> List[StockCandidate]:
        seen: Set[str] = set()
        result: List[StockCandidate] = []
        
        max_atr = self.config.max_atr_pct.get(regime.value, 0.05)
        
        for c in core:
            if c.symbol not in seen and c.atr_pct <= max_atr * 1.5:
                seen.add(c.symbol)
                result.append(c)
        
        for c in rotation:
            if c.symbol not in seen and c.atr_pct <= max_atr:
                if c.volume_ratio >= self.config.min_volume_ratio:
                    seen.add(c.symbol)
                    result.append(c)
        
        for c in candidates:
            if c.symbol not in seen and c.atr_pct <= max_atr:
                if c.volume_ratio >= self.config.min_volume_ratio:
                    seen.add(c.symbol)
                    result.append(c)
        
        return result
    
    def _get_cached_rotation_pool(self, as_of_date: date) -> List[StockCandidate]:
        if not self._sector_history:
            return []
        
        _, last_sectors = self._sector_history[-1]
        return self._build_rotation_pool(as_of_date, MarketRegime.RANGE_BOUND)
    
    def _get_sector_holdings(self, sector_etf: str) -> List[str]:
        query = "SELECT symbol FROM sector_holdings WHERE sector_etf = %s"
        with self.db.get_cursor() as cur:
            cur.execute(query, (sector_etf,))
            return [row[0] for row in cur.fetchall()]
    
    def _get_sector_for_symbol(self, symbol: str) -> Optional[str]:
        query = "SELECT sector_etf FROM sector_holdings WHERE symbol = %s LIMIT 1"
        with self.db.get_cursor() as cur:
            cur.execute(query, (symbol,))
            result = cur.fetchone()
            return result[0] if result else None
    
    def _get_all_tradeable_symbols(self) -> List[str]:
        query = "SELECT DISTINCT symbol FROM daily_prices WHERE symbol NOT LIKE '^%'"
        with self.db.get_cursor() as cur:
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]
    
    def _calc_atr_pct(self, indicators: Any, price: Any) -> float:
        atr = float(indicators.get("atr_14", 0) or 0) if hasattr(indicators, 'get') else 0
        close = float(price.get("close", 1) or 1) if hasattr(price, 'get') else 1
        return atr / close if close > 0 else 0.02
    
    def get_watchlist(
        self,
        as_of_date: date,
        regime: MarketRegime = MarketRegime.RANGE_BOUND,
    ) -> List[str]:
        candidates = self.build_universe(as_of_date, regime)
        return [c.symbol for c in candidates]
    
    def get_universe_summary(
        self,
        as_of_date: date,
        regime: MarketRegime = MarketRegime.RANGE_BOUND,
    ) -> Dict[str, Any]:
        candidates = self.build_universe(as_of_date, regime)
        
        by_pool = {p.value: [] for p in PoolType}
        by_sector: Dict[str, List[str]] = {}
        
        for c in candidates:
            by_pool[c.pool.value].append(c.symbol)
            if c.sector_etf:
                if c.sector_etf not in by_sector:
                    by_sector[c.sector_etf] = []
                by_sector[c.sector_etf].append(c.symbol)
        
        return {
            "date": str(as_of_date),
            "regime": regime.value,
            "total_count": len(candidates),
            "by_pool": {k: len(v) for k, v in by_pool.items()},
            "by_pool_symbols": by_pool,
            "by_sector": {k: len(v) for k, v in by_sector.items()},
            "watchlist": [c.symbol for c in candidates],
        }
