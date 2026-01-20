"""Tests for MarketRegimeDetector - 市场状态识别器测试"""
import pytest
from ai_trader_assist.risk_engine.market_regime import (
    MarketRegime,
    MarketRegimeDetector,
    RegimeSignals,
)


class TestMarketRegime:
    def test_regime_enum_values(self):
        assert MarketRegime.BULL_TREND.value == "bull_trend"
        assert MarketRegime.BEAR_TREND.value == "bear_trend"
        assert MarketRegime.RANGE_BOUND.value == "range_bound"

    def test_regime_from_string(self):
        assert MarketRegime("bull_trend") == MarketRegime.BULL_TREND
        assert MarketRegime("bear_rally") == MarketRegime.BEAR_RALLY


class TestMarketRegimeDetector:
    @pytest.fixture
    def detector(self):
        return MarketRegimeDetector()

    def test_detect_bull_trend(self, detector):
        # given: strong bullish market signals
        signals = RegimeSignals(
            spy_vs_sma200=5.0,
            spy_vs_sma50=2.0,
            sma50_slope=0.003,
            breadth=0.75,
            nh_nl_ratio=3.5,
            vix_value=14.0,
            vix_term_contango=True,
            spy_momentum_20d=0.08,
            qqq_momentum_20d=0.06,
        )

        # when: detect regime
        result = detector.detect(signals)

        # then: should identify bull trend
        assert result.regime == MarketRegime.BULL_TREND
        assert result.confidence > 0.5
        assert result.bull_score > 4.0

    def test_detect_bear_trend(self, detector):
        # given: bearish market signals
        signals = RegimeSignals(
            spy_vs_sma200=-8.0,
            spy_vs_sma50=-5.0,
            sma50_slope=-0.004,
            breadth=0.25,
            nh_nl_ratio=0.3,
            vix_value=32.0,
            vix_term_contango=False,
            spy_momentum_20d=-0.12,
            qqq_momentum_20d=-0.10,
        )

        # when: detect regime
        result = detector.detect(signals)

        # then: should identify bear trend
        assert result.regime == MarketRegime.BEAR_TREND
        assert result.confidence > 0.5

    def test_detect_range_bound(self, detector):
        # given: mixed/neutral signals
        signals = RegimeSignals(
            spy_vs_sma200=2.0,
            spy_vs_sma50=-1.0,
            sma50_slope=0.0002,
            breadth=0.50,
            nh_nl_ratio=1.0,
            vix_value=18.0,
            vix_term_contango=True,
            spy_momentum_20d=0.01,
            qqq_momentum_20d=0.01,
        )

        # when: detect regime
        result = detector.detect(signals)

        # then: should identify range bound or transitional state
        assert result.regime in [
            MarketRegime.RANGE_BOUND,
            MarketRegime.BULL_PULLBACK,
            MarketRegime.UNKNOWN,
        ]

    def test_detect_bull_pullback(self, detector):
        # given: bull trend with short-term weakness
        signals = RegimeSignals(
            spy_vs_sma200=3.0,
            spy_vs_sma50=-2.0,
            sma50_slope=0.001,
            breadth=0.45,
            nh_nl_ratio=1.5,
            vix_value=22.0,
            vix_term_contango=True,
            spy_momentum_20d=-0.02,
            qqq_momentum_20d=-0.01,
        )

        # when: detect regime
        result = detector.detect(signals)

        # then: should identify pullback in bull market
        assert result.regime in [MarketRegime.BULL_PULLBACK, MarketRegime.RANGE_BOUND, MarketRegime.BULL_TREND]

    def test_detect_with_default_signals(self, detector):
        # given: default signals (neutral)
        signals = RegimeSignals()

        # when: detect with defaults
        result = detector.detect(signals)

        # then: should return a valid result
        assert result.regime in MarketRegime
        assert 0 <= result.confidence <= 1

    def test_result_contains_signals_used(self, detector):
        # given: complete signals
        signals = RegimeSignals(
            spy_vs_sma200=3.0,
            spy_vs_sma50=1.0,
            sma50_slope=0.002,
            breadth=0.65,
            nh_nl_ratio=2.0,
            vix_value=16.0,
            vix_term_contango=True,
            spy_momentum_20d=0.05,
            qqq_momentum_20d=0.04,
        )

        # when: detect regime
        result = detector.detect(signals)

        # then: result should contain signal breakdown
        assert hasattr(result, "signals_used")
        assert isinstance(result.signals_used, dict)

    def test_extreme_vix_forces_bear(self, detector):
        # given: bullish structure but extreme VIX
        signals = RegimeSignals(
            spy_vs_sma200=5.0,
            spy_vs_sma50=2.0,
            sma50_slope=0.002,
            breadth=0.65,
            vix_value=40.0,
            vix_term_contango=False,
        )

        # when: detect regime
        result = detector.detect(signals)

        # then: extreme VIX should force bear trend
        assert result.regime == MarketRegime.BEAR_TREND
        assert result.recommended_exposure <= 0.40

    def test_from_config(self):
        # given: custom config with adjusted thresholds
        config = {
            "market_regimes": {
                "detection": {
                    "thresholds": {
                        "bull_score": 3.0,
                        "bear_score": 1.5,
                    }
                }
            }
        }

        # when: create detector from config
        detector = MarketRegimeDetector.from_config(config)

        # then: should use custom thresholds
        assert detector.config.thresholds.get("bull_score") == 3.0
        assert detector.config.thresholds.get("bear_score") == 1.5

    def test_recommended_exposure_varies_by_regime(self, detector):
        # given: bull signals
        bull_signals = RegimeSignals(
            spy_vs_sma200=5.0, spy_vs_sma50=2.0, sma50_slope=0.003,
            breadth=0.75, nh_nl_ratio=3.0, vix_value=14.0,
            vix_term_contango=True, spy_momentum_20d=0.08, qqq_momentum_20d=0.07,
        )
        # given: bear signals
        bear_signals = RegimeSignals(
            spy_vs_sma200=-8.0, spy_vs_sma50=-5.0, sma50_slope=-0.004,
            breadth=0.25, nh_nl_ratio=0.3, vix_value=30.0,
            vix_term_contango=False, spy_momentum_20d=-0.10, qqq_momentum_20d=-0.08,
        )

        # when: detect both
        bull_result = detector.detect(bull_signals)
        bear_result = detector.detect(bear_signals)

        # then: bull should have higher recommended exposure
        assert bull_result.recommended_exposure > bear_result.recommended_exposure
