"""Tests for backtest module and regime transition smoother"""
import pytest
from ai_trader_assist.risk_engine.macro_engine import (
    MacroRiskEngine,
    RegimeTransitionSmoother,
)
from ai_trader_assist.risk_engine.market_regime import MarketRegime


class TestRegimeTransitionSmoother:
    def test_initial_state_is_unknown(self):
        smoother = RegimeTransitionSmoother(confirmation_periods=3)
        assert smoother.smoothed_regime == MarketRegime.UNKNOWN

    def test_requires_confirmation_periods(self):
        smoother = RegimeTransitionSmoother(confirmation_periods=3)
        
        # first two updates should not confirm
        result1 = smoother.update(MarketRegime.BULL_TREND)
        assert result1 == MarketRegime.UNKNOWN
        
        result2 = smoother.update(MarketRegime.BULL_TREND)
        assert result2 == MarketRegime.UNKNOWN
        
        # third consecutive update confirms
        result3 = smoother.update(MarketRegime.BULL_TREND)
        assert result3 == MarketRegime.BULL_TREND

    def test_interruption_resets_confirmation(self):
        smoother = RegimeTransitionSmoother(confirmation_periods=3)
        
        smoother.update(MarketRegime.BULL_TREND)
        smoother.update(MarketRegime.BULL_TREND)
        # interrupt with different regime
        smoother.update(MarketRegime.BEAR_TREND)
        
        # should still be unknown
        assert smoother.smoothed_regime == MarketRegime.UNKNOWN

    def test_transition_requires_new_confirmation(self):
        smoother = RegimeTransitionSmoother(confirmation_periods=3)
        
        # confirm bull trend
        for _ in range(3):
            smoother.update(MarketRegime.BULL_TREND)
        assert smoother.smoothed_regime == MarketRegime.BULL_TREND
        
        # single bear signal shouldn't change
        smoother.update(MarketRegime.BEAR_TREND)
        assert smoother.smoothed_regime == MarketRegime.BULL_TREND
        
        # need 3 consecutive bear signals to switch
        smoother.update(MarketRegime.BEAR_TREND)
        smoother.update(MarketRegime.BEAR_TREND)
        assert smoother.smoothed_regime == MarketRegime.BEAR_TREND

    def test_reset_clears_history(self):
        smoother = RegimeTransitionSmoother(confirmation_periods=3)
        
        for _ in range(3):
            smoother.update(MarketRegime.BULL_TREND)
        assert smoother.smoothed_regime == MarketRegime.BULL_TREND
        
        smoother.reset()
        assert smoother.smoothed_regime == MarketRegime.UNKNOWN

    def test_raw_regime_property(self):
        smoother = RegimeTransitionSmoother(confirmation_periods=3)
        
        smoother.update(MarketRegime.BULL_TREND)
        smoother.update(MarketRegime.BEAR_TREND)
        
        assert smoother.raw_regime == MarketRegime.BEAR_TREND
        assert smoother.smoothed_regime == MarketRegime.UNKNOWN


class TestMacroRiskEngineSmoothing:
    @pytest.fixture
    def engine_with_smoothing(self):
        return MacroRiskEngine(use_smoothing=True)

    @pytest.fixture
    def engine_without_smoothing(self):
        return MacroRiskEngine(use_smoothing=False)

    def test_smoothing_enabled_by_default(self):
        engine = MacroRiskEngine()
        assert engine.use_smoothing is True

    def test_evaluate_returns_smoothing_info(self, engine_with_smoothing):
        features = {"RS_SPY": 0.5, "VIX_Z": -0.3, "BREADTH": 0.6}
        result = engine_with_smoothing.evaluate(features)
        
        assert "smoothing_enabled" in result
        assert "raw_regime" in result
        assert "smoothed_regime" in result
        assert result["smoothing_enabled"] is True

    def test_smoothing_disabled_returns_same_regime(self, engine_without_smoothing):
        features = {"RS_SPY": 0.5, "VIX_Z": -0.3, "BREADTH": 0.6}
        result = engine_without_smoothing.evaluate(features)
        
        assert result["smoothing_enabled"] is False
        assert result["raw_regime"] == result["smoothed_regime"]

    def test_smoothing_delays_regime_change(self, engine_with_smoothing):
        bull_features = {
            "RS_SPY": 0.8, "VIX_Z": -1.0, "BREADTH": 0.75,
            "spy_vs_sma200": 8.0, "spy_vs_sma50": 3.0,
            "spy_momentum_20d": 0.08,
        }
        
        # first evaluation
        result1 = engine_with_smoothing.evaluate(bull_features)
        initial_smoothed = result1["smoothed_regime"]
        
        # smoothed regime should lag behind raw detection initially
        assert initial_smoothed == "unknown"

    def test_from_config_loads_smoothing_params(self):
        config = {
            "market_regimes": {
                "use_smoothing": True,
                "smoothing_periods": 5,
            }
        }
        engine = MacroRiskEngine.from_config(config)
        
        assert engine.use_smoothing is True
        assert engine.smoother.confirmation_periods == 5


class TestBacktestIntegration:
    def test_backtest_module_imports(self):
        from ai_trader_assist.backtest.amd_backtest import (
            AMDBacktester,
            OldSystemBacktester,
            BacktestResult,
            TradeRecord,
        )
        assert AMDBacktester is not None
        assert OldSystemBacktester is not None

    def test_backtester_initialization(self):
        from ai_trader_assist.backtest.amd_backtest import AMDBacktester
        
        backtester = AMDBacktester(
            initial_capital=100000.0,
            use_smoothing=True,
            smoothing_periods=3,
        )
        
        assert backtester.initial_capital == 100000.0
        assert backtester.use_smoothing is True
