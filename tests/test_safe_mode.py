from __future__ import annotations

from ai_trader_assist.agent.safe_mode import SafeModeConfig, build_safe_outputs


def test_safe_mode_no_new_risk() -> None:
    payload = {
        "universe": {"watchlist": ["AAPL", "MSFT"]},
    }
    safe = build_safe_outputs(payload, "error", SafeModeConfig(max_exposure_cap=0.5))

    stock_view = safe["stock_classifier"]
    assert not stock_view["categories"]["Buy"]
    assert stock_view["categories"]["Hold"]

    exposure = safe["exposure_planner"]
    assert exposure["target_exposure"] <= 0.5
    for plan in exposure["allocation_plan"]:
        assert plan["weight"] == 0.0

    report = safe["report_composer"]
    assert "Safe Mode" in report["markdown"]
    assert safe["safe_mode"]["policy"] == "no_new_risk"
