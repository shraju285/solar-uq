"""Smoke test for Stage 1: proves the package installs and the config loads.

This is the only test at Stage 1 — there's no pipeline logic yet to test.
Run with:  pytest
"""

from solaruq.utils.config import load_config


def test_config_loads_and_has_expected_top_level_keys():
    config = load_config()
    expected_sections = {
        "project",
        "data",
        "forecast",
        "split",
        "models",
        "uncertainty",
        "evaluation",
        "paths",
    }
    assert expected_sections.issubset(config.keys())


def test_forecast_settings_match_locked_decisions():
    # These three numbers are the "must freeze early" decisions from
    # docs/decisions.md. This test exists so that if someone edits the
    # config later, they notice they're changing a recorded decision.
    config = load_config()
    assert config["forecast"]["resolution_minutes"] == 15
    assert config["forecast"]["lookback_steps"] == 24
    assert config["forecast"]["horizon_steps"] == 8
