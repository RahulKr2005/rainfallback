"""Unit tests for configuration and IMD rainfall threshold scaling."""

import pytest
from app.config import get_rainfall_category, IMD_RAINFALL_THRESHOLDS


def test_24h_rainfall_thresholds():
    """Verify 24h standard IMD thresholds."""
    # Light rain: 5 mm
    cat_light = get_rainfall_category(5.0, horizon_hours=24)
    assert cat_light["code"] == "LIGHT"
    assert cat_light["level"] == "GREEN"

    # Moderate rain: 35 mm
    cat_mod = get_rainfall_category(35.0, horizon_hours=24)
    assert cat_mod["code"] == "MODERATE"
    assert cat_mod["level"] == "YELLOW"

    # Heavy rain: 75 mm
    cat_heavy = get_rainfall_category(75.0, horizon_hours=24)
    assert cat_heavy["code"] == "HEAVY"
    assert cat_heavy["level"] == "ORANGE"

    # Very Heavy rain: 150 mm
    cat_vheavy = get_rainfall_category(150.0, horizon_hours=24)
    assert cat_vheavy["code"] == "VERY_HEAVY"
    assert cat_vheavy["level"] == "RED"


def test_scaled_6h_thresholds():
    """Verify that 6h horizon uses scaled thresholds, not 24h criteria.
    
    A 20 mm rainfall in 6 hours is HEAVY rain for a 6h window (rate ~3.3 mm/h),
    whereas 20 mm in 24 hours would only be MODERATE.
    """
    cat_6h = get_rainfall_category(20.0, horizon_hours=6)
    assert cat_6h["code"] == "HEAVY"
    assert cat_6h["level"] == "ORANGE"

    # In 24h, 20 mm is only Moderate
    cat_24h = get_rainfall_category(20.0, horizon_hours=24)
    assert cat_24h["code"] == "MODERATE"
    assert cat_24h["level"] == "YELLOW"


def test_zero_rainfall():
    """Verify zero or sub-threshold rainfall produces NO_WARNING."""
    cat_zero = get_rainfall_category(0.0, horizon_hours=24)
    assert cat_zero["code"] == "NO_WARNING"
    assert cat_zero["level"] == "GREEN"
