"""Unit tests for ML forecasting and inundation models."""

import pytest
import numpy as np
from app.ml.rainfall_model import RainfallForecaster
from app.ml.inundation_model import InundationPredictor


def test_rainfall_forecaster():
    """Verify RainfallForecaster training, prediction format, and non-negative output."""
    forecaster = RainfallForecaster(horizon_hours=24)
    sample_features = {
        "gfs_accum_mm": 25.0,
        "gfs_cape_j_kg": 1800.0,
        "gfs_humidity_pct": 85.0,
        "gfs_temp_c": 29.5,
        "gfs_wind_kmh": 22.0,
        "gpm_recent_6h_accum_mm": 12.0,
        "station_antecedent_24h_mm": 18.0,
        "upstream_hathnikund_discharge_cusecs": 75000.0
    }
    
    pred = forecaster.predict(sample_features)
    assert "predicted_accum_mm" in pred
    assert pred["predicted_accum_mm"] >= 0.0
    assert "persistence_baseline_mm" in pred
    assert "warning" in pred
    assert pred["warning"]["level"] in ["GREEN", "YELLOW", "ORANGE", "RED"]

    metrics = forecaster.get_evaluation_metrics()
    assert "continuous_metrics" in metrics
    assert "categorical_metrics" in metrics
    assert metrics["continuous_metrics"]["ml_model"]["mae_mm"] >= 0.0


def test_inundation_predictor():
    """Verify InundationPredictor susceptibility and classification bounds."""
    predictor = InundationPredictor(grid_size=30)
    dem = np.full((30, 30), 205.0)
    slope = np.zeros((30, 30))
    twi = np.full((30, 30), 8.0)
    runoff = np.full((30, 30), 0.7)
    dist_river = np.zeros((30, 30)) # On river
    dist_drain = np.zeros((30, 30))
    
    res = predictor.predict_inundation(
        dem=dem, slope_deg=slope, twi=twi, runoff_coeff=runoff,
        dist_river_norm=dist_river, dist_drain_norm=dist_drain,
        rainfall_accum_mm=100.0, upstream_discharge_cusecs=150000.0
    )
    
    assert 0.0 <= res["tfsi_mean"] <= 1.0
    assert 0.0 <= res["tfsi_max"] <= 1.0
    assert "disclaimer" in res
    assert "inundation_mask" in res
    assert res["inundation_mask"].shape == (30, 30)
