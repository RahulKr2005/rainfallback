"""Unit tests for geospatial pipeline and terrain indices."""

import numpy as np
import pytest
from app.data.adapters.srtm_dem import SRTMProcessor
from app.data.adapters.worldcover import WorldCoverProcessor
from app.data.adapters.sentinel1_flood import Sentinel1SARValidator
from app.data.pipeline import GeospatialDataPipeline


def test_srtm_dem_and_twi_math():
    """Verify elevation ranges, slope computation, and TWI bounds."""
    processor = SRTMProcessor(grid_size=30)
    dem = processor.get_delhi_dem_grid()
    
    assert dem.shape == (30, 30)
    assert np.min(dem) >= 200.0, "Minimum elevation should match Yamuna riverbed floor (~201m)"
    assert np.max(dem) <= 270.0, "Maximum elevation should not exceed Ridge peak (~265m)"
    assert not np.isnan(dem).any(), "Elevation grid contains NaNs"
    
    derivatives = processor.compute_terrain_derivatives(dem)
    slope = derivatives["slope_deg"]
    twi = derivatives["twi"]
    
    assert np.min(slope) >= 0.0, "Slope must be non-negative"
    assert not np.isnan(slope).any()
    assert not np.isinf(slope).any()
    
    assert np.min(twi) >= 1.0, "TWI lower bound must be positive"
    assert np.max(twi) <= 16.0, "TWI upper bound within standard hydrologic limits"
    assert not np.isnan(twi).any()
    assert not np.isinf(twi).any()


def test_worldcover_runoff():
    """Verify ESA WorldCover class mapping to runoff coefficients."""
    processor = WorldCoverProcessor(grid_size=20)
    lc = processor.get_landcover_grid()
    runoff = processor.get_runoff_coefficients(lc)
    
    assert runoff.shape == (20, 20)
    assert np.all(runoff >= 0.15) and np.all(runoff <= 1.0)
    # Built-up areas have higher runoff than forest
    assert np.mean(runoff) > 0.4


def test_sar_validation_metrics():
    """Verify SAR contingency calculations and IoU metrics."""
    validator = Sentinel1SARValidator(grid_size=25)
    gt = validator.get_july2023_flood_ground_truth()
    
    # Perfect prediction test
    metrics_perfect = validator.evaluate_extent(gt, gt)
    assert metrics_perfect["iou_jaccard"] == 1.0
    assert metrics_perfect["precision"] == 1.0
    assert metrics_perfect["recall"] == 1.0
    assert metrics_perfect["f1_score"] == 1.0

    # All dry prediction test
    dry = np.zeros_like(gt)
    metrics_dry = validator.evaluate_extent(dry, gt)
    assert metrics_dry["iou_jaccard"] == 0.0
    assert metrics_dry["recall"] == 0.0
