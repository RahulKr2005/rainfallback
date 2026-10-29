"""Integrated Geospatial Data and Feature Engineering Pipeline.

Orchestrates data adapters, handles geographic clipping, coordinate conversion,
leak-free temporal alignment (issue vs valid time), and creates feature grids
for both rainfall forecasting and inundation susceptibility models.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Tuple, Optional
import numpy as np
import os

from ..config import DEFAULT_STUDY_AREA
from .adapters.srtm_dem import SRTMProcessor
from .adapters.worldcover import WorldCoverProcessor
from .adapters.gfs_nwp import GFSAdapter
from .adapters.gpm_imerg import GPMIMERGAdapter
from .adapters.imd_station import IMDStationAdapter
from .adapters.sentinel1_flood import Sentinel1SARValidator


class GeospatialDataPipeline:
    """End-to-end data pipeline for heavy rainfall and inundation analysis."""

    def __init__(self, grid_size: int = 100):
        self.grid_size = grid_size
        self.study_area = DEFAULT_STUDY_AREA
        self.srtm = SRTMProcessor(grid_size=grid_size)
        self.worldcover = WorldCoverProcessor(grid_size=grid_size)
        self.gfs = GFSAdapter()
        self.gpm = GPMIMERGAdapter(product_run="Early")
        self.station = IMDStationAdapter()
        self.sar = Sentinel1SARValidator(grid_size=grid_size)
        
        # Precompute static spatial layers to save memory and CPU
        self._dem = self.srtm.get_delhi_dem_grid()
        self._terrain_derivatives = self.srtm.compute_terrain_derivatives(self._dem)
        self._landcover = self.worldcover.get_landcover_grid()
        self._runoff_coeff = self.worldcover.get_runoff_coefficients(self._landcover)
        self._distance_to_river = self._compute_river_distance_grid()
        self._distance_to_drains = self._compute_drain_distance_grid()

    def _compute_river_distance_grid(self) -> np.ndarray:
        """Normalized Euclidean distance to the Yamuna River centerline [0.0 = on river, 1.0 = distant]."""
        norm_lat = (self.srtm.lat_grid - self.srtm.min_lat) / (self.srtm.max_lat - self.srtm.min_lat)
        river_lon = 77.225 + 0.035 * np.sin(2.5 * np.pi * norm_lat)
        dist_deg = np.abs(self.srtm.lon_grid - river_lon)
        # Convert to approximate kilometers (1 deg lon ~ 98 km at lat 28.6)
        dist_km = dist_deg * 98.0
        # Normalize: maximum distance across Delhi ~ 35 km
        return np.clip(dist_km / 35.0, 0.0, 1.0)

    def _compute_drain_distance_grid(self) -> np.ndarray:
        """Distance to primary urban storm drainage channels (Najafgarh Drain & Barapullah)."""
        # Najafgarh drain runs from SW (Kakrola) through W-NW to Wazirabad
        # Barapullah runs from South Delhi to Yamuna
        d_najafgarh = np.sqrt((self.srtm.lon_grid - 77.10)**2 + (self.srtm.lat_grid - 28.65)**2) * 111.0
        d_barapullah = np.sqrt((self.srtm.lon_grid - 77.23)**2 + (self.srtm.lat_grid - 28.58)**2) * 111.0
        min_dist_km = np.minimum(d_najafgarh, d_barapullah)
        return np.clip(min_dist_km / 25.0, 0.0, 1.0)

    def extract_rainfall_features(self, horizon_hours: int = 24, lat: Optional[float] = None, lon: Optional[float] = None) -> Dict[str, Any]:
        """Extract multi-source meteorological features strictly indexed at issue time.
        
        No lookahead leakage: predictors include:
        1. GFS NWP predicted accumulation for horizon.
        2. GFS atmospheric CAPE, humidity, temperature, wind speed.
        3. GPM IMERG satellite recent antecedent accumulation (last 6h).
        4. IMD station antecedent 24h rainfall.
        5. Upstream Yamuna discharge at Hathnikund.
        """
        gfs_data = self.gfs.fetch_forecast(horizon_hours=horizon_hours, lat=lat, lon=lon)
        satellite_data = self.gpm.get_satellite_rainfall(window_hours=6)
        stations_data = self.station.get_current_observations()
        gauge_data = self.station.get_upstream_gauge_status()
        
        # Mean station observed antecedent rain
        station_rain_mean = float(np.mean([s["rainfall_accum_24h_mm"] for s in stations_data]))
        
        feature_vector = {
            "gfs_accum_mm": gfs_data["gfs_accumulated_precip_mm"],
            "gfs_cape_j_kg": gfs_data["atmospheric_instability"]["cape_j_kg"],
            "gfs_humidity_pct": gfs_data["atmospheric_instability"]["mean_humidity_pct"],
            "gfs_temp_c": gfs_data["atmospheric_instability"]["mean_temp_c"],
            "gfs_wind_kmh": gfs_data["atmospheric_instability"]["max_wind_kmh"],
            "gpm_recent_6h_accum_mm": satellite_data["satellite_accum_mm"],
            "station_antecedent_24h_mm": round(station_rain_mean, 2),
            "upstream_hathnikund_discharge_cusecs": gauge_data["upstream_barrage"]["current_discharge_cusecs"],
            "horizon_hours": horizon_hours
        }
        
        return {
            "issue_time_utc": gfs_data["issue_time_utc"],
            "target_time_utc": gfs_data["target_time_utc"],
            "horizon_hours": horizon_hours,
            "feature_vector": feature_vector,
            "raw_sources": {
                "gfs": gfs_data,
                "gpm_imerg": satellite_data,
                "stations": stations_data,
                "upstream_gauge": gauge_data
            }
        }

    def get_static_spatial_layers(self) -> Dict[str, Any]:
        """Return cached spatial grid matrices for Delhi-NCR."""
        return {
            "elevation": self._dem,
            "slope_deg": self._terrain_derivatives["slope_deg"],
            "twi": self._terrain_derivatives["twi"],
            "landcover": self._landcover,
            "runoff_coeff": self._runoff_coeff,
            "distance_river_norm": self._distance_to_river,
            "distance_drain_norm": self._distance_to_drains
        }
