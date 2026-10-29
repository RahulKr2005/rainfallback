"""Application Configuration and Hydrological Constants.

Defines spatial bounding boxes, IMD-calibrated precipitation thresholds
strictly scaled by forecast horizon, and directory paths.
"""

from pathlib import Path
import os
from typing import Dict, Any

# Root directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, LABELS_DIR, CACHE_DIR, ARTIFACTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "system.db"

# Study Area Configuration: Provisional Delhi-NCR
# Geographic bounds: Yamuna river basin Delhi segment, Najafgarh drain, urban center
DEFAULT_STUDY_AREA = {
    "name": "Delhi-NCR",
    "min_lat": float(os.getenv("BBOX_MIN_LAT", "28.30")),
    "max_lat": float(os.getenv("BBOX_MAX_LAT", "28.88")),
    "min_lon": float(os.getenv("BBOX_MIN_LON", "76.84")),
    "max_lon": float(os.getenv("BBOX_MAX_LON", "77.45")),
    "center_lat": 28.6139,
    "center_lon": 77.2090,
    "river": "Yamuna",
    "upstream_gauge": {
        "name": "Hathnikund Barrage (Haryana)",
        "lat": 30.3175,
        "lon": 77.5833,
        "danger_discharge_cusecs": 250000.0,
        "travel_time_hours": 48.0
    },
    "urban_drainage_basins": [
        {"name": "Najafgarh Basin", "area_sqkm": 950.0, "high_risk_impervious": True},
        {"name": "Barapullah Basin", "area_sqkm": 375.0, "high_risk_impervious": True},
        {"name": "Shahdara Basin", "area_sqkm": 210.0, "high_risk_impervious": True},
        {"name": "Yamuna Floodplain", "area_sqkm": 97.0, "high_risk_impervious": False}
    ]
}

# IMD Heavy Rainfall Thresholds (mm) calibrated by accumulation horizon
# Source: India Meteorological Department (IMD) Standard Operational Procedure for Weather Forecasting
IMD_RAINFALL_THRESHOLDS = {
    # 24-Hour Standard IMD Thresholds
    24: {
        "very_light": (0.1, 2.4),
        "light": (2.5, 15.5),
        "moderate": (15.6, 64.4),
        "heavy": (64.5, 115.5),
        "very_heavy": (115.6, 204.4),
        "extremely_heavy": (204.5, float("inf")),
    },
    # 12-Hour Scaled Thresholds (50% empirical scaling)
    12: {
        "very_light": (0.05, 1.2),
        "light": (1.3, 7.7),
        "moderate": (7.8, 32.2),
        "heavy": (32.3, 57.8),
        "very_heavy": (57.9, 102.2),
        "extremely_heavy": (102.3, float("inf")),
    },
    # 6-Hour Scaled Thresholds (25% empirical scaling)
    6: {
        "very_light": (0.02, 0.6),
        "light": (0.6, 3.8),
        "moderate": (3.9, 16.1),
        "heavy": (16.2, 28.9),
        "very_heavy": (29.0, 51.1),
        "extremely_heavy": (51.2, float("inf")),
    }
}


def get_rainfall_category(accumulation_mm: float, horizon_hours: int = 24) -> Dict[str, Any]:
    """Determine IMD rainfall category and warning severity based on accumulation and horizon.
    
    Ensures that 6h and 12h forecasts are evaluated against time-scaled thresholds,
    never against full 24h daily criteria.
    """
    thresholds = IMD_RAINFALL_THRESHOLDS.get(horizon_hours, IMD_RAINFALL_THRESHOLDS[24])
    
    if accumulation_mm < thresholds["light"][0]:
        return {
            "code": "NO_WARNING",
            "category": "No Rain / Very Light",
            "level": "GREEN",
            "description": "No significant rainfall expected.",
            "action": "No adverse impact anticipated."
        }
    elif accumulation_mm <= thresholds["light"][1]:
        return {
            "code": "LIGHT",
            "category": "Light Rain",
            "level": "GREEN",
            "description": "Scattered or light spells expected.",
            "action": "Normal activities can proceed."
        }
    elif accumulation_mm <= thresholds["moderate"][1]:
        return {
            "code": "MODERATE",
            "category": "Moderate Rain",
            "level": "YELLOW",
            "description": f"Accumulation of {accumulation_mm:.1f} mm in {horizon_hours}h may cause minor local waterlogging.",
            "action": "Be updated on local weather developments."
        }
    elif accumulation_mm <= thresholds["heavy"][1]:
        return {
            "code": "HEAVY",
            "category": "Heavy Rain",
            "level": "ORANGE",
            "description": f"Heavy rainfall accumulation of {accumulation_mm:.1f} mm in {horizon_hours}h. Waterlogging in low-lying areas and traffic disruption likely.",
            "action": "Be prepared. Keep storm-water drainage active."
        }
    elif accumulation_mm <= thresholds["very_heavy"][1]:
        return {
            "code": "VERY_HEAVY",
            "category": "Very Heavy Rain",
            "level": "RED",
            "description": f"Very heavy rain of {accumulation_mm:.1f} mm in {horizon_hours}h. Major inundation risk along riverbanks, drains, and underpasses.",
            "action": "Take action. Evacuate vulnerable floodplains and monitor drain surges."
        }
    else:
        return {
            "code": "EXTREMELY_HEAVY",
            "category": "Extremely Heavy Rain",
            "level": "RED",
            "description": f"Extremely intense downpour ({accumulation_mm:.1f} mm in {horizon_hours}h). Catastrophic urban and riverine flood threat.",
            "action": "Emergency action required."
        }
