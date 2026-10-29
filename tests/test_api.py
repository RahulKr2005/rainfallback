"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert "study_area" in data


def test_catalog_endpoint():
    response = client.get("/api/catalog")
    assert response.status_code == 200
    catalog = response.json()
    assert len(catalog) >= 10
    ids = [item["id"] for item in catalog]
    assert "NASA_GPM_IMERG" in ids
    assert "IMD_GRIDDED_RAINFALL" in ids
    assert "IMD_DWR_RADAR" in ids
    assert "COPERNICUS_SENTINEL1" in ids


def test_forecast_endpoint():
    response = client.get("/api/forecast?horizon=24")
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "supporting_evidence" in data
    assert "warning" in data["prediction"]


def test_map_layers_endpoint():
    response = client.get("/api/map/layers?horizon=24")
    assert response.status_code == 200
    data = response.json()
    assert "grid_points" in data
    assert len(data["grid_points"]) > 0
    assert "hydrological_features" in data


def test_evaluation_endpoint():
    response = client.get("/api/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert "rainfall_forecasting_evaluation" in data
    assert "inundation_sar_evaluation" in data


def test_station_csv_upload_validation():
    csv_payload = "station_id,rainfall_mm,temperature_c,humidity_pct\nDEL_TEST,25.4,32.0,80.0\n"
    response = client.post(
        "/api/upload",
        data={"source_type": "IMD_STATION_CSV"},
        files={"file": ("test_station.csv", csv_payload, "text/csv")}
    )
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "SUCCESS"
    assert res["records_parsed"] == 1


def test_auth_login_and_preferences():
    # 1. Login with seeded citizen
    login_res = client.post("/api/auth/login", json={
        "email": "citizen@delhi-weather.gov.in",
        "password": "Citizen123"
    })
    assert login_res.status_code == 200
    data = login_res.json()
    assert data["status"] == "SUCCESS"
    user = data["user"]
    assert user["role"] == "citizen"
    
    # 2. Update preferences
    pref_res = client.put("/api/auth/preferences", json={
        "user_id": user["id"],
        "default_location": "Palam Airport (IGI)",
        "alert_threshold": "HIGH",
        "temp_unit": "C",
        "rain_unit": "mm"
    })
    assert pref_res.status_code == 200
    updated = pref_res.json()["user"]
    assert updated["default_location"] == "Palam Airport (IGI)"
    assert updated["alert_threshold"] == "HIGH"


def test_analytics_multi_period():
    # 7-day query
    res_7 = client.get("/api/analytics?days=7")
    assert res_7.status_code == 200
    data_7 = res_7.json()
    assert data_7["period_days"] == 7
    assert len(data_7["daily_records"]) == 7
    assert "total_rainfall_mm" in data_7["summary"]

    # 30-day query
    res_30 = client.get("/api/analytics?days=30")
    assert res_30.status_code == 200
    data_30 = res_30.json()
    assert data_30["period_days"] == 30
    assert len(data_30["daily_records"]) == 30


def test_admin_telemetry():
    summary_res = client.get("/api/admin/summary")
    assert summary_res.status_code == 200
    summary = summary_res.json()
    assert summary["total_users"] >= 3
    assert summary["system_health"] == "OPTIMAL"

    users_res = client.get("/api/admin/users")
    assert users_res.status_code == 200
    users = users_res.json()
    assert len(users) >= 3


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "timestamp_utc" in data


def test_forecast_dynamic_coordinates():
    # Test with Mumbai coordinates
    res = client.get("/api/forecast?horizon=24&lat=19.076&lon=72.877&location_name=Mumbai")
    assert res.status_code == 200
    data = res.json()
    assert data["location_name"] == "Mumbai"
    assert "coordinates" in data
    assert abs(data["coordinates"]["lat"] - 19.076) < 0.001
    assert abs(data["coordinates"]["lon"] - 72.877) < 0.001
    assert "data_classification" in data
    assert "current_weather" in data["data_classification"]


