import pytest
from unittest.mock import patch

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_application_details(client):
    response = client.get("/application-details")
    assert response.status_code == 200
    data = response.json()
    assert "pipeline_status" in data
    assert "last_execution_time" in data

# Mock the model loading so the test runs without needing the .pkl file on disk
from unittest.mock import patch, MagicMock

@patch("app.main.joblib.load")
def test_predict_endpoint_success(mock_joblib, client, valid_payload):
    # 1. Setup the mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.2, 0.8]]
    mock_model.classes_ = [0, 1]
    
    # 2. Tell the joblib mock to return our fake model
    mock_joblib.return_value = mock_model
    
    # 3. Perform the request
    response = client.post("/predict", json=valid_payload)
    
    # 4. Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["probability"] == 0.8
    assert data["model_used"] == "MagicMock"

def test_predict_endpoint_invalid_input(client):
    # Negative test: Sending string instead of int for age
    invalid_payload = {"age": "too_old"}
    response = client.post("/predict", json=invalid_payload)
    # Expecting 422 Unprocessable Entity (FastAPI validation error)
    assert response.status_code == 422