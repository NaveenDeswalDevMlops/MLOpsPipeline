from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="Heart Disease UCI Model Service",
    version="1.0.0",
    description="Dedicated inference service that loads the MLflow Production model and serves predictions.",
)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "HeartDiseaseUCIModel")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/workspace/models"))
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class PredictionRequest(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: float = Field(..., ge=0)
    chol: float = Field(..., ge=0)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=0)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=3)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    timestamp: datetime
    response_time_ms: float
    model_used: str | None = None
    churn_prediction: int | None = None


MODEL = None
MODEL_VERSION = ""


def _load_model_from_registry() -> tuple[object, str]:
    client = mlflow.MlflowClient()
    selected_version = None

    try:
        selected_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_STAGE)
    except Exception:
        model_versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
        for model_version in model_versions:
            if model_version.current_stage == MODEL_STAGE:
                selected_version = model_version
                break

    if selected_version is None:
        raise RuntimeError(f"No model version found in stage {MODEL_STAGE} for {MODEL_NAME}")

    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    return loaded_model, str(selected_version.version)


@app.on_event("startup")
def startup_event():
    global MODEL, MODEL_VERSION
    MODEL, MODEL_VERSION = _load_model_from_registry()


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/model-info")
def model_info():
    return {"model_name": MODEL_NAME, "stage": MODEL_STAGE, "version": MODEL_VERSION, "status": "Ready for inference"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model service not initialized")

    start_time = perf_counter()
    features = pd.DataFrame([request.model_dump()])
    try:
        prediction = int(MODEL.predict(features)[0])
        probabilities = MODEL.predict_proba(features)[0]
        positive_index = list(getattr(MODEL, "classes_", [])).index(1) if hasattr(MODEL, "classes_") else 1
        probability = float(probabilities[positive_index]) if len(probabilities) > positive_index else float(probabilities[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    response_time_ms = round((perf_counter() - start_time) * 1000, 3)
    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        model_version=MODEL_VERSION,
        timestamp=datetime.now(timezone.utc),
        response_time_ms=response_time_ms,
        model_used=type(MODEL).__name__,
        churn_prediction=prediction,
    )
