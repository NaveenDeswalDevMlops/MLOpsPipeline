from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from .schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Heart Disease UCI API",
    version="1.0.0",
    description="FastAPI service for serving Heart Disease UCI predictions with Swagger, metrics, and health endpoints.",
)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.joblib"
LEGACY_MODEL_PATH = MODEL_DIR / "best_model.pkl"
PIPELINE_STATUS_PATH = Path("artifacts/pipeline_status.json")
MODEL_VERSION = "v1.0-heart-uci"


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Disease UCI API. Navigate to /docs for Swagger UI."}


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/application-details")
@app.get("/metadata")
def get_application_details():
    details = {
        "pipeline_status": "Active - Scheduled every 2 minutes",
        "deployment_info": "Dockerized Multi-Container Application (FastAPI, MLflow, Prometheus, Grafana)",
        "model_version": MODEL_VERSION,
        "dataset_info": "Heart Disease UCI Dataset",
        "flow_status": "Running via Prefect UI on port 4200",
        "metadata_last_checked": datetime.now(timezone.utc).isoformat(),
        "last_execution_time": None,
    }

    if PIPELINE_STATUS_PATH.exists():
        try:
            with PIPELINE_STATUS_PATH.open("r", encoding="utf-8") as status_file:
                status_data = status_file.read()
                status_payload = __import__("json").loads(status_data)
                details.update(status_payload)
                details["last_execution_time"] = status_payload.get("last_pipeline_run", details["last_execution_time"])
        except Exception:
            details["status_read_error"] = "Unable to read pipeline status file"

    return details


@app.get("/model-info")
def model_info():
    model_file = MODEL_PATH if MODEL_PATH.exists() else LEGACY_MODEL_PATH
    if not model_file.exists():
        raise HTTPException(status_code=404, detail="Model not trained yet.")
    return {"model_path": str(model_file), "status": "Ready for inference"}


def _build_features(request: PredictionRequest) -> pd.DataFrame:
    payload = request.model_dump()
    return pd.DataFrame([payload])


def _prediction_probability(model, features: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            index = classes.index(1)
            return float(probabilities[index])
        if len(probabilities) > 1:
            return float(probabilities[1])
        return float(probabilities[0])
    return 0.0


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start_time = perf_counter()
    model_file = MODEL_PATH if MODEL_PATH.exists() else LEGACY_MODEL_PATH
    if not model_file.exists():
        raise HTTPException(status_code=404, detail="Model not found. Run pipeline first.")

    try:
        model = joblib.load(model_file)
        features = _build_features(request)
        prediction = int(model.predict(features)[0])
        probability = float(_prediction_probability(model, features))
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Model input shape mismatch: {exc}") from exc
    except AttributeError as exc:
        raise HTTPException(status_code=500, detail=f"Loaded model does not support the expected prediction interface: {exc}") from exc

    response_time_ms = round((perf_counter() - start_time) * 1000, 3)
    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        model_version=MODEL_VERSION,
        timestamp=datetime.now(timezone.utc),
        response_time_ms=response_time_ms,
        model_used=type(model).__name__,
        churn_prediction=prediction,
    )
