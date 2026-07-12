import os
from pathlib import Path

WORKSPACE = Path(os.getenv("WORKSPACE", "/workspace"))
API_URL = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5010")

PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://localhost:4200/api")
EXTERNAL_API_URL = os.getenv("EXTERNAL_API_URL", "http://localhost:8000")
EXTERNAL_MLFLOW_URL = os.getenv("EXTERNAL_MLFLOW_URL", "http://localhost:5010")
EXTERNAL_PREFECT_URL = os.getenv("EXTERNAL_PREFECT_URL", "http://localhost:4200")
MODEL_INFO_URL = f"{API_URL}/model-info"
RAW_DATA_PATH = WORKSPACE / "artifacts" / "raw" / "raw_data.csv"
PROCESSED_DATA_PATH = WORKSPACE / "artifacts" / "processed" / "processed_data.csv"
PREPROCESS_DIR = WORKSPACE / "artifacts" / "preprocessing"
EDA_DIR = WORKSPACE / "artifacts" / "eda"
PIPELINE_STATUS_PATH = WORKSPACE / "artifacts" / "pipeline_status.json"
MODEL_PATH = WORKSPACE / "models" / "best_model.pkl"
METADATA_URL = API_URL + "/metadata"
HEALTH_URL = API_URL + "/health"
PREDICTION_DB = WORKSPACE / "dashboard" / "predictions.db"
LOG_DIR = WORKSPACE / "logs"
LOG_FILES = {
    "Pipeline Logs": LOG_DIR / "pipeline.log",
    "Training Logs": LOG_DIR / "training.log",
    "API Logs": LOG_DIR / "api.log",
    "Prediction Logs": LOG_DIR / "prediction.log"
}
