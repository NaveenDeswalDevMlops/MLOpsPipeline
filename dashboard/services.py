import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import pandas as pd
import plotly.express as px
import psutil
import requests
from mlflow.tracking import MlflowClient

from project_logger import get_logger

from .config import (
    API_URL,
    EDA_DIR,
    EXTERNAL_API_URL,
    HEALTH_URL,
    LOG_FILES,
    MLFLOW_URL,
    MODEL_INFO_URL,
    MODEL_PATH,
    METADATA_URL,
    PREDICTION_DB,
    PREFECT_API_URL,
    PREPROCESS_DIR,
    PIPELINE_STATUS_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)

API_LOGGER = get_logger("dashboard_api_service", "api.log")
PREDICTION_LOGGER = get_logger("dashboard_prediction_service", "prediction.log")


def safe_json_get(url: str, timeout: int = 5) -> Any:
    API_LOGGER.info("Fetching JSON metadata from %s", url)
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        API_LOGGER.info("JSON request succeeded for %s with status %s", url, response.status_code)
        return payload
    except ValueError:
        error_payload = {"error": "Invalid JSON response", "url": url, "text": response.text}
        API_LOGGER.warning("Invalid JSON response from %s: %s", url, response.text[:200])
        return error_payload
    except Exception as exc:
        API_LOGGER.warning("JSON request failed for %s: %s", url, exc)
        return {"error": str(exc), "url": url}


def safe_health_get(url: str, timeout: int = 5) -> Any:
    API_LOGGER.info("Health probing %s", url)
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = response.json()
            API_LOGGER.info("Health probe succeeded for %s with status %s", url, response.status_code)
            return payload
        text = response.text.strip()
        if text:
            API_LOGGER.info("Health probe succeeded for %s with textual payload", url)
            return text
        API_LOGGER.info("Health probe succeeded for %s with empty payload", url)
        return True
    except Exception as exc:
        API_LOGGER.warning("Health probe failed for %s: %s", url, exc)
        return {"error": str(exc), "url": url}


def parse_service_status(response: Any) -> str:
    if isinstance(response, dict):
        return response.get("status", response.get("health", response.get("state", "unknown")))
    if isinstance(response, bool):
        return "healthy" if response else "unhealthy"
    if isinstance(response, str):
        return "healthy" if response.strip().lower() in {"ok", "healthy", "up"} else "unknown"
    return "unknown"


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def get_app_metadata() -> Dict[str, Any]:
    return safe_json_get(METADATA_URL)


def get_app_health() -> Dict[str, Any]:
    return safe_json_get(HEALTH_URL)


def get_pipeline_status() -> Dict[str, Any]:
    if PIPELINE_STATUS_PATH.exists():
        try:
            with open(PIPELINE_STATUS_PATH, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return {"error": "Unable to parse pipeline status file"}
    return {"status": "Pipeline status not available"}


def get_data_preview() -> pd.DataFrame:
    df = load_csv(RAW_DATA_PATH)
    return df if df is not None else pd.DataFrame()


def get_preprocessing_report(name: str) -> pd.DataFrame:
    path = PREPROCESS_DIR / name
    df = load_csv(path)
    return df if df is not None else pd.DataFrame()


def get_eda_report(name: str) -> pd.DataFrame:
    path = EDA_DIR / name
    df = load_csv(path)
    return df if df is not None else pd.DataFrame()


def get_eda_images() -> List[Tuple[str, Path]]:
    preferred_order = [
        ("Correlation Heatmap", "correlation_heatmap.png"),
        ("Feature Distribution", "feature_distribution.png"),
        ("Boxplot Summary", "boxplot.png"),
        ("Target Countplot", "countplot.png"),
        ("Feature Importance", "feature_importance.png"),
        ("Class Balance", "class_balance.png"),
        ("Pairplot", "pairplot.png"),
        ("Missing Values", "missing_values.png"),
    ]

    preferred_names = {file_name for _, file_name in preferred_order}
    available: List[Tuple[str, Path]] = []
    for label, file_name in preferred_order:
        path = EDA_DIR / file_name
        if path.exists():
            available.append((label, path))

    for path in sorted(EDA_DIR.glob("*.png")):
        if path.name in preferred_names or path.name == "histograms.png":
            continue
        if path.name.startswith("histogram_"):
            label = path.stem.replace("histogram_", "Histogram ").replace("_", " ").title()
        else:
            label = path.stem.replace("_", " ").title()
        available.append((label, path))

    return available


def get_image(path: Path) -> Optional[bytes]:
    if path.exists():
        return path.read_bytes()
    return None


def get_mlflow_client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_URL)
    return MlflowClient(tracking_uri=MLFLOW_URL)


def _mlflow_entity_to_dict(entity: Any) -> Dict[str, Any]:
    if hasattr(entity, "to_dictionary"):
        return entity.to_dictionary()
    if hasattr(entity, "to_dict"):
        try:
            return entity.to_dict()
        except Exception:
            pass

    if hasattr(entity, "__dict__"):
        try:
            # Use public property names when available
            entity_dict = {k: v for k, v in entity.__dict__.items() if not k.startswith("_")}
            if entity_dict:
                return entity_dict
        except Exception:
            pass

    result: Dict[str, Any] = {}
    for key in dir(entity):
        if key.startswith("_"):
            continue
        if key in {"to_dictionary", "to_dict", "to_proto", "from_dictionary", "from_proto"}:
            continue
        try:
            value = getattr(entity, key)
        except Exception:
            continue
        if callable(value):
            continue
        if hasattr(value, "__dict__") or isinstance(value, (list, tuple, dict)):
            # Recursively convert nested MLflow entity objects
            try:
                if isinstance(value, dict):
                    result[key] = {
                        sub_k: _mlflow_entity_to_dict(sub_v) if hasattr(sub_v, "__dict__") else sub_v
                        for sub_k, sub_v in value.items()
                    }
                    continue
                if isinstance(value, (list, tuple)):
                    result[key] = [
                        _mlflow_entity_to_dict(item) if hasattr(item, "__dict__") else item
                        for item in value
                    ]
                    continue
                result[key] = _mlflow_entity_to_dict(value)
                continue
            except Exception:
                pass
        result[key] = value
    return result


def get_mlflow_experiments() -> List[Dict[str, Any]]:
    client = get_mlflow_client()
    try:
        experiments = client.list_experiments()
    except AttributeError:
        experiments = client.search_experiments()
    if experiments is None:
        return []
    return [_mlflow_entity_to_dict(exp) for exp in experiments]

# Backward compatibility if any code imports the old function name
list_mlflow_experiments = get_mlflow_experiments


def get_mlflow_runs(experiment_id: str) -> List[Dict[str, Any]]:
    client = get_mlflow_client()
    runs = client.search_runs([experiment_id], max_results=50)
    if runs is None:
        return []
    results: List[Dict[str, Any]] = []
    for run in runs:
        if hasattr(run, "to_dictionary"):
            results.append(run.to_dictionary())
        elif hasattr(run, "to_dict"):
            results.append(run.to_dict())
        else:
            results.append(_mlflow_entity_to_dict(run))
    return results


def get_mlflow_run_metrics(run_id: str) -> Dict[str, Any]:
    client = get_mlflow_client()
    return client.get_run(run_id).data.metrics


def get_mlflow_run_params(run_id: str) -> Dict[str, Any]:
    client = get_mlflow_client()
    return client.get_run(run_id).data.params


def get_mlflow_artifacts(run_id: str) -> List[Dict[str, Any]]:
    client = get_mlflow_client()
    try:
        return [artifact.to_dictionary() for artifact in client.list_artifacts(run_id)]
    except Exception:
        return []

def get_prefect_status() -> Dict[str, Any]:
    health = safe_health_get(f"{PREFECT_API_URL}/health")
    flows = safe_health_get(f"{PREFECT_API_URL}/flows?limit=20")
    return {
        "health": health,
        "health_status": parse_service_status(health),
        "flows": flows,
        "flows_status": parse_service_status(flows),
    }


def get_model_info() -> Dict[str, Any]:
    return safe_json_get(MODEL_INFO_URL)


def get_monitoring_stats() -> Dict[str, Any]:
    api_resp = safe_health_get(HEALTH_URL)
    mlflow_resp = safe_health_get(f"{MLFLOW_URL}/health")
    prefect_resp = safe_health_get(f"{PREFECT_API_URL}/health")
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "api_status": parse_service_status(api_resp),
        "mlflow_status": parse_service_status(mlflow_resp),
        "prefect_status": parse_service_status(prefect_resp),
    }


def init_prediction_db() -> None:
    PREDICTION_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(PREDICTION_DB)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            age INTEGER,
            sex INTEGER,
            cp INTEGER,
            trestbps REAL,
            chol REAL,
            fbs INTEGER,
            restecg INTEGER,
            thalach REAL,
            exang INTEGER,
            oldpeak REAL,
            slope INTEGER,
            ca INTEGER,
            thal INTEGER,
            prediction INTEGER,
            probability REAL,
            risk_level TEXT
        )
        """
    )
    conn.commit()

    cursor.execute("PRAGMA table_info(predictions)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    required_columns = {
        "age": "INTEGER",
        "sex": "INTEGER",
        "cp": "INTEGER",
        "trestbps": "REAL",
        "chol": "REAL",
        "fbs": "INTEGER",
        "restecg": "INTEGER",
        "thalach": "REAL",
        "exang": "INTEGER",
        "oldpeak": "REAL",
        "slope": "INTEGER",
        "ca": "INTEGER",
        "thal": "INTEGER",
    }
    for column_name, column_type in required_columns.items():
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {column_name} {column_type}")

    conn.commit()
    conn.close()


def save_prediction(record: Dict[str, Any]) -> None:
    init_prediction_db()
    conn = sqlite3.connect(PREDICTION_DB)
    cursor = conn.cursor()

    columns = [
        "timestamp", "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        "prediction", "probability", "risk_level",
    ]
    insert_columns = [column for column in columns if column in record]
    if not insert_columns:
        raise ValueError("Prediction record does not contain any supported fields.")

    placeholders = ", ".join("?" for _ in insert_columns)
    sql = f"INSERT INTO predictions ({', '.join(insert_columns)}) VALUES ({placeholders})"
    values = [record.get(column) for column in insert_columns]

    cursor.execute(sql, values)
    conn.commit()
    conn.close()

    PREDICTION_LOGGER.info(
        "Prediction saved: age=%s, risk=%s, prediction=%s, probability=%.4f",
        record.get("age"),
        record.get("risk_level"),
        record.get("prediction"),
        float(record.get("probability") or 0.0),
    )


def get_prediction_history() -> pd.DataFrame:
    init_prediction_db()
    conn = sqlite3.connect(PREDICTION_DB)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df


def make_prediction(payload: Dict[str, Any]) -> Dict[str, Any]:
    candidate_urls = [f"{API_URL}/predict"]
    if EXTERNAL_API_URL and EXTERNAL_API_URL != API_URL:
        candidate_urls.append(f"{EXTERNAL_API_URL}/predict")
    if "localhost" in API_URL:
        candidate_urls.append("http://api:8000/predict")
    candidate_urls = list(dict.fromkeys(candidate_urls))

    last_error = None
    for url in candidate_urls:
        try:
            PREDICTION_LOGGER.info("Submitting prediction request to %s with payload=%s", url, payload)
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            PREDICTION_LOGGER.info("Prediction response from %s: %s", url, result)
            return result
        except Exception as exc:
            PREDICTION_LOGGER.error("Prediction request failed for %s: %s", url, exc)
            last_error = exc
    raise last_error if last_error is not None else RuntimeError("Failed to call prediction endpoint")


def read_log(path: Path, level_filter: Optional[str] = None, search_text: Optional[str] = None) -> List[str]:
    if not path.exists():
        return [f"Log file not found: {path}"]
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    filtered = []
    for line in lines:
        if level_filter and level_filter not in line:
            continue
        if search_text and search_text.lower() not in line.lower():
            continue
        filtered.append(line)
    return filtered
