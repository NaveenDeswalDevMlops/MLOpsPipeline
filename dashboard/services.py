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

from .config import (
    API_URL,
    EDA_DIR,
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


def safe_json_get(url: str, timeout: int = 5) -> Any:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except ValueError:
        return {"error": "Invalid JSON response", "url": url, "text": response.text}
    except Exception as exc:
        return {"error": str(exc), "url": url}


def safe_health_get(url: str, timeout: int = 5) -> Any:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        text = response.text.strip()
        if text:
            return text
        return True
    except Exception as exc:
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
            tenure INTEGER,
            monthly_charges REAL,
            total_charges REAL,
            contract TEXT,
            partner TEXT,
            dependents TEXT,
            internet_service TEXT,
            payment_method TEXT,
            gender TEXT,
            senior_citizen TEXT,
            prediction INTEGER,
            probability REAL,
            risk_level TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction(record: Dict[str, Any]) -> None:
    init_prediction_db()
    conn = sqlite3.connect(PREDICTION_DB)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (
            timestamp, age, tenure, monthly_charges, total_charges,
            contract, partner, dependents, internet_service,
            payment_method, gender, senior_citizen,
            prediction, probability, risk_level
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            record.get("timestamp"),
            record.get("age"),
            record.get("tenure"),
            record.get("monthly_charges"),
            record.get("total_charges"),
            record.get("contract"),
            record.get("partner"),
            record.get("dependents"),
            record.get("internet_service"),
            record.get("payment_method"),
            record.get("gender"),
            record.get("senior_citizen"),
            record.get("prediction"),
            record.get("probability"),
            record.get("risk_level"),
        ],
    )
    conn.commit()
    conn.close()


def get_prediction_history() -> pd.DataFrame:
    init_prediction_db()
    conn = sqlite3.connect(PREDICTION_DB)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df


def make_prediction(payload: Dict[str, Any]) -> Dict[str, Any]:
    candidate_urls = [f"{API_URL}/predict"]
    if "localhost" in API_URL:
        candidate_urls.append("http://api:8000/predict")
    last_error = None
    for url in candidate_urls:
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
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
