import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.data_pipeline import _download_heart_disease_dataset
from project_logger import get_logger

# Silence the Git warning
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

LOG = get_logger("ml_pipeline", "training.log")
os.makedirs("models", exist_ok=True)
os.makedirs("artifacts/eda", exist_ok=True)

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Heart_Disease_UCI_Prediction")
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment(mlflow_experiment_name)

LOG.info("MLflow tracking URI set to %s", mlflow_tracking_uri)
LOG.info("MLflow experiment set to %s", mlflow_experiment_name)

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _save_confusion_matrix(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    path = Path("artifacts/eda") / f"confusion_matrix_{model_name.lower()}.png"
    plt.savefig(path)
    plt.close()
    return path


def _save_roc_curve(y_true, probabilities, model_name: str):
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{model_name} ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.tight_layout()
    path = Path("artifacts/eda") / f"roc_curve_{model_name.lower()}.png"
    plt.savefig(path)
    plt.close()
    return path


def _save_feature_importance(model, model_name: str):
    classifier = getattr(model, "named_steps", {}).get("classifier")
    if classifier is None or not hasattr(classifier, "feature_importances_"):
        return None

    preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
    transformed_names = preprocessor.get_feature_names_out().tolist()

    importances = pd.DataFrame({
        "feature": transformed_names,
        "importance": classifier.feature_importances_,
    }).sort_values(by="importance", ascending=False)

    summary_path = Path("artifacts/eda") / f"feature_importance_{model_name.lower()}.csv"
    importances.to_csv(summary_path, index=False)
    return summary_path


def evaluate_model(best_model, X_test, y_test, model_name: str):
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    report_path = Path("artifacts/eda") / f"classification_report_{model_name.lower()}.txt"
    report_path.write_text(classification_report(y_test, predictions), encoding="utf-8")

    cm_path = _save_confusion_matrix(y_test, predictions, model_name)
    roc_path = _save_roc_curve(y_test, probabilities, model_name)
    feature_importance_path = _save_feature_importance(best_model, model_name)

    print(f"--- {model_name} Evaluation ---")
    print(classification_report(y_test, predictions))
    return metrics, report_path, cm_path, roc_path, feature_importance_path


def run_ml_pipeline():
    LOG.info("Starting ML pipeline")
    raw_path = Path("artifacts/raw/heart_disease.csv")
    raw_legacy_path = Path("artifacts/raw/raw_data.csv")

    if not raw_path.exists():
        _download_heart_disease_dataset(raw_path)
    if not raw_legacy_path.exists():
        raw_legacy_path.write_text(raw_path.read_text(encoding="utf-8"), encoding="utf-8")

    df = pd.read_csv(raw_legacy_path)
    if "target" not in df.columns:
        raise KeyError("The Heart Disease dataset must contain a binary 'target' column.")

    df["target"] = df["target"].astype(int)
    X = df[FEATURE_COLUMNS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__solver": ["liblinear", "lbfgs"],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
            ),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_split": [2, 5],
            },
        ),
        "XGBoost": (
            XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.01, 0.1],
                "classifier__max_depth": [3, 5],
            },
        ),        
    }

    best_score = -1.0
    best_model_name = ""
    best_pipeline = None
    best_metrics = {}
    best_run_id = None

    for name, (base_model, param_grid) in models.items():
        start_time = time.perf_counter()
        pipeline = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor()),
                ("classifier", base_model),
            ]
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )

        with mlflow.start_run(run_name=name):
            LOG.info("Training %s with GridSearchCV", name)
            grid_search.fit(X_train, y_train)
            training_time = time.perf_counter() - start_time

            best_estimator = grid_search.best_estimator_
            metrics, report_path, cm_path, roc_path, feature_importance_path = evaluate_model(
                best_estimator,
                X_test,
                y_test,
                name,
            )

            mlflow.log_param("model_name", name)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("scoring", "roc_auc")
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("cross_validation_score", float(grid_search.best_score_))
            mlflow.log_metric("training_time_seconds", float(training_time))
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")
            mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
            mlflow.log_artifact(str(roc_path), artifact_path="evaluation")
            if feature_importance_path is not None:
                mlflow.log_artifact(str(feature_importance_path), artifact_path="evaluation")

            mlflow.sklearn.log_model(
                best_estimator,
                artifact_path="best_estimator",
                serialization_format="cloudpickle",
            )
            mlflow.sklearn.log_model(
                best_estimator.named_steps["preprocessor"],
                artifact_path="preprocessing_pipeline",
                serialization_format="cloudpickle",
            )

            if metrics["f1"] > best_score:
                best_score = metrics["f1"]
                best_model_name = name
                best_pipeline = best_estimator
                best_metrics = metrics
                best_run_id = mlflow.active_run().info.run_id

            LOG.info("Run %s completed with metrics: %s", name, metrics)

    if best_pipeline is None:
        raise RuntimeError("No model completed training successfully.")

    best_pipeline_path = Path("models/model.joblib")
    best_pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, best_pipeline_path)
    joblib.dump(best_pipeline.named_steps["preprocessor"], Path("models/pipeline.joblib"))
    joblib.dump(best_pipeline, Path("models/best_model.pkl"))

    model_registry_name = os.getenv("MLFLOW_MODEL_NAME", "HeartDiseaseUCIModel")
    if best_run_id is not None:
        client = mlflow.tracking.MlflowClient()
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{best_run_id}/best_estimator",
            name=model_registry_name,
        )
        client.transition_model_version_stage(
            name=model_registry_name,
            version=registered_model.version,
            stage="Production",
            archive_existing_versions=True,
        )
        client.set_registered_model_alias(
            model_registry_name,
            "Production",
            registered_model.version,
        )
        client.set_model_version_tag(
            model_registry_name,
            registered_model.version,
            "Stage",
            "Prod",
        )

    summary_payload = {
        "best_model": best_model_name,
        "best_f1": float(best_metrics["f1"]),
        "best_roc_auc": float(best_metrics["roc_auc"]),
        "model_version": "v1.0-heart-uci",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    Path("artifacts/pipeline_status.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    LOG.info("Pipeline complete. Best Model: %s (F1 Score: %.4f)", best_model_name, best_score)


if __name__ == "__main__":
    run_ml_pipeline()