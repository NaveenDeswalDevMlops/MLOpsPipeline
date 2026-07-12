import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from prefect import flow, get_run_logger, task
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from urllib.request import urlretrieve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_logger import get_logger

ARTIFACTS_DIR = Path("artifacts")
RAW_DIR = ARTIFACTS_DIR / "raw"
PREPROCESS_DIR = ARTIFACTS_DIR / "preprocessing"
EDA_DIR = ARTIFACTS_DIR / "eda"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
RAW_DATA_PATH = RAW_DIR / "heart_disease.csv"
PROCESSED_DATA_PATH = PROCESSED_DIR / "processed_data.csv"
PIPELINE_STATUS_PATH = ARTIFACTS_DIR / "pipeline_status.json"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PREPROCESS_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
LOG = get_logger("data_pipeline", "pipeline.log")
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
RAW_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def _download_heart_disease_dataset(target_path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Download the official UCI Heart Disease dataset and save it locally."""
    target_file = Path(target_path)
    target_file.parent.mkdir(parents=True, exist_ok=True)

    if not target_file.exists():
        urlretrieve(DATA_URL, target_file)

    frame = pd.read_csv(target_file, header=None, names=RAW_COLUMNS, na_values="?")
    frame["target"] = pd.to_numeric(frame["target"], errors="coerce")
    frame = frame.dropna(subset=["target"]).reset_index(drop=True)
    frame["target"] = (frame["target"] > 0).astype(int)
    frame.to_csv(target_file, index=False)

    legacy_alias = RAW_DIR / "raw_data.csv"
    if not legacy_alias.exists() or legacy_alias.stat().st_mtime < target_file.stat().st_mtime:
        frame.to_csv(legacy_alias, index=False)

    return frame


def compute_cramers_v(x: pd.Series, y: pd.Series) -> float:
    confusion = pd.crosstab(x, y)
    n = confusion.sum().sum()
    if n == 0:
        return 0.0

    row_totals = confusion.sum(axis=1).values.reshape(-1, 1)
    col_totals = confusion.sum(axis=0).values.reshape(1, -1)
    expected = row_totals.dot(col_totals) / n

    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = ((confusion.values - expected) ** 2 / expected)
        chi2 = np.nansum(chi2)

    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    return np.sqrt(phi2corr / denom) if denom > 0 else 0.0


@task(retries=2)
def ingest_data() -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("Starting Data Ingestion...")
    LOG.info("Starting Data Ingestion...")

    df = _download_heart_disease_dataset(RAW_DATA_PATH)
    df.to_csv(RAW_DIR / "raw_data.csv", index=False)
    logger.info(f"Rows Loaded: {df.shape[0]}")
    logger.info(f"Columns Loaded: {df.shape[1]}")
    logger.info(f"Ingested data shape: {df.shape}")
    return df


@task
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("Pipeline Started")
    logger.info("Starting Data Pre-processing...")
    LOG.info("Starting Data Pre-processing")

    df = df.copy()
    if "target" not in df.columns:
        raise KeyError("The official UCI Heart Disease dataset must contain the 'target' column.")

    numeric_features = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]
    categorical_features = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]
    feature_columns = numeric_features + categorical_features

    summary_statistics = df.describe(include="all")
    summary_statistics.to_csv(PREPROCESS_DIR / "summary_statistics.csv")
    logger.info("Summary Statistics Generated")

    data_types = df.dtypes.reset_index()
    data_types.columns = ["column", "dtype"]
    data_types.to_csv(PREPROCESS_DIR / "data_types.csv", index=False)
    logger.info("Data Types Generated")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({"missing_count": missing, "missing_percentage": missing_pct})
    missing_report.to_csv(PREPROCESS_DIR / "missing_values.csv")
    logger.info(f"Missing Value Report Generated: total missing values={int(missing.sum())}")

    numeric_df = df[numeric_features].copy()
    if numeric_df.isnull().any().any():
        medians = numeric_df.median()
        numeric_df = numeric_df.fillna(medians)
        logger.info("Numeric imputation completed using median values")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    transformed = preprocessor.fit_transform(df[feature_columns])
    transformed_columns = preprocessor.get_feature_names_out().tolist()
    transformed_df = pd.DataFrame(transformed, columns=transformed_columns)
    transformed_df["target"] = df["target"].values
    transformed_df.to_csv(PROCESSED_DATA_PATH, index=False)

    joblib_path = Path("models") / "pipeline.joblib"
    joblib_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(preprocessor, joblib_path)
    logger.info("Preprocessing pipeline saved to models/pipeline.joblib")

    #df = df.copy()
    # df[feature_columns] = numeric_df.join(df[categorical_features])
    # df = df[feature_columns + ["target"]]
    # df.to_csv(PROCESSED_DATA_PATH, index=False)

    logger.info("Pre-processing complete. Processed data saved.")
    return df


@task
def perform_eda(df: pd.DataFrame):
    logger = get_run_logger()
    logger.info("Generating EDA reports...")
    LOG.info("Generating EDA reports")

    df = df.apply(pd.to_numeric, errors='ignore')
   
    numeric_df = df.select_dtypes(include="number").copy()
    
    correlation_matrix = numeric_df.corr(numeric_only=True)
    correlation_matrix.to_csv(EDA_DIR / "correlation_matrix.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "correlation_heatmap.png")
    plt.close()

    numeric_columns = [column for column in numeric_df.columns if column != "target"]
    if numeric_columns:
        for column in numeric_columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(df[column], kde=True, color="#3B82F6")
            plt.title(f"Histogram: {column}")
            plt.xlabel(column)
            plt.tight_layout()
            plt.savefig(EDA_DIR / f"histogram_{column}.png", dpi=150)
            plt.close("all")

        distribution_df = df[numeric_columns].melt(value_name="value", var_name="feature")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=distribution_df, x="feature", y="value", palette="viridis")
        plt.title("Feature Distribution")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(EDA_DIR / "feature_distribution.png")
        plt.close("all")

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=distribution_df, x="feature", y="value", palette="viridis")
        plt.title("Boxplot Summary")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(EDA_DIR / "boxplot.png")
        plt.close("all")

        plt.figure(figsize=(8, 6))
        sns.histplot(df["target"], discrete=True)
        plt.title("Class Balance")
        plt.savefig(EDA_DIR / "class_balance.png")
        plt.close("all")

        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="target")
        plt.title("Target Countplot")
        plt.tight_layout()
        plt.savefig(EDA_DIR / "countplot.png")
        plt.close("all")

        sample = numeric_df.sample(n=min(200, len(numeric_df)), random_state=42)
        feature_subset = list(sample.columns[:6])
        try:
            sns.pairplot(sample[feature_subset])
            plt.savefig(EDA_DIR / "pairplot.png")
            plt.close("all")
            logger.info("Pairplot generated")
        except Exception as exc:
            logger.info(f"Pairplot skipped due to size or plotting issue: {exc}")

    missing_values_plot = df.isnull().sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_values_plot.index, y=missing_values_plot.values, palette="viridis")
    plt.title("Missing Value Visualization")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "missing_values.png")
    plt.close()

    if "target" in df.columns:
        count_df = df["target"].value_counts().reset_index()
        count_df.columns = ["target", "count"]
        count_df.to_csv(EDA_DIR / "class_balance.csv", index=False)

    if numeric_df.shape[1] > 1:
        feature_importance_df = pd.DataFrame(
            {
                "feature": numeric_df.drop(columns=["target"]).columns,
                "importance": np.abs(numeric_df.drop(columns=["target"]).corrwith(df["target"])).values,
            }
        ).sort_values("importance", ascending=False)
        feature_importance_df.to_csv(EDA_DIR / "feature_importance.csv", index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="importance", y="feature", data=feature_importance_df, palette="viridis")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(EDA_DIR / "feature_importance.png")
        plt.close()

    missing_count = int(df.isnull().sum().sum())
    status_payload = {
        "last_pipeline_run": datetime.now(timezone.utc).isoformat(),
        "rows_processed": int(df.shape[0]),
        "features_processed": int(df.shape[1]),
        "missing_values_found": missing_count,
        "correlation_matrix_generated": True,
        "eda_status": "completed",
        "pipeline_status": "completed",
    }
    with PIPELINE_STATUS_PATH.open("w", encoding="utf-8") as status_file:
        json.dump(status_payload, status_file, indent=2)

    logger.info("Visualizations Generated")
    logger.info("Pipeline Completed Successfully")
    LOG.info("Pipeline Completed Successfully")


@flow(name="DataOps-Pipeline", log_prints=True)
def data_pipeline_flow():
    df = ingest_data()
    processed_df = preprocess_data(df)
    perform_eda(processed_df)


if __name__ == "__main__":
    data_pipeline_flow.serve(name="dataops-deployment", interval=12000)
