# Heart Disease UCI Assignment Report

## Dataset Migration
The project was migrated from the telecom churn dataset to the UCI Cleveland Heart Disease dataset.

Key dataset-specific updates now in place:
- `download_dataset.py` downloads the official Cleveland heart disease source.
- `pipelines/data_pipeline.py` reads the Heart Disease schema and creates Heart Disease-specific preprocessing artifacts.
- `pipelines/ml_pipeline.py` trains a classifier using the Heart Disease feature set and tracks the run in MLflow.
- `app/schemas.py` validates the Heart Disease request contract.

## Runtime Validation
The FastAPI service now expects the Heart Disease fields:
- `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

The persisted model artifact was rebuilt against that same contract and verified with a fresh endpoint regression check.

## Deployment and DevOps
The repository keeps the established Docker, Compose, and Kubernetes deployment structure. A CI workflow is included under `.github/workflows/ci.yml` to execute the API and pipeline regression suite automatically.

## Verification Evidence
Fresh validation run:
- `pytest -q tests/test_app.py tests/test_heart_pipeline.py`
- Result: `6 passed in 2.90s`
