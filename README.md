# Heart Disease UCI Prediction

## Overview
This repository contains an end-to-end Heart Disease UCI prediction solution built with:
- **FastAPI** for serving real-time predictions
- **Streamlit** for dashboard visualization
- **MLflow** for experiment tracking
- **Prefect** for pipeline orchestration
- **Docker Compose** for containerized deployment

The project uses the Cleveland Heart Disease dataset from the UCI repository to predict whether a patient is likely to have heart disease and stores prediction history for dashboard monitoring.

## Prerequisites
Before running the project, install or enable the following:
- **Docker** (Docker Desktop for macOS / Linux)
- **Docker Compose** (included with Docker Desktop)
- **GNU Make** (`make` command)
- Optional: **Python 3.10+** if you want to run tests or scripts locally outside Docker

## Quick Start
1. Open a terminal and change into the project folder:
   ```bash
   cd /Users/naveendeswal/Documents/Semester\ 3/Mlops/MLopsAssignment1/aiml_assignment
   ```
2. Start the Docker stack:
   ```bash
   make up
   ```
3. If Docker is still setting up and you want to run the code locally right away, use the fallback workflow below in parallel terminals.

### Run the project without Docker while Docker is starting up
Use these steps when you want the app to be available locally before the Docker Compose stack finishes booting.

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Start the local MLflow server in one terminal:
   ```bash
   mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns
   ```
3. Start the local Prefect server in a second terminal:
   ```bash
   prefect server start --host 0.0.0.0 --port 4200
   ```
4. Start the FastAPI service in a third terminal:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
5. Start the Streamlit dashboard in a fourth terminal:
   ```bash
   WORKSPACE=$PWD API_URL=http://localhost:8000 MLFLOW_URL=http://localhost:5000 PREFECT_API_URL=http://localhost:4200/api EXTERNAL_API_URL=http://localhost:8000 EXTERNAL_MLFLOW_URL=http://localhost:5000 EXTERNAL_PREFECT_URL=http://localhost:4200 streamlit run dashboard/streamlit_app.py --server.port 8501
   ```
6. Run the data pipeline locally:
   ```bash
   python pipelines/data_pipeline.py
   ```
7. Train the model locally and push its metrics to MLflow:
   ```bash
   MLFLOW_TRACKING_URI=http://localhost:5000 python pipelines/ml_pipeline.py
   ```

> The Docker targets remain the recommended production-style path. The commands above are a useful fallback when Docker is still initializing.

## What the Make targets do
- `make up`
  - Builds and starts all services in detached mode using `docker compose up --build -d`
- `make pipeline`
  - Runs the data preprocessing pipeline inside the API container
  - Internally executes: `docker exec -it aiml_assignment-api-1 python pipelines/data_pipeline.py`
- `make train`
  - Runs the ML training pipeline inside the API container
  - Internally executes: `docker exec -it aiml_assignment-api-1 python pipelines/ml_pipeline.py`
- `make down`
  - Stops and removes the Docker Compose services
- `make test`
  - Runs the pytest suite for API and pipeline validation

## Application Services
Once `make up` is running, the following services are available:
- **Streamlit dashboard:** http://localhost:8501
- **FastAPI docs:** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5010
- **Prefect UI:** http://localhost:4200

## Data Flow
1. **Data ingestion and preprocessing** are handled by `pipelines/data_pipeline.py`.
2. **Model training** is performed by `pipelines/ml_pipeline.py`.
3. **Inference** is served through `app/main.py` at `POST /predict`.
4. **Dashboard** saves request history to `dashboard/predictions.db` and visualizes it in Streamlit.

## Prediction API
Use the `/predict` endpoint with JSON input like:
```json
{
  "age": 53,
  "sex": 1,
  "cp": 3,
  "trestbps": 140,
  "chol": 233,
  "fbs": 0,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```
Response includes:
- `prediction` (0 or 1)
- `probability` (0.0 - 1.0)
- `model_used`

## Where to find the data
- Raw dataset: `data/raw/heart_disease.csv`
- Legacy raw copy: `data/raw_data.csv`
- Processed dataset: `data/processed_data.csv`
- Trained model artifact: `models/model.joblib`
- Legacy model artifact: `models/best_model.pkl`
- Prediction history DB: `dashboard/predictions.db`
- MLflow experiment storage: `mlruns/`
- Preprocessing reports: `artifacts/preprocessing/`
- EDA reports and plots: `artifacts/eda/`
- Logs: `logs/`

## Viewing results in the dashboard
- Open the Streamlit app at `http://localhost:8501`
- Go to the **Prediction** page to run live Heart Disease scoring
- Go to the **ML Pipeline** page to inspect model metrics
- Go to the **Data Pipeline** page to inspect the raw and processed datasets
- The **Recent Predictions** section shows the last five saved scores

## Notes
- If the API or dashboard cannot reach the prediction service, ensure the Docker Compose stack is running with `make up`.
- Re-run `make pipeline` whenever data preprocessing changes.
- Re-run `make train` whenever training logic or the model should be updated.
- Use `make down` to stop containers when you are done.
