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
   MODEL_SERVICE_URL=http://localhost:8001 uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
5. Start the dedicated model service in a fourth terminal:
   ```bash
   MLFLOW_TRACKING_URI=http://localhost:5000 MLFLOW_MODEL_NAME=HeartDiseaseUCIModel MLFLOW_MODEL_STAGE=Production uvicorn model_service.main:app --host 0.0.0.0 --port 8001
   ```
6. Start the Streamlit dashboard in a fifth terminal:
   ```bash
   WORKSPACE=$PWD API_URL=http://localhost:8000 MLFLOW_URL=http://localhost:5000 PREFECT_API_URL=http://localhost:4200/api EXTERNAL_API_URL=http://localhost:8000 EXTERNAL_MLFLOW_URL=http://localhost:5000 EXTERNAL_PREFECT_URL=http://localhost:4200 streamlit run dashboard/streamlit_app.py --server.port 8501
   ```
7. Run the data pipeline locally:
   ```bash
   python pipelines/data_pipeline.py
   ```
8. Train the model locally, register the best run in MLflow Model Registry, and promote the selected model to the `Production` stage:
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

## End-to-End Assignment Execution Guide
Use the following sequence to execute the full assignment locally from start to finish.

### 1. Prepare the environment
```bash
cd /Users/naveendeswal/Documents/Semester\ 3/Mlops/MLopsAssignment1/aiml_assignment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the local MLflow and Prefect services
Start MLflow in one terminal:
```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns
```

Start Prefect in a second terminal:
```bash
prefect server start --host 0.0.0.0 --port 4200
```

### 3. Run the data pipeline
This step downloads the UCI Heart Disease dataset, writes the raw artifact, builds the processed artifact, and generates the EDA plots under `artifacts/eda/`.
```bash
python pipelines/data_pipeline.py
```

### 4. Train the model and register the Production model
Run the ML training pipeline. The model is trained, evaluated, saved locally, and registered to MLflow Model Registry. The winning model is then promoted to the `Production` stage and tagged with `Stage=Prod`.
```bash
MLFLOW_TRACKING_URI=http://localhost:5000 python pipelines/ml_pipeline.py
```

### 5. Start the API and model inference split
Start the FastAPI API in one terminal and point it at the dedicated model service:
```bash
MODEL_SERVICE_URL=http://localhost:8001 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Start the dedicated model service in a second terminal:
```bash
MLFLOW_TRACKING_URI=http://localhost:5000 MLFLOW_MODEL_NAME=HeartDiseaseUCIModel MLFLOW_MODEL_STAGE=Production uvicorn model_service.main:app --host 0.0.0.0 --port 8001
```

### 6. Start the dashboard
```bash
WORKSPACE=$PWD API_URL=http://localhost:8000 MLFLOW_URL=http://localhost:5000 PREFECT_API_URL=http://localhost:4200/api EXTERNAL_API_URL=http://localhost:8000 EXTERNAL_MLFLOW_URL=http://localhost:5000 EXTERNAL_PREFECT_URL=http://localhost:4200 streamlit run dashboard/streamlit_app.py --server.port 8501
```

### 7. Verify the end-to-end flow
Open the following endpoints:
- FastAPI docs: `http://localhost:8000/docs`
- FastAPI health: `http://localhost:8000/health`
- Model service health: `http://localhost:8001/health`
- Streamlit dashboard: `http://localhost:8501`
- MLflow UI: `http://localhost:5000`
- Prefect UI: `http://localhost:4200`

Use the Swagger UI or `curl` to send a prediction request to the API:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### 8. Run the Docker-based stack if needed
```bash
make up
```

### 9. Trigger the data and training workflows in the container path
```bash
make pipeline
make train
```

### 10. Deploy locally on Minikube
Follow the steps in the next section, `Deploying on Minikube`, to build the two required images, load them into Minikube, and apply the Kubernetes manifests.

## Deploying on Minikube
The project is configured for local Kubernetes deployment on Minikube. GitHub Actions is used for CI validation only; it does not connect to your laptop's Minikube cluster.

### 1. Start Minikube
```bash
minikube start --driver=docker
minikube addons enable ingress
```

### 2. Build and load the images into Minikube
Use the local Docker daemon so Minikube can reuse the images without pushing to a remote registry. Build the API image, the dedicated model-service image, and the dashboard image separately:
```bash
cd /Users/naveendeswal/Documents/Semester\ 3/Mlops/MLopsAssignment1/aiml_assignment
eval $(minikube docker-env)
docker build -t heart-disease-api:latest .
docker build -f model_service/Dockerfile -t heart-disease-model-service:latest .
docker build -f dashboard/Dockerfile -t heart-disease-dashboard:latest .
minikube image load heart-disease-api:latest
minikube image load heart-disease-model-service:latest
minikube image load heart-disease-dashboard:latest
```

### 3. Deploy the Kubernetes manifests
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

### 4. Verify the rollout
```bash
kubectl rollout status deployment/heart-disease-api -n heart-disease
kubectl get pods -n heart-disease
kubectl get svc -n heart-disease
kubectl get ingress -n heart-disease
kubectl describe pod -n heart-disease -l app=heart-disease-api
kubectl logs -n heart-disease deployment/heart-disease-api
```

### 5. Access the service locally
If the Ingress addon is enabled, add the host entry for the local Minikube IP:
```bash
minikube ip
```
Then update your `/etc/hosts` file with the Minikube IP and `heart-disease.local`:
```text
<MINIKUBE_IP> heart-disease.local
```

Open the application in the browser:
- Swagger UI: `http://heart-disease.local/docs`
- Health endpoint: `http://heart-disease.local/health`
- Prediction endpoint: `http://heart-disease.local/predict`

### 6. Optional local service access with Minikube
```bash
minikube service heart-disease-api-service -n heart-disease --url
```

### 7. Local deployment helper script
A reusable deployment helper script is available at `scripts/deploy_minikube.sh`.
```bash
chmod +x scripts/deploy_minikube.sh
./scripts/deploy_minikube.sh
```

### 8. CI/CD behavior
The GitHub Actions workflow in `.github/workflows/ci.yml` performs the following validation order:
1. Checkout repository
2. Set up Python
3. Cache dependencies
4. Install requirements
5. Run linting
6. Run pytest
7. Validate FastAPI imports,Dashboard, DataPipeline, MLpipeline, K8s scripts, DockerImage
8. Build the Docker image
10. Upload the image artifact

> The workflow intentionally does not try to connect to your local Minikube cluster. Local Minikube deployment is a manual follow-up step after the CI pipeline succeeds.

## Notes
- If the API or dashboard cannot reach the prediction service, ensure the Docker Compose stack is running with `make up`.
- Re-run `make pipeline` whenever data preprocessing changes.
- Re-run `make train` whenever training logic or the model should be updated.
- Use `make down` to stop containers when you are done.
