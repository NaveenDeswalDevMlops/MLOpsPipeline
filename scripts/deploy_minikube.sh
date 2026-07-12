#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="heart-disease"
DEPLOYMENT_NAME="heart-disease-api"
IMAGE_NAME="heart-disease-api:latest"

if ! command -v minikube >/dev/null 2>&1; then
  echo "minikube is required but not found in PATH" >&2
  exit 1
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required but not found in PATH" >&2
  exit 1
fi

if ! minikube status >/dev/null 2>&1; then
  echo "Starting Minikube..."
  minikube start --driver=docker
else
  echo "Minikube is already running"
fi

minikube addons enable ingress

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "Building API Docker image for Minikube..."
  eval "$(minikube docker-env)"
  docker build -t "$IMAGE_NAME" .
fi

MODEL_SERVICE_IMAGE="heart-disease-model-service:latest"
if ! docker image inspect "$MODEL_SERVICE_IMAGE" >/dev/null 2>&1; then
  echo "Building model-service Docker image for Minikube..."
  eval "$(minikube docker-env)"
  docker build -f model_service/Dockerfile -t "$MODEL_SERVICE_IMAGE" .
fi

DASHBOARD_IMAGE="heart-disease-dashboard:latest"
if ! docker image inspect "$DASHBOARD_IMAGE" >/dev/null 2>&1; then
  echo "Building dashboard Docker image for Minikube..."
  eval "$(minikube docker-env)"
  docker build -f dashboard/Dockerfile -t "$DASHBOARD_IMAGE" .
fi

minikube image load "$IMAGE_NAME"
minikube image load "$MODEL_SERVICE_IMAGE"
minikube image load "$DASHBOARD_IMAGE"

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/persistent-volume.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

kubectl rollout status deployment/$DEPLOYMENT_NAME -n "$NAMESPACE"
kubectl get pods -n "$NAMESPACE"
kubectl get svc -n "$NAMESPACE"
kubectl get ingress -n "$NAMESPACE"

URL="http://heart-disease.local"
echo ""
echo "Minikube deployment complete."
echo "Access the API at: $URL"
echo "Swagger UI: $URL/docs"
echo "Health check: $URL/health"
