# MLOPS PyTorch Pipeline - Assignment 3

A small PyTorch image-classification project using CIFAR-10, Docker, and Kubernetes.

## Structure

```text
mlops-pytorch-pipeline/
├── .github/workflows/ci.yml
├── configs/training_config.yaml
├── docker/Dockerfile.train
├── docker/Dockerfile.serve
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
├── requirements/
├── src/
└── tests/
```

## Architecture

```text
CIFAR-10
   │
   ▼
Training Job ──► Persistent Storage ──► Model Serving
                                      │
                                      ├── /health
                                      └── /predict
                                            │
                                            ▼
                                         Service
                                            │
                                            ▼
                                           HPA
```

## Local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/train.txt
pytest tests/ -v
```

## Docker

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
mkdir -p data checkpoints
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-train:v1

docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-serve:v1
```

Test serving:

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Kubernetes

This project uses the Kubernetes cluster provided by Docker Desktop.

Apply the training resources:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml
```

After training completes:

```bash
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
```

Verify:

```bash
kubectl get jobs,pods,svc,hpa -n ml-training
kubectl describe deployment model-serving -n ml-training
```

Test locally:

```bash
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## CI

GitHub Actions runs linting and tests for pushes to `main` and `develop` and for pull requests targeting those branches.

```bash
ruff check src tests
python -m pytest tests/ -v
```

## Git workflow

```bash
git clone https://github.com/3rpk1/mlops-pytorch-pipeline.git
cd mlops-pytorch-pipeline
git checkout -b develop
git checkout -b feature/<name>
```

Feature branches are merged into `develop` through pull requests. After the feature branches are merged and validated, `develop` is merged into `main`.
