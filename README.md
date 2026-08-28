# MLOPS PyTorch Pipeline - Assignment 3

A small PyTorch image-classification project using CIFAR-10, Docker, and Kubernetes.

## Structure

```text
mlops-pytorch-pipeline/
├── .github/workflows/
├── configs/training_config.yaml
├── docker/Dockerfile.train
├── docker/Dockerfile.serve
├── k8s/
├── requirements/
├── src/
└── tests/
```

## Architecture

```text
CIFAR-10
   │
   ▼
Training Job ──► checkpoint PVC ──► Model Serving
      │                                  │
      └── data PVC                        ├── /health
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
python src/train.py
```

The default checkpoint is written to `./checkpoints/classifier_v1.pt`.

## Docker

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker run --rm \\
  -v "$(pwd)/data:/app/data" \\
  -v "$(pwd)/checkpoints:/app/checkpoints" \\
  mlops-train:v1

docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 \\
  -v "$(pwd)/checkpoints:/app/checkpoints" \\
  mlops-serve:v1
```

Then test the service:

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Kubernetes

Build the images first. For Minikube:

```bash
minikube image load mlops-train:v1
minikube image load mlops-serve:v1
```

Apply the resources:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/storage.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml
```

After training completes:

```bash
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training
```

For local prediction testing:

```bash
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Git workflow

```bash
git clone https://github.com/3rpk1/mlops-pytorch-pipeline.git
cd mlops-pytorch-pipeline
git checkout -b develop
git checkout -b feature/<name>
```