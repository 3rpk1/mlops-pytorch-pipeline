import os

import torch
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms

from src.model import get_model

app = Flask(__name__)
MODEL_PATH = os.getenv("MODEL_PATH", "/app/checkpoints/classifier_v1.pt")

model = get_model()
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
])


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "image is required"}), 400
    image = Image.open(file.stream).convert("RGB")
    with torch.no_grad():
        probs = torch.softmax(model(transform(image).unsqueeze(0)), dim=1)[0]
    return jsonify({
        "predicted_class": int(probs.argmax()),
        "probabilities": [round(float(p), 6) for p in probs],
    })
