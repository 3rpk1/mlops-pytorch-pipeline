import argparse
import json
from pathlib import Path

import torch
import yaml
from torch import nn

from src.dataset import get_dataloaders
from src.model import get_model


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * inputs.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()
    container_config = Path("/app/configs/training_config.yaml")
    config_path = container_config if container_config.exists() and args.config == parser.get_default("config") else Path(args.config)
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config["model"]["architecture"], config["model"]["num_classes"]).to(device)
    train_loader, val_loader = get_dataloaders(config["data"]["data_dir"], config["training"]["batch_size"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    stale = 0
    patience = config["training"]["early_stopping_patience"]
    output = checkpoint_dir / config["output"]["model_name"]

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        print(json.dumps({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }), flush=True)

        if val_loss < best_loss:
            best_loss = val_loss
            stale = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": config["model"]["architecture"],
                "num_classes": config["model"]["num_classes"],
            }, output)
            print(json.dumps({"event": "checkpoint_saved", "path": str(output)}), flush=True)
        else:
            stale += 1
            if stale >= patience:
                print(json.dumps({"event": "early_stopping", "epoch": epoch}), flush=True)
                break

    print(json.dumps({"event": "training_complete", "best_val_loss": round(best_loss, 4)}), flush=True)


if __name__ == "__main__":
    main()
