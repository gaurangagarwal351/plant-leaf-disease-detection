"""Evaluate the saved model once on the held-out test split."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import load_split
from .metrics import classification_metrics
from .runtime import load_model, select_device
from .train import run_epoch


def evaluate(data_dir: Path, checkpoint: Path, output: Path, batch_size: int, device_name: str):
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    device = select_device(device_name)
    model, class_to_idx, _ = load_model(checkpoint, device)
    test_set = load_split(data_dir, "test")
    if test_set.class_to_idx != class_to_idx:
        raise ValueError("Test data classes differ from checkpoint classes.")
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    loss, truth, predictions = run_epoch(model, loader, nn.CrossEntropyLoss(), device)
    metrics = classification_metrics(truth, predictions, class_to_idx)
    metrics["test_loss"] = loss
    metrics["examples"] = len(test_set)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved test metrics to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/best_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/test_metrics.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    try:
        evaluate(args.data_dir, args.checkpoint, args.output, args.batch_size, args.device)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
