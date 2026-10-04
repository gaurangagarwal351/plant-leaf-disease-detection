"""Shared checkpoint and device utilities."""

from pathlib import Path

import torch

from .model import LeafCNN


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable on this machine.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable on this machine.")
    return device


def load_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}. Train the model first.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_to_idx = checkpoint["class_to_idx"]
    if set(class_to_idx) != {"healthy", "diseased"}:
        raise ValueError("Checkpoint classes are not healthy and diseased.")
    model = LeafCNN(num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_to_idx, int(checkpoint["image_size"])
