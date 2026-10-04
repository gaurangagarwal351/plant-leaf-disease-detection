"""Classify one leaf image with a trained checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .data import image_transform
from .runtime import load_model, select_device


def predict_image(model, image: Image.Image, class_to_idx: dict[str, int], image_size: int, device):
    image = ImageOps.exif_transpose(image).convert("RGB")
    tensor = image_transform(image_size=image_size)(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    scores = {name: probabilities[index] for name, index in class_to_idx.items()}
    return {"prediction": max(scores, key=scores.get), "probabilities": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/best_model.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    model, class_to_idx, image_size = load_model(args.checkpoint, device)
    with Image.open(args.image) as image:
        result = predict_image(model, image, class_to_idx, image_size, device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
