"""Train the leaf CNN and save the best validation checkpoint."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import IMAGE_SIZE, load_split
from .metrics import classification_metrics
from .model import LeafCNN
from .runtime import select_device


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    truths, predictions = [], []
    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            truths.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return total_loss / len(loader.dataset), truths, predictions


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.learning_rate <= 0:
        raise ValueError("Epochs, batch size, patience and learning rate must be positive.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)
    train_set = load_split(args.data_dir, "train", training=True)
    val_set = load_split(args.data_dir, "val")
    if train_set.class_to_idx != val_set.class_to_idx:
        raise ValueError("Training and validation class order differs.")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LeafCNN().to(device)
    counts = Counter(train_set.targets)
    class_weights = torch.tensor(
        [len(train_set) / (2 * counts[index]) for index in range(2)],
        dtype=torch.float32, device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    stalled = 0
    history = []
    print(f"Device: {device}; train={len(train_set)}, val={len(val_set)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_true, train_pred = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_true, val_pred = run_epoch(model, val_loader, criterion, device)
        train_metrics = classification_metrics(train_true, train_pred, train_set.class_to_idx)
        val_metrics = classification_metrics(val_true, val_pred, val_set.class_to_idx)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "train_metrics": train_metrics, "val_metrics": val_metrics})
        print(f"Epoch {epoch:02d}: train loss={train_loss:.4f}, val loss={val_loss:.4f}, "
              f"val accuracy={val_metrics['accuracy']:.3f}, diseased F1={val_metrics['diseased_f1']:.3f}")

        if val_loss < best_loss:
            best_loss = val_loss
            stalled = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "class_to_idx": train_set.class_to_idx,
                        "image_size": IMAGE_SIZE,
                        "epoch": epoch,
                        "val_metrics": val_metrics}, args.output_dir / "best_model.pt")
        else:
            stalled += 1
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stalled >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break
    print(f"Best checkpoint: {args.output_dir / 'best_model.pt'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    try:
        train(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
