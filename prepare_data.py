"""Create reproducible, stratified train/validation/test folders.

Source layouts:
  binary:       source/healthy/* and source/diseased/*
  plantvillage: source/<crop___condition>/* (condition 'healthy' is healthy)
"""

import argparse
import csv
import hashlib
import random
import shutil
from collections import Counter
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(source: Path, layout: str) -> dict[str, list[Path]]:
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    folders = sorted(path for path in source.iterdir() if path.is_dir() and not path.name.startswith("."))
    if layout == "binary":
        if {folder.name for folder in folders} != {"healthy", "diseased"}:
            raise ValueError("Binary source must contain only healthy/ and diseased/ folders.")
    elif not folders:
        raise ValueError("PlantVillage source has no class folders.")

    groups: dict[str, list[Path]] = {"healthy": [], "diseased": []}
    for folder in folders:
        label = folder.name if layout == "binary" else (
            "healthy" if folder.name.lower().split("___")[-1] == "healthy" else "diseased"
        )
        groups[label].extend(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    for label, images in groups.items():
        if len(images) < 3:
            raise ValueError(f"Need at least 3 {label} images; found {len(images)}.")
    return groups


def prepare(source: Path, output: Path, layout: str, seed: int) -> Counter:
    source = source.resolve()
    output = output.resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Source and output folders must be separate and not nested.")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output folder is not empty: {output}")

    groups = collect_images(source, layout)
    rng = random.Random(seed)
    counts: Counter = Counter()
    rows = []
    for label, images in groups.items():
        images = sorted(images)
        rng.shuffle(images)
        n_val = max(1, round(len(images) * 0.15))
        n_test = max(1, round(len(images) * 0.15))
        partitions = (
            ("val", images[:n_val]),
            ("test", images[n_val:n_val + n_test]),
            ("train", images[n_val + n_test:]),
        )
        for split, files in partitions:
            for path in files:
                relative = path.relative_to(source)
                digest = hashlib.sha256(str(relative).encode()).hexdigest()[:12]
                filename = f"{digest}_{path.name}"
                destination = output / split / label / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                counts[(split, label)] += 1
                rows.append((str(relative), split, label, str(destination.relative_to(output))))

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("source", "split", "label", "destination"))
        writer.writerows(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--layout", choices=("binary", "plantvillage"), default="binary")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        counts = prepare(args.source, args.output, args.layout, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    for split in ("train", "val", "test"):
        print(f"{split}: healthy={counts[(split, 'healthy')]}, diseased={counts[(split, 'diseased')]}")
    print(f"Prepared data at {args.output.resolve()}")


if __name__ == "__main__":
    main()
