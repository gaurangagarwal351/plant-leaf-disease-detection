"""Image preprocessing shared by training and inference."""

from pathlib import Path

from torchvision import datasets, transforms

IMAGE_SIZE = 224
CLASSES = {"diseased", "healthy"}
NORMALIZE = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


def image_transform(training: bool = False, image_size: int = IMAGE_SIZE):
    if training:
        steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        ]
    else:
        steps = [transforms.Resize(image_size + 32), transforms.CenterCrop(image_size)]
    return transforms.Compose([*steps, transforms.ToTensor(), NORMALIZE])


def load_split(data_dir: Path, split: str, training: bool = False):
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise ValueError(f"Missing {split_dir}. Run prepare_data.py first.")
    dataset = datasets.ImageFolder(split_dir, transform=image_transform(training))
    if set(dataset.classes) != CLASSES:
        raise ValueError(
            f"{split_dir} must contain nonempty 'healthy' and 'diseased' folders; "
            f"found {dataset.classes}"
        )
    return dataset
