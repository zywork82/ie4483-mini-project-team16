
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

from utils import set_seed, get_num_workers

# ImageNet normalisation values
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Class for loading unlabelled test images
class TestDataset(Dataset):

    def __init__(self, folder, transform):
        self.folder = Path(folder)

        extensions = {".jpg", ".jpeg", ".png", ".webp"}

        self.paths = sorted(
            [
                p for p in self.folder.rglob("*")
                if p.is_file() and p.suffix.lower() in extensions
            ],
            key=lambda p: (
                0, int(p.stem)
            ) if p.stem.isdigit() else (1, p.name)
        )

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        with Image.open(path) as img:
            image = img.convert("RGB")
            image = self.transform(image)

        return image, path.stem


# Select a reproducible subset
def make_subset(dataset, size, seed):

    if size is None:
        return dataset

    if size <= 0:
        raise ValueError("Subset size must be positive")

    if size >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        len(dataset),
        generator=generator
    )[:size].tolist()

    return Subset(dataset, indices)


# Main function for teammates
def get_loaders(
    data_dir="/content/datasets",
    batch_size=32,
    img_size=224,
    augment=True,
    train_subset=None,
    val_subset=None,
    seed=42
):

    set_seed(seed)

    data_dir = Path(data_dir)

    # Training preprocessing
    train_steps = [
        transforms.Resize((img_size, img_size))
    ]

    if augment:
        train_steps.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.85, 1.0)
            )
        ])

    train_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    train_transform = transforms.Compose(train_steps)

    # Validation and test preprocessing
    # No random augmentation!
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    # Labelled training data
    train_data = datasets.ImageFolder(
        root=data_dir / "train",
        transform=train_transform
    )

    # Labelled validation data
    val_data = datasets.ImageFolder(
        root=data_dir / "val",
        transform=eval_transform
    )

    # Unlabelled test data
    test_data = TestDataset(
        folder=data_dir / "test",
        transform=eval_transform
    )

    # Verify labels
    assert train_data.class_to_idx == {
        "cat": 0,
        "dog": 1
    }

    assert val_data.class_to_idx == {
        "cat": 0,
        "dog": 1
    }

    # Apply subsets if requested
    train_data = make_subset(
        train_data, train_subset, seed
    )

    val_data = make_subset(
        val_data, val_subset, seed
    )

    workers = min(get_num_workers(), 2)

    generator = torch.Generator().manual_seed(seed)

    # Training loader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        generator=generator
    )

    # Validation loader
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers
    )

    # Test loader
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers
    )

    return train_loader, val_loader, test_loader
