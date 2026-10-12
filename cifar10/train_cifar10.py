import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from small_cnn import SmallCNN
from utils import set_seed


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Using device:", device)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    full_train = datasets.CIFAR10(ROOT / "notebooks" / "data", train=True, download=True, transform=tf)
    test_set = datasets.CIFAR10(ROOT / "notebooks" / "data", train=False, download=True, transform=tf)

    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(full_train, [45000, 5000], generator=gen)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model = SmallCNN(num_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss_fn(model(images), labels).backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}: val accuracy {accuracy(model, val_loader, device):.3f}")

    print(f"Final test accuracy: {accuracy(model, test_loader, device):.3f}")


if __name__ == "__main__":
    main()