import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from data import get_loaders
from small_cnn import SmallCNN
from utils import set_seed

ROOT = Path(__file__).resolve().parent.parent


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(ROOT / "dataset"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--val-subset", type=int, default=None)
    p.add_argument("--out", default=str(ROOT / "results" / "small_cnn_dogs_vs_cats.csv"))
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Using device:", device)

    train_loader, val_loader, _ = get_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        augment=not args.no_augment,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        seed=args.seed,
    )

    model = SmallCNN(num_classes=2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    rows = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        val_acc = correct / total
        print(f"Epoch {epoch + 1}: train loss {avg_loss:.3f}, val accuracy {val_acc:.3f}")
        rows.append([epoch + 1, avg_loss, val_acc])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_accuracy"])
        writer.writerows(rows)
    print("Saved:", out)


if __name__ == "__main__":
    main()