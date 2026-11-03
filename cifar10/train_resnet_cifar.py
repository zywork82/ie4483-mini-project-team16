"""Pretrained ResNet-18 on CIFAR-10 for question (g).

Applies the Dogs vs Cats approach (ImageNet-pretrained ResNet, new last layer)
to CIFAR-10. The 32 x 32 images are upscaled to 224 x 224 on the GPU so they
match the input size the pretrained weights expect.

Usage (from the repo root):
  python cifar10/train_resnet_cifar.py --mode frozen   --epochs 5 --seed 42
  python cifar10/train_resnet_cifar.py --mode finetune --epochs 5 --seed 42
Add --augment for random crop (padding 4) and horizontal flip.

frozen   = only the new 10-output layer is trained (same as the Dogs vs Cats final model)
finetune = the whole network is trained, starting from ImageNet weights

The checkpoint with the best validation accuracy is evaluated once on the
10,000 test images. One summary row per run is appended to
results/resnet_cifar_summary.csv.
"""
import argparse
import copy
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prep(x, device, size):
    """Upscale a batch of 32 x 32 images on the device, then apply ImageNet normalisation."""
    x = x.to(device, non_blocking=True)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return (x - MEAN.to(device)) / STD.to(device)


def get_loaders(data_dir, batch_size, augment, split_seed, workers):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if augment else []
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True,
                                transform=transforms.Compose(aug + [transforms.ToTensor()]))
    val_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=transforms.ToTensor())

    # 45,000 train / 5,000 validation, fixed by split_seed so every run uses the same split
    perm = torch.randperm(50000, generator=torch.Generator().manual_seed(split_seed)).tolist()
    val_idx, train_idx = perm[:5000], perm[5000:]

    def make(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=workers, pin_memory=torch.cuda.is_available())

    return make(Subset(train_ds, train_idx), True), make(Subset(val_ds, val_idx), False), make(test_ds, False)


def build_model(mode):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    if mode == "frozen":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc.")
    return model


@torch.no_grad()
def evaluate(model, loader, device, size):
    model.eval()
    correct, total = torch.zeros(10), torch.zeros(10)
    for x, y in loader:
        pred = model(prep(x, device, size)).argmax(1).cpu()
        for c in range(10):
            mask = y == c
            total[c] += mask.sum()
            correct[c] += (pred[mask] == c).sum()
    return (correct.sum() / total.sum()).item(), (correct / total).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frozen", "finetune"], default="frozen")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-3 frozen, 1e-4 finetune")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/resnet_cifar_summary.csv")
    args = ap.parse_args()
    lr = args.lr if args.lr is not None else (1e-3 if args.mode == "frozen" else 1e-4)

    set_seed(args.seed)
    device = get_device()
    use_amp = device.type == "cuda"
    train_dl, val_dl, test_dl = get_loaders(args.data_dir, args.batch_size, args.augment,
                                            args.split_seed, args.workers)
    model = build_model(args.mode).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Device {device} | mode {args.mode} | trainable {sum(p.numel() for p in params):,} "
          f"of {sum(p.numel() for p in model.parameters()):,} | lr {lr}")

    optimiser = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_acc, best_epoch, best_state, epoch_times = -1.0, 0, None, []

    for epoch in range(1, args.epochs + 1):
        # Frozen mode keeps BatchNorm in eval mode so the pretrained statistics are not changed
        model.train() if args.mode == "finetune" else model.eval()
        start, running, n = time.perf_counter(), 0.0, 0
        for x, y in train_dl:
            y = y.to(device)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(prep(x, device, args.img_size)), y)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            running += loss.item() * y.size(0)
            n += y.size(0)
        epoch_times.append(time.perf_counter() - start)
        val_acc, _ = evaluate(model, val_dl, device, args.img_size)
        print(f"Epoch {epoch}: train loss {running / n:.4f} | val acc {val_acc:.4f} | {epoch_times[-1]:.0f} s")
        if val_acc > best_acc:
            best_acc, best_epoch, best_state = val_acc, epoch, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_acc, per_class = evaluate(model, test_dl, device, args.img_size)
    print(f"Best val {best_acc:.4f} (epoch {best_epoch}) | TEST {test_acc:.4f}")
    for name, acc in zip(CLASSES, per_class):
        print(f"  {name:<11}{acc:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    row = {"model": "resnet18", "mode": args.mode, "augment": int(args.augment), "epochs": args.epochs,
           "lr": lr, "batch_size": args.batch_size, "img_size": args.img_size, "seed": args.seed,
           "best_val_acc": round(best_acc, 4), "best_epoch": best_epoch, "test_acc": round(test_acc, 4),
           **{f"acc_{c}": round(a, 4) for c, a in zip(CLASSES, per_class)},
           "sec_per_epoch": round(sum(epoch_times) / len(epoch_times), 1), "device": device.type}
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(f"Saved summary row to {args.out}")


if __name__ == "__main__":
    main()
