"""CIFAR-10 class-imbalance experiments for (g) and (h).

One run = one setup + one seed. Each run appends one row to the summary CSV
and saves its test confusion matrix to results/cm/<run>.csv.

The 45,000 / 5,000 train/val split and the images dropped for the imbalance
are fixed by --split-seed, so every run trains on the same images.
--seed only changes training randomness (weights init, batch order, augmentation).

Examples:
  python -u cifar10/train_imbalance.py --imbalance none --fix none --seed 42
  python -u cifar10/train_imbalance.py --imbalance step --fix weighted --seed 42
  python -u cifar10/train_imbalance.py --imbalance step --fix oversample --aug --seed 42
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from small_cnn import SmallCNN
from utils import set_seed

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
RARE = [2, 3, 4]  # bird, cat, deer
NUM_VAL = 5000


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_split(targets, imbalance, rare_keep, split_seed):
    """Return (train_idx, val_idx). Val is balanced-ish and never touched by the imbalance."""
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(len(targets))
    val_idx = perm[:NUM_VAL]
    train_idx = perm[NUM_VAL:]
    if imbalance == "step":
        keep, seen = [], {c: 0 for c in RARE}
        for i in train_idx:
            c = targets[i]
            if c in seen:
                if seen[c] >= rare_keep:
                    continue
                seen[c] += 1
            keep.append(i)
        train_idx = np.array(keep)
    return train_idx, val_idx


def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).argmax(dim=1).cpu())
            labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def confusion(preds, labels, n=10):
    cm = np.zeros((n, n), dtype=int)  # rows = true class, cols = predicted
    np.add.at(cm, (labels, preds), 1)
    return cm


def macro_f1(cm):
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return f1.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imbalance", choices=["none", "step"], default="none")
    p.add_argument("--fix", choices=["none", "weighted", "oversample"], default="none")
    p.add_argument("--aug", action="store_true", help="random crop (pad 4) + horizontal flip on train images")
    p.add_argument("--rare-keep", type=int, default=450, help="images kept per rare class with --imbalance step")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--summary", default=str(ROOT / "results" / "cifar_imbalance_summary.csv"))
    args = p.parse_args()

    run = f"{args.imbalance}_{args.fix}{'_aug' if args.aug else ''}_ep{args.epochs}_s{args.seed}"
    set_seed(args.seed)
    device = get_device()
    print(f"Run {run} on {device}")

    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    eval_tf = transforms.Compose([transforms.ToTensor(), norm])
    train_tf = eval_tf
    if args.aug:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), norm,
        ])

    data_dir = ROOT / "notebooks" / "data"
    train_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    val_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=eval_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_tf)

    targets = np.array(train_full.targets)
    train_idx, val_idx = make_split(targets, args.imbalance, args.rare_keep, args.split_seed)
    counts = np.bincount(targets[train_idx], minlength=10)
    print("Train images per class:")
    for c, n in zip(CLASSES, counts):
        print(f"  {c:<10} {n}")
    print(f"  total      {counts.sum()}  (val {len(val_idx)}, test {len(test_set)})")

    train_set = Subset(train_full, train_idx)
    val_set = Subset(val_full, val_idx)
    gen = torch.Generator().manual_seed(args.seed)

    if args.fix == "oversample":
        # Each sample drawn with prob proportional to 1 / its class count, so every
        # class is equally likely per draw. Same number of draws per epoch as without the fix.
        sample_w = 1.0 / counts[targets[train_idx]]
        sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                        num_samples=len(train_idx), replacement=True, generator=gen)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_set, batch_size=256)
    test_loader = DataLoader(test_set, batch_size=256)

    class_w = None
    if args.fix == "weighted":
        # weight_c = total / (10 * count_c): 1.0 when balanced, bigger for rare classes
        class_w = torch.tensor(counts.sum() / (10 * counts), dtype=torch.float32)
        print("Class weights:", ", ".join(f"{c} {w:.2f}" for c, w in zip(CLASSES, class_w.tolist())))
        class_w = class_w.to(device)

    model = SmallCNN(num_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            n += y.size(0)
        vp, vl = predict(model, val_loader, device)
        val_acc = (vp == vl).mean()
        print(f"Epoch {epoch + 1}: train loss {total_loss / n:.3f}, val accuracy {val_acc:.3f}")
    secs = time.time() - start

    # Test set is used once, with the final-epoch model
    tp, tl = predict(model, test_loader, device)
    cm = confusion(tp, tl)
    per_class = np.diag(cm) / cm.sum(axis=1)
    common = [c for c in range(10) if c not in RARE]
    row = {
        "run": run, "imbalance": args.imbalance, "fix": args.fix, "aug": int(args.aug),
        "epochs": args.epochs, "seed": args.seed, "split_seed": args.split_seed,
        "rare_keep": args.rare_keep if args.imbalance == "step" else "",
        "train_size": int(counts.sum()),
        "val_acc": round(float(val_acc), 4),
        "test_acc": round(float((tp == tl).mean()), 4),
        "rare_acc": round(float(per_class[RARE].mean()), 4),
        "common_acc": round(float(per_class[common].mean()), 4),
        "macro_f1": round(float(macro_f1(cm)), 4),
        **{f"acc_{c}": round(float(a), 4) for c, a in zip(CLASSES, per_class)},
        "train_seconds": round(secs, 1), "device": device,
    }
    print(f"Test accuracy {row['test_acc']:.3f} | rare (bird/cat/deer) {row['rare_acc']:.3f} | "
          f"common {row['common_acc']:.3f} | macro-F1 {row['macro_f1']:.3f}")
    for c, a in zip(CLASSES, per_class):
        print(f"  {c:<10} {a:.3f}")

    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    new_file = not summary.exists()
    with open(summary, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            w.writeheader()
        w.writerow(row)

    cm_dir = summary.parent / "cm"
    cm_dir.mkdir(exist_ok=True)
    with open(cm_dir / f"{run}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + CLASSES)
        for c, r in zip(CLASSES, cm):
            w.writerow([c] + r.tolist())
    print(f"Saved row to {summary} and confusion matrix to {cm_dir / (run + '.csv')}")


if __name__ == "__main__":
    main()
