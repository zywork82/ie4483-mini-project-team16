"""Training script. Put this file at src/train.py.

Example (quick smoke test on ~200 images):
    python src/train.py --model resnet18 --lr 0.001 --epochs 2 --subset 200

Example (real run):
    python src/train.py --model resnet18 --lr 0.001 --batch-size 32 --epochs 10 --augment

Every run appends one row to results/experiment_log.csv and saves
the best checkpoint (by validation accuracy) plus a curves plot.
"""
import argparse
import csv
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed (works on Colab, Windows, Mac)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from models import build_model, unfreeze_last_block
from utils import get_device, get_num_workers, set_seed

LOG_COLUMNS = ["who", "model", "learning_rate", "batch_size", "epochs",
               "img_size", "augmentation", "train_size", "val_accuracy"]

# Timing goes in its own file so the shared experiment_log.csv keeps its agreed columns.
TIMING_COLUMNS = ["run_name", "who", "model", "device", "train_size",
                  "epochs_run", "total_seconds", "seconds_per_epoch"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--who", default="P2", help="your tag in the log, e.g. P2")
    p.add_argument("--model", default="resnet18")
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--augment", action="store_true", help="turn augmentation on")
    p.add_argument("--unfreeze", action="store_true",
                   help="also fine-tune the last conv block (use a small lr)")
    p.add_argument("--subset", type=int, default=None,
                   help="use only this many TRAIN images (val gets a quarter)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--patience", type=int, default=5, help="early stopping patience")
    return p.parse_args()


def get_data(args):
    """Use Eunice's get_loaders if data.py exists, else a plain ImageFolder fallback."""
    try:
        from data import get_loaders
    except ImportError:
        get_loaders = None
        print("data.py not found: using plain ImageFolder fallback")

    if get_loaders is not None:
        val_n = max(args.subset // 4, 50) if args.subset else None
        train_loader, val_loader, _ = get_loaders(
            args.data_dir, batch_size=args.batch_size, img_size=args.img_size,
            augment=args.augment, train_subset=args.subset, val_subset=val_n,
            seed=args.seed)
        return train_loader, val_loader

    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = [transforms.Resize((args.img_size, args.img_size))]
    if args.augment:
        train_tf.append(transforms.RandomHorizontalFlip())
    train_tf += [transforms.ToTensor(), norm]
    val_tf = [transforms.Resize((args.img_size, args.img_size)),
              transforms.ToTensor(), norm]

    root = Path(args.data_dir)
    train_ds = datasets.ImageFolder(root / "train", transforms.Compose(train_tf))
    val_ds = datasets.ImageFolder(root / "val", transforms.Compose(val_tf))
    print("Class mapping:", train_ds.class_to_idx, "(want cat=0, dog=1)")

    if args.subset:
        rng = random.Random(args.seed)
        tr_idx = rng.sample(range(len(train_ds)), min(args.subset, len(train_ds)))
        va_idx = rng.sample(range(len(val_ds)), min(max(args.subset // 4, 50), len(val_ds)))
        train_ds, val_ds = Subset(train_ds, tr_idx), Subset(val_ds, va_idx)

    workers = get_num_workers()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=workers)
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, device, train, freeze_bn):
    model.train(train)
    if train and freeze_bn:
        # Frozen backbone: keep BatchNorm layers fixed so pretrained statistics
        # are not overwritten by our small dataset.
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def append_log(row: dict, path="results/experiment_log.csv"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def append_timing(row: dict, path="results/timing_log.csv"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TIMING_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def save_curves(history, run_name):
    out = Path("results/curves")
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / f"{run_name}.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    train_loader, val_loader = get_data(args)

    model = build_model(args.model, num_classes=2, pretrained=True,
                        freeze_backbone=True)
    if args.unfreeze:
        unfreeze_last_block(model, args.model)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    print("Trainable parameter tensors:", len(params))
    criterion = nn.CrossEntropyLoss()  # 2 logits; softmax is inside the loss
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr)
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)

    run_name = (f"{args.model}_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}"
                f"_img{args.img_size}_{'aug' if args.augment else 'noaug'}"
                f"{'_ft' if args.unfreeze else ''}_s{args.seed}")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, bad_epochs = 0.0, 0
    freeze_bn = not args.unfreeze
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer,
                                    device, True, freeze_bn)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer,
                                    device, False, freeze_bn)
        for k, v in zip(history, [tr_loss, va_loss, tr_acc, va_acc]):
            history[k].append(v)
        print(f"Epoch {epoch}/{args.epochs}  train loss {tr_loss:.4f} acc {tr_acc:.4f}"
              f"  |  val loss {va_loss:.4f} acc {va_acc:.4f}"
              f"  [{time.perf_counter() - epoch_start:.0f}s]")

        if va_acc > best_acc:
            best_acc, bad_epochs = va_acc, 0
            torch.save(model.state_dict(), ckpt_dir / f"{run_name}.pth")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping")
                break

    save_curves(history, run_name)
    append_log({
        "who": args.who, "model": args.model, "learning_rate": args.lr,
        "batch_size": args.batch_size, "epochs": args.epochs,
        "img_size": args.img_size, "augmentation": args.augment,
        "train_size": len(train_loader.dataset),
        "val_accuracy": round(best_acc, 4),
    })
    total_s = time.perf_counter() - train_start
    epochs_run = len(history["val_acc"])
    append_timing({
        "run_name": run_name, "who": args.who, "model": args.model,
        "device": str(device), "train_size": len(train_loader.dataset),
        "epochs_run": epochs_run, "total_seconds": round(total_s, 1),
        "seconds_per_epoch": round(total_s / epochs_run, 1),
    })
    print(f"Training time: {total_s:.0f}s total, {total_s / epochs_run:.0f}s per epoch "
          f"({epochs_run} epochs, device {device})")
    print(f"Best val accuracy: {best_acc:.4f}  (checkpoint: {ckpt_dir / (run_name + '.pth')})")


if __name__ == "__main__":
    main()