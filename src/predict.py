import argparse
from pathlib import Path
import pandas as pd
import torch
from data import get_loaders
from models import build_model
from utils import get_device, set_seed

def evaluate_val(model, val_loader, device):
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct, total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out", default="results/submission.csv")
    ap.add_argument("--details", default="results/test_predictions_detail.csv")
    ap.add_argument("--sample", default="sampleSubmission.csv")
    ap.add_argument("--check-val", action="store_true")
    args = ap.parse_args()

    set_seed(42)
    device = get_device()
    print("Device:", device, flush=True)

    model = build_model(args.model, num_classes=2, pretrained=False)
    model.load_state_dict(
        torch.load(args.ckpt, map_location=device, weights_only=True)
    )
    model.to(device).eval()

    _, val_loader, test_loader = get_loaders(
        args.data_dir,
        batch_size=64,
        img_size=args.img_size,
        augment=False
    )

    if args.check_val:
        correct, total = evaluate_val(model, val_loader, device)
        print(
            f"Validation accuracy: {correct}/{total} "
            f"= {100 * correct / total:.2f}%",
            flush=True
        )

    ids, labels, p_cat, p_dog = [], [], [], []

    with torch.no_grad():
        for x, batch_ids in test_loader:
            probs = torch.softmax(
                model(x.to(device)), dim=1
            ).cpu()

            ids.extend(int(i) for i in batch_ids)
            labels.extend(probs.argmax(1).tolist())
            p_cat.extend(probs[:, 0].tolist())
            p_dog.extend(probs[:, 1].tolist())

    sub = pd.DataFrame({
        "id": ids,
        "label": labels
    }).sort_values("id")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)

    detail = pd.DataFrame({
        "id": ids,
        "label": labels,
        "p_cat": p_cat,
        "p_dog": p_dog
    })
    detail["confidence"] = detail[["p_cat", "p_dog"]].max(axis=1)
    Path(args.details).parent.mkdir(parents=True, exist_ok=True)
    detail.sort_values("id").round(4).to_csv(
        args.details, index=False
    )

    dogs = int((sub["label"] == 1).sum())

    print(
        f"Wrote {args.out}: {len(sub)} rows, "
        f"{dogs} dogs, {len(sub) - dogs} cats",
        flush=True
    )
    print(f"Wrote {args.details}", flush=True)

    if Path(args.sample).exists():
        sample = pd.read_csv(args.sample)
        same = (
            set(sample["id"]) == set(sub["id"])
            and len(sample) == len(sub)
        )
        print(
            "ids match sampleSubmission.csv:",
            "YES" if same else "NO"
        )
    else:
        print("Sample CSV missing; ID comparison skipped")

if __name__ == "__main__":
    main()