"""Check that a saved checkpoint gives the expected validation accuracy.
Put this file at src/verify_checkpoint.py. Run from the repo root:

    python src/verify_checkpoint.py --ckpt checkpoints\\resnet50_lr0.001_bs32_ep5_img224_noaug_s42.pth

Expected for the final ResNet-50: about 99.5% on the 5,000 validation images.
Eunice can use the same loading steps inside predict.py.
"""
import argparse

import torch

from data import get_loaders
from models import build_model
from utils import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to the .pth file")
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    device = get_device()
    print("Device:", device)

    # pretrained=False: the weights come from the checkpoint, so no download is needed
    model = build_model(args.model, num_classes=2, pretrained=False)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # augment=False: validation images get resize + normalise only
    _, val_loader, _ = get_loaders(args.data_dir, batch_size=64,
                                   img_size=args.img_size, augment=False)

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.numel()
    print(f"Validation accuracy: {correct}/{total} = {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
