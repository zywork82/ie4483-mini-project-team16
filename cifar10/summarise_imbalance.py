"""Summarise the CIFAR-10 imbalance runs written by train_imbalance.py.

Outputs (next to the summary CSV):
  cifar_imbalance_table.csv      mean ± std over seeds per setup (overall, rare, common, macro-F1)
  cifar_imbalance_perclass.csv   mean per-class test accuracy per setup
  cm/png/<setup>_s42.png         confusion matrix for seed 42 of each setup
  per_class_accuracy.png         per-class bar chart for the (h) setups

Usage:  python cifar10/summarise_imbalance.py [--summary results/cifar_imbalance_summary.csv]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
SETUP_COLS = ["imbalance", "fix", "aug", "epochs"]

# Readable names for the report, in the order they should appear
LABELS = {
    ("none", "none", 0, 5): "(g) Baseline: balanced, 5 ep",
    ("none", "none", 0, 15): "(g) Check: balanced, 15 ep, no aug",
    ("none", "none", 1, 15): "(g) Improved: balanced + aug, 15 ep",
    ("step", "none", 0, 5): "(h) Imbalanced, no fix",
    ("step", "weighted", 0, 5): "(h) Approach 1: weighted loss",
    ("step", "oversample", 0, 5): "(h) Oversample only",
    ("step", "oversample", 1, 5): "(h) Approach 2: oversample + aug",
    ("step", "none", 1, 5): "(h) Aug only (ablation)",
}


def setup_name(key):
    imb, fix, aug, ep = key
    return LABELS.get(key, f"{imb}_{fix}{'_aug' if aug else ''}_ep{ep}")


def run_stem(key):
    imb, fix, aug, ep = key
    return f"{imb}_{fix}{'_aug' if aug else ''}_ep{ep}"


def pm(series):
    # Sample std (ddof=1), same as the Dogs vs Cats tables
    if len(series) < 2:
        return f"{100 * series.mean():.1f}"
    return f"{100 * series.mean():.1f} ± {100 * series.std(ddof=1):.1f}"


def plot_cm(cm_csv, out_png, title):
    cm = pd.read_csv(cm_csv, index_col=0).values
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Fraction of true class")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default=str(ROOT / "results" / "cifar_imbalance_summary.csv"))
    args = p.parse_args()
    summary = Path(args.summary)
    out_dir = summary.parent
    df = pd.read_csv(summary)

    # If a run was repeated, keep the latest row
    df = df.drop_duplicates(subset=SETUP_COLS + ["seed"], keep="last")

    order = [k for k in LABELS if k in set(map(tuple, df[SETUP_COLS].values))]
    order += [k for k in map(tuple, df[SETUP_COLS].drop_duplicates().values) if k not in order]

    rows, per_class_rows = [], []
    for key in order:
        g = df[(df[SETUP_COLS] == pd.Series(key, index=SETUP_COLS)).all(axis=1)]
        rows.append({
            "setup": setup_name(key), "seeds": ",".join(map(str, sorted(g["seed"]))),
            "train_size": int(g["train_size"].iloc[0]),
            "overall_acc": pm(g["test_acc"]), "rare_acc": pm(g["rare_acc"]),
            "common_acc": pm(g["common_acc"]), "macro_f1": pm(g["macro_f1"]),
        })
        per_class_rows.append({"setup": setup_name(key),
                               **{c: round(100 * g[f"acc_{c}"].mean(), 1) for c in CLASSES}})

    table = pd.DataFrame(rows)
    per_class = pd.DataFrame(per_class_rows)
    table.to_csv(out_dir / "cifar_imbalance_table.csv", index=False)
    per_class.to_csv(out_dir / "cifar_imbalance_perclass.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("Test results, mean ± std over seeds (percent). Rare = bird, cat, deer.\n")
    print(table.to_string(index=False))
    print("\nMean per-class test accuracy (percent):\n")
    print(per_class.to_string(index=False))

    # Confusion matrices for seed 42
    png_dir = out_dir / "cm" / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    for key in order:
        cm_csv = out_dir / "cm" / f"{run_stem(key)}_s42.csv"
        if cm_csv.exists():
            plot_cm(cm_csv, png_dir / f"{run_stem(key)}_s42.png", f"{setup_name(key)} (seed 42)")

    # Per-class bar chart: balanced reference vs imbalanced setups
    bar_keys = [k for k in order if k[0] == "step" or k == ("none", "none", 0, 5)]
    if bar_keys:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        width = 0.8 / len(bar_keys)
        x = np.arange(10)
        for i, key in enumerate(bar_keys):
            vals = per_class.loc[per_class["setup"] == setup_name(key), CLASSES].values[0]
            ax.bar(x + (i - (len(bar_keys) - 1) / 2) * width, vals, width, label=setup_name(key))
        for c in (2, 3, 4):
            ax.axvspan(c - 0.45, c + 0.45, color="grey", alpha=0.12, zorder=0)
        ax.set_xticks(x, CLASSES)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Per-class test accuracy (shaded = rare classes, 450 train images each)")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "per_class_accuracy.png", dpi=150)
        plt.close(fig)

    print(f"\nSaved tables and figures to {out_dir}")


if __name__ == "__main__":
    main()
