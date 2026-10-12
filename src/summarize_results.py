"""Print the experiment log as a clean table. Put at src/summarize_results.py.

    python src/summarize_results.py                      # all runs
    python src/summarize_results.py --from-row 9         # skip early practice runs
    python src/summarize_results.py --who P2 --sort-by val_accuracy
    python src/summarize_results.py --from-row 9 --markdown   # table you can paste into the report
    python src/summarize_results.py --from-row 9 --save results/summary.md

Run numbers are the row numbers in experiment_log.csv, so they stay the same when you filter.
"""
import argparse
from pathlib import Path

import pandas as pd

COLUMNS = ["run", "who", "model", "learning_rate", "batch_size", "epochs",
           "img_size", "augmentation", "train_size", "val_accuracy"]


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="results/experiment_log.csv")
    ap.add_argument("--from-row", type=int, default=1, help="first run number to show")
    ap.add_argument("--who", default=None, help="only this person, e.g. P2")
    ap.add_argument("--model", default=None, help="only this model, e.g. resnet18")
    ap.add_argument("--sort-by", default=None, choices=COLUMNS,
                    help="sort by a column (default: run order)")
    ap.add_argument("--markdown", action="store_true", help="print as a markdown table")
    ap.add_argument("--save", default=None, help="save the table to this file")
    args = ap.parse_args()

    df = pd.read_csv(args.log)
    df.insert(0, "run", range(1, len(df) + 1))

    df = df[df["run"] >= args.from_row]
    if args.who:
        df = df[df["who"] == args.who]
    if args.model:
        df = df[df["model"] == args.model]
    if args.sort_by:
        df = df.sort_values(args.sort_by, ascending=(args.sort_by != "val_accuracy"))

    if df.empty:
        print("No runs match those filters.")
        return

    show = df.copy()
    show["val_accuracy"] = (show["val_accuracy"] * 100).round(1).astype(str) + "%"
    text = to_markdown(show) if args.markdown else show.to_string(index=False)

    print(text)
    best = df.loc[df["val_accuracy"].idxmax()]
    print(f"\n{len(df)} runs shown. Best: run {int(best['run'])} "
          f"({best['model']}, lr {best['learning_rate']}, batch {int(best['batch_size'])}) "
          f"= {best['val_accuracy'] * 100:.1f}%")
    print("Note: with ~500 validation images, differences under about 1% are within noise.")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(text + "\n")
        print("Saved to", args.save)


if __name__ == "__main__":
    main()