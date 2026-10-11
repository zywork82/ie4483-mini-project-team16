# ie4483-mini-project-team16
# IE4483 Mini Project: Dogs vs. Cats (Option 2)

Classify images as dog (1) or cat (0), then extend the approach to CIFAR-10
(including class imbalance).

**Team:** [Member 1], [Member 2], [Member 3]

## 1. Setup

Requires Python 3.10 or 3.11.

```bash
git clone https://github.com/<username>/ie4483-mini-project.git
cd ie4483-mini-project

python -m venv venv
# Mac / Linux:      source venv/bin/activate
# Windows (PS):     venv\Scripts\Activate.ps1
# Windows (Git Bash): source venv/Scripts/activate

pip install -r requirements.txt
python src/utils.py      # checks your setup, prints the detected device
```

## 2. Dataset

Download the dataset from [shared Google Drive link] and place it so the
folders look like this (the `dataset/` folder is NOT stored in Git):

```
dataset/
  train/dog, train/cat
  val/dog,   val/cat
  test/
```

## 3. Project structure

```
src/
  data.py        data loading, pre-processing, augmentation
  models.py      model definitions
  train.py       training script
  predict.py     creates results/submission.csv
  utils.py       seed, device, helpers
cifar10/         CIFAR-10 and class-imbalance code
notebooks/       Colab notebooks (one per person)
results/         experiment log, plots, submission.csv
report/          report drafts and figures
```

## 4. How to run

Train the final Dogs vs Cats model:

```bash
python src/train.py --model [resnet18] --lr [0.001] --epochs [10] --seed 42
```

Create the submission file:

```bash
python src/predict.py --checkpoint [path/to/best_model.pth]
# writes results/submission.csv  (columns: id, label; 1 = dog, 0 = cat)
```

CIFAR-10:

```bash
python cifar10/[train_cifar10.py] [arguments]
```

(Update the commands above to match the final scripts.)

## 5. Reproducibility

- Random seed fixed to 42 (Python, NumPy, PyTorch) via `set_seed()` in `src/utils.py`.
- Final results were produced on: [hardware, e.g. Colab T4 GPU], PyTorch [version].
- [Mean and std over N runs, if applicable.]

## 6. Results

| Task | Model | Accuracy |
|---|---|---|
| Dogs vs Cats (validation) | [ ] | [ ]% |
| CIFAR-10 (test) | [ ] | [ ]% |

## 7. Git workflow

- Work on your own branch: `person1-data`, `person2-main-model`, `person3-cifar10`.
- `git pull` before you start each day.
- Open a pull request to merge into `main`.
- Never commit the dataset, model files, or tokens.