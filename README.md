# ie4483-mini-project-team16
# IE4483 Mini Project: Dogs vs. Cats (Option 2)

Classify images as dog (1) or cat (0), then extend the approach to CIFAR-10
(including class imbalance).

**Team 16:**

| Member | Matric no. | Role |
|---|---|---|
| Lee Yu Ying, Eunice | U2321640A | Person 1: data, pre-processing, prediction |
| Quek Zhi Yu | U2321638H | Person 2: main model (ResNet-50) |
| Sng Xiang Ling Chloe | U2320927G | Person 3: small CNN, CIFAR-10, class imbalance |

## 1. Setup

Requires Python 3.10 or 3.11.

```bash
git clone https://github.com/zywork82/ie4483-mini-project-team16.git
cd ie4483-mini-project-team16

python -m venv venv
# Mac / Linux:      source venv/bin/activate
# Windows (PS):     venv\Scripts\Activate.ps1
# Windows (Git Bash): source venv/Scripts/activate

pip install -r requirements.txt
python src/utils.py      # checks your setup, prints the detected device
```

**Windows with an NVIDIA GPU:** plain `pip install` gives the CPU-only build of
PyTorch. Install the CUDA 12.6 build first, then the rest:

```bash
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Mac (Apple MPS) and CPU-only machines need only `pip install -r requirements.txt`.

## 2. Dataset

Download the dataset from https://drive.google.com/drive/folders/1haV9JuVwcm29QMpoxVctJMP9vBibXaWG?usp=drive_link and place it so the
folders look like this (the `dataset/` folder is NOT stored in Git):

```
dataset/
  train/dog, train/cat
  val/dog,   val/cat
  test/
```

20,000 training images and 5,000 validation images, balanced between cats and dogs.
Labels: cat = 0, dog = 1.

**Pre-processing** (all images, in `src/data.py`): convert to RGB, resize to 224 × 224,
convert to a tensor, normalise with the ImageNet mean (0.485, 0.456, 0.406) and
std (0.229, 0.224, 0.225).

**Augmentation** (training images only, applied on the fly; the image files are not changed):

| Augmentation | Setting |
|---|---|
| Random horizontal flip | p = 0.5 |
| Random rotation | ±10 degrees |
| Random resized crop | scale 0.85 to 1.0 |

Validation and test images get only the pre-processing above. Augmentation is off by
default in `train.py` (`--augment` turns it on) and on by default in `train_small_cnn.py`
(`--no-augment` turns it off). The final ResNet-50 was trained without augmentation,
because it did not improve validation accuracy.

CIFAR-10 downloads itself (about 170 MB) into `notebooks/data/` on the first run.
This folder is not stored in Git.

## 3. Project structure

```
src/
  data.py               data loading, pre-processing, augmentation
  models.py             model definitions (build_model: resnet18, resnet50, vgg16, small_cnn)
  small_cnn.py          small CNN trained from scratch (second model)
  train.py              shared training script, logs to results/experiment_log.csv
  train_small_cnn.py    small CNN training script, one CSV per run
  predict.py            creates results/submission.csv
  summarize_results.py  prints a table from results/experiment_log.csv
  utils.py              seed, device, helpers
cifar10/
  train_cifar10.py      CIFAR-10 baseline (small CNN, 10 classes)
  train_imbalance.py    CIFAR-10 class-imbalance experiments
  summarise_imbalance.py  mean ± std tables, per-class accuracy, confusion matrices
notebooks/              first versions, kept for reference only
results/                experiment log, plots, submission.csv
report/                 report drafts and figures
checkpoints/            saved .pth files (not stored in Git)
```

## 4. How to run

Run all commands from the project folder with the venv active.

### 4.1 Final Dogs vs Cats model (ResNet-50)

ResNet-50 pretrained on ImageNet, backbone frozen, new 2-output last layer,
Adam, lr 0.001, batch 32, 5 epochs, no augmentation, all 20,000 training images.

```bash
python src/train.py --model resnet50 --who P2 --epochs 5 --batch-size 32 --lr 0.001 --seed 42
```

Repeat with `--seed 43` and `--seed 44`. Augmentation is off by default in
`train.py` (`--augment` turns it on).

The final checkpoint is on the shared Google Drive:
`resnet50_lr0.001_bs32_ep5_img224_noaug_s42.pth`. To load it:

```python
import sys
import torch
sys.path.insert(0, "src")
from models import build_model

model = build_model("resnet50", num_classes=2, pretrained=False)
model.load_state_dict(torch.load("checkpoints/resnet50_lr0.001_bs32_ep5_img224_noaug_s42.pth", map_location="cpu"))
model.eval()
```

Evaluation transform: resize to 224, ImageNet normalisation.

### 4.2 Submission file

Put the final checkpoint from the shared Google Drive in `checkpoints/`, then:

```bash
python src/predict.py --ckpt checkpoints/resnet50_lr0.001_bs32_ep5_img224_noaug_s42.pth --check-val
# writes results/submission.csv  (columns: id, label; 1 = dog, 0 = cat)
```

This predicts all 500 test images with the final ResNet-50. `--check-val` first
prints the validation accuracy (99.54 percent for this checkpoint). The script also
writes `results/test_predictions_detail.csv` (p_cat, p_dog and confidence for each
image) and, if `sampleSubmission.csv` is in the project folder, checks that the ids
match it. The prediction files are also on the shared Google Drive under `results/`.

### 4.3 Small CNN (second model, for comparison)

Full data (20,000 train, 5,000 val), augmentation off, 5 epochs, lr 0.001, batch 32:

```bash
python src/train_small_cnn.py --no-augment --seed 42 --out results/small_cnn_noaug_seed42.csv
```

Repeat with seeds 43 and 44. Drop `--no-augment` for the augmentation-on runs
(augmentation is on by default in this script). Each run takes about 20 minutes on a Mac with Apple MPS.

Matched comparison with ResNet-18 (2,000 train images, 500 val images):

```bash
python src/train.py --who P3 --model small_cnn --subset 2000 --epochs 5 --seed 42
```

Repeat with `--seed 1` and `--seed 2`.

Parameter counts for all models: `python src/models.py`

### 4.4 CIFAR-10

Baseline (45,000 / 5,000 train/val split, 10,000 test images used once at the end):

```bash
python cifar10/train_cifar10.py --seed 42
```

Class imbalance. `--imbalance step` keeps only 450 training images (10 percent)
for bird, cat and deer. `--fix` picks the correction, `--aug` adds random crop
(padding 4) and horizontal flip. The split is fixed by `--split-seed 42`, so
`--seed` only changes training randomness. Each run appends one row to
`results/cifar_imbalance_summary.csv` and saves its test confusion matrix to `results/cm/`.

```bash
for s in 42 43 44; do
  python -u cifar10/train_imbalance.py --imbalance none --fix none --seed $s
  python -u cifar10/train_imbalance.py --imbalance none --fix none --aug --epochs 15 --seed $s
  python -u cifar10/train_imbalance.py --imbalance step --fix none --seed $s
  python -u cifar10/train_imbalance.py --imbalance step --fix weighted --seed $s
  python -u cifar10/train_imbalance.py --imbalance step --fix oversample --seed $s
  python -u cifar10/train_imbalance.py --imbalance step --fix oversample --aug --seed $s
  python -u cifar10/train_imbalance.py --imbalance step --fix none --aug --seed $s
done
python cifar10/summarise_imbalance.py
```

The 21 runs take about 20 minutes on Apple MPS. `summarise_imbalance.py` writes
`results/cifar_imbalance_table.csv`, `results/cifar_imbalance_perclass.csv`,
`results/per_class_accuracy.png` and confusion matrix plots in `results/cm/png/`.

## 5. Reproducibility

- All reported results are mean ± sample std over seeds 42, 43 and 44, unless stated otherwise
  (the 2,000-image small CNN runs used seeds 42, 1 and 2).
- The seed is fixed for Python, NumPy and PyTorch via `set_seed()` in `src/utils.py`,
  and also passed to the data loader, which uses it to pick subsets.
- Rerunning with the same seed on the same machine gave identical validation accuracy at every epoch.
- Hardware and versions used for the final results:

| Experiments | Hardware | PyTorch | Time |
|---|---|---|---|
| ResNet-50 / ResNet-18 / VGG16 (Dogs vs Cats) | NVIDIA RTX 4060 (CUDA 12.6) | 2.14.1+cu126 | about 180 s per epoch (ResNet-50, full data) |
| Small CNN (Dogs vs Cats) | Mac, Apple MPS | 2.14.1 (Python 3.11.7) | about 220 s per epoch (full data) |
| CIFAR-10 and class imbalance | Mac, Apple MPS | 2.14.1 (Python 3.11.7) | about 8–9 s per epoch |

- GPU and MPS training is not guaranteed to be fully deterministic, so small differences
  may appear on other hardware.

## 6. Results

All numbers are mean ± std over seeds 42, 43 and 44.

**Dogs vs Cats (validation, 20,000 train / 5,000 val images, no augmentation, 5 epochs)**

| Model | Trainable / total parameters | Val accuracy (%) |
|---|---|---|
| ResNet-50, pretrained, frozen (**final model**) | 4,098 / 23.5 M | 99.57 ± 0.02 |
| ResNet-18, pretrained, frozen | 1,026 / 11.2 M | 98.66 ± 0.02 |
| Small CNN, from scratch | 355,778 / 355,778 | 83.6 ± 0.5 |

**CIFAR-10 (test, 10,000 images, small CNN from scratch)**

Rare = bird, cat, deer. Common = the other 7 classes.

| Setup | Train images | Overall (%) | Rare (%) | Common (%) | Macro-F1 |
|---|---|---|---|---|---|
| Balanced, 5 epochs (baseline) | 45,000 | 74.3 ± 0.8 | 58.8 ± 4.6 | 81.0 ± 0.9 | 74.0 ± 1.0 |
| Balanced + augmentation, 15 epochs (improved) | 45,000 | 78.7 ± 0.9 | 67.5 ± 3.6 | 83.6 ± 1.5 | 78.6 ± 0.9 |
| Imbalanced, no fix | 32,890 | 63.4 ± 0.2 | 13.4 ± 0.7 | 84.8 ± 0.1 | 57.9 ± 0.4 |
| Imbalanced, weighted loss | 32,890 | 65.8 ± 0.8 | 45.3 ± 3.7 | 74.6 ± 2.2 | 65.3 ± 0.7 |
| Imbalanced, oversampling + augmentation | 32,890 | 65.5 ± 0.7 | 47.9 ± 5.2 | 73.0 ± 2.9 | 65.0 ± 0.7 |

## 7. Git workflow

- Work on your own branch: `eunice` (Person 1, data), `zhiyu` (Person 2, main model), `chloe` (Person 3, CIFAR-10).
- `git pull` before you start each day.
- Open a pull request to merge into `main`.
- Never commit the dataset, model files, or tokens.
