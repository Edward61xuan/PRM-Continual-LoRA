# PRM Companion Code

This branch is the minimal companion-code release for PRM
(Proximity-Regularized Merging). The repository root is directly runnable:
`train_prm.py`, the `prm/` package, sweep scripts, figure scripts, and
environment files all live at the top level.

## Repository Layout

- `train_prm.py`: main PRM training entrypoint.
- `prm/`: LoRA modules, PRM write-back, Fisher utilities, continual-learning
  data wrappers, metrics, alpha-sweep helpers, and ViT registration.
- `scripts/run_alpha_sweep.py`: PRM-fixed run with diagnostic alpha sweeps.
- `scripts/run_lambda_sweep.py`: PRM-fixed lambda_prox sweep.
- `scripts/norm_matched_control.py`: norm-matched control diagnostic.
- `create_figures/`: Figure 2/3 scripts plus the small bundled data needed to
  regenerate the paper figures.
- `requirements.txt`, `environment.yml`, `requirements-lock.txt`: environment
  specifications.

## Environment

The experiments were run with the existing local conda environment:

```bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate /vepfs-dev/liuyixuan/P-M-for-Continual-Learning/env
python --version
```

Expected Python version: `3.10.20`.

To recreate a similar environment elsewhere:

```bash
conda env create -f environment.yml
conda activate prm
```

Alternatively, install the pip requirements:

```bash
python -m pip install -r requirements.txt
```

`requirements-lock.txt` records the package versions from the working
environment used for this code release.

## Dataset Format

The loader expects every dataset to be an image-folder tree with this shape:

```text
DATA_ROOT/
  train/
    class_000/
      image_000.jpg
      ...
    class_001/
      ...
  val/
    class_000/
      image_000.jpg
      ...
    class_001/
      ...
```

Class folders are sorted lexicographically. For CIFAR-100, folder names should
be integer class IDs (`0`, `1`, ..., `99`). For ImageNet-R, the original WordNet
IDs (`n01443537`, etc.) are fine. For CUB-200, use stable zero-padded folders
such as `000`, `001`, ..., `199`.

The supported `--dataset` values are:

- `imagenet_r`
- `cifar100`
- `sdomainet`
- `cub200`

## Download and Prepare Data

ImageNet-R: download from the official ImageNet-R repository
<https://github.com/hendrycks/imagenet-r>. It provides 200 class folders and
about 30k images. If your download has only one split, make a deterministic
train/val split per class, then arrange it as:

```text
data/imagenet-r/train/<wnid>/*.jpg
data/imagenet-r/val/<wnid>/*.jpg
```

CIFAR-100: download the official Python archive from
<https://www.cs.toronto.edu/~kriz/cifar.html>. Export the pickle batches or use
`torchvision.datasets.CIFAR100` to write images into:

```text
data/cifar100/train/0/*.png
...
data/cifar100/train/99/*.png
data/cifar100/val/0/*.png
...
data/cifar100/val/99/*.png
```

DomainNet / SDomainNet: download the cleaned DomainNet release from
<http://ai.bu.edu/DomainNet/>. The full DomainNet has 345 classes across
clipart, infograph, painting, quickdraw, real, and sketch. This code's
`sdomainet` option expects the processed Small-DomainNet-style subset already
converted to the common `train/<class>/...` and `val/<class>/...` layout.

CUB-200-2011: download the official archive from CaltechDATA
<https://data.caltech.edu/records/65de6-vp158>. Use the official metadata files
(`images.txt`, `image_class_labels.txt`, `train_test_split.txt`) to copy or
symlink images into:

```text
data/cub200/train/000/*.jpg
...
data/cub200/train/199/*.jpg
data/cub200/val/000/*.jpg
...
data/cub200/val/199/*.jpg
```

## Run PRM

Example ImageNet-R 10-task PRM-fixed run:

```bash
python train_prm.py \
  --dataset imagenet_r \
  --data_root data/imagenet-r \
  --num_tasks 10 \
  --seed 0 \
  --epochs 10 \
  --batch_size 256 \
  --eval_batch_size 100 \
  --model vit_base_patch16_clip_quickgelu_224.openai \
  --prm_method fixed \
  --prm_alpha 0.8 \
  --prm_prox_weight 0.01 \
  --results_file outputs/imagenet_r_T10_prm_fixed_s0.json
```

For a quick smoke test, reduce `--epochs`, `--batch_size`, and
`--num_tasks`. For the paper-scale run, keep the paper's dataset split,
`--num_tasks`, seeds, and hyperparameters fixed.

## Sweep Scripts

Alpha sweep at selected tasks:

```bash
python scripts/run_alpha_sweep.py \
  --dataset imagenet_r \
  --data_root data/imagenet-r \
  --num_tasks 10 \
  --tasks 2 5 10 \
  --seeds 0 1 2 \
  --epochs 10 \
  --out_root outputs/alpha_sweep
```

Lambda sweep:

```bash
python scripts/run_lambda_sweep.py \
  --dataset imagenet_r \
  --data_root data/imagenet-r \
  --num_tasks 10 \
  --seeds 0 1 2 \
  --lambdas 0 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 1 \
  --epochs 10 \
  --out_root outputs/lambda_sweep
```

Both wrappers forward unknown extra arguments to `train_prm.py`, so you can add
training options such as `--lr`, `--rank`, `--module`, or `--workers`.

## Norm-Matched Control

The norm-matched diagnostic compares task vectors from the same prefix. First
save a prefix before the target task:

```bash
python train_prm.py \
  --dataset imagenet_r \
  --data_root data/imagenet-r \
  --num_tasks 10 \
  --seed 0 \
  --epochs 10 \
  --prm_method fixed \
  --prm_prox_weight 0.01 \
  --save_prefix_at_task 5 \
  --save_prefix_dir outputs/mechanism/prefix
```

Then resume that prefix twice and save Stage-1 deltas, once without proximity
regularization and once with PRM's proximity regularization:

```bash
python train_prm.py \
  --dataset imagenet_r \
  --data_root data/imagenet-r \
  --num_tasks 10 \
  --seed 0 \
  --epochs 10 \
  --resume_prefix outputs/mechanism/prefix/prefix_t05_seed0.pt \
  --train_task 5 \
  --prm_method fixed \
  --prm_prox_weight 0.0 \
  --stage1_only_save_delta outputs/mechanism/delta/pm

python train_prm.py \
  --dataset imagenet_r \
  --data_root data/imagenet-r \
  --num_tasks 10 \
  --seed 0 \
  --epochs 10 \
  --resume_prefix outputs/mechanism/prefix/prefix_t05_seed0.pt \
  --train_task 5 \
  --prm_method fixed \
  --prm_prox_weight 0.01 \
  --stage1_only_save_delta outputs/mechanism/delta/prm
```

Finally run:

```bash
python scripts/norm_matched_control.py \
  --prefix_dir outputs/mechanism/prefix \
  --delta_pm_dir outputs/mechanism/delta/pm \
  --delta_prm_dir outputs/mechanism/delta/prm \
  --task 5 \
  --seeds 0 1 2 \
  --dataset imagenet_r \
  --data_root data/imagenet-r \
  --num_tasks 10 \
  --out_root outputs/mechanism/norm_matched
```

## Reproduce Figure 2 and Figure 3

The figure scripts use the bundled data under `create_figures/data`; they do
not require downloading the full datasets.

```bash
python create_figures/plot_figure2.py
python create_figures/plot_figure3.py
```

Outputs are written to `create_figures/outputs/`.

## Common Parameters

- `--dataset`: one of `imagenet_r`, `cifar100`, `sdomainet`, `cub200`.
- `--data_root`: root containing `train/` and `val/`.
- `--num_tasks`: class-incremental task count.
- `--seed`: class order and random seed.
- `--epochs`, `--batch_size`, `--eval_batch_size`: training scale.
- `--lr`, `--weight_decay`, `--lr_sch`: optimizer/schedule controls.
- `--rank`: LoRA rank.
- `--module`: adapted projections, default `kv`.
- `--eps`, `--p0`: perturbation magnitude and sampling probability.
- `--prm_alpha`: fixed write-back coefficient.
- `--prm_prox_weight`: lambda_prox / proximity regularization strength.
- `--alpha_sweep_tasks`: tasks where dense alpha-sweep JSONs are saved.
- `--workers`, `--eval_workers`: dataloader workers.
