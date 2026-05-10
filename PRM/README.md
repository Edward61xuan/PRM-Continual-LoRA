# PRM Companion Code

This folder is the minimal code release for the paper's main method, PRM
(Proximity-Regularized Merging). It keeps the runnable PRM implementation,
diagnostic sweep entrypoints, the norm-matched control script, and the bundled
data needed to reproduce Figure 2 and Figure 3.

## Layout

- `train_prm.py`: main PRM training entrypoint.
- `prm/`: PRM implementation, LoRA layers, continual-learning utilities, Fisher
  accumulation, metrics, and alpha-sweep helpers.
- `scripts/run_alpha_sweep.py`: PRM-fixed run with selected-task alpha sweeps.
- `scripts/run_lambda_sweep.py`: lambda_prox sweep for PRM-fixed.
- `scripts/norm_matched_control.py`: norm-matched control diagnostic.
- `create_figures/`: scripts and bundled lightweight data for Figure 2/3.

## Setup

```bash
cd PRM
pip install -r requirements.txt
```

Datasets are expected as path-based image folders with `train/` and `val/`
subdirectories.

## Run PRM

```bash
python train_prm.py \
  --dataset imagenet_r \
  --data_root /path/to/imagenet-r \
  --num_tasks 10 \
  --epochs 10 \
  --prm_method fixed \
  --prm_alpha 0.8 \
  --prm_prox_weight 0.01 \
  --results_file outputs/prm_fixed_seed0.json
```

## Alpha Sweep

```bash
python scripts/run_alpha_sweep.py \
  --dataset imagenet_r \
  --data_root /path/to/imagenet-r \
  --num_tasks 10 \
  --tasks 2 5 10 \
  --seeds 0 1 2 \
  --out_root outputs/alpha_sweep
```

## Lambda Sweep

```bash
python scripts/run_lambda_sweep.py \
  --dataset imagenet_r \
  --data_root /path/to/imagenet-r \
  --num_tasks 10 \
  --seeds 0 1 2 \
  --out_root outputs/lambda_sweep
```

## Norm-Matched Control

First save a common prefix before the target task, then train PM and PRM
Stage-1 branches from that same prefix with different `--prm_prox_weight`
values and `--stage1_only_save_delta`. Then run:

```bash
python scripts/norm_matched_control.py \
  --prefix_dir outputs/mechanism/prefix \
  --delta_pm_dir outputs/mechanism/delta/pm \
  --delta_prm_dir outputs/mechanism/delta/prm \
  --task 5 \
  --seeds 0 1 2 \
  --dataset imagenet_r \
  --data_root /path/to/imagenet-r \
  --num_tasks 10
```

## Reproduce Paper Figures

The figure scripts use the included data under `create_figures/data` and write
outputs to `create_figures/outputs`.

```bash
python create_figures/plot_figure2.py
python create_figures/plot_figure3.py
```
