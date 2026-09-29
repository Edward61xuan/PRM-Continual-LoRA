# Proximity Regularized Merging (PRM)

Official companion implementation for **When the Merge Coefficient Stops
Mattering: Proximity Regularized Merging for Continual LoRA Adaptation**
([arXiv:2609.32332](https://arxiv.org/abs/2609.32332)).

Authors: Yixuan Liu, Yuhao Sun, Sen Song, and Jin Li.

This release focuses on the two variants used for the paper's main claims:

- **PRM-fixed** uses proximity-regularized Stage 1 and `alpha=0.8`.
- **PRM-Fisher** uses the same Stage 1 and P&M's Fisher-derived coefficient.

P&M-compatible controls use the same code with `prm_prox_weight=0`. Unrelated
baseline trainers, Figure 1, and main-table aggregation are out of scope.
Figure 2/3 diagnostics are included.

## Installation

Paper runs require Linux, a CUDA GPU, Python 3.10, and access to the pretrained
timm checkpoint. CPU is supported for data preparation and tests, not training.

```bash
conda env create -f environment.yml
conda activate prm
```

Or install the direct requirements:

```bash
python -m pip install -r requirements.txt
```

`requirements-lock.txt` records the complete release environment.

## Prepare data

Loaders expect `DATA_ROOT/{train,val}/<class>/<image>`. The preparation utility
uses the bundled paper/P&M-compatible manifests and verifies their SHA-256 digests:

```bash
python tools/dataset_prep/prepare.py imagenet_r \
  --source /path/to/imagenet-r --output data/imagenet-r
python tools/dataset_prep/prepare.py cifar100 --output data/cifar100
python tools/dataset_prep/prepare.py sdomainet \
  --source /path/to/domainnet --output data/domainnet
python tools/dataset_prep/prepare.py cub200 \
  --source /path/to/CUB_200_2011 --output data/cub200
```

Use `--copy` when symlinks are unsuitable. Validate an existing tree with
`python tools/dataset_prep/prepare.py verify --output DATA_ROOT`. See
[the data guide](tools/dataset_prep/README.md) for provenance and split details.

## Reproduce the paper claims

The seven paper settings are encoded in `configs/`. Run either published
variant over all three seeds with:

```bash
python scripts/run_config.py configs/imagenet_r_10t.yaml \
  --method fixed --seeds 0 1 2 --data-root data/imagenet-r
python scripts/run_config.py configs/imagenet_r_10t.yaml \
  --method fisher --seeds 0 1 2 --data-root data/imagenet-r
```

For matched P&M controls, add `--pm-control`; this sets
`prm_prox_weight=0` while keeping the selected fixed/Fisher rule. Results go to
`outputs/<config>/<method>/seed<N>.json`.

The ImageNet-R 10-task AugReg targets reported on arXiv are:

| Variant | Final accuracy | AAA |
| --- | ---: | ---: |
| PRM-fixed | 81.32 ± 0.25 | 86.39 ± 0.53 |
| PRM-Fisher | 80.87 ± 0.44 | 85.96 ± 0.48 |

These are three-seed references, not hard pass/fail thresholds. See
[the reproduction guide](docs/reproduction.md) for every target and protocol.
Ready-to-run single- and all-setting launchers are under [examples/](examples/).

## Quick smoke test

After preparing CIFAR-100, verify the GPU path with one task and one epoch:

```bash
python train_prm.py --dataset cifar100 --data_root data/cifar100 \
  --num_tasks 1 --epochs 1 --batch_size 16 --eval_batch_size 16 \
  --workers 0 --eval_workers 0 --expand_times 1 \
  --model vit_base_patch16_224.augreg2_in21k_ft_in1k \
  --temperature 30 --prm_method fixed --prm_prox_weight 0.01 \
  --results_file outputs/smoke.json
```

AMP and class shuffling can be disabled with `--no-use_amp` and
`--no-shuffle_classes`.

## Diagnostics, figures, and tests

`scripts/run_alpha_sweep.py`, `scripts/run_lambda_sweep.py`, and
`scripts/norm_matched_control.py` reproduce selected diagnostics.

```bash
python create_figures/plot_figure2.py
python create_figures/plot_figure3.py
python -m unittest discover -s tests -v
```

CI checks imports, CLI/config expansion, metrics, deterministic class splits,
data-tree validation, sweep dry-runs, and figure regeneration. It does not
claim GPU end-to-end reproducibility.

## Citation and license

Use [CITATION.cff](CITATION.cff) when citing this work. The code is available
under the [MIT License](LICENSE).
