# Proximity Regularized Merging (PRM)

Official companion implementation for **When the Merge Coefficient Stops
Mattering: Proximity Regularized Merging for Continual LoRA Adaptation**
([arXiv:2609.32332](https://arxiv.org/abs/2609.32332)).

Authors: Yixuan Liu, Yuhao Sun, Sen Song, and Jin Li.

This release focuses on the two variants used for the paper's main claims:

- **PRM-fixed** uses proximity-regularized Stage 1 and `alpha=0.8`.
- **PRM-Fisher** uses the same Stage 1 and the Fisher-derived coefficient from
  P&M ([Qiu et al., 2025](https://papers.nips.cc/paper_files/paper/2025/hash/908b3e443a15f1d664f691fdc4303e71-Abstract-Conference.html)).

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

## Paper results and reproduction

The seven paper settings are encoded in `configs/`. Run either published
variant over seeds 0, 1, and 2. The following results are reported in the paper;
Acc is final average seen-task accuracy and AAA is average anytime accuracy.

| Setting | PRM-fixed Acc | PRM-fixed AAA | PRM-Fisher Acc | PRM-Fisher AAA |
| --- | ---: | ---: | ---: | ---: |
| ImageNet-R 5T | 82.05 ± 0.39 | 86.03 ± 0.52 | 81.64 ± 0.43 | 85.78 ± 0.42 |
| ImageNet-R 10T | 81.32 ± 0.25 | 86.39 ± 0.53 | 80.87 ± 0.44 | 85.96 ± 0.48 |
| ImageNet-R 20T | 78.12 ± 0.57 | 83.70 ± 0.81 | 77.93 ± 0.30 | 83.14 ± 0.92 |
| CIFAR-100 10T | 88.10 ± 0.52 | 92.03 ± 1.03 | 88.25 ± 0.40 | 92.08 ± 0.93 |
| CIFAR-100 20T | 85.43 ± 0.68 | 89.65 ± 1.72 | 85.87 ± 0.54 | 89.87 ± 1.64 |
| S-DomainNet 5T | 84.74 ± 0.46 | 89.33 ± 0.59 | 84.76 ± 0.40 | 89.26 ± 0.52 |
| CUB-200 10T | 73.65 ± 1.89 | 80.81 ± 0.43 | 74.07 ± 1.66 | 81.04 ± 0.74 |

These are three-seed references, not hard pass/fail thresholds. See
[the reproduction guide](docs/reproduction.md) for the full protocol.

### Run ImageNet-R

Choose `fixed` or `fisher` and the desired task split:

```bash
python scripts/run_config.py configs/imagenet_r_5t.yaml \
  --method fixed --seeds 0 1 2 --data-root data/imagenet-r
python scripts/run_config.py configs/imagenet_r_10t.yaml \
  --method fisher --seeds 0 1 2 --data-root data/imagenet-r
python scripts/run_config.py configs/imagenet_r_20t.yaml \
  --method fixed --seeds 0 1 2 --data-root data/imagenet-r
```

For matched P&M controls, add `--pm-control`; this sets
`prm_prox_weight=0` while keeping the selected fixed/Fisher rule. Results go to
`outputs/<config>/<method>/seed<N>.json`.

### Run CIFAR-100, S-DomainNet, and CUB-200

The same launcher covers every other paper setting. Change `--method fixed` to
`--method fisher` to run PRM-Fisher.

```bash
python scripts/run_config.py configs/cifar100_10t.yaml \
  --method fixed --seeds 0 1 2 --data-root data/cifar100
python scripts/run_config.py configs/cifar100_20t.yaml \
  --method fixed --seeds 0 1 2 --data-root data/cifar100
python scripts/run_config.py configs/sdomainet_5t.yaml \
  --method fixed --seeds 0 1 2 --data-root data/domainnet
python scripts/run_config.py configs/cub200_10t.yaml \
  --method fixed --seeds 0 1 2 --data-root data/cub200
```

Ready-to-run smoke and all-setting launchers are under [examples/](examples/).

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

## Citation

```bibtex
@article{liu2026mergecoefficient,
  title   = {When the Merge Coefficient Stops Mattering: Proximity Regularized
             Merging for Continual LoRA Adaptation},
  author  = {Liu, Yixuan and Sun, Yuhao and Song, Sen and Li, Jin},
  journal = {arXiv preprint arXiv:2609.32332},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.32332}
}
```

Machine-readable metadata is also available in [CITATION.cff](CITATION.cff).
The code is released under the [MIT License](LICENSE).
