# Reproduction protocol

The release reproduces PRM-fixed and PRM-Fisher. Both use the same
proximity-regularized proposal; they differ only in the Stage-2 coefficient.
Setting `prm_prox_weight: 0` gives the matched P&M proposal for either rule.

## Shared protocol

- ViT-B/16 AugReg: `vit_base_patch16_224.augreg2_in21k_ft_in1k`
- rank-10 LoRA on attention K/V, without extra LoRA scaling
- AdamW, base LR `1e-3`, classifier LR `1e-2`, batch size 256
- perturbation magnitude 0.5 and `p0=0.33`
- 10 epochs/task, except S-DomainNet (5)
- PRM proximal weight `1e-2`
- PRM-fixed coefficient `0.8`; PRM-Fisher uses the P&M Fisher rule
- PCG64 class permutation with seeds 0, 1, and 2
- standard train split for Stage 1 and standard validation/test split for evaluation

The bundled S-DomainNet manifests reflect the actual released experiment
environment and span all six DomainNet domains. They are authoritative for
code-level reproduction.

## arXiv reference values

Values are mean ± standard deviation over three seeds. Acc is final average
seen-task accuracy and AAA is average anytime accuracy, in percentage points.

| Setting | PRM-fixed Acc | PRM-fixed AAA | PRM-Fisher Acc | PRM-Fisher AAA |
| --- | ---: | ---: | ---: | ---: |
| ImageNet-R 5T | 82.05 ± 0.39 | 86.03 ± 0.52 | 81.64 ± 0.43 | 85.78 ± 0.42 |
| ImageNet-R 10T | 81.32 ± 0.25 | 86.39 ± 0.53 | 80.87 ± 0.44 | 85.96 ± 0.48 |
| ImageNet-R 20T | 78.12 ± 0.57 | 83.70 ± 0.81 | 77.93 ± 0.30 | 83.14 ± 0.92 |
| CIFAR-100 10T | 88.10 ± 0.52 | 92.03 ± 1.03 | 88.25 ± 0.40 | 92.08 ± 0.93 |
| CIFAR-100 20T | 85.43 ± 0.68 | 89.65 ± 1.72 | 85.87 ± 0.54 | 89.87 ± 1.64 |
| S-DomainNet 5T | 84.74 ± 0.46 | 89.33 ± 0.59 | 84.76 ± 0.40 | 89.26 ± 0.52 |
| CUB-200 10T | 73.65 ± 1.89 | 80.81 ± 0.43 | 74.07 ± 1.66 | 81.04 ± 0.74 |

The repository records raw per-seed JSON rather than asserting a tolerance:
minor nondeterminism can arise from CUDA kernels and dependency/hardware
differences. Report the environment, GPU, three raw seeds, mean, and population
standard deviation when comparing a reproduction to these values.
