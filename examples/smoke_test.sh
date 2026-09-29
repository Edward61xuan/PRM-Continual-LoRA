#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 PREPARED_CIFAR100_ROOT" >&2
  exit 2
fi

python train_prm.py \
  --dataset cifar100 --data_root "$1" --num_tasks 1 --seed 0 \
  --epochs 1 --batch_size 16 --eval_batch_size 16 \
  --workers 0 --eval_workers 0 --expand_times 1 \
  --model vit_base_patch16_224.augreg2_in21k_ft_in1k \
  --temperature 30 --prm_method fixed --prm_prox_weight 0.01 \
  --results_file outputs/smoke.json
