#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 PREPARED_DATASETS_ROOT" >&2
  exit 2
fi

data_roots=$1
settings=(
  "imagenet_r_5t.yaml:imagenet-r"
  "imagenet_r_10t.yaml:imagenet-r"
  "imagenet_r_20t.yaml:imagenet-r"
  "cifar100_10t.yaml:cifar100"
  "cifar100_20t.yaml:cifar100"
  "sdomainet_5t.yaml:domainnet"
  "cub200_10t.yaml:cub200"
)

for item in "${settings[@]}"; do
  config=${item%%:*}
  dataset_dir=${item##*:}
  for method in fixed fisher; do
    python scripts/run_config.py "configs/${config}" \
      --method "$method" --seeds 0 1 2 \
      --data-root "${data_roots}/${dataset_dir}"
  done
done
