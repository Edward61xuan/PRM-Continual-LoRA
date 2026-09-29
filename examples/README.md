# Examples

- `smoke_test.sh DATA_ROOT` runs a one-task, one-epoch CIFAR-100 GPU check.
- `reproduce_all.sh DATA_ROOTS` launches PRM-fixed and PRM-Fisher for all seven
  paper settings and seeds 0/1/2.

For `reproduce_all.sh`, `DATA_ROOTS` is a directory containing the prepared
subdirectories `imagenet-r`, `cifar100`, `domainnet`, and `cub200`. Each command
writes one incremental JSON result and one `_per_task.json` sidecar below
`outputs/<setting>/<variant>/`.

Run one setting directly when scheduling jobs separately:

```bash
python scripts/run_config.py configs/cub200_10t.yaml \
  --method fisher --seeds 0 1 2 --data-root /datasets/cub200
```
