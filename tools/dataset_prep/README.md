# Dataset preparation

The training code uses an image-folder tree with identical class directories
under `train/` and `val/`.

- **ImageNet-R:** download and extract the
  [official archive](https://github.com/hendrycks/imagenet-r). The preparer uses
  the train/validation manifests from the P&M-compatible experiment environment
  used for the paper.
- **CIFAR-100:** the preparer uses `torchvision.datasets.CIFAR100` and the
  [official archive](https://www.cs.toronto.edu/~kriz/cifar.html); the official
  training split becomes `train` and the official test split becomes `val`.
- **S-DomainNet:** download and extract all six domains from the
  [official DomainNet release](http://ai.bu.edu/DomainNet/) below
  `--source`. The bundled P&M-compatible manifests span `clipart`, `infograph`,
  `painting`, `quickdraw`, `real`, and `sketch`, and map them to the 200-class
  subset used by the released runs.
- **CUB-200-2011:** extract the
  [official archive](https://data.caltech.edu/records/65de6-vp158) and pass its `CUB_200_2011`
  directory. The official `train_test_split.txt` defines `train` and `val`.

The exact ImageNet-R and S-DomainNet manifests are bundled under `manifests/`
and versioned with the repository. `tools/imagenet_r_classnames.txt` contains
the 200 names used by the optional image-text head.
