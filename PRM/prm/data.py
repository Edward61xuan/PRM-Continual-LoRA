"""
Dataset loading for class-incremental continual learning.

Supports ImageNet-R, CIFAR-100, and SDomainNet with:
  - Per-class image path indexing
  - Class-incremental dataset slicing (only current task's classes)
  - Label remapping (global -> local for per-task heads, global -> sequential for eval)
  - Data augmentation via AutoAugment + RandomResizedCrop
"""

import os
from os.path import join, exists, realpath, dirname
from copy import deepcopy
from functools import partial

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


# ---------------------------------------------------------------------------
# Path-indexed dataset classes
# ---------------------------------------------------------------------------

class ImagePathDataset(Dataset):
    """Base class: maps class_int -> list of image file paths."""

    def __init__(self):
        super().__init__()
        self.class_imgpath_dict = {}
        self.class_int_str_map = {}
        self.split_path = ""

    def __getitem__(self, index):
        return self.class_imgpath_dict[index]

    def __len__(self):
        return len(self.class_imgpath_dict)

    @property
    def class_list(self):
        return sorted(list(self.class_imgpath_dict.keys()))

    @property
    def num_classes(self):
        return len(self)


class ImageNetRPath(ImagePathDataset):
    """ImageNet-R dataset with 200 classes."""

    def __init__(self, root_dir, train=True):
        super().__init__()
        split = "train" if train else "val"
        split_path = join(root_dir, split)
        assert exists(split_path), f"Path not found: {split_path}"
        self.split_path = split_path

        cls_dir_list = sorted(os.listdir(split_path))
        # Try to load class names
        classnames_file = join(
            dirname(dirname(realpath(__file__))), "tools", "imagenet_r_classnames.txt"
        )
        name_map = {}
        if exists(classnames_file):
            with open(classnames_file, "r") as f:
                for line in f.readlines():
                    sn = line.split(" ")[0]
                    dn = line.strip("\n")[len(sn) + 1 :]
                    name_map[sn] = dn

        for cls_int, cls_dir in enumerate(cls_dir_list):
            cls_path = realpath(join(split_path, cls_dir))
            self.class_imgpath_dict[cls_int] = [
                join(cls_path, f) for f in os.listdir(cls_path)
            ]
            self.class_int_str_map[cls_int] = name_map.get(cls_dir, cls_dir)


class CIFAR100Path(ImagePathDataset):
    """CIFAR-100 dataset with 100 classes."""

    def __init__(self, root_dir, train=True):
        super().__init__()
        split = "train" if train else "val"
        split_path = join(root_dir, split)
        assert exists(split_path), f"Path not found: {split_path}"
        self.split_path = split_path

        cls_dir_list = sorted(os.listdir(split_path))
        for cls_dir in cls_dir_list:
            cls_int = int(cls_dir)
            cls_path = realpath(join(split_path, cls_dir))
            self.class_imgpath_dict[cls_int] = [
                join(cls_path, f) for f in os.listdir(cls_path)
            ]
            self.class_int_str_map[cls_int] = cls_dir


class SDomainNetPath(ImagePathDataset):
    """SDomainNet dataset with 200 classes."""

    def __init__(self, root_dir, train=True):
        super().__init__()
        split = "train" if train else "val"
        split_path = join(root_dir, split)
        assert exists(split_path), f"Path not found: {split_path}"
        self.split_path = split_path

        cls_dir_list = sorted(os.listdir(split_path))
        for cls_int, cls_dir in enumerate(cls_dir_list):
            cls_path = realpath(join(split_path, cls_dir))
            self.class_imgpath_dict[cls_int] = [
                join(cls_path, f) for f in os.listdir(cls_path)
            ]
            self.class_int_str_map[cls_int] = cls_dir


class CUB200Path(ImagePathDataset):
    """CUB-200-2011 dataset with 200 classes (fine-grained bird species).

    Expects `root_dir/{train,val}/<cls_00..199>/<img>` layout, as produced by
    the CUB-200 data preparation utility. Class folder names are 3-digit-padded
    ("000", "001", ...) to be sort-stable.
    """

    def __init__(self, root_dir, train=True):
        super().__init__()
        split = "train" if train else "val"
        split_path = join(root_dir, split)
        assert exists(split_path), f"Path not found: {split_path}"
        self.split_path = split_path

        cls_dir_list = sorted(os.listdir(split_path))
        for cls_int, cls_dir in enumerate(cls_dir_list):
            cls_path = realpath(join(split_path, cls_dir))
            self.class_imgpath_dict[cls_int] = [
                join(cls_path, f) for f in os.listdir(cls_path)
            ]
            self.class_int_str_map[cls_int] = cls_dir


# Dataset registry
DATASET_REGISTRY = {
    "imagenet_r": ImageNetRPath,
    "cifar100": CIFAR100Path,
    "sdomainet": SDomainNetPath,
    "cub200": CUB200Path,
}


def get_path_dataset(dataset_name, data_root, train=True):
    """Create a path-indexed dataset by name.

    Args:
        dataset_name: One of 'imagenet_r', 'cifar100', 'sdomainet'.
        data_root:    Root directory containing train/ and val/ subdirs.
        train:        If True, load training split; else validation.

    Returns:
        ImagePathDataset instance.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name](data_root, train=train)


# ---------------------------------------------------------------------------
# Class-incremental dataset
# ---------------------------------------------------------------------------

class ClassIncrementalDataset(Dataset):
    """Dataset that slices a full dataset down to specific task classes.

    Handles label remapping and optional data expansion (repeat factor
    to increase effective epoch size for small class subsets).
    """

    def __init__(
        self,
        path_dataset,
        task_class_list,
        transforms=None,
        target_transforms=None,
        expand_times=1,
    ):
        super().__init__()
        self.path_dataset = path_dataset
        self.task_class_list = tuple(deepcopy(task_class_list))
        self.expand_times = expand_times
        self.transforms = transforms
        self.target_transforms = target_transforms

        self.samples, self.labels = self._collect_samples()
        self.num_samples = len(self.labels)

    def _collect_samples(self):
        samples, labels = [], []
        for cls_int in self.task_class_list:
            for img_path in sorted(self.path_dataset[cls_int]):
                samples.append(img_path)
                label = cls_int
                if self.target_transforms is not None:
                    label = self.target_transforms(label)
                labels.append(label)
        return samples, labels

    def __getitem__(self, index):
        index %= self.num_samples
        img = Image.open(self.samples[index]).convert("RGB")
        label = self.labels[index]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return self.num_samples * self.expand_times


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def make_label_maps(taskid, task_classes):
    """Create global -> (taskid, local_label, sequential_label) mapping.

    Args:
        taskid:       Current task index.
        task_classes:  List of original class indices for this task.

    Returns:
        dict: {original_class: (taskid, local_idx, global_sequential_idx)}
    """
    g2l_map = {}
    for idx, cls in enumerate(task_classes):
        g2l_map[cls] = (taskid, idx, taskid * len(task_classes) + idx)
    return g2l_map


class TargetTransform:
    """Callable that remaps labels using a global-to-local map."""

    def __init__(self, label_map_g2l, map_to_local=True):
        self.label_map_g2l = deepcopy(label_map_g2l)
        self.map_to_local = map_to_local

    def __call__(self, target):
        if self.map_to_local:
            return self.label_map_g2l[target][1]
        else:
            return self.label_map_g2l[target][2]


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

def build_train_transform(dataset_name, pretrained_cfg):
    """Build training data augmentation transform.

    Uses AutoAugment + RandomResizedCrop + normalization with
    dataset-appropriate interpolation and augmentation policy.

    CUB-200 uses Resize+CenterCrop (no RandomResizedCrop) because the bird
    often occupies only 30-50% of the frame and random-crop regularly
    removes the object entirely.
    """
    mean = pretrained_cfg.get("mean", (0.485, 0.456, 0.406))
    std = pretrained_cfg.get("std", (0.229, 0.224, 0.225))
    bilinear = T.InterpolationMode.BILINEAR

    if dataset_name == "cifar100":
        interp = T.InterpolationMode.BICUBIC
        policy = T.AutoAugmentPolicy.CIFAR10
    else:
        interp = T.InterpolationMode.BILINEAR
        policy = T.AutoAugmentPolicy.IMAGENET

    if dataset_name == "cub200":
        return T.Compose([
            T.Resize(256, interpolation=interp, antialias=True),
            T.CenterCrop((224, 224)),
            T.AutoAugment(policy, bilinear),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    return T.Compose([
        T.AutoAugment(policy, bilinear),
        T.RandomResizedCrop((224, 224), interpolation=interp, antialias=True),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def build_eval_transform(dataset_name, pretrained_cfg):
    """Build evaluation transform (resize + center crop + normalize)."""
    mean = pretrained_cfg.get("mean", (0.485, 0.456, 0.406))
    std = pretrained_cfg.get("std", (0.229, 0.224, 0.225))

    if dataset_name == "cifar100":
        interp = T.InterpolationMode.BICUBIC
        return T.Compose([
            T.Resize((224, 224), antialias=True, interpolation=interp),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        interp = T.InterpolationMode.BILINEAR
        return T.Compose([
            T.Resize((256, 256), antialias=True, interpolation=interp),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
