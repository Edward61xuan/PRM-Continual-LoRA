#!/usr/bin/env python3
"""Prepare the four PRM datasets in a common train/val image-folder layout."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


MANIFESTS = {
    "imagenet_r_train.txt": "bd1536ce73d52328d697d983652ce02eb10c5c9fc085d4b2434c67c32b81ee2c",
    "imagenet_r_val.txt": "c6719b71b1181cba536a16d77a3815295eaf3b9c844f5db4452b9bb6dac85d2f",
    "sdomain_train.txt": "87671cd9a6309dc5e9ebfa8eab8c14abbc270d605c9430c7525ff8e9d7e7e74e",
    "sdomain_val.txt": "112d39337e19ea1b5b81f1a423145be007c5a42e16aeea84473a98c28462a8d2",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _verified_manifest(name: str) -> Path:
    path = Path(__file__).resolve().parent / "manifests" / name
    if not path.exists():
        raise FileNotFoundError(f"bundled split manifest missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MANIFESTS[name]:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {name}: {digest}")
    return path


def _place(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def _from_manifest(dataset: str, source: Path, output: Path, copy: bool) -> None:
    prefix = "imagenet_r" if dataset == "imagenet_r" else "sdomain"
    for split in ("train", "val"):
        manifest = _verified_manifest(f"{prefix}_{split}.txt")
        count = 0
        for raw in manifest.read_text().splitlines():
            fields = raw.strip().split()
            if not fields:
                continue
            rel = Path(fields[0])
            src = source / rel
            if not src.is_file():
                raise FileNotFoundError(f"manifest file missing: {src}")
            cls = fields[1].zfill(3) if len(fields) == 2 else rel.parent.name
            _place(src, output / split / cls / src.name, copy)
            count += 1
        print(f"[{dataset}] {split}: {count} images")


def _cifar100(output: Path) -> None:
    from torchvision.datasets import CIFAR100

    raw = output / "raw"
    for train, split in ((True, "train"), (False, "val")):
        ds = CIFAR100(raw, train=train, download=True)
        for index, (image, label) in enumerate(ds):
            dst = output / split / str(label) / f"{index:05d}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                image.save(dst)
        print(f"[cifar100] {split}: {len(ds)} images")


def _cub200(source: Path, output: Path, copy: bool) -> None:
    images = {}
    for line in (source / "images.txt").read_text().splitlines():
        idx, rel = line.split(maxsplit=1)
        images[int(idx)] = rel
    labels = {}
    for line in (source / "image_class_labels.txt").read_text().splitlines():
        idx, label = line.split()
        labels[int(idx)] = int(label) - 1
    splits = {}
    for line in (source / "train_test_split.txt").read_text().splitlines():
        idx, is_train = line.split()
        splits[int(idx)] = "train" if is_train == "1" else "val"
    for idx, rel in images.items():
        src = source / "images" / rel
        _place(src, output / splits[idx] / f"{labels[idx]:03d}" / src.name, copy)
    print(f"[cub200] prepared {len(images)} images using the official split")


def verify_tree(root: Path) -> dict[str, int]:
    counts = {}
    class_sets = []
    for split in ("train", "val"):
        split_root = root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"missing directory: {split_root}")
        classes = {p.name for p in split_root.iterdir() if p.is_dir()}
        if not classes:
            raise ValueError(f"no class directories in {split_root}")
        class_sets.append(classes)
        count = sum(
            1 for p in split_root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if count == 0:
            raise ValueError(f"no images in {split_root}")
        counts[split] = count
    if class_sets[0] != class_sets[1]:
        raise ValueError("train and val class directories differ")
    counts["classes"] = len(class_sets[0])
    print(f"[verify] classes={counts['classes']} train={counts['train']} val={counts['val']}")
    return counts


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("imagenet_r", "cifar100", "sdomainet", "cub200", "verify"))
    parser.add_argument("--source", type=Path, help="Extracted raw dataset root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="Copy rather than symlink images")
    args = parser.parse_args(argv)

    if args.dataset == "verify":
        verify_tree(args.output)
        return
    if args.dataset == "cifar100":
        _cifar100(args.output)
    else:
        if args.source is None:
            parser.error(f"{args.dataset} requires --source")
        if args.dataset in ("imagenet_r", "sdomainet"):
            _from_manifest(args.dataset, args.source, args.output, args.copy)
        else:
            _cub200(args.source, args.output, args.copy)
    verify_tree(args.output)


if __name__ == "__main__":
    main()
