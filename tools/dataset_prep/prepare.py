#!/usr/bin/env python3
"""Prepare the four PRM datasets in a common train/val image-folder layout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_DATASETS = {
    "imagenet_r": {"classes": 200, "train": 24000, "val": 6000},
    "cifar100": {"classes": 100, "train": 50000, "val": 10000},
    "sdomainet": {"classes": 200, "train": 47532, "val": 20450},
    "cub200": {"classes": 200, "train": 5994, "val": 5794},
}


def _bundled_manifest(name: str) -> Path:
    path = Path(__file__).resolve().parent / "manifests" / name
    if not path.is_file():
        raise FileNotFoundError(f"bundled split manifest missing: {path}")

    entries = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError(f"bundled split manifest is empty: {path}")
    for line_number, fields in enumerate(entries, start=1):
        rel = Path(fields[0])
        if len(fields) not in (1, 2) or rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"invalid manifest entry at {path}:{line_number}")
        if len(fields) == 2 and not fields[1].isdigit():
            raise ValueError(f"invalid class label at {path}:{line_number}")
    return path


def _place(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        try:
            actual = dst.resolve(strict=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"broken output symlink: {dst}") from exc
        expected = src.resolve(strict=True)
        if actual != expected:
            raise RuntimeError(
                f"output symlink points to {actual}, expected {expected}: {dst}"
            )
        return
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def _from_manifest(dataset: str, source: Path, output: Path, copy: bool) -> None:
    prefix = "imagenet_r" if dataset == "imagenet_r" else "sdomain"
    for split in ("train", "val"):
        manifest = _bundled_manifest(f"{prefix}_{split}.txt")
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


def _cifar100(output: Path, source: Path | None, copy: bool) -> None:
    from torchvision.datasets import CIFAR100

    download = source is None
    if source is None:
        raw = output / "raw"
    elif source.is_file():
        if not source.name.endswith(".tar.gz"):
            raise ValueError("CIFAR-100 --source must be the official .tar.gz archive or extracted directory")
        raw = output / "raw"
        _place(source, raw / "cifar-100-python.tar.gz", copy)
        # torchvision's download path also performs archive extraction. With
        # the official archive already present, it validates and extracts it
        # without fetching it again.
        download = True
    elif source.is_dir():
        if source.name == "cifar-100-python":
            raw = source.parent
        elif (source / "cifar-100-python").is_dir():
            raw = source
        elif (source / "cifar-100-python.tar.gz").is_file():
            raw = source
            download = True
        else:
            raise ValueError(
                "CIFAR-100 --source directory must contain cifar-100-python/ "
                "or cifar-100-python.tar.gz"
            )
    else:
        raise FileNotFoundError(f"CIFAR-100 source not found: {source}")

    for train, split in ((True, "train"), (False, "val")):
        ds = CIFAR100(raw, train=train, download=download)
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


def _decode_images(paths: list[Path], mode: str, samples: int) -> None:
    if mode == "none":
        return
    from PIL import Image

    selected = paths if mode == "all" else paths[:samples]
    for path in selected:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"image cannot be decoded: {path}") from exc


def verify_tree(
    root: Path,
    dataset: str | None = None,
    decode_images: str = "none",
    decode_samples: int = 32,
) -> dict[str, int]:
    """Validate a prepared tree without changing it."""
    counts = {}
    class_sets = []
    resolved_by_split = {}
    for split in ("train", "val"):
        split_root = root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"missing directory: {split_root}")
        classes = {p.name for p in split_root.iterdir() if p.is_dir()}
        if not classes:
            raise ValueError(f"no class directories in {split_root}")
        class_sets.append(classes)

        broken = [p for p in split_root.rglob("*") if p.is_symlink() and not p.exists()]
        if broken:
            preview = ", ".join(map(str, broken[:3]))
            raise ValueError(f"broken symlink(s) under {split_root}: {preview}")

        paths = sorted(
            (p for p in split_root.rglob("*")
             if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda p: str(p),
        )
        empty_classes = [name for name in classes if not any(
            p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            for p in (split_root / name).iterdir()
        )]
        if empty_classes:
            raise ValueError(f"empty class directories in {split_root}: {empty_classes[:3]}")

        count = len(paths)
        if count == 0:
            raise ValueError(f"no images in {split_root}")
        counts[split] = count
        resolved_by_split[split] = {p.resolve(strict=True) for p in paths}
        _decode_images(paths, decode_images, decode_samples)

    if class_sets[0] != class_sets[1]:
        raise ValueError("train and val class directories differ")
    counts["classes"] = len(class_sets[0])

    overlap = resolved_by_split["train"] & resolved_by_split["val"]
    if overlap:
        preview = ", ".join(map(str, sorted(overlap, key=str)[:3]))
        raise ValueError(
            f"train/val resolved-path overlap ({len(overlap)} images): {preview}"
        )

    if dataset is not None:
        expected = EXPECTED_DATASETS[dataset]
        mismatches = {
            key: (counts[key], expected[key])
            for key in ("classes", "train", "val")
            if counts[key] != expected[key]
        }
        if mismatches:
            detail = ", ".join(
                f"{key}={actual} (expected {wanted})"
                for key, (actual, wanted) in mismatches.items()
            )
            raise ValueError(f"{dataset} count mismatch: {detail}")

    print(f"[verify] classes={counts['classes']} train={counts['train']} val={counts['val']}")
    return counts


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("imagenet_r", "cifar100", "sdomainet", "cub200", "verify"))
    parser.add_argument("--source", type=Path, help="Extracted raw dataset root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="Copy rather than symlink images")
    parser.add_argument("--expected-dataset", choices=tuple(EXPECTED_DATASETS),
                        help="For 'verify', require the paper dataset's exact counts")
    parser.add_argument("--decode-images", choices=("none", "sample", "all"), default="none",
                        help="Optionally verify that a sample or every image decodes")
    parser.add_argument("--decode-samples", type=int, default=32,
                        help="Images per split checked by --decode-images sample")
    args = parser.parse_args(argv)

    if args.dataset == "verify":
        verify_tree(args.output, args.expected_dataset, args.decode_images, args.decode_samples)
        return
    if args.dataset == "cifar100":
        _cifar100(args.output, args.source, args.copy)
    else:
        if args.source is None:
            parser.error(f"{args.dataset} requires --source")
        if args.dataset in ("imagenet_r", "sdomainet"):
            _from_manifest(args.dataset, args.source, args.output, args.copy)
        else:
            _cub200(args.source, args.output, args.copy)
    verify_tree(args.output, args.dataset, args.decode_images, args.decode_samples)


if __name__ == "__main__":
    main()
