"""
Rebuild local VTAB-style Caltech101 and DTD folders with class-balanced splits.

The original helper in this workspace wrote the first N samples from torchvision
datasets. Caltech101 and DTD are class-ordered, so sequential splits can contain
only the first few classes. This script samples stratified train/val splits and
rewrites the image folders plus split text files.
"""
import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import torchvision


def allocate_counts(class_to_items, total, rng, min_per_class=1):
    classes = sorted(class_to_items)
    available = {c: len(class_to_items[c]) for c in classes}
    counts = {c: 0 for c in classes}

    eligible = [c for c in classes if available[c] >= min_per_class]
    if total >= len(eligible) * min_per_class:
        for c in eligible:
            counts[c] = min_per_class

    remaining = total - sum(counts.values())
    while remaining > 0:
        candidates = [c for c in classes if counts[c] < available[c]]
        if not candidates:
            break
        weights = [available[c] - counts[c] for c in candidates]
        chosen = rng.choices(candidates, weights=weights, k=1)[0]
        counts[chosen] += 1
        remaining -= 1

    return counts


def reset_split_dir(base, split):
    out_dir = base / "images" / split
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_split(base, split, samples):
    out_dir = reset_split_dir(base, split)
    lines = []
    for i, (ds, idx, label) in enumerate(samples):
        img, _ = ds[idx]
        if img.mode != "RGB":
            img = img.convert("RGB")
        filename = f"{i:06d}.jpg"
        img.resize((224, 224)).save(out_dir / filename)
        img.close()
        lines.append(f"images/{split}/{filename} {label}\n")
    with (base / f"{split}.txt").open("w") as f:
        f.writelines(lines)
    print(f"{base.name}/{split}: {len(samples)} images, {len(set(label for _, _, label in samples))} classes")


def stratified_take(class_to_items, counts, rng):
    taken = []
    remaining = {}
    for c, items in class_to_items.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        n = counts.get(c, 0)
        taken.extend(shuffled[:n])
        remaining[c] = shuffled[n:]
    rng.shuffle(taken)
    return taken, remaining


def rebuild_caltech101(data_root, source_root, seed):
    rng = random.Random(seed)
    ds = torchvision.datasets.Caltech101(root=source_root, download=True)
    class_to_items = defaultdict(list)
    for idx in range(len(ds)):
        img, label = ds[idx]
        img.close()
        class_to_items[int(label)].append((ds, idx, int(label)))

    base = data_root / "caltech101"
    base.mkdir(parents=True, exist_ok=True)

    trainval_counts = allocate_counts(class_to_items, total=1000, rng=rng, min_per_class=1)
    trainval, remaining_by_class = stratified_take(class_to_items, trainval_counts, rng)

    trainval_by_class = defaultdict(list)
    for sample in trainval:
        trainval_by_class[sample[1]].append(sample)

    train_counts = allocate_counts(trainval_by_class, total=800, rng=rng, min_per_class=1)
    train, val_by_class = stratified_take(trainval_by_class, train_counts, rng)
    val = [sample for items in val_by_class.values() for sample in items]
    test = [sample for items in remaining_by_class.values() for sample in items]

    rng.shuffle(val)
    rng.shuffle(test)
    save_split(base, "train800", train)
    save_split(base, "val200", val)
    save_split(base, "train800val200", train + val)
    save_split(base, "test", test)


def rebuild_dtd(data_root, source_root, seed):
    rng = random.Random(seed)
    train_ds = torchvision.datasets.DTD(root=source_root, split="train", download=True)
    test_ds = torchvision.datasets.DTD(root=source_root, split="test", download=True)

    class_to_items = defaultdict(list)
    for idx in range(len(train_ds)):
        img, label = train_ds[idx]
        img.close()
        class_to_items[int(label)].append((train_ds, idx, int(label)))

    base = data_root / "dtd"
    base.mkdir(parents=True, exist_ok=True)

    train_counts = allocate_counts(class_to_items, total=800, rng=rng, min_per_class=1)
    train, remaining_by_class = stratified_take(class_to_items, train_counts, rng)

    val_counts = allocate_counts(remaining_by_class, total=200, rng=rng, min_per_class=1)
    val, _ = stratified_take(remaining_by_class, val_counts, rng)

    test = []
    for idx in range(len(test_ds)):
        img, label = test_ds[idx]
        img.close()
        test.append((test_ds, idx, int(label)))
    rng.shuffle(test)

    save_split(base, "train800", train)
    save_split(base, "val200", val)
    save_split(base, "train800val200", train + val)
    save_split(base, "test", test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["caltech101", "dtd"],
                        choices=["caltech101", "dtd"])
    parser.add_argument("--data_root", default="data/vtab-1k")
    parser.add_argument("--source_root", default="/tmp")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    source_root = Path(args.source_root)
    for dataset in args.datasets:
        if dataset == "caltech101":
            rebuild_caltech101(data_root, str(source_root / "caltech101"), args.seed)
        elif dataset == "dtd":
            rebuild_dtd(data_root, str(source_root / "dtd"), args.seed)


if __name__ == "__main__":
    main()
