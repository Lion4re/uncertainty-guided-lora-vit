#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_item(item):
    if "=" not in item:
        path = item
        label = Path(path).stem
    else:
        label, path = item.split("=", 1)
    return label.strip(), path.strip()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_reliability(items, out_path, title):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Perfect calibration")
    centers = np.linspace(1 / 30, 1 - 1 / 30, 15)

    for label, path in items:
        result = load_json(path)
        acc = np.asarray(result.get("bin_accs", []), dtype=float)
        conf = np.asarray(result.get("bin_confs", []), dtype=float)
        counts = np.asarray(result.get("bin_counts", []), dtype=float)
        if len(acc) == 0 or len(conf) == 0:
            print(f"skip {path}: missing bin_accs/bin_confs")
            continue
        active = counts > 0
        full_label = (
            f"{label} "
            f"(acc={result.get('accuracy', float('nan')):.3f}, "
            f"ECE={result.get('ece', float('nan')):.3f}, "
            f"NLL={result.get('nll', float('nan')):.3f})"
        )
        ax.plot(conf[active], acc[active], marker="o", linewidth=1.8, label=full_label)
        ax.scatter(centers[active], acc[active], s=np.maximum(counts[active], 1) / np.maximum(counts.max(), 1) * 120,
                   alpha=0.18)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--item",
        action="append",
        required=True,
        help="Reliability item as label=/path/to/result.json. Can be repeated.",
    )
    parser.add_argument("--out", default="plots_dtd_reliability/reliability_dtd_methods.png")
    parser.add_argument("--title", default="DTD reliability: PAC-Bayes vs PACE-KL uncertainty variants")
    args = parser.parse_args()
    items = [parse_item(item) for item in args.item]
    plot_reliability(items, args.out, args.title)


if __name__ == "__main__":
    main()

