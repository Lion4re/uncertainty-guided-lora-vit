import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "pace": "PACE-MSE",
    "pace_kl": "PACE-KL",
}

METRICS = [
    ("accuracy", "Accuracy", True),
    ("avg_confidence", "Average confidence", True),
    ("ece", "ECE", False),
    ("mean_logit_norm", "Mean logit norm", True),
]


def parse_result(path):
    name = path.name
    match = re.search(r"(pace(?:_kl)?)_Lbd([0-9.]+)_S", name)
    if not match:
        return None
    method = match.group(1)
    lbd = float(match.group(2))
    if method not in METHOD_LABELS:
        return None

    data = json.loads(path.read_text())
    dataset = data.get("dataset")
    if not dataset:
        for candidate in ["cifar", "caltech101", "dtd", "svhn", "oxford_flowers102", "oxford_iiit_pet"]:
            if f"_{candidate}_{candidate}" in name:
                dataset = candidate
                break
    if not dataset:
        return None

    seed_match = re.search(r"_Seed([0-9]+)_", name)
    seed = int(seed_match.group(1)) if seed_match else 42

    row = {
        "path": str(path),
        "dataset": dataset,
        "method": method,
        "lambda": lbd,
        "seed": seed,
    }
    for key, _, _ in METRICS:
        row[key] = data.get(key)
    return row


def collect(results_dir):
    rows = []
    for path in Path(results_dir).rglob("*.json"):
        row = parse_result(path)
        if row is not None:
            rows.append(row)
    return rows


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"], row["lambda"])].append(row)

    summary = []
    for (dataset, method, lbd), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        out = {
            "dataset": dataset,
            "method": method,
            "lambda": lbd,
            "n": len(group),
        }
        for key, _, _ in METRICS:
            avg, std = mean_std([row.get(key) for row in group])
            out[f"{key}_mean"] = avg
            out[f"{key}_std"] = std
        summary.append(out)
    return summary


def save_csv(summary, out_path):
    fieldnames = ["dataset", "method", "lambda", "n"]
    for key, _, _ in METRICS:
        fieldnames.extend([f"{key}_mean", f"{key}_std"])
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def plot_dataset(summary, dataset, out_dir):
    rows = [row for row in summary if row["dataset"] == dataset]
    if not rows:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)
    axes = axes.ravel()

    colors = {
        "pace": "#4C78A8",
        "pace_kl": "#E45756",
    }

    for ax, (metric, ylabel, as_percent) in zip(axes, METRICS):
        for method in ["pace", "pace_kl"]:
            selected = [row for row in rows if row["method"] == method and row.get(f"{metric}_mean") is not None]
            selected.sort(key=lambda row: row["lambda"])
            if not selected:
                continue
            xs = [row["lambda"] for row in selected]
            means = [row[f"{metric}_mean"] for row in selected]
            stds = [row[f"{metric}_std"] or 0.0 for row in selected]
            if as_percent and metric != "mean_logit_norm":
                means = [100.0 * value for value in means]
                stds = [100.0 * value for value in stds]
            ax.plot(xs, means, marker="o", linewidth=1.8, color=colors[method], label=METHOD_LABELS[method])
            if any(std > 0 for std in stds):
                lower = np.asarray(means) - np.asarray(stds)
                upper = np.asarray(means) + np.asarray(stds)
                ax.fill_between(xs, lower, upper, color=colors[method], alpha=0.14, linewidth=0)
        ax.set_xscale("log")
        ax.set_xlabel("lambda")
        ax.set_ylabel(ylabel + (" (%)" if as_percent and metric != "mean_logit_norm" else ""))
        ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    out_path = Path(out_dir) / f"pace_lambda_sweep_{dataset}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--out_dir", default="outputs/plots/pace_lambda_sweep")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect(args.results_dir)
    summary = summarize(rows)
    save_csv(summary, out_dir / "pace_lambda_sweep.csv")

    datasets = sorted({row["dataset"] for row in summary})
    for dataset in datasets:
        out_path = plot_dataset(summary, dataset, out_dir)
        if out_path:
            print(out_path)


if __name__ == "__main__":
    main()
