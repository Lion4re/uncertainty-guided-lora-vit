"""
Plot aggregate experiment results from evaluate_all_metrics.py JSON files.

Examples:
  python plotting/plot_main_results.py --results_dir results/main_3seed_vtab
  python plotting/plot_main_results.py --results_dir results/main_3seed_vtab --out_dir outputs/plots/main_results
"""
import argparse
import json
import os
import re
from collections import defaultdict
from glob import glob
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "LoRAmul_VPTadd_R10": "Baseline",
    "pace_Lbd1_S1.2_LoRAmul_VPTadd_R10": "PACE-MSE",
    "pace_kl_Lbd0.5_S1.2_LoRAmul_VPTadd_R10": "PACE-KL T=2",
    "pace_kl_Lbd0.5_S1.2_T1_LoRAmul_VPTadd_R10": "PACE-KL T=1",
    "pace_kl_margin_Lbd0.5_S1.2_T1_LoRAmul_VPTadd_R10": "PACE-KL+Margin",
    "pace_kl_flat_Lbd0.5_S1.2_T1_LoRAmul_VPTadd_R10": "Flat-LoRA+KL",
    "pace_kl_jacobian_Lbd0.5_S1.2_T1_LoRAmul_VPTadd_R10": "PACE-KL+Jacobian",
}

METHOD_ORDER = [
    "LoRAmul_VPTadd_R10",
    "pace_Lbd1_S1.2_LoRAmul_VPTadd_R10",
    "pace_kl_Lbd0.5_S1.2_T1_LoRAmul_VPTadd_R10",
    "pace_kl_Lbd0.5_S1.2_LoRAmul_VPTadd_R10",
    "pace_kl_margin_Lbd0.5_S1.2_T1_LoRAmul_VPTadd_R10",
    "pace_kl_flat_Lbd0.5_S1.2_T1_LoRAmul_VPTadd_R10",
    "pace_kl_jacobian_Lbd0.5_S1.2_T1_LoRAmul_VPTadd_R10",
]

METRICS = [
    ("accuracy", "Accuracy", True),
    ("ece", "ECE", False),
    ("nll", "NLL", False),
    ("brier", "Brier", False),
    ("avg_confidence", "Avg. Confidence", True),
]


def method_key(filename, dataset):
    stem = os.path.basename(filename).removesuffix(".json")
    suffix = f"_{dataset}"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return re.sub(r"_Seed\d+", "", stem)


def collect(results_dir):
    groups = defaultdict(list)
    for path in sorted(glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            result = json.load(f)
        dataset = result.get("dataset", "unknown")
        groups[(dataset, method_key(path, dataset))].append(result)
    return groups


def metric_values(results, metric):
    values = []
    for result in results:
        value = result.get(metric)
        if value is not None:
            values.append(value)
    return values


def ts_metric_values(results, metric):
    values = []
    for result in results:
        posthoc = result.get("posthoc_temperature_scaled") or {}
        value = posthoc.get(metric)
        if value is not None:
            values.append(value)
    return values


def mean_std(values):
    if not values:
        return np.nan, 0.0
    return mean(values), pstdev(values)


def available_methods(groups, dataset):
    methods = {method for ds, method in groups if ds == dataset}
    ordered = [m for m in METHOD_ORDER if m in methods]
    ordered += sorted(methods - set(ordered))
    return ordered


def plot_metric_bars(groups, out_dir):
    datasets = sorted({ds for ds, _ in groups})
    for metric, label, _higher_is_better in METRICS:
        fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 4.2), sharey=False)
        if len(datasets) == 1:
            axes = [axes]
        for ax, dataset in zip(axes, datasets):
            methods = available_methods(groups, dataset)
            vals, errs, labels = [], [], []
            for method in methods:
                v, e = mean_std(metric_values(groups[(dataset, method)], metric))
                vals.append(v)
                errs.append(e)
                labels.append(METHOD_LABELS.get(method, method))
            x = np.arange(len(methods))
            ax.bar(x, vals, yerr=errs, capsize=3)
            ax.set_title(dataset)
            ax.set_ylabel(label)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"{label} Across Datasets")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"metric_{metric}.png"), dpi=200)
        plt.close(fig)


def plot_pre_post_temperature(groups, out_dir):
    datasets = sorted({ds for ds, _ in groups})
    for metric, label in [("ece", "ECE"), ("nll", "NLL"), ("brier", "Brier")]:
        fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 4.2), sharey=False)
        if len(datasets) == 1:
            axes = [axes]
        for ax, dataset in zip(axes, datasets):
            methods = available_methods(groups, dataset)
            pre, pre_err, post, post_err, labels = [], [], [], [], []
            for method in methods:
                v, e = mean_std(metric_values(groups[(dataset, method)], metric))
                tv, te = mean_std(ts_metric_values(groups[(dataset, method)], metric))
                pre.append(v)
                pre_err.append(e)
                post.append(tv)
                post_err.append(te)
                labels.append(METHOD_LABELS.get(method, method))
            x = np.arange(len(methods))
            width = 0.38
            ax.bar(x - width / 2, pre, width, yerr=pre_err, capsize=3, label="Before TS")
            ax.bar(x + width / 2, post, width, yerr=post_err, capsize=3, label="After TS")
            ax.set_title(dataset)
            ax.set_ylabel(label)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
        fig.suptitle(f"Post-hoc Temperature Scaling: {label}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"temperature_scaling_{metric}.png"), dpi=200)
        plt.close(fig)


def plot_accuracy_ece(groups, out_dir):
    datasets = sorted({ds for ds, _ in groups})
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 4.2), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        for method in available_methods(groups, dataset):
            acc, acc_err = mean_std(metric_values(groups[(dataset, method)], "accuracy"))
            ece, ece_err = mean_std(metric_values(groups[(dataset, method)], "ece"))
            ax.errorbar(acc, ece, xerr=acc_err, yerr=ece_err, marker="o", capsize=3, linestyle="none")
            ax.annotate(METHOD_LABELS.get(method, method), (acc, ece), xytext=(5, 3),
                        textcoords="offset points", fontsize=8)
        ax.set_title(dataset)
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("ECE")
        ax.grid(alpha=0.25)
    fig.suptitle("Accuracy vs Calibration Error")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_vs_ece.png"), dpi=200)
    plt.close(fig)


def plot_confidence_gap(groups, out_dir):
    datasets = sorted({ds for ds, _ in groups})
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 4.2), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        methods = available_methods(groups, dataset)
        vals, errs, labels = [], [], []
        for method in methods:
            gaps = []
            for result in groups[(dataset, method)]:
                if result.get("avg_confidence") is not None and result.get("accuracy") is not None:
                    gaps.append(result["avg_confidence"] - result["accuracy"])
            v, e = mean_std(gaps)
            vals.append(v)
            errs.append(e)
            labels.append(METHOD_LABELS.get(method, method))
        x = np.arange(len(methods))
        ax.axhline(0, color="black", linewidth=1)
        ax.bar(x, vals, yerr=errs, capsize=3)
        ax.set_title(dataset)
        ax.set_ylabel("Confidence - Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Overconfidence / Underconfidence Gap")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confidence_gap.png"), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/main_3seed_vtab")
    parser.add_argument("--out_dir", default="outputs/plots/main_results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    groups = collect(args.results_dir)
    if not groups:
        raise SystemExit(f"No JSON result files found in {args.results_dir}")

    plot_metric_bars(groups, args.out_dir)
    plot_pre_post_temperature(groups, args.out_dir)
    plot_accuracy_ece(groups, args.out_dir)
    plot_confidence_gap(groups, args.out_dir)
    print(f"Saved plots to {args.out_dir}")


if __name__ == "__main__":
    main()
