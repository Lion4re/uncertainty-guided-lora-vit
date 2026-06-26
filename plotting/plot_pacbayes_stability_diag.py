#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    "total_loss",
    "cls_loss",
    "pace_loss",
    "pac_bayes_kl",
    "pac_bayes_sigma_mean",
    "pac_bayes_sigma_max",
    "ce_grad_norm",
    "pace_grad_norm",
    "grad_kl_ce_ratio",
]


def read_log(path):
    steps, values = [], []
    if not path.exists():
        return steps, values
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        step, value = line.split(":", 1)
        try:
            steps.append(int(step.strip()))
            values.append(float(value.strip()))
        except ValueError:
            continue
    return steps, values


def short_name(run_name):
    if run_name.startswith("pace_kl_pacbayes"):
        return "PACE-KL + PAC"
    if run_name.startswith("pace_pacbayes"):
        return "PACE + PAC"
    return run_name[:32]


def plot_metric(root, out_dir, metric):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        steps, values = read_log(run_dir / f"{metric}.log")
        if not values:
            continue
        ax.plot(steps, values, linewidth=1.5, label=short_name(run_dir.name))
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_title(metric.replace("_", " "))
    ax.set_xlabel("step")
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = out_dir / f"{metric}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_grid(root, out_dir):
    available = []
    for metric in METRICS:
        if any((run_dir / f"{metric}.log").exists() for run_dir in root.iterdir() if run_dir.is_dir()):
            available.append(metric)
    if not available:
        return None

    cols = 3
    rows = (len(available) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for ax, metric in zip(axes.ravel(), available):
        ax.axis("on")
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            steps, values = read_log(run_dir / f"{metric}.log")
            if not values:
                continue
            ax.plot(steps, values, linewidth=1.2, label=short_name(run_dir.name))
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("PAC-Bayes Diagnostic", fontsize=14)
    fig.tight_layout()
    out_path = out_dir / "pacbayes_diagnostic_grid.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def write_summary(root, out_dir):
    lines = ["# PAC-Bayes Diagnostic Summary", ""]
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        lines.append(f"## {short_name(run_dir.name)}")
        lines.append("")
        lines.append("| Metric | Start | End | Min | Max |")
        lines.append("|---|---:|---:|---:|---:|")
        for metric in METRICS:
            _, values = read_log(run_dir / f"{metric}.log")
            if not values:
                continue
            lines.append(
                f"| `{metric}` | {values[0]:.6g} | {values[-1]:.6g} | "
                f"{min(values):.6g} | {max(values):.6g} |"
            )
        lines.append("")
    out_path = out_dir / "pacbayes_diagnostic_summary.md"
    out_path.write_text("\n".join(lines))
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="outputs/pacbayes_stability_diag",
        help="Directory containing diagnostic run folders.",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/plots/pacbayes_stability_diag",
        help="Directory where plots and summary are saved.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for metric in METRICS:
        path = plot_metric(root, out_dir, metric)
        if path is not None:
            saved.append(path)
    grid = plot_grid(root, out_dir)
    if grid is not None:
        saved.append(grid)
    summary = write_summary(root, out_dir)
    saved.append(summary)

    for path in saved:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
