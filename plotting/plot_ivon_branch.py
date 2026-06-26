#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(root):
    rows = []
    root = Path(root)
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*.json")):
        with open(path, "r") as f:
            result = json.load(f)
        name = path.name
        text = str(path)
        rows.append({
            "path": str(path),
            "name": name,
            "mode": "mc10" if (result.get("ivon_mc_samples", 0) or "MCIVON10" in name) else "mean",
            "lambda": parse_float(r"Lbd([0-9.]+)", text),
            "ess": parse_ess(text, result),
            "accuracy": result.get("accuracy"),
            "ece": result.get("ece"),
            "nll": result.get("nll"),
            "brier": result.get("brier"),
            "avg_confidence": result.get("avg_confidence"),
            "confidence_gap": (
                result.get("accuracy") - result.get("avg_confidence")
                if result.get("accuracy") is not None and result.get("avg_confidence") is not None
                else None
            ),
            "mean_logit_norm": result.get("mean_logit_norm"),
            "posthoc_temperature": result.get("posthoc_temperature"),
            "ts_ece": (result.get("posthoc_temperature_scaled") or {}).get("ece"),
            "ts_nll": (result.get("posthoc_temperature_scaled") or {}).get("nll"),
            "ts_brier": (result.get("posthoc_temperature_scaled") or {}).get("brier"),
            "ts_avg_confidence": (result.get("posthoc_temperature_scaled") or {}).get("avg_confidence"),
        })
    return rows


def parse_float(pattern, text):
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_ess(text, result):
    match = re.search(r"Ess([0-9.eE+\-]+)", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    if result.get("ivon_ess") is not None:
        return float(result["ivon_ess"])
    return None


def sort_key(row, key):
    val = row.get(key)
    return float("inf") if val is None or (isinstance(val, float) and math.isnan(val)) else val


def filter_rows(rows, mode="mean", key=None):
    selected = [r for r in rows if r["mode"] == mode]
    if key:
        selected = [r for r in selected if r.get(key) is not None]
        selected.sort(key=lambda r: sort_key(r, key))
    return selected


def save_csv(rows, out_path):
    cols = [
        "path", "mode", "lambda", "ess", "accuracy", "ece", "nll", "brier",
        "avg_confidence", "confidence_gap", "mean_logit_norm",
        "posthoc_temperature", "ts_ece", "ts_nll", "ts_brier", "ts_avg_confidence",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in cols})


def fmt(x):
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def save_markdown(lambda_rows, ess_rows, out_path):
    def table(title, rows, sweep_key):
        lines = [f"## {title}", ""]
        lines.append("| mode | " + sweep_key + " | acc | ECE | NLL | Brier | avg conf | acc-conf gap | logit norm |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                f"| {row['mode']} | {fmt(row.get(sweep_key))} | {fmt(row['accuracy'])} | "
                f"{fmt(row['ece'])} | {fmt(row['nll'])} | {fmt(row['brier'])} | "
                f"{fmt(row['avg_confidence'])} | {fmt(row['confidence_gap'])} | "
                f"{fmt(row['mean_logit_norm'])} |"
            )
        return lines

    lines = ["# PACE + IVON-LoRA Branch Summary", ""]
    lines += table("Lambda Sweep", lambda_rows, "lambda")
    lines += [""] + table("ESS Sweep", ess_rows, "ess")
    lines += [
        "",
        "## Reading",
        "",
        "- The lambda sweep tests whether reducing raw-logit PACE consistency repairs underconfidence.",
        "- The ESS sweep tests whether IVON posterior temperature makes MC10 better than posterior-mean inference.",
        "- Positive Bayesian effect would mean MC10 improves ECE/NLL/Brier without a meaningful accuracy drop.",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def savefig(out_dir, name):
    path = Path(out_dir) / name
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"saved {path}")


def plot_lambda_main(rows, out_dir):
    rows = filter_rows(rows, mode="mean", key="lambda")
    xs = [r["lambda"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(xs, [r["ece"] for r in rows], marker="o", label="ECE", color="#1f77b4")
    ax1.plot(xs, [r["nll"] for r in rows], marker="s", label="NLL", color="#d62728")
    ax1.set_xlabel("PACE lambda")
    ax1.set_ylabel("ECE / NLL")
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(xs, [r["avg_confidence"] for r in rows], marker="^", label="Avg confidence", color="#2ca02c")
    ax2.set_ylabel("Avg confidence")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.set_title("PACE + IVON-LoRA: lambda controls calibration and confidence")
    savefig(out_dir, "ivon_lambda_ece_nll_confidence.png")


def plot_lambda_conf_gap(rows, out_dir):
    rows = filter_rows(rows, mode="mean", key="lambda")
    xs = [r["lambda"] for r in rows]
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(xs, [r["confidence_gap"] for r in rows], marker="o", color="#9467bd")
    plt.gca().invert_xaxis()
    plt.xlabel("PACE lambda")
    plt.ylabel("Accuracy - avg confidence")
    plt.title("Confidence gap shrinks when PACE lambda is reduced")
    plt.grid(True, alpha=0.25)
    savefig(out_dir, "ivon_lambda_confidence_gap.png")


def plot_lambda_logit_norm(rows, out_dir):
    rows = filter_rows(rows, mode="mean", key="lambda")
    rows = [r for r in rows if r.get("mean_logit_norm") is not None]
    if not rows:
        print("skip ivon_lambda_logit_norm.png: no mean_logit_norm values")
        return
    xs = [r["lambda"] for r in rows]
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(xs, [r["mean_logit_norm"] for r in rows], marker="o", color="#ff7f0e")
    plt.gca().invert_xaxis()
    plt.xlabel("PACE lambda")
    plt.ylabel("Mean logit norm")
    plt.title("Logit norm under PACE + IVON-LoRA lambda sweep")
    plt.grid(True, alpha=0.25)
    savefig(out_dir, "ivon_lambda_logit_norm.png")


def paired_by_key(rows, key):
    groups = {}
    for row in rows:
        if row.get(key) is None:
            continue
        groups.setdefault(row[key], {})[row["mode"]] = row
    return [(k, v.get("mean"), v.get("mc10")) for k, v in sorted(groups.items())]


def plot_mean_vs_mc(rows, key, out_dir, prefix):
    pairs = paired_by_key(rows, key)
    pairs = [(k, mean, mc) for k, mean, mc in pairs if mean and mc]
    if not pairs:
        print(f"skip {prefix}_mean_vs_mc10_ece_nll.png: no paired rows")
        return
    xs = [p[0] for p in pairs]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].plot(xs, [p[1]["ece"] for p in pairs], marker="o", label="Mean", color="#1f77b4")
    axes[0].plot(xs, [p[2]["ece"] for p in pairs], marker="s", label="MC10", color="#ff7f0e")
    axes[0].set_ylabel("ECE")
    axes[0].set_title("ECE")
    axes[1].plot(xs, [p[1]["nll"] for p in pairs], marker="o", label="Mean", color="#1f77b4")
    axes[1].plot(xs, [p[2]["nll"] for p in pairs], marker="s", label="MC10", color="#ff7f0e")
    axes[1].set_ylabel("NLL")
    axes[1].set_title("NLL")
    for ax in axes:
        ax.set_xlabel("PACE lambda" if key == "lambda" else "IVON ESS")
        ax.grid(True, alpha=0.25)
        ax.legend()
        if key == "ess":
            ax.set_xscale("log")
    fig.suptitle("Mean vs MC10: posterior averaging effect")
    savefig(out_dir, f"{prefix}_mean_vs_mc10_ece_nll.png")


def plot_ess_conf_gap(rows, out_dir):
    rows = filter_rows(rows, mode="mean", key="ess")
    plt.figure(figsize=(7.0, 4.2))
    plt.plot([r["ess"] for r in rows], [r["confidence_gap"] for r in rows], marker="o", color="#9467bd")
    plt.xscale("log")
    plt.xlabel("IVON ESS")
    plt.ylabel("Accuracy - avg confidence")
    plt.title("Confidence gap across IVON ESS values")
    plt.grid(True, alpha=0.25)
    savefig(out_dir, "ivon_ess_confidence_gap.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_root", default="results/pace_ivon_lbd_sweep")
    parser.add_argument("--ess_root", default="results/pace_ivon_lbd01_ess_sweep")
    parser.add_argument("--out_dir", default="plots_ivon_branch")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lambda_rows = load_results(args.lambda_root)
    ess_rows = load_results(args.ess_root)
    all_rows = lambda_rows + ess_rows

    save_csv(lambda_rows, out_dir / "ivon_lambda_sweep_table.csv")
    save_csv(ess_rows, out_dir / "ivon_ess_sweep_table.csv")
    save_csv(all_rows, out_dir / "ivon_all_results_table.csv")
    save_markdown(
        sorted(lambda_rows, key=lambda r: (sort_key(r, "lambda"), r["mode"])),
        sorted(ess_rows, key=lambda r: (sort_key(r, "ess"), r["mode"])),
        out_dir / "ivon_branch_summary.md",
    )

    plot_lambda_main(lambda_rows, out_dir)
    plot_lambda_conf_gap(lambda_rows, out_dir)
    plot_lambda_logit_norm(lambda_rows, out_dir)
    plot_mean_vs_mc(lambda_rows, "lambda", out_dir, "ivon_lambda")
    plot_mean_vs_mc(ess_rows, "ess", out_dir, "ivon_ess")
    plot_ess_conf_gap(ess_rows, out_dir)

    print(f"Saved IVON branch plots/tables to {out_dir}")


if __name__ == "__main__":
    main()

