"""
Aggregate evaluation JSON files into mean/std tables grouped by method prefix.

Usage:
  python aggregate_results.py --results_dir results/main_3seed_vtab
"""
import json
import os
import re
from argparse import ArgumentParser
from glob import glob
from statistics import mean, pstdev


METRICS = [
    "accuracy",
    "ece",
    "nll",
    "brier",
    "avg_confidence",
    "posthoc_temperature",
]

POSTHOC_METRICS = [
    ("ts_accuracy", "accuracy"),
    ("ts_ece", "ece"),
    ("ts_nll", "nll"),
    ("ts_brier", "brier"),
    ("ts_avg_confidence", "avg_confidence"),
]


def method_key(filename, dataset):
    stem = filename.removesuffix(".json")
    suffix = f"_{dataset}"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    stem = re.sub(r"_Seed\d+", "", stem)
    return stem


def fmt(values):
    if not values:
        return "---"
    return f"{mean(values):.4f} +/- {pstdev(values):.4f}"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_dir", default="results/main_3seed_vtab")
    args = parser.parse_args()

    groups = {}
    for path in sorted(glob(os.path.join(args.results_dir, "*.json"))):
        with open(path) as f:
            result = json.load(f)
        dataset = result.get("dataset", "unknown")
        key = (dataset, method_key(os.path.basename(path), dataset))
        groups.setdefault(key, []).append(result)

    for (dataset, method), results in sorted(groups.items()):
        print(f"\n{dataset} | {method} | n={len(results)}")
        for metric in METRICS:
            values = [r[metric] for r in results if r.get(metric) is not None]
            print(f"  {metric}: {fmt(values)}")
        for out_name, inner_name in POSTHOC_METRICS:
            values = []
            for r in results:
                posthoc = r.get("posthoc_temperature_scaled")
                if posthoc and posthoc.get(inner_name) is not None:
                    values.append(posthoc[inner_name])
            print(f"  {out_name}: {fmt(values)}")
