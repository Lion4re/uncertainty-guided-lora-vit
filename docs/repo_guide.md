# Repository Guide

The repository keeps the upstream PACE-style layout to avoid breaking reproducibility.

```text
train.py                         main training entrypoint
evaluate_all_metrics.py          single-model evaluation, calibration, shift/OOD, IVON MC
evaluate_two_model_ensemble.py   probability-averaging ensembles
aggregate_results.py             aggregates JSON result files
run_main_experiments.sh          configurable experiment runner
pace/                            adapters, consistency losses, PAC-Bayes, IVON, Bayesian-LoRA
utils/                           VTAB/few-shot data loading and utilities
runners/                         focused experiment runners
configs/                         shell-env configs for thesis runs and sweeps
plotting/                        reusable plotting scripts
results/                         curated lightweight thesis CSVs and figures only
docs/                            reproducibility and mapping documentation
tests/                           smoke/unit tests for core components
```

## Artifact Policy

Tracked:

- source code,
- runner scripts,
- reproducibility configs,
- documentation,
- lightweight CSV tables,
- selected thesis figures.

Not tracked:

- datasets,
- checkpoints,
- full raw result JSON directories,
- training/evaluation logs,
- transfer archives,
- exploratory notebooks or plots.

Use local ignored directories such as `outputs/`, `outs/`, `logs/`, and non-curated `results/` subfolders for regenerated artifacts.

## Checkpoints and Data

Place `ViT-B_16.npz` at the repository root. Place VTAB data under `data/`. Both are ignored by Git. See `checkpoints/README.md` and `data/README.md`.
