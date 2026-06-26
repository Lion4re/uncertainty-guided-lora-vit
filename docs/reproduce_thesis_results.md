# Reproducing Thesis Results

Run commands from the repository root. The scripts use shell environment variables rather than a custom config parser.

## Setup

```bash
pip install -r requirements.txt
```

Place the pretrained ViT checkpoint at:

```text
ViT-B_16.npz
```

Place VTAB datasets under `data/` as described in `data/README.md`.

## Main Primary Runs

```bash
set -a
source configs/thesis_runs/main_primary.env
set +a
bash run_main_experiments.sh
```

This runs the main LoRA, PACE, PACE-KL, margin, PAC-Bayes, and KL+PAC-Bayes rows over CIFAR-100, Caltech-101, and DTD.

## Uncertainty-Guided Runs

```bash
set -a
source configs/thesis_runs/uncertainty_guided_main.env
set +a
bash runners/run_uncertainty_guided_experiments.sh
```

This includes clean, controlled-shift, and OOD-proxy evaluation for the uncertainty-guided soft/top-k branches.

## Single-Model Held-Out Temperature Scaling

```bash
set -a
source configs/thesis_runs/single_model_tssplit.env
set +a
bash runners/run_single_model_tssplit_main_june26.sh
```

This re-evaluates existing checkpoints, fits temperature on a disjoint calibration split, and reports raw/temperature-scaled metrics on the held-out split.

## Extra Dataset Breadth Runs

```bash
set -a
source configs/thesis_runs/extra_datasets.env
set +a
bash run_main_experiments.sh
```

This covers SVHN, Oxford Flowers-102, and Oxford-IIIT Pet for the selected methods.

## Ensemble Evaluation

Use `evaluate_two_model_ensemble.py` once the relevant checkpoints exist:

```bash
python evaluate_two_model_ensemble.py \
  --checkpoint_a outputs/pace_pacbayes/weight.pt \
  --label_a PACE_PACBayes \
  --pacbayes_a \
  --checkpoint_b outputs/pace_kl_uncert_topk/weight.pt \
  --label_b PACEKL_uncert_topk \
  --dataset cifar \
  --adapter LoRAmul_VPTadd \
  --rank 10 \
  --sigma 1.2 \
  --posthoc_temp_scaling \
  --temperature_protocol split \
  --save_dir results/ensemble_topk \
  --plot_dir outputs/plots/ensemble_topk
```

## Sweeps

PACE/PACE-KL lambda sweep:

```bash
set -a
source configs/sweeps/pace_lambda.env
set +a
bash runners/run_pace_lambda_sweep.sh
```

IVON lambda and ESS sweeps:

```bash
set -a
source configs/sweeps/pace_ivon_lbd.env
set +a
bash runners/run_pace_ivon_lbd_sweep.sh

set -a
source configs/sweeps/pace_ivon_ess.env
set +a
bash runners/run_pace_ivon_lbd01_ess_sweep.sh
```

## Aggregation and Curated Artifacts

Aggregate a result directory:

```bash
python aggregate_results.py --results_dir results/main_primary
```

List the curated thesis CSVs and figures:

```bash
bash scripts/export_thesis_tables.sh
```

Verify the expected curated artifacts are present:

```bash
bash scripts/check_artifact_inventory.sh
```
