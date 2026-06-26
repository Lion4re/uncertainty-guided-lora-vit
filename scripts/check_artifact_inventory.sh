#!/usr/bin/env bash
set -euo pipefail

required_files=(
  "results/tables/main_results_long.csv"
  "results/tables/ensemble_clean.csv"
  "results/tables/single_model_tssplit_summary_clean.csv"
  "results/tables/extra_datasets_with_lora_pace_compact.csv"
  "results/tables/ivon_all_results_table.csv"
  "results/tables/ivon_lambda_sweep_table.csv"
  "results/tables/ivon_ess_sweep_table.csv"
  "results/figures/fig_pipeline_overview.png"
  "results/figures/fig_acc_ece.png"
  "results/figures/fig_main_core.png"
  "results/figures/fig_ens_ece.png"
  "results/figures/fig_ens_scores.png"
  "results/figures/fig_reliability_overlay_dtd.png"
  "results/figures/fig_shift_robustness.png"
  "results/figures/fig_training_dynamics.png"
  "results/figures/fig_grad_diag.png"
  "results/figures/fig_lambda_sweep.png"
  "results/figures/fig_ivon_ess.png"
)

missing=0
for path in "${required_files[@]}"; do
  if [[ -f "$path" ]]; then
    echo "OK   $path"
  else
    echo "MISS $path"
    missing=1
  fi
done

echo
echo "Checking for forbidden large/generated files in curated artifact area..."
forbidden="$(find results -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.npz' -o -name '*.tar.gz' -o -name '*.xlsx' -o -name '*.json' -o -name '*.log' \) -print)"
if [[ -n "$forbidden" ]]; then
  echo "$forbidden"
  exit 1
fi

if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

echo "Artifact inventory complete."
