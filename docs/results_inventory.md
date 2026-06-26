# Results Inventory

This page maps the included lightweight artifacts to thesis/report sections.

## Tables

| File | Purpose |
| --- | --- |
| `results/tables/main_results_long.csv` | Main 3-dataset comparison table source. |
| `results/tables/ensemble_clean.csv` | Ensemble/control comparison source. |
| `results/tables/single_model_tssplit_summary_clean.csv` | Held-out split temperature-scaling control. |
| `results/tables/extra_datasets_with_lora_pace_compact.csv` | Extra dataset appendix table. |
| `results/tables/ivon_all_results_table.csv` | IVON branch summary table. |
| `results/tables/ivon_lambda_sweep_table.csv` | IVON lambda sweep source. |
| `results/tables/ivon_ess_sweep_table.csv` | IVON ESS sweep source. |

## Figures

| File | Purpose |
| --- | --- |
| `results/figures/fig_acc_ece.png` | Accuracy/ECE overview. |
| `results/figures/fig_main_core.png` | Main method comparison. |
| `results/figures/fig_ens_ece.png` | Ensemble ECE comparison. |
| `results/figures/fig_ens_scores.png` | Ensemble NLL/Brier comparison. |
| `results/figures/fig_reliability_overlay_dtd.png` | DTD reliability overlay. |
| `results/figures/fig_shift_robustness.png` | Shift robustness accuracy/ECE figure. |
| `results/figures/fig_training_dynamics.png` | CIFAR training dynamics diagnostic. |
| `results/figures/fig_grad_diag.png` | PAC-Bayes gradient/noise diagnostic. |
| `results/figures/fig_lambda_sweep.png` | Consistency-weight sweep figure. |
| `results/figures/fig_ivon_ess.png` | IVON ESS/MC diagnostic figure. |

The filenames above are the canonical GitHub artifact names. If the thesis/Overleaf project copies
the same images under section-specific names, cite these files as the source artifacts in the repo.

Full raw JSONs, per-seed reliability diagrams, logs, and appendix bundles are intentionally excluded from the GitHub artifact.
