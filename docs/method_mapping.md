# Method Mapping

This document maps thesis-facing method names to runner names and implementation locations.

| Thesis name | Runner/config name | Main implementation | Notes |
| --- | --- | --- | --- |
| LoRA | `baseline`, `lora` | `pace/residual_adapters.py`, `train.py` | Frozen ViT-B/16 with adapter and linear head. |
| PACE | `pace` | `pace/pace_ops.py`, `train.py` | Raw-logit MSE consistency branch. |
| PACE-OF | `pace_offset` | `pace/pace_ops.py`, `run_main_experiments.sh` | Offset-free MSE diagnostic branch. |
| PACE-KL | `pace_kl_t1`, `pace_kl_t2` | `pace/pace_ops.py`, `train.py` | Probability-space KL consistency. |
| PACE-KL + Margin | `pace_kl_margin_t1` | `pace/pace_ops.py`, `train.py` | KL consistency with margin regularization. |
| PACE + PAC-Bayes | `pace_pacbayes` | `pace/residual_adapters.py`, `pace/pace_ops.py` | Learned adapter-output noise with PAC-Bayes KL term. |
| PACE-KL + PAC-Bayes | `pace_kl_pacbayes_t1` | `pace/residual_adapters.py`, `pace/pace_ops.py` | KL consistency plus PAC-Bayes adapter-output noise. |
| Uncertainty soft/top-k | `pace_uncert_soft`, `pace_uncert_topk`, `pace_kl_uncert_soft`, `pace_kl_uncert_topk` | `pace/pace_ops.py`, `runners/run_uncertainty_guided_experiments.sh` | Selects or weights consistency examples by predictive uncertainty. |
| PACE + IVON-LoRA | `pace_ivon` | `pace/ivon_utils.py`, `train.py` | Uses additive `LoRAadd` adapter because IVON places a posterior over trainable LoRA weights. |
| PACE-KL + IVON-LoRA | `pace_kl_ivon_t1`, `pace_kl_ivon_t2` | `pace/ivon_utils.py`, `train.py` | KL consistency plus IVON optimizer state. |
| Bayesian-LoRA diagnostics | `full_bayes_lora`, `pace_kl_full_bayes_lora_t1` | `pace/bayesian_lora.py` | Experimental diagnostic branch. |
| Ensemble | `evaluate_two_model_ensemble.py` | `evaluate_two_model_ensemble.py` | Probability averaging of separately trained checkpoints. |
| Split temperature scaling | `run_single_model_tssplit_main_june26.sh` | `evaluate_all_metrics.py` | Fits scalar temperature on a disjoint split and reports held-out metrics. |

The main PACE/PAC-Bayes/KL results use `LoRAmul_VPTadd`. IVON-LoRA uses `LoRAadd`; this is noted in the thesis because it changes the adapter parameterization.
