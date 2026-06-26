#!/usr/bin/env bash
set -euo pipefail

# PACE-MSE vs PACE-KL lambda sweep.
#
# Produces checkpoints and evaluation JSONs for:
#   - pace      with raw-logit MSE consistency
#   - pace_kl   with probability-space KL consistency
#
# Defaults are intentionally single-seed and single-dataset because this sweep is
# meant to isolate the consistency-weight mechanism. Override DATASETS/SEEDS if
# you want a wider run.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EPOCHS="${EPOCHS:-300}"
DATASETS="${DATASETS:-cifar}"
SEEDS="${SEEDS:-42}"
LBD_VALUES="${LBD_VALUES:-0.1 0.25 0.5 1}"
SIGMA="${SIGMA:-1.2}"
TEMPERATURE="${TEMPERATURE:-1}"
ADAPTER="${ADAPTER:-LoRAmul_VPTadd}"
RANK="${RANK:-10}"
BS="${BS:-16}"
LR="${LR:-1e-3}"
WD="${WD:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

OUT_ROOT="${OUT_ROOT:-outputs/pace_lambda_sweep}"
EVAL_ROOT="${EVAL_ROOT:-results/pace_lambda_sweep}"
LOG_ROOT="${LOG_ROOT:-logs/pace_lambda_sweep}"

mkdir -p "$OUT_ROOT" "$EVAL_ROOT" "$LOG_ROOT"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

for dataset in $DATASETS; do
    for seed in $SEEDS; do
        for method in pace pace_kl; do
            for lbd in $LBD_VALUES; do
                lbd_tag="$(python - "$lbd" <<'PY'
import sys
print(f"{float(sys.argv[1]):g}")
PY
)"
                run_name="${method}_Lbd${lbd_tag}_S${SIGMA}_${ADAPTER}_R${RANK}"
                if [ "$seed" != "42" ]; then
                    run_name="${run_name}_Seed${seed}"
                fi
                run_name="${run_name}_${dataset}"

                log "START train_${run_name}"
                train_args=(
                    python train.py
                    --dataset "$dataset"
                    --lr "$LR"
                    --wd "$WD"
                    --adapter "$ADAPTER"
                    --rank "$RANK"
                    --epoch "$EPOCHS"
                    --bs "$BS"
                    --num_workers "$NUM_WORKERS"
                    --test_every "$EPOCHS"
                    --seed "$seed"
                    --out_dir "$OUT_ROOT"
                    --pace_type "$method"
                    --lbd "$lbd"
                    --sigma "$SIGMA"
                )
                if [ "$method" = "pace_kl" ]; then
                    train_args+=(--temperature "$TEMPERATURE")
                fi
                "${train_args[@]}" > "${LOG_ROOT}/train_${run_name}.out" 2> "${LOG_ROOT}/train_${run_name}.err"
                log "DONE  train_${run_name}"

                checkpoint="${OUT_ROOT}/${run_name}/weight.pt"
                log "START eval_${run_name}"
                eval_args=(
                    python evaluate_all_metrics.py
                    --checkpoint "$checkpoint"
                    --dataset "$dataset"
                    --adapter "$ADAPTER"
                    --rank "$RANK"
                    --sigma "$SIGMA"
                    --pace
                    --posthoc_temp_scaling
                    --save_dir "$EVAL_ROOT"
                )
                "${eval_args[@]}" > "${LOG_ROOT}/eval_${run_name}.out" 2> "${LOG_ROOT}/eval_${run_name}.err"
                log "DONE  eval_${run_name}"
            done
        done
    done
done

log "PACE lambda sweep finished."
