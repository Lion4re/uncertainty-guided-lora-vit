#!/usr/bin/env bash
set -uo pipefail

# Focused PACE + IVON-LoRA ESS sweep after the lambda sweep.
#
# Purpose:
#   Keep the winning PACE + IVON setup fixed and test whether IVON posterior
#   temperature / effective sample size can make MC inference useful.
#
# Fixed setup:
#   lambda = 0.1
#   sigma = 1.2
#   IVON lr = 0.03
#   hess_init = 0.1
#
# Default grid:
#   IVON_ESS in {1600, 8000, 40000, 200000, 1000000}
#
# Run from the repository root:
#   bash runners/run_pace_ivon_lbd01_ess_sweep.sh

ROOT_DIR="$(pwd)"
LOG_ROOT="${LOG_ROOT:-logs/pace_ivon_lbd01_ess_sweep}"
OUT_ROOT="${OUT_ROOT:-outputs/pace_ivon_lbd01_ess_sweep}"
EVAL_ROOT="${EVAL_ROOT:-results/pace_ivon_lbd01_ess_sweep}"
mkdir -p "$LOG_ROOT" "$OUT_ROOT" "$EVAL_ROOT"

RUN_LOG="${LOG_ROOT}/run_pace_ivon_lbd01_ess_sweep_$(date +%Y%m%d_%H%M%S).log"
touch "$RUN_LOG"

EPOCHS="${EPOCHS:-300}"
DATASET="${DATASET:-caltech101}"
SEED="${SEED:-42}"
IVON_ESS_VALUES="${IVON_ESS_VALUES:-1600 8000 40000 200000 1000000}"

ADAPTER="${ADAPTER:-LoRAadd}"
RANK="${RANK:-10}"
BS="${BS:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WD="${WD:-1e-4}"
LBD="${LBD:-0.1}"
SIGMA="${SIGMA:-1.2}"
IVON_LR="${IVON_LR:-0.03}"
IVON_HESS_INIT="${IVON_HESS_INIT:-0.1}"
IVON_CLIP_RADIUS="${IVON_CLIP_RADIUS:-1e-3}"
IVON_BETA2="${IVON_BETA2:-0.99999}"
IVON_MC_SAMPLES="${IVON_MC_SAMPLES:-10}"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$RUN_LOG"
}

run_step() {
    local name="$1"
    shift
    log "START: ${name}"
    "$@" > "${LOG_ROOT}/${name}.out" 2> "${LOG_ROOT}/${name}.err"
    local status=$?
    if [ "$status" -eq 0 ]; then
        log "DONE:  ${name}"
    else
        log "FAIL:  ${name} (exit ${status}). See ${LOG_ROOT}/${name}.err"
    fi
    return 0
}

fmt_g() {
    python - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):g}")
PY
}

log "PACE + IVON-LoRA lambda=0.1 ESS sweep starting in ${ROOT_DIR}"
log "Config: EPOCHS=${EPOCHS} DATASET=${DATASET} SEED=${SEED} IVON_ESS_VALUES=[${IVON_ESS_VALUES}]"
log "Adapter=${ADAPTER} Rank=${RANK} BS=${BS} LBD=${LBD} SIGMA=${SIGMA} IVON_LR=${IVON_LR} IVON_HESS_INIT=${IVON_HESS_INIT} IVON_MC_SAMPLES=${IVON_MC_SAMPLES}"
log "Dirs: OUT_ROOT=${OUT_ROOT} EVAL_ROOT=${EVAL_ROOT} LOG_ROOT=${LOG_ROOT}"

for ess in $IVON_ESS_VALUES; do
    tag="Ess$(fmt_g "$ess")"
    out_dir="${OUT_ROOT}/${tag}"
    eval_dir="${EVAL_ROOT}/${tag}"
    mkdir -p "$out_dir" "$eval_dir"

    run_step "train_pace_ivon_Lbd$(fmt_g "$LBD")_${tag}_${DATASET}" python train.py \
        --out_dir "$out_dir" \
        --dataset "$DATASET" --lr "$IVON_LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
        --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
        --test_every "$EPOCHS" --seed "$SEED" \
        --pace_type pace --lbd "$LBD" --sigma "$SIGMA" \
        --optimizer ivon --ivon_ess "$ess" --ivon_hess_init "$IVON_HESS_INIT" \
        --ivon_clip_radius "$IVON_CLIP_RADIUS" --ivon_beta2 "$IVON_BETA2"

    checkpoint="$(find "$out_dir" -name weight.pt | sort | tail -1)"
    if [ -z "$checkpoint" ]; then
        log "SKIP: eval_pace_ivon_${tag}_${DATASET} missing checkpoint under ${out_dir}"
        continue
    fi

    run_step "eval_pace_ivon_Lbd$(fmt_g "$LBD")_${tag}_${DATASET}_mean" python evaluate_all_metrics.py \
        --checkpoint "$checkpoint" --dataset "$DATASET" --adapter "$ADAPTER" --pace --ivon \
        --ivon_ess "$ess" --ivon_hess_init "$IVON_HESS_INIT" \
        --ivon_clip_radius "$IVON_CLIP_RADIUS" --ivon_beta2 "$IVON_BETA2" \
        --posthoc_temp_scaling --save_dir "$eval_dir"

    run_step "eval_pace_ivon_Lbd$(fmt_g "$LBD")_${tag}_${DATASET}_mc${IVON_MC_SAMPLES}" python evaluate_all_metrics.py \
        --checkpoint "$checkpoint" --dataset "$DATASET" --adapter "$ADAPTER" --pace --ivon \
        --ivon_ess "$ess" --ivon_hess_init "$IVON_HESS_INIT" \
        --ivon_clip_radius "$IVON_CLIP_RADIUS" --ivon_beta2 "$IVON_BETA2" \
        --ivon_mc_samples "$IVON_MC_SAMPLES" \
        --posthoc_temp_scaling --save_dir "$eval_dir"
done

log "PACE + IVON-LoRA lambda=0.1 ESS sweep finished."
