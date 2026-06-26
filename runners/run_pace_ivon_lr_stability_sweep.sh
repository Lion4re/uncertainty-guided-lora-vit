#!/usr/bin/env bash
set -uo pipefail

# Focused PACE + IVON-LoRA stability sweep.
#
# Purpose:
#   Recover stable PACE + IVON-LoRA training after the small-ESS sweep collapsed.
#   This sweep keeps PACE and IVON posterior-scale settings fixed, and varies
#   only the IVON learning rate.
#
# Default grid:
#   IVON_LR in {1e-3, 3e-3, 1e-2, 3e-2}
#
# Run from the repository root:
#   bash runners/run_pace_ivon_lr_stability_sweep.sh

ROOT_DIR="$(pwd)"
LOG_ROOT="${LOG_ROOT:-logs/pace_ivon_lr_stability_sweep}"
OUT_ROOT="${OUT_ROOT:-outputs/pace_ivon_lr_stability_sweep}"
EVAL_ROOT="${EVAL_ROOT:-results/pace_ivon_lr_stability_sweep}"
mkdir -p "$LOG_ROOT" "$OUT_ROOT" "$EVAL_ROOT"

RUN_LOG="${LOG_ROOT}/run_pace_ivon_lr_stability_sweep_$(date +%Y%m%d_%H%M%S).log"
touch "$RUN_LOG"

EPOCHS="${EPOCHS:-300}"
DATASETS="${DATASETS:-caltech101}"
SEEDS="${SEEDS:-42}"
IVON_LR_VALUES="${IVON_LR_VALUES:-1e-3 3e-3 1e-2 3e-2}"

ADAPTER="${ADAPTER:-LoRAadd}"
RANK="${RANK:-10}"
BS="${BS:-16}"
SIGMA="${SIGMA:-1.2}"
IVON_ESS="${IVON_ESS:-1e6}"
IVON_HESS_INIT="${IVON_HESS_INIT:-0.1}"
IVON_CLIP_RADIUS="${IVON_CLIP_RADIUS:-1e-3}"
IVON_BETA2="${IVON_BETA2:-0.99999}"
IVON_MC_SAMPLES="${IVON_MC_SAMPLES:-10}"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$RUN_LOG"
}

log "PACE + IVON-LoRA LR stability sweep starting in ${ROOT_DIR}"
log "Config: EPOCHS=${EPOCHS} DATASETS=[${DATASETS}] SEEDS=[${SEEDS}] IVON_LR_VALUES=[${IVON_LR_VALUES}]"
log "Adapter=${ADAPTER} Rank=${RANK} BS=${BS} SIGMA=${SIGMA} IVON_ESS=${IVON_ESS} IVON_HESS_INIT=${IVON_HESS_INIT} IVON_CLIP_RADIUS=${IVON_CLIP_RADIUS} IVON_MC_SAMPLES=${IVON_MC_SAMPLES}"
log "Dirs: OUT_ROOT=${OUT_ROOT} EVAL_ROOT=${EVAL_ROOT} LOG_ROOT=${LOG_ROOT}"

for lr in $IVON_LR_VALUES; do
    tag="lr${lr}"
    log "START GRID: ${tag}"

    EPOCHS="$EPOCHS" \
    DATASETS="$DATASETS" \
    SEEDS="$SEEDS" \
    METHODS="pace_ivon" \
    ADAPTER="$ADAPTER" \
    RANK="$RANK" \
    BS="$BS" \
    SIGMA="$SIGMA" \
    IVON_LR="$lr" \
    IVON_ESS="$IVON_ESS" \
    IVON_HESS_INIT="$IVON_HESS_INIT" \
    IVON_CLIP_RADIUS="$IVON_CLIP_RADIUS" \
    IVON_BETA2="$IVON_BETA2" \
    IVON_MC_SAMPLES="$IVON_MC_SAMPLES" \
    OUT_DIR="${OUT_ROOT}/${tag}" \
    EVAL_DIR="${EVAL_ROOT}/${tag}" \
    LOG_DIR="${LOG_ROOT}/${tag}" \
    bash run_main_experiments.sh

    status=$?
    if [ "$status" -eq 0 ]; then
        log "DONE GRID:  ${tag}"
    else
        log "FAIL GRID:  ${tag} (exit ${status})"
    fi
done

log "PACE + IVON-LoRA LR stability sweep finished."
