#!/usr/bin/env bash
set -uo pipefail

# Focused PACE + IVON-LoRA sweep.
#
# Purpose:
#   Tune the two uncertainty-strength knobs that matter most for PACE + IVON:
#   1) PACE adapter-noise scale SIGMA
#   2) IVON effective sample size / posterior strength IVON_ESS
#
# Default grid:
#   SIGMA in {0.5, 0.8, 1.2}
#   IVON_ESS in {200, 400, 800, 1600}
#
# Run from the repository root:
#   bash runners/run_pace_ivon_sigma_ess_sweep.sh
#
# Smaller 6-run version:
#   SIGMAS="0.5 0.8" IVON_ESS_VALUES="400 800 1600" bash runners/run_pace_ivon_sigma_ess_sweep.sh

ROOT_DIR="$(pwd)"
LOG_ROOT="${LOG_ROOT:-logs/pace_ivon_sigma_ess_sweep}"
OUT_ROOT="${OUT_ROOT:-outputs/pace_ivon_sigma_ess_sweep}"
EVAL_ROOT="${EVAL_ROOT:-results/pace_ivon_sigma_ess_sweep}"
mkdir -p "$LOG_ROOT" "$OUT_ROOT" "$EVAL_ROOT"

RUN_LOG="${LOG_ROOT}/run_pace_ivon_sigma_ess_sweep_$(date +%Y%m%d_%H%M%S).log"
touch "$RUN_LOG"

EPOCHS="${EPOCHS:-300}"
DATASETS="${DATASETS:-caltech101}"
SEEDS="${SEEDS:-42}"
SIGMAS="${SIGMAS:-0.5 0.8 1.2}"
IVON_ESS_VALUES="${IVON_ESS_VALUES:-200 400 800 1600}"

ADAPTER="${ADAPTER:-LoRAadd}"
RANK="${RANK:-10}"
BS="${BS:-16}"
IVON_LR="${IVON_LR:-0.03}"
IVON_HESS_INIT="${IVON_HESS_INIT:-1e-3}"
IVON_CLIP_RADIUS="${IVON_CLIP_RADIUS:-1e-3}"
IVON_BETA2="${IVON_BETA2:-0.99999}"
IVON_MC_SAMPLES="${IVON_MC_SAMPLES:-10}"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$RUN_LOG"
}

log "PACE + IVON-LoRA sigma/ESS sweep starting in ${ROOT_DIR}"
log "Config: EPOCHS=${EPOCHS} DATASETS=[${DATASETS}] SEEDS=[${SEEDS}] SIGMAS=[${SIGMAS}] IVON_ESS_VALUES=[${IVON_ESS_VALUES}]"
log "Adapter=${ADAPTER} Rank=${RANK} BS=${BS} IVON_LR=${IVON_LR} IVON_HESS_INIT=${IVON_HESS_INIT} IVON_CLIP_RADIUS=${IVON_CLIP_RADIUS} IVON_MC_SAMPLES=${IVON_MC_SAMPLES}"
log "Dirs: OUT_ROOT=${OUT_ROOT} EVAL_ROOT=${EVAL_ROOT} LOG_ROOT=${LOG_ROOT}"

for sigma in $SIGMAS; do
    for ess in $IVON_ESS_VALUES; do
        tag="S${sigma}_Ess${ess}"
        log "START GRID: ${tag}"

        EPOCHS="$EPOCHS" \
        DATASETS="$DATASETS" \
        SEEDS="$SEEDS" \
        METHODS="pace_ivon" \
        ADAPTER="$ADAPTER" \
        RANK="$RANK" \
        BS="$BS" \
        SIGMA="$sigma" \
        IVON_LR="$IVON_LR" \
        IVON_ESS="$ess" \
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
done

log "PACE + IVON-LoRA sigma/ESS sweep finished."
