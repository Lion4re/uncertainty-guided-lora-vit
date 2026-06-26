#!/usr/bin/env bash
set -uo pipefail

# Main experiment batch:
# - full PACE-paper epoch budget (default 300)
# - post-hoc temperature scaling for all selected methods
# - training timing and inference timing
# - more seeds / more datasets when available
#
# Run from Vision:
#   bash run_main_experiments.sh

LOG_DIR="${LOG_DIR:-logs/main_experiments}"
EVAL_DIR="${EVAL_DIR:-results/main_3seed_vtab}"
OUT_DIR="${OUT_DIR:-outputs/checkpoints_and_training_logs}"
mkdir -p "$LOG_DIR" "$EVAL_DIR" "$OUT_DIR"
RUN_LOG="${LOG_DIR}/run_main_experiments_$(date +%Y%m%d_%H%M%S).log"
touch "$RUN_LOG"

EPOCHS="${EPOCHS:-300}"
BS="${BS:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-03}"
WD="${WD:-1e-4}"
IVON_LR="${IVON_LR:-0.03}"
IVON_ESS="${IVON_ESS:-1e6}"
IVON_HESS_INIT="${IVON_HESS_INIT:-1e-3}"
IVON_CLIP_RADIUS="${IVON_CLIP_RADIUS:-1e-3}"
IVON_BETA2="${IVON_BETA2:-0.99999}"
IVON_MC_SAMPLES="${IVON_MC_SAMPLES:-10}"
RANK="${RANK:-10}"
LORA_ALPHA="${LORA_ALPHA:-$RANK}"
ADAPTER="${ADAPTER:-LoRAmul_VPTadd}"
SIGMA="${SIGMA:-1.2}"
LBD="${LBD:-0.5}"
PAC_LBD="${PAC_LBD:-1e-3}"
PAC_PRIOR_SIGMA="${PAC_PRIOR_SIGMA:-1.2}"
PAC_PGD_STAGE1_EPOCHS="${PAC_PGD_STAGE1_EPOCHS:-100}"
PAC_PGD_LBD="${PAC_PGD_LBD:-1.0}"
PAC_PGD_GAMMA="${PAC_PGD_GAMMA:-0.1}"
PAC_PGD_INIT_FLOOR="${PAC_PGD_INIT_FLOOR:-1e-4}"
PAC_PGD_PRIOR_FLOOR="${PAC_PGD_PRIOR_FLOOR:-1e-4}"
PAC_PGD_MC_SAMPLES="${PAC_PGD_MC_SAMPLES:-10}"
BLOB_LBD="${BLOB_LBD:-1e-3}"
BLOB_PRIOR_SIGMA="${BLOB_PRIOR_SIGMA:-1.0}"
BLOB_INIT_SIGMA="${BLOB_INIT_SIGMA:-1e-4}"
BLOB_KL_REDUCTION="${BLOB_KL_REDUCTION:-mean}"
BLOB_MC_SAMPLES="${BLOB_MC_SAMPLES:-10}"
BAYES_LORA_LBD_U="${BAYES_LORA_LBD_U:-1e-5}"
BAYES_LORA_LBD_W="${BAYES_LORA_LBD_W:-1e-5}"
BAYES_LORA_FLOW="${BAYES_LORA_FLOW:-none}"
BAYES_LORA_FLOW_DEPTH="${BAYES_LORA_FLOW_DEPTH:-1}"
BAYES_LORA_INIT_SIGMA="${BAYES_LORA_INIT_SIGMA:-1e-4}"
BAYES_LORA_PRIOR_SIGMA="${BAYES_LORA_PRIOR_SIGMA:-0.1}"
BAYES_LORA_MAX_SIGMA_U="${BAYES_LORA_MAX_SIGMA_U:-0.1}"
BAYES_LORA_LAMBDA_INIT="${BAYES_LORA_LAMBDA_INIT:-1e-3}"
BAYES_LORA_LAMBDA_MAX="${BAYES_LORA_LAMBDA_MAX:-3e-2}"
BAYES_LORA_MC_SAMPLES="${BAYES_LORA_MC_SAMPLES:-4}"
SEEDS="${SEEDS:-42 123 456}"
DATASETS="${DATASETS:-cifar caltech101 dtd svhn eurosat resisc45 patch_camelyon oxford_flowers102 oxford_iiit_pet}"
METHODS="${METHODS:-baseline pace pace_kl_t1 pace_kl_t2}"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$RUN_LOG"
}

run_step() {
    local name="$1"
    shift
    log "START: $name"
    "$@" > "${LOG_DIR}/${name}.out" 2> "${LOG_DIR}/${name}.err"
    local status=$?
    if [ "$status" -eq 0 ]; then
        log "DONE:  $name"
    else
        log "FAIL:  $name (exit $status). See ${LOG_DIR}/${name}.err"
    fi
    return 0
}

dataset_exists() {
    local dataset="$1"
    [ -f "data/vtab-1k/${dataset}/train800.txt" ] && [ -f "data/vtab-1k/${dataset}/test.txt" ]
}

fmt_g() {
    python - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):g}")
PY
}

float_isclose() {
    python - "$1" "$2" <<'PY'
import math
import sys
print("1" if math.isclose(float(sys.argv[1]), float(sys.argv[2]), rel_tol=0.0, abs_tol=1e-12) else "0")
PY
}

method_name() {
    local method="$1"
    local dataset="$2"
    local seed="$3"
    case "$method" in
        baseline)
            name="${ADAPTER}_R${RANK}"
            ;;
        blob)
            name="${ADAPTER}_R${RANK}_BLoB"
            if [ "$(float_isclose "$BLOB_LBD" "1e-3")" != "1" ]; then
                name="${name}_Blob$(fmt_g "$BLOB_LBD")"
            fi
            if [ "$BLOB_KL_REDUCTION" != "mean" ]; then
                name="${name}_Bkl${BLOB_KL_REDUCTION}"
            fi
            ;;
        pace)
            name="pace_Lbd1_S${SIGMA}_${ADAPTER}_R${RANK}"
            ;;
        pace_offset)
            name="pace_OF_Lbd1_S${SIGMA}_${ADAPTER}_R${RANK}"
            ;;
        pace_ivon)
            name="pace_Lbd1_S${SIGMA}_${ADAPTER}_R${RANK}_IVON"
            if [ "$(float_isclose "$IVON_ESS" "1e6")" != "1" ]; then
                name="${name}_Ess$(fmt_g "$IVON_ESS")"
            fi
            if [ "$(float_isclose "$IVON_HESS_INIT" "1e-3")" != "1" ]; then
                name="${name}_Hi$(fmt_g "$IVON_HESS_INIT")"
            fi
            if [ "$(float_isclose "$IVON_CLIP_RADIUS" "1e-3")" != "1" ]; then
                name="${name}_Cr$(fmt_g "$IVON_CLIP_RADIUS")"
            fi
            if [ "$(float_isclose "$IVON_LR" "1e-3")" != "1" ]; then
                name="${name}_lr$(fmt_g "$IVON_LR")"
            fi
            ;;
        pace_pacbayes)
            name="pace_pacbayes_Lbd1_S${SIGMA}"
            if [ "$(float_isclose "$PAC_LBD" "1e-3")" != "1" ]; then
                name="${name}_Pac$(fmt_g "$PAC_LBD")"
            fi
            if [ "$(float_isclose "$PAC_PRIOR_SIGMA" "1.2")" != "1" ]; then
                name="${name}_Ps$(fmt_g "$PAC_PRIOR_SIGMA")"
            fi
            name="${name}_${ADAPTER}_R${RANK}"
            ;;
        pace_pacpgd)
            name="pace_pacpgd_Lbd1_S${SIGMA}_PacPGD_St1${PAC_PGD_STAGE1_EPOCHS}"
            if [ "$(float_isclose "$PAC_PGD_GAMMA" "0.1")" != "1" ]; then
                name="${name}_Pg$(fmt_g "$PAC_PGD_GAMMA")"
            fi
            if [ "$(float_isclose "$PAC_PGD_LBD" "1.0")" != "1" ]; then
                name="${name}_Pgl$(fmt_g "$PAC_PGD_LBD")"
            fi
            name="${name}_${ADAPTER}_R${RANK}"
            ;;
        pace_dropout_p01)
            name="pace_Lbd1_S${SIGMA}_Drop0.1_${ADAPTER}_R${RANK}"
            ;;
        pace_dropout_p02)
            name="pace_Lbd1_S${SIGMA}_Drop0.2_${ADAPTER}_R${RANK}"
            ;;
        pace_dropout_p03)
            name="pace_Lbd1_S${SIGMA}_Drop0.3_${ADAPTER}_R${RANK}"
            ;;
        pace_blob)
            name="pace_Lbd1_S${SIGMA}_${ADAPTER}_R${RANK}_BLoB"
            if [ "$(float_isclose "$BLOB_LBD" "1e-3")" != "1" ]; then
                name="${name}_Blob$(fmt_g "$BLOB_LBD")"
            fi
            if [ "$BLOB_KL_REDUCTION" != "mean" ]; then
                name="${name}_Bkl${BLOB_KL_REDUCTION}"
            fi
            ;;
        pace_kl_t1)
            name="pace_kl_Lbd${LBD}_S${SIGMA}_T1_${ADAPTER}_R${RANK}"
            ;;
        pace_kl_detach_t1)
            name="pace_kl_Lbd${LBD}_S${SIGMA}_T1_Detach_${ADAPTER}_R${RANK}"
            ;;
        pace_kl_t2)
            name="pace_kl_Lbd${LBD}_S${SIGMA}_${ADAPTER}_R${RANK}"
            ;;
        pace_kl_uncert_topk_t1)
            name="pace_kl_uncert_topk_Lbd${LBD}_S${SIGMA}_T1_${ADAPTER}_R${RANK}"
            ;;
        pace_kl_uncert_topk_detach_t1)
            name="pace_kl_uncert_topk_Lbd${LBD}_S${SIGMA}_T1_Detach_${ADAPTER}_R${RANK}"
            ;;
        pace_kl_ivon_t1)
            name="pace_kl_Lbd${LBD}_S${SIGMA}_T1_${ADAPTER}_R${RANK}_IVON"
            if [ "$(float_isclose "$IVON_ESS" "1e6")" != "1" ]; then
                name="${name}_Ess$(fmt_g "$IVON_ESS")"
            fi
            if [ "$(float_isclose "$IVON_HESS_INIT" "1e-3")" != "1" ]; then
                name="${name}_Hi$(fmt_g "$IVON_HESS_INIT")"
            fi
            if [ "$(float_isclose "$IVON_CLIP_RADIUS" "1e-3")" != "1" ]; then
                name="${name}_Cr$(fmt_g "$IVON_CLIP_RADIUS")"
            fi
            if [ "$(float_isclose "$IVON_LR" "1e-3")" != "1" ]; then
                name="${name}_lr$(fmt_g "$IVON_LR")"
            fi
            ;;
        pace_kl_ivon_t2)
            name="pace_kl_Lbd${LBD}_S${SIGMA}_${ADAPTER}_R${RANK}_IVON"
            if [ "$(float_isclose "$IVON_ESS" "1e6")" != "1" ]; then
                name="${name}_Ess$(fmt_g "$IVON_ESS")"
            fi
            if [ "$(float_isclose "$IVON_HESS_INIT" "1e-3")" != "1" ]; then
                name="${name}_Hi$(fmt_g "$IVON_HESS_INIT")"
            fi
            if [ "$(float_isclose "$IVON_CLIP_RADIUS" "1e-3")" != "1" ]; then
                name="${name}_Cr$(fmt_g "$IVON_CLIP_RADIUS")"
            fi
            if [ "$(float_isclose "$IVON_LR" "1e-3")" != "1" ]; then
                name="${name}_lr$(fmt_g "$IVON_LR")"
            fi
            ;;
        full_bayes_lora)
            name="full_bayes_lora_${ADAPTER}_R${RANK}_FullBayesLoRA"
            if [ "$(float_isclose "$LORA_ALPHA" "$RANK")" != "1" ]; then
                name="${name}_Alpha$(fmt_g "$LORA_ALPHA")"
            fi
            if [ "$BAYES_LORA_FLOW" != "none" ]; then
                name="${name}_BFlow${BAYES_LORA_FLOW}"
            fi
            if [ "$BAYES_LORA_FLOW" != "none" ] && [ "$BAYES_LORA_FLOW_DEPTH" != "1" ]; then
                name="${name}_Bfd${BAYES_LORA_FLOW_DEPTH}"
            fi
            if [ "$(float_isclose "$BAYES_LORA_LBD_U" "1e-5")" != "1" ]; then
                name="${name}_Bu$(fmt_g "$BAYES_LORA_LBD_U")"
            fi
            if [ "$(float_isclose "$BAYES_LORA_LBD_W" "1e-5")" != "1" ]; then
                name="${name}_Bw$(fmt_g "$BAYES_LORA_LBD_W")"
            fi
            ;;
        pace_kl_learnsigma_t1)
            name="pace_kl_learnsigma_Lbd${LBD}_S${SIGMA}_T1"
            if [ "$(float_isclose "$PAC_PRIOR_SIGMA" "1.2")" != "1" ]; then
                name="${name}_Ps$(fmt_g "$PAC_PRIOR_SIGMA")"
            fi
            name="${name}_${ADAPTER}_R${RANK}"
            ;;
        pace_kl_wp_t1)
            name="pace_kl_wp_Lbd${LBD}_S${SIGMA}_T1_${ADAPTER}_R${RANK}_BLoB"
            if [ "$(float_isclose "$BLOB_LBD" "1e-3")" != "1" ]; then
                name="${name}_Blob$(fmt_g "$BLOB_LBD")"
            fi
            if [ "$BLOB_KL_REDUCTION" != "mean" ]; then
                name="${name}_Bkl${BLOB_KL_REDUCTION}"
            fi
            ;;
        pace_kl_blob_t2)
            name="pace_kl_Lbd${LBD}_S${SIGMA}_${ADAPTER}_R${RANK}_BLoB"
            if [ "$(float_isclose "$BLOB_LBD" "1e-3")" != "1" ]; then
                name="${name}_Blob$(fmt_g "$BLOB_LBD")"
            fi
            if [ "$BLOB_KL_REDUCTION" != "mean" ]; then
                name="${name}_Bkl${BLOB_KL_REDUCTION}"
            fi
            ;;
        pace_kl_margin_t1)
            name="pace_kl_margin_Lbd${LBD}_S${SIGMA}_T1_${ADAPTER}_R${RANK}"
            ;;
        pace_kl_full_bayes_lora_t1)
            name="pace_kl_full_bayes_lora_Lbd${LBD}_S${SIGMA}_T1_${ADAPTER}_R${RANK}_FullBayesLoRA"
            if [ "$(float_isclose "$LORA_ALPHA" "$RANK")" != "1" ]; then
                name="${name}_Alpha$(fmt_g "$LORA_ALPHA")"
            fi
            if [ "$BAYES_LORA_FLOW" != "none" ]; then
                name="${name}_BFlow${BAYES_LORA_FLOW}"
            fi
            if [ "$BAYES_LORA_FLOW" != "none" ] && [ "$BAYES_LORA_FLOW_DEPTH" != "1" ]; then
                name="${name}_Bfd${BAYES_LORA_FLOW_DEPTH}"
            fi
            if [ "$(float_isclose "$BAYES_LORA_LBD_U" "1e-5")" != "1" ]; then
                name="${name}_Bu$(fmt_g "$BAYES_LORA_LBD_U")"
            fi
            if [ "$(float_isclose "$BAYES_LORA_LBD_W" "1e-5")" != "1" ]; then
                name="${name}_Bw$(fmt_g "$BAYES_LORA_LBD_W")"
            fi
            ;;
        pace_kl_margin_full_bayes_lora_t1)
            name="pace_kl_margin_full_bayes_lora_Lbd${LBD}_S${SIGMA}_T1_${ADAPTER}_R${RANK}_FullBayesLoRA"
            if [ "$(float_isclose "$LORA_ALPHA" "$RANK")" != "1" ]; then
                name="${name}_Alpha$(fmt_g "$LORA_ALPHA")"
            fi
            if [ "$BAYES_LORA_FLOW" != "none" ]; then
                name="${name}_BFlow${BAYES_LORA_FLOW}"
            fi
            if [ "$BAYES_LORA_FLOW" != "none" ] && [ "$BAYES_LORA_FLOW_DEPTH" != "1" ]; then
                name="${name}_Bfd${BAYES_LORA_FLOW_DEPTH}"
            fi
            if [ "$(float_isclose "$BAYES_LORA_LBD_U" "1e-5")" != "1" ]; then
                name="${name}_Bu$(fmt_g "$BAYES_LORA_LBD_U")"
            fi
            if [ "$(float_isclose "$BAYES_LORA_LBD_W" "1e-5")" != "1" ]; then
                name="${name}_Bw$(fmt_g "$BAYES_LORA_LBD_W")"
            fi
            ;;
        pace_kl_margin_blob_t1)
            name="pace_kl_margin_Lbd${LBD}_S${SIGMA}_T1_${ADAPTER}_R${RANK}_BLoB"
            if [ "$(float_isclose "$BLOB_LBD" "1e-3")" != "1" ]; then
                name="${name}_Blob$(fmt_g "$BLOB_LBD")"
            fi
            if [ "$BLOB_KL_REDUCTION" != "mean" ]; then
                name="${name}_Bkl${BLOB_KL_REDUCTION}"
            fi
            ;;
        pace_kl_flat_t1)
            name="pace_kl_flat_Lbd${LBD}_S${SIGMA}_T1_${ADAPTER}_R${RANK}"
            ;;
        pace_kl_jacobian_t1)
            name="pace_kl_jacobian_Lbd${LBD}_S${SIGMA}_T1_${ADAPTER}_R${RANK}"
            ;;
        pace_kl_pacbayes_t1)
            name="pace_kl_pacbayes_Lbd${LBD}_S${SIGMA}_T1"
            if [ "$(float_isclose "$PAC_LBD" "1e-3")" != "1" ]; then
                name="${name}_Pac$(fmt_g "$PAC_LBD")"
            fi
            if [ "$(float_isclose "$PAC_PRIOR_SIGMA" "1.2")" != "1" ]; then
                name="${name}_Ps$(fmt_g "$PAC_PRIOR_SIGMA")"
            fi
            name="${name}_${ADAPTER}_R${RANK}"
            ;;
        pace_kl_pacpgd_t1)
            name="pace_kl_pacpgd_Lbd${LBD}_S${SIGMA}_T1_PacPGD_St1${PAC_PGD_STAGE1_EPOCHS}"
            if [ "$(float_isclose "$PAC_PGD_GAMMA" "0.1")" != "1" ]; then
                name="${name}_Pg$(fmt_g "$PAC_PGD_GAMMA")"
            fi
            if [ "$(float_isclose "$PAC_PGD_LBD" "1.0")" != "1" ]; then
                name="${name}_Pgl$(fmt_g "$PAC_PGD_LBD")"
            fi
            name="${name}_${ADAPTER}_R${RANK}"
            ;;
        pace_kl_margin_pacbayes_t1)
            name="pace_kl_margin_pacbayes_Lbd${LBD}_S${SIGMA}_T1"
            if [ "$(float_isclose "$PAC_LBD" "1e-3")" != "1" ]; then
                name="${name}_Pac$(fmt_g "$PAC_LBD")"
            fi
            if [ "$(float_isclose "$PAC_PRIOR_SIGMA" "1.2")" != "1" ]; then
                name="${name}_Ps$(fmt_g "$PAC_PRIOR_SIGMA")"
            fi
            name="${name}_${ADAPTER}_R${RANK}"
            ;;
        *)
            log "Unknown method: $method"
            return 1
            ;;
    esac
    if [ "$seed" != "42" ]; then
        name="${name}_Seed${seed}"
    fi
    echo "${name}_${dataset}"
}

train_method() {
    local method="$1"
    local dataset="$2"
    local seed="$3"
    local name
    name="$(method_name "$method" "$dataset" "$seed")" || return 0
    local checkpoint="${OUT_DIR}/${name}/weight.pt"
    if [ -f "$checkpoint" ]; then
        log "SKIP:  train_${name} checkpoint exists at $checkpoint"
    else
        case "$method" in
            baseline)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed"
                ;;
            blob)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --blob --blob_lbd "$BLOB_LBD" --blob_prior_sigma "$BLOB_PRIOR_SIGMA" \
                    --blob_init_sigma "$BLOB_INIT_SIGMA" --blob_kl_reduction "$BLOB_KL_REDUCTION"
                ;;
            pace)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace --lbd 1 --sigma "$SIGMA"
                ;;
            pace_offset)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_offset --lbd 1 --sigma "$SIGMA"
                ;;
            pace_ivon)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$IVON_LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace --lbd 1 --sigma "$SIGMA" \
                    --optimizer ivon --ivon_ess "$IVON_ESS" --ivon_hess_init "$IVON_HESS_INIT" \
                    --ivon_clip_radius "$IVON_CLIP_RADIUS" --ivon_beta2 "$IVON_BETA2"
                ;;
            pace_pacbayes)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_pacbayes --lbd 1 --sigma "$SIGMA" \
                    --pac_lbd "$PAC_LBD" --pac_prior_sigma "$PAC_PRIOR_SIGMA"
                ;;
            pace_pacpgd)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_pacpgd --lbd 1 --sigma "$SIGMA" \
                    --pac_pgd_stage1_epochs "$PAC_PGD_STAGE1_EPOCHS" \
                    --pac_pgd_lbd "$PAC_PGD_LBD" --pac_pgd_gamma "$PAC_PGD_GAMMA" \
                    --pac_pgd_init_floor "$PAC_PGD_INIT_FLOOR" \
                    --pac_pgd_prior_floor "$PAC_PGD_PRIOR_FLOOR"
                ;;
            pace_dropout_p01)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace --lbd 1 --sigma "$SIGMA" --adapter_dropout 0.1
                ;;
            pace_dropout_p02)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace --lbd 1 --sigma "$SIGMA" --adapter_dropout 0.2
                ;;
            pace_dropout_p03)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace --lbd 1 --sigma "$SIGMA" --adapter_dropout 0.3
                ;;
            pace_blob)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace --lbd 1 --sigma "$SIGMA" \
                    --blob --blob_lbd "$BLOB_LBD" --blob_prior_sigma "$BLOB_PRIOR_SIGMA" \
                    --blob_init_sigma "$BLOB_INIT_SIGMA" --blob_kl_reduction "$BLOB_KL_REDUCTION"
                ;;
            pace_kl_t1)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl --lbd "$LBD" --sigma "$SIGMA" --temperature 1.0
                ;;
            pace_kl_detach_t1)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl --lbd "$LBD" --sigma "$SIGMA" --temperature 1.0 \
                    --pace_kl_detach_target
                ;;
            pace_kl_t2)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl --lbd "$LBD" --sigma "$SIGMA" --temperature 2.0
                ;;
            pace_kl_uncert_topk_t1)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl_uncert_topk --lbd "$LBD" --sigma "$SIGMA" --temperature 1.0
                ;;
            pace_kl_uncert_topk_detach_t1)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl_uncert_topk --lbd "$LBD" --sigma "$SIGMA" --temperature 1.0 \
                    --pace_kl_detach_target
                ;;
            pace_kl_ivon_t1)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$IVON_LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl --lbd "$LBD" --sigma "$SIGMA" --temperature 1.0 \
                    --optimizer ivon --ivon_ess "$IVON_ESS" --ivon_hess_init "$IVON_HESS_INIT" \
                    --ivon_clip_radius "$IVON_CLIP_RADIUS" --ivon_beta2 "$IVON_BETA2"
                ;;
            pace_kl_ivon_t2)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$IVON_LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl --lbd "$LBD" --sigma "$SIGMA" --temperature 2.0 \
                    --optimizer ivon --ivon_ess "$IVON_ESS" --ivon_hess_init "$IVON_HESS_INIT" \
                    --ivon_clip_radius "$IVON_CLIP_RADIUS" --ivon_beta2 "$IVON_BETA2"
                ;;
            full_bayes_lora)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" --lora_alpha "$LORA_ALPHA" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type full_bayes_lora --full_bayes_lora \
                    --bayes_lora_flow "$BAYES_LORA_FLOW" \
                    --bayes_lora_flow_depth "$BAYES_LORA_FLOW_DEPTH" \
                    --bayes_lora_lbd_u "$BAYES_LORA_LBD_U" --bayes_lora_lbd_w "$BAYES_LORA_LBD_W" \
                    --bayes_lora_init_sigma "$BAYES_LORA_INIT_SIGMA" \
                    --bayes_lora_prior_sigma "$BAYES_LORA_PRIOR_SIGMA" \
                    --bayes_lora_max_sigma_u "$BAYES_LORA_MAX_SIGMA_U" \
                    --bayes_lora_lambda_init "$BAYES_LORA_LAMBDA_INIT" \
                    --bayes_lora_lambda_max "$BAYES_LORA_LAMBDA_MAX"
                ;;
            pace_kl_learnsigma_t1)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl_learnsigma --lbd "$LBD" --sigma "$SIGMA" --temperature 1.0 \
                    --pac_prior_sigma "$PAC_PRIOR_SIGMA"
                ;;
            pace_kl_wp_t1)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl_wp --lbd "$LBD" --sigma "$SIGMA" --temperature 1.0 \
                    --blob_lbd "$BLOB_LBD" --blob_prior_sigma "$BLOB_PRIOR_SIGMA" \
                    --blob_init_sigma "$BLOB_INIT_SIGMA" --blob_kl_reduction "$BLOB_KL_REDUCTION"
                ;;
            pace_kl_blob_t2)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl --lbd "$LBD" --sigma "$SIGMA" --temperature 2.0 \
                    --blob --blob_lbd "$BLOB_LBD" --blob_prior_sigma "$BLOB_PRIOR_SIGMA" \
                    --blob_init_sigma "$BLOB_INIT_SIGMA" --blob_kl_reduction "$BLOB_KL_REDUCTION"
                ;;
            pace_kl_margin_t1)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl_margin --lbd "$LBD" --sigma "$SIGMA" --temperature 1.0
                ;;
            pace_kl_full_bayes_lora_t1)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" --lora_alpha "$LORA_ALPHA" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl_full_bayes_lora --lbd "$LBD" --sigma "$SIGMA" --temperature 1.0 \
                    --full_bayes_lora --bayes_lora_flow "$BAYES_LORA_FLOW" \
                    --bayes_lora_flow_depth "$BAYES_LORA_FLOW_DEPTH" \
                    --bayes_lora_lbd_u "$BAYES_LORA_LBD_U" --bayes_lora_lbd_w "$BAYES_LORA_LBD_W" \
                    --bayes_lora_init_sigma "$BAYES_LORA_INIT_SIGMA" \
                    --bayes_lora_prior_sigma "$BAYES_LORA_PRIOR_SIGMA" \
                    --bayes_lora_max_sigma_u "$BAYES_LORA_MAX_SIGMA_U" \
                    --bayes_lora_lambda_init "$BAYES_LORA_LAMBDA_INIT" \
                    --bayes_lora_lambda_max "$BAYES_LORA_LAMBDA_MAX"
                ;;
            pace_kl_margin_full_bayes_lora_t1)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" --lora_alpha "$LORA_ALPHA" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl_margin_full_bayes_lora --lbd "$LBD" --sigma "$SIGMA" --temperature 1.0 \
                    --full_bayes_lora --bayes_lora_flow "$BAYES_LORA_FLOW" \
                    --bayes_lora_flow_depth "$BAYES_LORA_FLOW_DEPTH" \
                    --bayes_lora_lbd_u "$BAYES_LORA_LBD_U" --bayes_lora_lbd_w "$BAYES_LORA_LBD_W" \
                    --bayes_lora_init_sigma "$BAYES_LORA_INIT_SIGMA" \
                    --bayes_lora_prior_sigma "$BAYES_LORA_PRIOR_SIGMA" \
                    --bayes_lora_max_sigma_u "$BAYES_LORA_MAX_SIGMA_U" \
                    --bayes_lora_lambda_init "$BAYES_LORA_LAMBDA_INIT" \
                    --bayes_lora_lambda_max "$BAYES_LORA_LAMBDA_MAX"
                ;;
            pace_kl_margin_blob_t1)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl_margin --lbd "$LBD" --sigma "$SIGMA" --temperature 1.0 \
                    --blob --blob_lbd "$BLOB_LBD" --blob_prior_sigma "$BLOB_PRIOR_SIGMA" \
                    --blob_init_sigma "$BLOB_INIT_SIGMA" --blob_kl_reduction "$BLOB_KL_REDUCTION"
                ;;
            pace_kl_flat_t1)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl_flat --lbd "$LBD" --sigma "$SIGMA" --temperature 1.0
                ;;
            pace_kl_jacobian_t1)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl_jacobian --lbd "$LBD" --sigma "$SIGMA" --temperature 1.0
                ;;
            pace_kl_pacbayes_t1)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl_pacbayes --lbd "$LBD" --sigma "$SIGMA" --temperature 1.0 \
                    --pac_lbd "$PAC_LBD" --pac_prior_sigma "$PAC_PRIOR_SIGMA"
                ;;
            pace_kl_pacpgd_t1)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl_pacpgd --lbd "$LBD" --sigma "$SIGMA" --temperature 1.0 \
                    --pac_pgd_stage1_epochs "$PAC_PGD_STAGE1_EPOCHS" \
                    --pac_pgd_lbd "$PAC_PGD_LBD" --pac_pgd_gamma "$PAC_PGD_GAMMA" \
                    --pac_pgd_init_floor "$PAC_PGD_INIT_FLOOR" \
                    --pac_pgd_prior_floor "$PAC_PGD_PRIOR_FLOOR"
                ;;
            pace_kl_margin_pacbayes_t1)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" \
                    --dataset "$dataset" --lr "$LR" --wd "$WD" --adapter "$ADAPTER" --rank "$RANK" \
                    --epoch "$EPOCHS" --bs "$BS" --num_workers "$NUM_WORKERS" \
                    --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl_margin_pacbayes --lbd "$LBD" --sigma "$SIGMA" --temperature 1.0 \
                    --pac_lbd "$PAC_LBD" --pac_prior_sigma "$PAC_PRIOR_SIGMA"
                ;;
        esac
    fi

    eval_method "$method" "$dataset" "$seed"
}

eval_method() {
    local method="$1"
    local dataset="$2"
    local seed="$3"
    local name
    name="$(method_name "$method" "$dataset" "$seed")" || return 0
    local checkpoint="${OUT_DIR}/${name}/weight.pt"
    if [ ! -f "$checkpoint" ]; then
        log "SKIP:  eval_${name} missing checkpoint $checkpoint"
        return 0
    fi
    case "$method" in
        baseline)
            run_step "eval_${name}" python evaluate_all_metrics.py \
                --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" \
                --posthoc_temp_scaling --save_dir "$EVAL_DIR"
            ;;
        blob)
            run_step "eval_${name}_m0" python evaluate_all_metrics.py \
                --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" \
                --blob --blob_prior_sigma "$BLOB_PRIOR_SIGMA" --blob_init_sigma "$BLOB_INIT_SIGMA" \
                --posthoc_temp_scaling --save_dir "$EVAL_DIR"
            run_step "eval_${name}_m${BLOB_MC_SAMPLES}" python evaluate_all_metrics.py \
                --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" \
                --blob --blob_prior_sigma "$BLOB_PRIOR_SIGMA" --blob_init_sigma "$BLOB_INIT_SIGMA" \
                --blob_mc_samples "$BLOB_MC_SAMPLES" \
                --posthoc_temp_scaling --save_dir "$EVAL_DIR"
            ;;
        pace_kl_learnsigma_t1)
            run_step "eval_${name}" python evaluate_all_metrics.py \
                --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" --pace --learnsigma \
                --pac_prior_sigma "$PAC_PRIOR_SIGMA" \
                --posthoc_temp_scaling --save_dir "$EVAL_DIR"
            ;;
        pace_kl_wp_t1)
            run_step "eval_${name}_m0" python evaluate_all_metrics.py \
                --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" --pace \
                --blob --blob_prior_sigma "$BLOB_PRIOR_SIGMA" --blob_init_sigma "$BLOB_INIT_SIGMA" \
                --posthoc_temp_scaling --save_dir "$EVAL_DIR"
            run_step "eval_${name}_m${BLOB_MC_SAMPLES}" python evaluate_all_metrics.py \
                --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" --pace \
                --blob --blob_prior_sigma "$BLOB_PRIOR_SIGMA" --blob_init_sigma "$BLOB_INIT_SIGMA" \
                --blob_mc_samples "$BLOB_MC_SAMPLES" \
                --posthoc_temp_scaling --save_dir "$EVAL_DIR"
            ;;
        full_bayes_lora)
            run_step "eval_${name}_m0" python evaluate_all_metrics.py \
                --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" --lora_alpha "$LORA_ALPHA" \
                --full_bayes_lora --bayes_lora_flow "$BAYES_LORA_FLOW" \
                --bayes_lora_flow_depth "$BAYES_LORA_FLOW_DEPTH" \
                --bayes_lora_init_sigma "$BAYES_LORA_INIT_SIGMA" \
                --bayes_lora_prior_sigma "$BAYES_LORA_PRIOR_SIGMA" \
                --bayes_lora_max_sigma_u "$BAYES_LORA_MAX_SIGMA_U" \
                --bayes_lora_lambda_init "$BAYES_LORA_LAMBDA_INIT" \
                --bayes_lora_lambda_max "$BAYES_LORA_LAMBDA_MAX" \
                --posthoc_temp_scaling --save_dir "$EVAL_DIR"
            run_step "eval_${name}_m${BAYES_LORA_MC_SAMPLES}" python evaluate_all_metrics.py \
                --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" --lora_alpha "$LORA_ALPHA" \
                --full_bayes_lora --bayes_lora_flow "$BAYES_LORA_FLOW" \
                --bayes_lora_flow_depth "$BAYES_LORA_FLOW_DEPTH" \
                --bayes_lora_init_sigma "$BAYES_LORA_INIT_SIGMA" \
                --bayes_lora_prior_sigma "$BAYES_LORA_PRIOR_SIGMA" \
                --bayes_lora_max_sigma_u "$BAYES_LORA_MAX_SIGMA_U" \
                --bayes_lora_lambda_init "$BAYES_LORA_LAMBDA_INIT" \
                --bayes_lora_lambda_max "$BAYES_LORA_LAMBDA_MAX" \
                --bayes_lora_mc_samples "$BAYES_LORA_MC_SAMPLES" \
                --posthoc_temp_scaling --save_dir "$EVAL_DIR"
            ;;
        *full_bayes_lora*)
            run_step "eval_${name}_m0" python evaluate_all_metrics.py \
                --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" --lora_alpha "$LORA_ALPHA" --pace \
                --full_bayes_lora --bayes_lora_flow "$BAYES_LORA_FLOW" \
                --bayes_lora_flow_depth "$BAYES_LORA_FLOW_DEPTH" \
                --bayes_lora_init_sigma "$BAYES_LORA_INIT_SIGMA" \
                --bayes_lora_prior_sigma "$BAYES_LORA_PRIOR_SIGMA" \
                --bayes_lora_max_sigma_u "$BAYES_LORA_MAX_SIGMA_U" \
                --bayes_lora_lambda_init "$BAYES_LORA_LAMBDA_INIT" \
                --bayes_lora_lambda_max "$BAYES_LORA_LAMBDA_MAX" \
                --posthoc_temp_scaling --save_dir "$EVAL_DIR"
            run_step "eval_${name}_m${BAYES_LORA_MC_SAMPLES}" python evaluate_all_metrics.py \
                --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" --lora_alpha "$LORA_ALPHA" --pace \
                --full_bayes_lora --bayes_lora_flow "$BAYES_LORA_FLOW" \
                --bayes_lora_flow_depth "$BAYES_LORA_FLOW_DEPTH" \
                --bayes_lora_init_sigma "$BAYES_LORA_INIT_SIGMA" \
                --bayes_lora_prior_sigma "$BAYES_LORA_PRIOR_SIGMA" \
                --bayes_lora_max_sigma_u "$BAYES_LORA_MAX_SIGMA_U" \
                --bayes_lora_lambda_init "$BAYES_LORA_LAMBDA_INIT" \
                --bayes_lora_lambda_max "$BAYES_LORA_LAMBDA_MAX" \
                --bayes_lora_mc_samples "$BAYES_LORA_MC_SAMPLES" \
                --posthoc_temp_scaling --save_dir "$EVAL_DIR"
            ;;
        *_blob*)
            run_step "eval_${name}_m0" python evaluate_all_metrics.py \
                --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" --pace \
                --blob --blob_prior_sigma "$BLOB_PRIOR_SIGMA" --blob_init_sigma "$BLOB_INIT_SIGMA" \
                --posthoc_temp_scaling --save_dir "$EVAL_DIR"
            run_step "eval_${name}_m${BLOB_MC_SAMPLES}" python evaluate_all_metrics.py \
                --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" --pace \
                --blob --blob_prior_sigma "$BLOB_PRIOR_SIGMA" --blob_init_sigma "$BLOB_INIT_SIGMA" \
                --blob_mc_samples "$BLOB_MC_SAMPLES" \
                --posthoc_temp_scaling --save_dir "$EVAL_DIR"
            ;;
        *pacbayes*)
            run_step "eval_${name}" python evaluate_all_metrics.py \
                --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" --pace --pacbayes \
                --pac_lbd "$PAC_LBD" --pac_prior_sigma "$PAC_PRIOR_SIGMA" \
                --posthoc_temp_scaling --save_dir "$EVAL_DIR"
            ;;
        *pacpgd*)
            run_step "eval_${name}" python evaluate_all_metrics.py \
                --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" --pace --pac_pgd \
                --pac_pgd_stage1_epochs "$PAC_PGD_STAGE1_EPOCHS" \
                --pac_pgd_lbd "$PAC_PGD_LBD" --pac_pgd_gamma "$PAC_PGD_GAMMA" \
                --pac_pgd_init_floor "$PAC_PGD_INIT_FLOOR" \
                --pac_pgd_prior_floor "$PAC_PGD_PRIOR_FLOOR" \
                --posthoc_temp_scaling --save_dir "$EVAL_DIR"
            run_step "eval_${name}_mc${PAC_PGD_MC_SAMPLES}" python evaluate_all_metrics.py \
                --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" --pace --pac_pgd \
                --pac_pgd_stage1_epochs "$PAC_PGD_STAGE1_EPOCHS" \
                --pac_pgd_lbd "$PAC_PGD_LBD" --pac_pgd_gamma "$PAC_PGD_GAMMA" \
                --pac_pgd_init_floor "$PAC_PGD_INIT_FLOOR" \
                --pac_pgd_prior_floor "$PAC_PGD_PRIOR_FLOOR" \
                --pac_pgd_mc_samples "$PAC_PGD_MC_SAMPLES" \
                --posthoc_temp_scaling --save_dir "$EVAL_DIR"
            ;;
        *ivon*)
            run_step "eval_${name}_mean" python evaluate_all_metrics.py \
                --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" --pace --ivon \
                --ivon_ess "$IVON_ESS" --ivon_hess_init "$IVON_HESS_INIT" \
                --ivon_clip_radius "$IVON_CLIP_RADIUS" --ivon_beta2 "$IVON_BETA2" \
                --posthoc_temp_scaling --save_dir "$EVAL_DIR"
            run_step "eval_${name}_mc${IVON_MC_SAMPLES}" python evaluate_all_metrics.py \
                --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" --pace --ivon \
                --ivon_ess "$IVON_ESS" --ivon_hess_init "$IVON_HESS_INIT" \
                --ivon_clip_radius "$IVON_CLIP_RADIUS" --ivon_beta2 "$IVON_BETA2" \
                --ivon_mc_samples "$IVON_MC_SAMPLES" \
                --posthoc_temp_scaling --save_dir "$EVAL_DIR"
            ;;
        *)
            run_step "eval_${name}" python evaluate_all_metrics.py \
                --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" --pace \
                --posthoc_temp_scaling --save_dir "$EVAL_DIR"
            ;;
    esac
}

log "Experiment run starting in $(pwd)"
log "Config: EPOCHS=$EPOCHS SEEDS=[$SEEDS] DATASETS=[$DATASETS] METHODS=[$METHODS] ADAPTER=$ADAPTER"
log "Dirs: OUT_DIR=$OUT_DIR EVAL_DIR=$EVAL_DIR LOG_DIR=$LOG_DIR"

for dataset in $DATASETS; do
    if ! dataset_exists "$dataset"; then
        log "SKIP:  dataset $dataset not found under data/vtab-1k"
        continue
    fi
    for seed in $SEEDS; do
        for method in $METHODS; do
            train_method "$method" "$dataset" "$seed"
        done
    done
done

log "Experiment run finished."
