#!/usr/bin/env bash
set -uo pipefail

# Focused uncertainty-guided runner:
# LoRA/PACE baselines + uncertainty-guided PACE/PACE-KL, clean/shift/OOD evaluation.
#
# Run from the repository root:
#   bash runners/run_uncertainty_guided_experiments.sh

LOG_DIR="${LOG_DIR:-logs/uncertainty_guided}"
EVAL_DIR="${EVAL_DIR:-results/uncertainty_guided_vtab}"
OUT_DIR="${OUT_DIR:-outputs/uncertainty_guided_checkpoints}"
mkdir -p "$LOG_DIR" "$EVAL_DIR" "$OUT_DIR"
RUN_LOG="${LOG_DIR}/run_uncertainty_guided_$(date +%Y%m%d_%H%M%S).log"
touch "$RUN_LOG"

EPOCHS="${EPOCHS:-300}"
BS="${BS:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-03}"
WD="${WD:-1e-4}"
RANK="${RANK:-10}"
ADAPTER="${ADAPTER:-LoRAmul_VPTadd}"
SIGMA="${SIGMA:-1.2}"
LBD="${LBD:-0.5}"
SEEDS="${SEEDS:-42 123 456}"
DATASETS="${DATASETS:-cifar caltech101 dtd}"
METHODS="${METHODS:-baseline pace pace_uncert_soft pace_uncert_topk pace_kl_t2 pace_kl_uncert_soft pace_kl_uncert_topk}"
SHIFTS="${SHIFTS:-clean gaussian_noise gaussian_blur brightness contrast cutout}"
SHIFT_SEVERITIES="${SHIFT_SEVERITIES:-1 2 3}"
OOD_SCORE="${OOD_SCORE:-entropy}"
UNCERTAINTY_FRACTION="${UNCERTAINTY_FRACTION:-0.30}"
UNCERTAINTY_WEIGHT="${UNCERTAINTY_WEIGHT:-1.0}"
UNCERTAINTY_SCORE="${UNCERTAINTY_SCORE:-entropy}"

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

method_name() {
    local method="$1"
    local dataset="$2"
    local seed="$3"
    case "$method" in
        baseline)
            name="${ADAPTER}_R${RANK}"
            ;;
        pace)
            name="pace_Lbd1_S${SIGMA}_${ADAPTER}_R${RANK}"
            ;;
        pace_uncert_soft)
            name="pace_uncert_soft_Lbd1_S${SIGMA}"
            if [ "$UNCERTAINTY_SCORE" != "entropy" ]; then
                name="${name}_Us${UNCERTAINTY_SCORE}"
            fi
            if [ "$(fmt_g "$UNCERTAINTY_WEIGHT")" != "1" ]; then
                name="${name}_Uw$(fmt_g "$UNCERTAINTY_WEIGHT")"
            fi
            name="${name}_${ADAPTER}_R${RANK}"
            ;;
        pace_uncert_topk)
            name="pace_uncert_topk_Lbd1_S${SIGMA}"
            if [ "$UNCERTAINTY_SCORE" != "entropy" ]; then
                name="${name}_Us${UNCERTAINTY_SCORE}"
            fi
            if [ "$(fmt_g "$UNCERTAINTY_FRACTION")" != "0.3" ]; then
                name="${name}_Uf$(fmt_g "$UNCERTAINTY_FRACTION")"
            fi
            name="${name}_${ADAPTER}_R${RANK}"
            ;;
        pace_kl_t2)
            name="pace_kl_Lbd${LBD}_S${SIGMA}_${ADAPTER}_R${RANK}"
            ;;
        pace_kl_uncert_soft)
            name="pace_kl_uncert_soft_Lbd${LBD}_S${SIGMA}"
            if [ "$UNCERTAINTY_SCORE" != "entropy" ]; then
                name="${name}_Us${UNCERTAINTY_SCORE}"
            fi
            if [ "$(fmt_g "$UNCERTAINTY_WEIGHT")" != "1" ]; then
                name="${name}_Uw$(fmt_g "$UNCERTAINTY_WEIGHT")"
            fi
            name="${name}_${ADAPTER}_R${RANK}"
            ;;
        pace_kl_uncert_topk)
            name="pace_kl_uncert_topk_Lbd${LBD}_S${SIGMA}"
            if [ "$UNCERTAINTY_SCORE" != "entropy" ]; then
                name="${name}_Us${UNCERTAINTY_SCORE}"
            fi
            if [ "$(fmt_g "$UNCERTAINTY_FRACTION")" != "0.3" ]; then
                name="${name}_Uf$(fmt_g "$UNCERTAINTY_FRACTION")"
            fi
            name="${name}_${ADAPTER}_R${RANK}"
            ;;
        *)
            log "SKIP: unknown method $method"
            return 1
            ;;
    esac
    if [ "$seed" != "42" ]; then
        name="${name}_Seed${seed}"
    fi
    echo "${name}_${dataset}"
}

default_ood_dataset() {
    case "$1" in
        cifar) echo "svhn" ;;
        caltech101) echo "dtd" ;;
        dtd) echo "caltech101" ;;
        *) echo "" ;;
    esac
}

train_method() {
    local method="$1"
    local dataset="$2"
    local seed="$3"
    local name
    name="$(method_name "$method" "$dataset" "$seed")" || return 0
    local checkpoint="${OUT_DIR}/${name}/weight.pt"
    if [ -f "$checkpoint" ]; then
        log "SKIP:  train_${name} existing checkpoint"
    else
        case "$method" in
            baseline)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" --dataset "$dataset" --lr "$LR" --wd "$WD" \
                    --adapter "$ADAPTER" --rank "$RANK" --epoch "$EPOCHS" --bs "$BS" \
                    --num_workers "$NUM_WORKERS" --test_every "$EPOCHS" --seed "$seed"
                ;;
            pace)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" --dataset "$dataset" --lr "$LR" --wd "$WD" \
                    --adapter "$ADAPTER" --rank "$RANK" --epoch "$EPOCHS" --bs "$BS" \
                    --num_workers "$NUM_WORKERS" --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace --lbd 1 --sigma "$SIGMA"
                ;;
            pace_uncert_soft)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" --dataset "$dataset" --lr "$LR" --wd "$WD" \
                    --adapter "$ADAPTER" --rank "$RANK" --epoch "$EPOCHS" --bs "$BS" \
                    --num_workers "$NUM_WORKERS" --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_uncert_soft --lbd 1 --sigma "$SIGMA" \
                    --uncertainty_score "$UNCERTAINTY_SCORE" \
                    --uncertainty_weight "$UNCERTAINTY_WEIGHT"
                ;;
            pace_uncert_topk)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" --dataset "$dataset" --lr "$LR" --wd "$WD" \
                    --adapter "$ADAPTER" --rank "$RANK" --epoch "$EPOCHS" --bs "$BS" \
                    --num_workers "$NUM_WORKERS" --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_uncert_topk --lbd 1 --sigma "$SIGMA" \
                    --uncertainty_score "$UNCERTAINTY_SCORE" \
                    --uncertainty_fraction "$UNCERTAINTY_FRACTION"
                ;;
            pace_kl_t2)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" --dataset "$dataset" --lr "$LR" --wd "$WD" \
                    --adapter "$ADAPTER" --rank "$RANK" --epoch "$EPOCHS" --bs "$BS" \
                    --num_workers "$NUM_WORKERS" --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl --lbd "$LBD" --sigma "$SIGMA" --temperature 2.0
                ;;
            pace_kl_uncert_soft)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" --dataset "$dataset" --lr "$LR" --wd "$WD" \
                    --adapter "$ADAPTER" --rank "$RANK" --epoch "$EPOCHS" --bs "$BS" \
                    --num_workers "$NUM_WORKERS" --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl_uncert_soft --lbd "$LBD" --sigma "$SIGMA" \
                    --temperature 2.0 --uncertainty_score "$UNCERTAINTY_SCORE" \
                    --uncertainty_weight "$UNCERTAINTY_WEIGHT"
                ;;
            pace_kl_uncert_topk)
                run_step "train_${name}" python train.py \
                    --out_dir "$OUT_DIR" --dataset "$dataset" --lr "$LR" --wd "$WD" \
                    --adapter "$ADAPTER" --rank "$RANK" --epoch "$EPOCHS" --bs "$BS" \
                    --num_workers "$NUM_WORKERS" --test_every "$EPOCHS" --seed "$seed" \
                    --pace_type pace_kl_uncert_topk --lbd "$LBD" --sigma "$SIGMA" \
                    --temperature 2.0 --uncertainty_score "$UNCERTAINTY_SCORE" \
                    --uncertainty_fraction "$UNCERTAINTY_FRACTION"
                ;;
        esac
    fi
    eval_method "$method" "$dataset" "$seed"
}

eval_once() {
    local method="$1"
    local name="$2"
    local checkpoint="$3"
    local dataset="$4"
    local shift="$5"
    local severity="$6"
    shift_args=(--eval_shift "$shift" --shift_severity "$severity")
    if [ "$shift" = "clean" ]; then
        shift_args=(--eval_shift clean --shift_severity 0)
    fi
    pace_args=()
    if [ "$method" != "baseline" ]; then
        pace_args=(--pace)
    fi
    run_step "eval_${name}_${shift}_s${severity}" python evaluate_all_metrics.py \
        --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" "${pace_args[@]}" \
        "${shift_args[@]}" --posthoc_temp_scaling --save_dir "$EVAL_DIR"
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
    for shift in $SHIFTS; do
        if [ "$shift" = "clean" ]; then
            eval_once "$method" "$name" "$checkpoint" "$dataset" clean 0
        else
            for severity in $SHIFT_SEVERITIES; do
                eval_once "$method" "$name" "$checkpoint" "$dataset" "$shift" "$severity"
            done
        fi
    done
    local ood_dataset
    ood_dataset="$(default_ood_dataset "$dataset")"
    if [ -n "$ood_dataset" ] && dataset_exists "$ood_dataset"; then
        pace_args=()
        if [ "$method" != "baseline" ]; then
            pace_args=(--pace)
        fi
        run_step "eval_${name}_ood_${ood_dataset}" python evaluate_all_metrics.py \
            --checkpoint "$checkpoint" --dataset "$dataset" --adapter "$ADAPTER" "${pace_args[@]}" \
            --ood_dataset "$ood_dataset" --ood_score "$OOD_SCORE" \
            --posthoc_temp_scaling --save_dir "$EVAL_DIR"
    fi
}

log "Uncertainty-guided run starting in $(pwd)"
log "Config: EPOCHS=$EPOCHS SEEDS=[$SEEDS] DATASETS=[$DATASETS] METHODS=[$METHODS]"

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

log "Uncertainty-guided run finished."
