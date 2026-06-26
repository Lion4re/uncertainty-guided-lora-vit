#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="${RESULT_DIR:-results/single_model_tssplit_main_june26}"
LOG_DIR="${LOG_DIR:-logs/single_model_tssplit_main_june26}"
DATASETS="${DATASETS:-cifar caltech101 dtd}"
SEEDS="${SEEDS:-42 123 456}"
METHODS="${METHODS:-lora pace pace_kl pace_pacbayes pace_kl_pacbayes}"
CALIB_FRACTION="${CALIB_FRACTION:-0.5}"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

checkpoint_for() {
    local method="$1"
    local seedtag="$2"
    local dataset="$3"

    case "$method" in
        lora)
            for candidate in \
                "outs/LoRAmul_VPTadd_R10${seedtag}_${dataset}/weight.pt" \
                "outputs/LoRAmul_VPTadd_R10${seedtag}_${dataset}/weight.pt" \
                "outs/lora_LoRAmul_VPTadd_R10${seedtag}_${dataset}/weight.pt" \
                "outputs/lora_LoRAmul_VPTadd_R10${seedtag}_${dataset}/weight.pt"; do
                if [[ -f "$candidate" ]]; then
                    echo "$candidate"
                    return 0
                fi
            done
            ;;
        pace)
            echo "outs/pace_Lbd1_S1.2_LoRAmul_VPTadd_R10${seedtag}_${dataset}/weight.pt"
            ;;
        pace_kl)
            echo "outs/pace_kl_Lbd0.5_S1.2_LoRAmul_VPTadd_R10${seedtag}_${dataset}/weight.pt"
            ;;
        pace_pacbayes)
            echo "outs/pace_pacbayes_Lbd1_S1.2_LoRAmul_VPTadd_R10${seedtag}_${dataset}/weight.pt"
            ;;
        pace_kl_pacbayes)
            echo "outs/pace_kl_pacbayes_Lbd0.5_S1.2_T1_LoRAmul_VPTadd_R10${seedtag}_${dataset}/weight.pt"
            ;;
        *)
            return 1
            ;;
    esac
}

flags_for() {
    local method="$1"
    case "$method" in
        lora)
            ;;
        pace|pace_kl)
            printf '%s\n' --pace
            ;;
        pace_pacbayes|pace_kl_pacbayes)
            printf '%s\n' --pace --pacbayes --pac_prior_sigma 1.2
            ;;
    esac
}

echo "Single-model held-out temperature scaling run"
echo "RESULT_DIR=$RESULT_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "DATASETS=[$DATASETS]"
echo "SEEDS=[$SEEDS]"
echo "METHODS=[$METHODS]"

for dataset in $DATASETS; do
    for seed in $SEEDS; do
        if [[ "$seed" == "42" ]]; then
            seedtag=""
        else
            seedtag="_Seed${seed}"
        fi

        for method in $METHODS; do
            ckpt="$(checkpoint_for "$method" "$seedtag" "$dataset")"
            run_name="${method}${seedtag}_${dataset}"

            if [[ ! -f "$ckpt" ]]; then
                echo "SKIP missing $run_name: $ckpt"
                continue
            fi

            mapfile -t method_flags < <(flags_for "$method")
            echo "START $run_name"
            python evaluate_all_metrics.py \
                --checkpoint "$ckpt" \
                --dataset "$dataset" \
                --adapter LoRAmul_VPTadd \
                --rank 10 \
                --sigma 1.2 \
                "${method_flags[@]}" \
                --posthoc_temp_scaling \
                --temperature_protocol split \
                --temperature_calib_fraction "$CALIB_FRACTION" \
                --temperature_split_seed "$seed" \
                --save_dir "$RESULT_DIR" \
                > "$LOG_DIR/${run_name}.out" \
                2> "$LOG_DIR/${run_name}.err"
            echo "DONE $run_name"
        done
    done
done

echo "FINISHED single-model held-out temperature scaling run"
