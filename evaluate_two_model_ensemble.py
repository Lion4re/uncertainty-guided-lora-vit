#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from timm.models import create_model

import pace
import utils
from evaluate_all_metrics import fit_temperature, summarize_metrics


def build_model(args, checkpoint, adapter_name, pacbayes=False):
    class_dim = utils.get_vtab_classes_num(args.dataset)
    model = create_model("vit_base_patch16_224_in21k", checkpoint_path="./ViT-B_16.npz", drop_path_rate=0.1)
    model.reset_classifier(class_dim)
    pace.inject_residual_adapter(model, adapter=adapter_name, rank=args.rank)

    if args.pace:
        adapters_and_block_ids = pace.get_adapters_and_block_ids(model)
        num_blocks = max(bid for _, _, bid in adapters_and_block_ids) + 1
        sigmas = np.concatenate([np.zeros(1), np.linspace(0, args.sigma, num_blocks)[-1:0:-1]])
        for parent, name, bid in adapters_and_block_ids:
            adapter = getattr(parent, name)
            if pacbayes:
                wrapped = pace.LearnableMultiplicativeNoiseAdapter(
                    adapter,
                    init_sigma=sigmas[bid],
                    prior_sigma=args.pac_prior_sigma,
                    adapter_dropout=0.0,
                )
            else:
                wrapped = pace.MultiplicativeNoiseAdapter(adapter, sigma=sigmas[bid], adapter_dropout=0.0)
            setattr(parent, name, wrapped)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    status = model.load_state_dict(ckpt, strict=False)
    return model, status


def compute_ece(logits, labels, n_bins=15):
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, dim=1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        n_in_bin = in_bin.float().sum()
        if n_in_bin > 0:
            avg_conf = confidences[in_bin].mean().item()
            avg_acc = accuracies[in_bin].float().mean().item()
            ece += abs(avg_conf - avg_acc) * n_in_bin
            bin_accs.append(avg_acc)
            bin_confs.append(avg_conf)
            bin_counts.append(n_in_bin.item())
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_counts.append(0)
    return (ece / len(labels)).item(), bin_accs, bin_confs, bin_counts


def plot_reliability(metrics, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = np.asarray(metrics["bin_accs"], dtype=float)
    conf = np.asarray(metrics["bin_confs"], dtype=float)
    counts = np.asarray(metrics["bin_counts"], dtype=float)
    active = counts > 0

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Perfect calibration")
    ax.plot(conf[active], acc[active], marker="o", label="Ensemble")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_a", required=True)
    parser.add_argument("--checkpoint_b", required=True)
    parser.add_argument("--label_a", default="model_a")
    parser.add_argument("--label_b", default="model_b")
    parser.add_argument("--pacbayes_a", action="store_true")
    parser.add_argument("--pacbayes_b", action="store_true")
    parser.add_argument("--dataset", default="dtd")
    parser.add_argument("--adapter", default="LoRAmul_VPTadd")
    parser.add_argument("--adapter_a", default=None)
    parser.add_argument("--adapter_b", default=None)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--pace", action="store_true", default=True)
    parser.add_argument("--pac_prior_sigma", type=float, default=1.2)
    parser.add_argument("--posthoc_temp_scaling", action="store_true")
    parser.add_argument(
        "--temperature_protocol",
        choices=["oracle", "split"],
        default="oracle",
        help=(
            "oracle fits T on the full eval set; split fits T on a random calibration "
            "subset and reports calibrated metrics on the remaining subset."
        ),
    )
    parser.add_argument("--temperature_calib_fraction", type=float, default=0.5)
    parser.add_argument("--temperature_split_seed", type=int, default=42)
    parser.add_argument("--save_dir", default="results/ensemble_dtd")
    parser.add_argument("--plot_dir", default="plots_ensemble_dtd")
    parser.add_argument("--timing_warmup", type=int, default=5)
    parser.add_argument("--timing_batches", type=int, default=20)
    args = parser.parse_args()
    args.adapter_a = args.adapter_a or args.adapter
    args.adapter_b = args.adapter_b or args.adapter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    _, val_dl = utils.get_vtab_data(
        args.dataset,
        evaluate=True,
        batch_size=64,
        num_workers=4,
        is_hdf5=False,
    )

    model_a, status_a = build_model(args, args.checkpoint_a, args.adapter_a, pacbayes=args.pacbayes_a)
    model_b, status_b = build_model(args, args.checkpoint_b, args.adapter_b, pacbayes=args.pacbayes_b)
    model_a = model_a.to(device).eval()
    model_b = model_b.to(device).eval()

    timing_total_images = 0
    timing_total_time = 0.0
    all_logits, all_labels = [], []
    with torch.no_grad():
        for bi, batch in enumerate(val_dl):
            x, y = batch[0].to(device), batch[1].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            probs_a = F.softmax(model_a(x), dim=1)
            probs_b = F.softmax(model_b(x), dim=1)
            avg_probs = ((probs_a + probs_b) / 2.0).clamp_min(1e-12)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if bi >= args.timing_warmup and bi < args.timing_warmup + args.timing_batches:
                timing_total_images += x.shape[0]
                timing_total_time += elapsed
            all_logits.append(avg_probs.log().cpu())
            all_labels.append(y.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = summarize_metrics(logits, labels)
    ece, bin_accs, bin_confs, bin_counts = compute_ece(logits, labels)
    metrics["ece"] = ece
    metrics["bin_accs"] = bin_accs
    metrics["bin_confs"] = bin_confs
    metrics["bin_counts"] = bin_counts
    metrics["mean_logit_norm"] = None
    metrics["mean_max_logit"] = None
    metrics["std_max_logit"] = None

    calibrated_metrics = None
    posthoc_temperature = None
    temperature_eval_raw_metrics = None
    temperature_protocol_details = None
    if args.posthoc_temp_scaling:
        if args.temperature_protocol == "oracle":
            posthoc_temperature = fit_temperature(logits, labels)
            calibrated_metrics = summarize_metrics(logits / posthoc_temperature, labels)
            temperature_protocol_details = {
                "protocol": "oracle_eval_set",
                "fit_count": int(labels.numel()),
                "eval_count": int(labels.numel()),
            }
        else:
            if not 0.0 < args.temperature_calib_fraction < 1.0:
                raise ValueError("--temperature_calib_fraction must be between 0 and 1 for split protocol")
            generator = torch.Generator().manual_seed(args.temperature_split_seed)
            perm = torch.randperm(labels.numel(), generator=generator)
            calib_count = max(1, min(labels.numel() - 1, int(round(labels.numel() * args.temperature_calib_fraction))))
            calib_idx = perm[:calib_count]
            eval_idx = perm[calib_count:]
            posthoc_temperature = fit_temperature(logits[calib_idx], labels[calib_idx])
            temperature_eval_raw_metrics = summarize_metrics(logits[eval_idx], labels[eval_idx])
            calibrated_metrics = summarize_metrics(logits[eval_idx] / posthoc_temperature, labels[eval_idx])
            temperature_protocol_details = {
                "protocol": "random_split",
                "split_seed": args.temperature_split_seed,
                "calibration_fraction": args.temperature_calib_fraction,
                "fit_count": int(calib_idx.numel()),
                "eval_count": int(eval_idx.numel()),
            }

    timing = {
        "timed_images": timing_total_images,
        "total_seconds": timing_total_time,
        "seconds_per_image": timing_total_time / timing_total_images if timing_total_images else None,
        "images_per_second": timing_total_images / timing_total_time if timing_total_time else None,
    }

    result = {
        "dataset": args.dataset,
        "ensemble": True,
        "ensemble_type": "softmax_average",
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "pacbayes_a": args.pacbayes_a,
        "pacbayes_b": args.pacbayes_b,
        "adapter": args.adapter,
        "adapter_a": args.adapter_a,
        "adapter_b": args.adapter_b,
        "sigma": args.sigma,
        "load_state_a_missing_key_count": len(status_a.missing_keys),
        "load_state_a_unexpected_key_count": len(status_a.unexpected_keys),
        "load_state_b_missing_key_count": len(status_b.missing_keys),
        "load_state_b_unexpected_key_count": len(status_b.unexpected_keys),
        **metrics,
        "posthoc_temperature": posthoc_temperature,
        "temperature_protocol": temperature_protocol_details,
        "temperature_eval_raw_metrics": temperature_eval_raw_metrics,
        "posthoc_temperature_scaled": calibrated_metrics,
        "inference_timing": timing,
    }

    save_name = f"ensemble_{args.label_a}_plus_{args.label_b}_{args.dataset}".replace("/", "_").replace(" ", "_")
    save_path = Path(args.save_dir) / f"{save_name}.json"
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    plot_path = Path(args.plot_dir) / f"{save_name}_reliability.png"
    plot_reliability(metrics, plot_path, f"{args.dataset.upper()} ensemble reliability")
    calibrated_plot_path = None
    if calibrated_metrics:
        calibrated_plot_path = Path(args.plot_dir) / f"{save_name}_temperature_scaled_reliability.png"
        plot_reliability(
            calibrated_metrics,
            calibrated_plot_path,
            f"{args.dataset.upper()} ensemble reliability after temperature scaling",
        )

    print("=" * 60)
    print(f"Ensemble: {args.label_a} + {args.label_b}")
    print(f"Dataset:  {args.dataset}")
    print("=" * 60)
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"ECE:            {metrics['ece']:.4f}")
    print(f"NLL:            {metrics['nll']:.4f}")
    print(f"Brier:          {metrics['brier']:.4f}")
    print(f"Avg confidence: {metrics['avg_confidence']:.4f}")
    if calibrated_metrics:
        print(f"Post-hoc Temp:  {posthoc_temperature:.4f}")
        if temperature_protocol_details:
            print(f"TS protocol:    {temperature_protocol_details['protocol']}")
            print(f"TS fit/eval n:  {temperature_protocol_details['fit_count']}/{temperature_protocol_details['eval_count']}")
        if temperature_eval_raw_metrics:
            print(f"Split raw ECE:  {temperature_eval_raw_metrics['ece']:.4f}")
            print(f"Split raw NLL:  {temperature_eval_raw_metrics['nll']:.4f}")
            print(f"Split raw Brier:{temperature_eval_raw_metrics['brier']:.4f}")
        print(f"TS ECE:         {calibrated_metrics['ece']:.4f}")
        print(f"TS NLL:         {calibrated_metrics['nll']:.4f}")
        print(f"TS Brier:       {calibrated_metrics['brier']:.4f}")
    print(f"Results saved to {save_path}")
    print(f"Reliability plot saved to {plot_path}")
    if calibrated_plot_path:
        print(f"Temperature-scaled reliability plot saved to {calibrated_plot_path}")


if __name__ == "__main__":
    main()
