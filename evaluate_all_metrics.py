"""
Evaluate all metrics: Accuracy, ECE, NLL, Brier Score, Confidence stats, Logit stats.
Saves reliability diagram data and prints everything.
Usage: python evaluate_all_metrics.py --checkpoint PATH [--pace] [--rank 10] [--dataset cifar]
"""
import torch
import torch.nn.functional as F
import os
import json
import numpy as np
import time
import re
from argparse import ArgumentParser
from timm.models import create_model
import utils
import pace

def compute_ece(logits, labels, n_bins=15):
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, dim=1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i+1]
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

def compute_nll(logits, labels):
    return F.cross_entropy(logits, labels).item()

def compute_brier(logits, labels):
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, num_classes=probs.shape[1]).float()
    return ((probs - one_hot) ** 2).sum(dim=1).mean().item()

def summarize_metrics(logits, labels):
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    ece, bin_accs, bin_confs, bin_counts = compute_ece(logits, labels)
    nll = compute_nll(logits, labels)
    brier = compute_brier(logits, labels)
    softmaxes = F.softmax(logits, dim=1)
    confidences = softmaxes.max(dim=1)[0]
    correct_mask = preds == labels
    wrong_mask = ~correct_mask
    max_logits = logits.max(dim=1)[0]
    logit_norms = logits.norm(dim=1)
    return {
        'accuracy': acc,
        'ece': ece,
        'nll': nll,
        'brier': brier,
        'avg_confidence': confidences.mean().item(),
        'correct_confidence': confidences[correct_mask].mean().item() if correct_mask.any() else None,
        'wrong_confidence': confidences[wrong_mask].mean().item() if wrong_mask.any() else None,
        'mean_max_logit': max_logits.mean().item(),
        'std_max_logit': max_logits.std().item(),
        'mean_logit_norm': logit_norms.mean().item(),
        'bin_accs': bin_accs,
        'bin_confs': bin_confs,
        'bin_counts': bin_counts,
        'confidence_histogram': confidences.numpy().tolist(),
        'max_logit_histogram': max_logits.numpy().tolist(),
    }

def compute_uncertainty_scores(logits, score="entropy"):
    probs = F.softmax(logits, dim=1)
    if score == "entropy":
        return -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)
    if score == "max_conf":
        return 1.0 - probs.max(dim=1).values
    if score == "margin":
        top2 = probs.topk(k=min(2, probs.shape[1]), dim=1).values
        if top2.shape[1] == 1:
            return torch.ones_like(top2[:, 0])
        return 1.0 - (top2[:, 0] - top2[:, 1])
    raise ValueError(f"Unknown OOD score: {score}")

def compute_ood_detection_metrics(id_scores, ood_scores):
    id_scores = torch.as_tensor(id_scores).float()
    ood_scores = torch.as_tensor(ood_scores).float()
    scores = torch.cat([id_scores, ood_scores])
    labels = torch.cat([torch.zeros_like(id_scores), torch.ones_like(ood_scores)])
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=scores.dtype, device=scores.device)
    pos_ranks = ranks[labels == 1].sum()
    n_pos = ood_scores.numel()
    n_neg = id_scores.numel()
    auroc = (pos_ranks - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1)

    threshold = torch.quantile(ood_scores, 0.05) if n_pos > 1 else ood_scores.min()
    fpr95 = (id_scores >= threshold).float().mean()
    return {
        'auroc': auroc.item(),
        'fpr95': fpr95.item(),
        'id_score_mean': id_scores.mean().item(),
        'ood_score_mean': ood_scores.mean().item(),
        'id_score_std': id_scores.std(unbiased=False).item() if n_neg > 1 else 0.0,
        'ood_score_std': ood_scores.std(unbiased=False).item() if n_pos > 1 else 0.0,
        'threshold_95_tpr': threshold.item(),
    }

def collect_deterministic_logits(model, data_loader, device):
    logits, labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            logits.append(model(x).cpu())
            labels.append(batch[1].cpu())
    return torch.cat(logits), torch.cat(labels)

def fit_temperature(logits, labels, max_iter=50):
    temperature = torch.ones(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature.clamp_min(1e-6), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.detach().clamp_min(1e-6).item()

def fmt_metric(value, precision=4):
    return "NA" if value is None else f"{value:.{precision}f}"

def infer_adapter_dropout_from_checkpoint(checkpoint):
    match = re.search(r"Drop([0-9]+(?:\.[0-9]+)?)", checkpoint)
    if not match:
        return 0.0
    return float(match.group(1))

def infer_uncertainty_config_from_checkpoint(checkpoint):
    ckpt_name = os.path.basename(os.path.dirname(checkpoint))
    guided = "pace_kl_uncert_" in ckpt_name
    score_match = re.search(r"_Us(entropy|max_conf|margin)(?:_|$)", ckpt_name)
    fraction_match = re.search(r"_Uf([0-9]+(?:\.[0-9]+)?)", ckpt_name)
    weight_match = re.search(r"_Uw([0-9]+(?:\.[0-9]+)?)", ckpt_name)
    return {
        "uncertainty_guided": guided,
        "uncertainty_score": (score_match.group(1) if score_match else "entropy") if guided else None,
        "uncertainty_fraction": (float(fraction_match.group(1)) if fraction_match else 0.30)
        if "pace_kl_uncert_topk" in ckpt_name else None,
        "uncertainty_weight": (float(weight_match.group(1)) if weight_match else 1.0)
        if "pace_kl_uncert_soft" in ckpt_name else None,
    }

def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--adapter', type=str, default='LoRAmul_VPTadd')
    parser.add_argument('--rank', type=int, default=10)
    parser.add_argument('--lora_alpha', type=float, default=None,
                        help='LoRA scaling alpha. Defaults to rank, so alpha/rank keeps existing scale.')
    parser.add_argument('--sigma', type=float, default=1.2)
    parser.add_argument('--adapter_dropout', type=float, default=0.0,
                        help='Adapter-branch dropout probability used by a dropout checkpoint. If omitted, inferred from DropX in the checkpoint path when possible.')
    parser.add_argument('--pace', action='store_true')
    parser.add_argument('--pacbayes', action='store_true',
                        help='Reconstruct PAC-Bayes learnable adapter noise before loading checkpoint.')
    parser.add_argument('--learnsigma', action='store_true',
                        help='Reconstruct learnable adapter noise without marking the method as PAC-Bayes.')
    parser.add_argument('--pac_lbd', type=float, default=1e-3)
    parser.add_argument('--pac_prior_sigma', type=float, default=1.2)
    parser.add_argument('--pac_pgd', action='store_true',
                        help='Reconstruct PAC-PGD parameter-noise state before loading checkpoint.')
    parser.add_argument('--pac_pgd_mc_samples', type=int, default=0,
                        help='Number of PAC-PGD parameter-noise samples to average at inference. 0 uses deterministic weights.')
    parser.add_argument('--pac_pgd_stage1_epochs', type=int, default=100)
    parser.add_argument('--pac_pgd_lbd', type=float, default=1.0)
    parser.add_argument('--pac_pgd_gamma', type=float, default=0.1)
    parser.add_argument('--pac_pgd_init_floor', type=float, default=1e-4)
    parser.add_argument('--pac_pgd_prior_floor', type=float, default=1e-4)
    parser.add_argument('--blob', action='store_true',
                        help='Enable BLoB adapter parameters before loading checkpoint.')
    parser.add_argument('--blob_mc_samples', type=int, default=0,
                        help='Number of stochastic BLoB posterior samples to average at inference. 0 uses posterior mean.')
    parser.add_argument('--blob_prior_sigma', type=float, default=1.0)
    parser.add_argument('--blob_init_sigma', type=float, default=1e-4)
    parser.add_argument('--full_bayes_lora', action='store_true',
                        help='Reconstruct full Bayesian-LoRA inducing-variable adapter posterior.')
    parser.add_argument('--bayes_lora_mc_samples', type=int, default=0,
                        help='Number of full Bayesian-LoRA posterior samples to average at inference. 0 uses posterior mean.')
    parser.add_argument('--mc_dropout_samples', type=int, default=0,
                        help='Number of MC adapter-dropout forward passes to average at inference. 0 disables MC dropout.')
    parser.add_argument('--bayes_lora_flow', type=str, default='none', choices=['none', 'maf', 'row_maf'])
    parser.add_argument('--bayes_lora_flow_depth', type=int, default=1)
    parser.add_argument('--bayes_lora_init_sigma', type=float, default=1e-4)
    parser.add_argument('--bayes_lora_prior_sigma', type=float, default=0.1)
    parser.add_argument('--bayes_lora_max_sigma_u', type=float, default=0.1)
    parser.add_argument('--bayes_lora_lambda_init', type=float, default=1e-3)
    parser.add_argument('--bayes_lora_lambda_max', type=float, default=3e-2)
    parser.add_argument('--posthoc_temp_scaling', action='store_true',
                        help='Fit one scalar temperature on the evaluation split and report calibrated metrics.')
    parser.add_argument('--temperature_protocol', type=str, default='oracle',
                        choices=['oracle', 'split'],
                        help='Temperature scaling protocol: oracle uses the full eval set; split fits on a random calibration subset and reports on the held-out subset.')
    parser.add_argument('--temperature_calib_fraction', type=float, default=0.5,
                        help='Fraction of evaluation examples used to fit temperature when --temperature_protocol split.')
    parser.add_argument('--temperature_split_seed', type=int, default=42,
                        help='Random seed for the held-out temperature scaling split.')
    parser.add_argument('--eval_shift', type=str, default='clean',
                        choices=['clean', 'gaussian_noise', 'gaussian_blur', 'brightness', 'contrast', 'cutout'],
                        help='Controlled distribution shift applied to the evaluation split.')
    parser.add_argument('--shift_severity', type=int, default=0, choices=[0, 1, 2, 3],
                        help='Severity for controlled distribution shifts. clean should use 0.')
    parser.add_argument('--ood_dataset', type=str, default=None,
                        help='Optional VTAB dataset used as an open-world/OOD proxy. Labels are ignored.')
    parser.add_argument('--ood_score', type=str, default='entropy',
                        choices=['entropy', 'max_conf', 'margin'],
                        help='Uncertainty score used for OOD detection. Larger scores mean more uncertain.')
    parser.add_argument('--ivon', action='store_true',
                        help='Reconstruct IVON optimizer state for variational LoRA MC inference.')
    parser.add_argument('--ivon_state', type=str, default=None,
                        help='Path to ivon_state.pt. Defaults to checkpoint directory / ivon_state.pt.')
    parser.add_argument('--ivon_mc_samples', type=int, default=0,
                        help='Number of IVON posterior samples to average at inference. 0 uses checkpoint mean weights.')
    parser.add_argument('--ivon_ess', type=float, default=1e6)
    parser.add_argument('--ivon_hess_init', type=float, default=1e-3)
    parser.add_argument('--ivon_clip_radius', type=float, default=1e-3)
    parser.add_argument('--ivon_beta2', type=float, default=0.99999)
    parser.add_argument('--timing_warmup', type=int, default=5)
    parser.add_argument('--timing_batches', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default='results/manual_evaluations')
    args = parser.parse_args()
    if args.lora_alpha is None:
        args.lora_alpha = float(args.rank)
    if args.full_bayes_lora and args.adapter != 'LoRAadd':
        raise ValueError("Full Bayesian-LoRA follows the paper's additive LoRA form; evaluate with --adapter LoRAadd.")
    if args.adapter_dropout == 0.0:
        args.adapter_dropout = infer_adapter_dropout_from_checkpoint(args.checkpoint)
    if args.mc_dropout_samples > 0 and not args.pace:
        raise ValueError("--mc_dropout_samples requires --pace so adapter dropout wrappers are reconstructed.")
    if args.mc_dropout_samples > 0 and args.adapter_dropout <= 0:
        raise ValueError("MC dropout requested, but adapter dropout is 0. Pass --adapter_dropout or use a checkpoint path containing Drop0.1/Drop0.2/Drop0.3.")
    if args.pac_pgd_mc_samples > 0 and not args.pac_pgd:
        raise ValueError("--pac_pgd_mc_samples requires --pac_pgd so parameter-noise state is reconstructed.")
    if args.ivon_mc_samples > 0 and not args.ivon:
        raise ValueError("--ivon_mc_samples requires --ivon so the optimizer posterior can be reconstructed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    class_dim = utils.get_vtab_classes_num(args.dataset)
    if args.eval_shift == 'clean':
        args.shift_severity = 0
    _, val_dl = utils.get_vtab_data(
        args.dataset,
        evaluate=True,
        batch_size=64,
        num_workers=4,
        is_hdf5=False,
        eval_shift=args.eval_shift,
        shift_severity=args.shift_severity,
    )

    model = create_model('vit_base_patch16_224_in21k',
                         checkpoint_path='./ViT-B_16.npz', drop_path_rate=0.1)
    model.reset_classifier(class_dim)
    pace.inject_residual_adapter(model, adapter=args.adapter, rank=args.rank)
    if args.blob:
        pace.enable_blob(model, init_sigma=args.blob_init_sigma, prior_sigma=args.blob_prior_sigma)
    if args.full_bayes_lora:
        pace.enable_full_bayesian_lora(
            model,
            pace.BayesianLoRAConfig(
                rank=args.rank,
                lora_alpha=args.lora_alpha,
                flow=args.bayes_lora_flow,
                flow_depth=args.bayes_lora_flow_depth,
                init_sigma=args.bayes_lora_init_sigma,
                prior_sigma=args.bayes_lora_prior_sigma,
                max_sigma_u=args.bayes_lora_max_sigma_u,
                lambda_init=args.bayes_lora_lambda_init,
                lambda_max=args.bayes_lora_lambda_max,
            )
        )

    if args.pace:
        adapters_and_block_ids = pace.get_adapters_and_block_ids(model)
        num_blocks = max(bid for _, _, bid in adapters_and_block_ids) + 1
        sigmas = np.concatenate([np.zeros(1), np.linspace(0, args.sigma, num_blocks)[-1:0:-1]])
        for parent, name, bid in adapters_and_block_ids:
            adapter = getattr(parent, name)
            if args.pacbayes or args.learnsigma:
                noise_adapter = pace.LearnableMultiplicativeNoiseAdapter(
                    adapter, init_sigma=sigmas[bid],
                    prior_sigma=args.pac_prior_sigma,
                    adapter_dropout=args.adapter_dropout)
            else:
                noise_adapter = pace.MultiplicativeNoiseAdapter(
                    adapter, sigma=sigmas[bid],
                    adapter_dropout=args.adapter_dropout)
            setattr(parent, name, noise_adapter)

    pac_pgd_state = None
    if args.pac_pgd:
        pac_pgd_state = pace.build_pac_pgd_state(
            model,
            pace.PACPGDConfig(
                stage1_epochs=args.pac_pgd_stage1_epochs,
                lbd=args.pac_pgd_lbd,
                gamma=args.pac_pgd_gamma,
                init_floor=args.pac_pgd_init_floor,
                prior_floor=args.pac_pgd_prior_floor,
            )
        )
        model.pac_pgd_state = pac_pgd_state

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    load_status = model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()
    ivon_optimizer = None
    ivon_state_config = None
    if args.ivon and args.ivon_mc_samples > 0:
        ivon_params = []
        for name, param in model.named_parameters():
            if name in ckpt:
                param.requires_grad = True
                ivon_params.append(param)
            else:
                param.requires_grad = False
        ivon_optimizer = pace.create_optimizer(
            ivon_params,
            optimizer_name='ivon',
            lr=0.03,
            weight_decay=0.0,
            ivon_ess=args.ivon_ess,
            ivon_hess_init=args.ivon_hess_init,
            ivon_clip_radius=args.ivon_clip_radius,
            ivon_beta2=args.ivon_beta2,
        )
        ivon_state_path = args.ivon_state or pace.default_ivon_state_path(args.checkpoint)
        if args.ivon_mc_samples > 0:
            if not os.path.exists(ivon_state_path):
                raise FileNotFoundError(
                    f"IVON MC inference requires optimizer posterior state at {ivon_state_path}. "
                    "Run IVON training first or pass --ivon_state."
                )
            ivon_state_config = pace.load_ivon_state(ivon_state_path, ivon_optimizer, map_location=device)
    if ivon_optimizer is not None:
        for key, value in list(ivon_optimizer.state.items()):
            ivon_optimizer.state[key] = move_to_device(value, device)

    # Inference timing uses the same evaluation loader and reports rough latency/throughput.
    timing_total_images = 0
    timing_total_time = 0.0
    with torch.no_grad():
        for bi, batch in enumerate(val_dl):
            if bi >= args.timing_warmup + args.timing_batches:
                break
            x = batch[0].to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            if args.mc_dropout_samples > 0:
                pace.set_adapter_mc_dropout(model, True)
                probs = []
                for _sample_idx in range(args.mc_dropout_samples):
                    probs.append(F.softmax(model(x), dim=1))
                pace.set_adapter_mc_dropout(model, False)
                _ = torch.stack(probs).mean(dim=0)
            elif args.pac_pgd and args.pac_pgd_mc_samples > 0:
                probs = []
                for _sample_idx in range(args.pac_pgd_mc_samples):
                    noises = pac_pgd_state.inject_noise()
                    try:
                        probs.append(F.softmax(model(x), dim=1))
                    finally:
                        pac_pgd_state.remove_noise(noises)
                _ = torch.stack(probs).mean(dim=0)
            elif args.full_bayes_lora and args.bayes_lora_mc_samples > 0:
                pace.set_full_bayes_mc_sample(model, True)
                probs = []
                for _sample_idx in range(args.bayes_lora_mc_samples):
                    probs.append(F.softmax(model(x), dim=1))
                pace.set_full_bayes_mc_sample(model, False)
                _ = torch.stack(probs).mean(dim=0)
            elif args.ivon and args.ivon_mc_samples > 0:
                probs = []
                for _sample_idx in range(args.ivon_mc_samples):
                    with pace.sampled_params_context(ivon_optimizer, train=False):
                        probs.append(F.softmax(model(x), dim=1))
                _ = torch.stack(probs).mean(dim=0)
            elif args.blob and args.blob_mc_samples > 0:
                pace.set_blob_mc_sample(model, True)
                probs = []
                for _sample_idx in range(args.blob_mc_samples):
                    probs.append(F.softmax(model(x), dim=1))
                pace.set_blob_mc_sample(model, False)
                _ = torch.stack(probs).mean(dim=0)
            else:
                _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if bi >= args.timing_warmup:
                timing_total_time += elapsed
                timing_total_images += x.shape[0]

    inference_seconds_per_image = timing_total_time / timing_total_images if timing_total_images else None
    inference_images_per_second = timing_total_images / timing_total_time if timing_total_time else None

    active_mc_modes = [
        name for name, enabled in [
            ("dropout", args.mc_dropout_samples > 0),
            ("pac_pgd", args.pac_pgd and args.pac_pgd_mc_samples > 0),
            ("blob", args.blob and args.blob_mc_samples > 0),
            ("full_bayes_lora", args.full_bayes_lora and args.bayes_lora_mc_samples > 0),
            ("ivon", args.ivon and args.ivon_mc_samples > 0),
        ]
        if enabled
    ]
    if len(active_mc_modes) > 1:
        raise ValueError(f"Use one MC inference mode at a time; got {active_mc_modes}")
    mc_mode = bool(active_mc_modes)
    mc_mode_name = active_mc_modes[0] if active_mc_modes else None
    if mc_mode_name == "dropout":
        mc_samples = args.mc_dropout_samples
    elif mc_mode_name == "pac_pgd":
        mc_samples = args.pac_pgd_mc_samples
    elif mc_mode_name == "full_bayes_lora":
        mc_samples = args.bayes_lora_mc_samples
    elif mc_mode_name == "ivon":
        mc_samples = args.ivon_mc_samples
    else:
        mc_samples = args.blob_mc_samples
    all_logits, all_labels = [], []
    mc_prob_std_means = []
    mc_prob_std_maxes = []
    mc_logit_std_means = []
    mc_logit_std_maxes = []
    with torch.no_grad():
        for batch in val_dl:
            x, y = batch[0].to(device), batch[1].to(device)
            if mc_mode:
                if mc_mode_name == "dropout":
                    pace.set_adapter_mc_dropout(model, True)
                if mc_mode_name == "full_bayes_lora":
                    pace.set_full_bayes_mc_sample(model, True)
                if mc_mode_name == "blob":
                    pace.set_blob_mc_sample(model, True)
                probs = []
                logits_samples = []
                for _sample_idx in range(mc_samples):
                    if mc_mode_name == "pac_pgd":
                        noises = pac_pgd_state.inject_noise()
                        try:
                            logits_sample = model(x)
                        finally:
                            pac_pgd_state.remove_noise(noises)
                    elif mc_mode_name == "ivon":
                        with pace.sampled_params_context(ivon_optimizer, train=False):
                            logits_sample = model(x)
                    else:
                        logits_sample = model(x)
                    logits_samples.append(logits_sample)
                    probs.append(F.softmax(logits_sample, dim=1))
                if mc_mode_name == "dropout":
                    pace.set_adapter_mc_dropout(model, False)
                if mc_mode_name == "full_bayes_lora":
                    pace.set_full_bayes_mc_sample(model, False)
                if mc_mode_name == "blob":
                    pace.set_blob_mc_sample(model, False)
                probs = torch.stack(probs)
                logits_samples = torch.stack(logits_samples)
                mc_prob_std_means.append(probs.std(dim=0).mean().item())
                mc_prob_std_maxes.append(probs.std(dim=0).max().item())
                mc_logit_std_means.append(logits_samples.std(dim=0).mean().item())
                mc_logit_std_maxes.append(logits_samples.std(dim=0).max().item())
                avg_probs = probs.mean(dim=0).clamp_min(1e-12)
                # Metrics remain valid because softmax(log(avg_probs)) = avg_probs.
                # Logit diagnostics are disabled below because log-probabilities are
                # not comparable to the model's raw logits.
                all_logits.append(avg_probs.log().cpu())
            else:
                all_logits.append(model(x).cpu())
            all_labels.append(y.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    metrics = summarize_metrics(all_logits, all_labels)
    if mc_mode:
        metrics['mean_max_logit'] = None
        metrics['std_max_logit'] = None
        metrics['mean_logit_norm'] = None
        metrics['max_logit_histogram'] = []
        metrics['mc_sample_prob_std_mean'] = float(np.mean(mc_prob_std_means)) if mc_prob_std_means else None
        metrics['mc_sample_prob_std_max'] = float(np.max(mc_prob_std_maxes)) if mc_prob_std_maxes else None
        metrics['mc_sample_logit_std_mean'] = float(np.mean(mc_logit_std_means)) if mc_logit_std_means else None
        metrics['mc_sample_logit_std_max'] = float(np.max(mc_logit_std_maxes)) if mc_logit_std_maxes else None
    pac_bayes_kl = None
    pac_sigma_summary = None
    if args.pacbayes or args.learnsigma:
        with torch.no_grad():
            pac_kl = pace.pac_bayes_noise_kl(model)
            pac_bayes_kl = pac_kl.item() if pac_kl is not None else None
            pac_sigma_summary = pace.pac_bayes_sigma_summary(model)
    pac_pgd_summary = None
    pac_pgd_kl = None
    pac_pgd_bound = None
    if args.pac_pgd:
        with torch.no_grad():
            pac_pgd_summary = pac_pgd_state.summary()
            pac_pgd_kl_t = pac_pgd_state.kl_term_layer_pb(pac_pgd_state.weight_decay_mulb())
            pac_pgd_kl = pac_pgd_kl_t.item()
            pac_pgd_bound = pac_pgd_state.pac_bound(pac_pgd_kl_t, len(val_dl.dataset)).item()
    blob_kl = None
    blob_sigma_summary = None
    if args.blob:
        with torch.no_grad():
            blob_kl_t = pace.blob_kl(model, reduction='mean')
            blob_kl = blob_kl_t.item() if blob_kl_t is not None else None
            blob_sigma_summary = pace.blob_sigma_summary(model)
    bayes_lora_kl_u = None
    bayes_lora_kl_w = None
    bayes_lora_kl_total = None
    bayes_lora_summary = None
    if args.full_bayes_lora:
        with torch.no_grad():
            terms = pace.full_bayes_lora_kl(model, reduction='mean')
            if terms is not None:
                bayes_lora_kl_u = terms['kl_u'].item()
                bayes_lora_kl_w = terms['kl_w'].item()
                bayes_lora_kl_total = terms['kl_total'].item()
            bayes_lora_summary = pace.full_bayes_lora_summary(model)
    posthoc_temperature = None
    calibrated_metrics = None
    temperature_eval_raw_metrics = None
    temperature_protocol_details = None
    if args.posthoc_temp_scaling:
        if args.temperature_protocol == 'oracle':
            posthoc_temperature = fit_temperature(all_logits, all_labels)
            calibrated_metrics = summarize_metrics(all_logits / posthoc_temperature, all_labels)
            temperature_protocol_details = {
                'protocol': 'oracle_eval_set',
                'fit_count': int(all_labels.numel()),
                'eval_count': int(all_labels.numel()),
            }
        else:
            if not 0.0 < args.temperature_calib_fraction < 1.0:
                raise ValueError("--temperature_calib_fraction must be between 0 and 1 for split protocol")
            generator = torch.Generator().manual_seed(args.temperature_split_seed)
            perm = torch.randperm(all_labels.numel(), generator=generator)
            calib_count = max(
                1,
                min(all_labels.numel() - 1, int(round(all_labels.numel() * args.temperature_calib_fraction))),
            )
            calib_idx = perm[:calib_count]
            eval_idx = perm[calib_count:]
            posthoc_temperature = fit_temperature(all_logits[calib_idx], all_labels[calib_idx])
            temperature_eval_raw_metrics = summarize_metrics(all_logits[eval_idx], all_labels[eval_idx])
            calibrated_metrics = summarize_metrics(all_logits[eval_idx] / posthoc_temperature, all_labels[eval_idx])
            temperature_protocol_details = {
                'protocol': 'random_split',
                'split_seed': args.temperature_split_seed,
                'calibration_fraction': args.temperature_calib_fraction,
                'fit_count': int(calib_idx.numel()),
                'eval_count': int(eval_idx.numel()),
            }

    # Print results
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset:    {args.dataset}")
    print(f"{'='*60}")
    print(f"Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"ECE (15 bins):      {metrics['ece']:.4f}")
    print(f"NLL:                {metrics['nll']:.4f}")
    print(f"Brier Score:        {metrics['brier']:.4f}")
    print(f"Avg Confidence:     {metrics['avg_confidence']:.4f}")
    print(f"Correct Conf:       {fmt_metric(metrics['correct_confidence'])}")
    print(f"Wrong Conf:         {fmt_metric(metrics['wrong_confidence'])}")
    print(f"Mean Max Logit:     {fmt_metric(metrics['mean_max_logit'])}")
    print(f"Std Max Logit:      {fmt_metric(metrics['std_max_logit'])}")
    print(f"Mean Logit Norm:    {fmt_metric(metrics['mean_logit_norm'])}")
    if mc_mode:
        print(f"MC mode:            {mc_mode_name}")
        print(f"MC samples:         {mc_samples}")
        if mc_mode_name == "dropout":
            print(f"Adapter dropout p:  {args.adapter_dropout:g}")
        print(f"MC prob std mean:   {fmt_metric(metrics['mc_sample_prob_std_mean'], precision=8)}")
        print(f"MC prob std max:    {fmt_metric(metrics['mc_sample_prob_std_max'], precision=8)}")
        print(f"MC logit std mean:  {fmt_metric(metrics['mc_sample_logit_std_mean'], precision=8)}")
        print(f"MC logit std max:   {fmt_metric(metrics['mc_sample_logit_std_max'], precision=8)}")
    if inference_images_per_second is not None:
        print(f"Inference img/s:    {inference_images_per_second:.2f}")
        print(f"Inference sec/img:  {inference_seconds_per_image:.6f}")
    if args.pacbayes or args.learnsigma:
        label = "PAC-Bayes KL" if args.pacbayes else "Noise KL probe"
        print(f"{label + ':':<20}{fmt_metric(pac_bayes_kl)}")
        if pac_sigma_summary is not None:
            print(f"PAC sigma mean:     {pac_sigma_summary['mean']:.6f}")
    if args.pac_pgd:
        print(f"PAC-PGD KL:         {fmt_metric(pac_pgd_kl)}")
        print(f"PAC-PGD bound:      {fmt_metric(pac_pgd_bound)}")
        if pac_pgd_summary is not None:
            print(f"PAC-PGD sigma mean: {pac_pgd_summary['pac_pgd_sigma_mean']:.6f}")
    if args.blob:
        print(f"Weight KL mean:     {fmt_metric(blob_kl)}")
        if blob_sigma_summary is not None:
            print(f"Weight sigma mean:  {blob_sigma_summary['mean']:.6f}")
    if args.full_bayes_lora:
        print(f"Bayes-LoRA KL_U:    {fmt_metric(bayes_lora_kl_u)}")
        print(f"Bayes-LoRA KL_W:    {fmt_metric(bayes_lora_kl_w)}")
        print(f"Bayes-LoRA KL total:{fmt_metric(bayes_lora_kl_total)}")
        print(f"Bayes-LoRA alpha:   {args.lora_alpha:g}")
        print(f"Bayes-LoRA alpha/r: {args.lora_alpha / float(args.rank):.6f}")
        if bayes_lora_summary is not None:
            print(f"Bayes U sigma mean: {bayes_lora_summary['u_sigma_mean']:.6f}")
            print(f"Bayes lambda mean:  {bayes_lora_summary['lambda']:.6f}")
    if calibrated_metrics is not None:
        print(f"Post-hoc Temp:      {posthoc_temperature:.4f}")
        if temperature_protocol_details is not None:
            print(f"TS protocol:        {temperature_protocol_details['protocol']}")
            print(f"TS fit/eval n:      {temperature_protocol_details['fit_count']}/{temperature_protocol_details['eval_count']}")
        if temperature_eval_raw_metrics is not None:
            print(f"Split Raw ECE:      {temperature_eval_raw_metrics['ece']:.4f}")
            print(f"Split Raw NLL:      {temperature_eval_raw_metrics['nll']:.4f}")
            print(f"Split Raw Brier:    {temperature_eval_raw_metrics['brier']:.4f}")
        print(f"TS ECE (15 bins):   {calibrated_metrics['ece']:.4f}")
        print(f"TS NLL:             {calibrated_metrics['nll']:.4f}")
        print(f"TS Brier Score:     {calibrated_metrics['brier']:.4f}")
        print(f"TS Avg Confidence:  {calibrated_metrics['avg_confidence']:.4f}")
    ood_detection = None
    if args.ood_dataset:
        _, ood_dl = utils.get_vtab_data(
            args.ood_dataset,
            evaluate=True,
            batch_size=64,
            num_workers=4,
            is_hdf5=False,
            eval_shift='clean',
            shift_severity=0,
        )
        ood_logits, _ = collect_deterministic_logits(model, ood_dl, device)
        id_scores = compute_uncertainty_scores(all_logits, score=args.ood_score)
        ood_scores = compute_uncertainty_scores(ood_logits, score=args.ood_score)
        ood_detection = compute_ood_detection_metrics(id_scores, ood_scores)
        ood_detection.update({
            'id_dataset': args.dataset,
            'ood_dataset': args.ood_dataset,
            'ood_score': args.ood_score,
        })
        print(f"OOD dataset:        {args.ood_dataset}")
        print(f"OOD score:          {args.ood_score}")
        print(f"OOD AUROC:          {ood_detection['auroc']:.4f}")
        print(f"OOD FPR@95TPR:      {ood_detection['fpr95']:.4f}")
    print(f"{'='*60}")

    # Save results
    ckpt_name = os.path.basename(os.path.dirname(args.checkpoint))
    uncertainty_config = infer_uncertainty_config_from_checkpoint(args.checkpoint)
    results = {
        'checkpoint': args.checkpoint,
        'dataset': args.dataset,
        'eval_shift': args.eval_shift,
        'shift_severity': args.shift_severity,
        **uncertainty_config,
        'pace': args.pace,
        'adapter_dropout': args.adapter_dropout,
        'lora_alpha': args.lora_alpha,
        'lora_scaling': args.lora_alpha / float(args.rank),
        'mc_dropout_samples': args.mc_dropout_samples,
        'mc_mode': mc_mode_name,
        'mc_samples': mc_samples if mc_mode else 0,
        'ivon': args.ivon,
        'ivon_mc_samples': args.ivon_mc_samples,
        'ivon_state_path': args.ivon_state or (pace.default_ivon_state_path(args.checkpoint) if args.ivon else None),
        'ivon_state_config': ivon_state_config,
        'ivon_ess': args.ivon_ess if args.ivon else None,
        'ivon_hess_init': args.ivon_hess_init if args.ivon else None,
        'ivon_clip_radius': args.ivon_clip_radius if args.ivon else None,
        'ivon_beta2': args.ivon_beta2 if args.ivon else None,
        'learnsigma': args.learnsigma,
        'pacbayes': args.pacbayes,
        'pac_lbd': args.pac_lbd if args.pacbayes else None,
        'pac_prior_sigma': args.pac_prior_sigma if (args.pacbayes or args.learnsigma) else None,
        'pac_bayes_kl': pac_bayes_kl,
        'pac_sigma_summary': pac_sigma_summary,
        'pac_pgd': args.pac_pgd,
        'pac_pgd_mc_samples': args.pac_pgd_mc_samples,
        'pac_pgd_stage1_epochs': args.pac_pgd_stage1_epochs if args.pac_pgd else None,
        'pac_pgd_lbd': args.pac_pgd_lbd if args.pac_pgd else None,
        'pac_pgd_gamma': args.pac_pgd_gamma if args.pac_pgd else None,
        'pac_pgd_kl': pac_pgd_kl,
        'pac_pgd_bound': pac_pgd_bound,
        'pac_pgd_summary': pac_pgd_summary,
        'blob': args.blob,
        'blob_mc_samples': args.blob_mc_samples,
        'blob_kl_mean': blob_kl,
        'blob_sigma_summary': blob_sigma_summary,
        'full_bayes_lora': args.full_bayes_lora,
        'bayes_lora_paper_faithful_scaling': bool(args.full_bayes_lora),
        'bayes_lora_flow': args.bayes_lora_flow if args.full_bayes_lora else None,
        'bayes_lora_flow_depth': args.bayes_lora_flow_depth if args.full_bayes_lora else None,
        'bayes_lora_init_sigma': args.bayes_lora_init_sigma if args.full_bayes_lora else None,
        'bayes_lora_prior_sigma': args.bayes_lora_prior_sigma if args.full_bayes_lora else None,
        'bayes_lora_max_sigma_u': args.bayes_lora_max_sigma_u if args.full_bayes_lora else None,
        'bayes_lora_lambda_init': args.bayes_lora_lambda_init if args.full_bayes_lora else None,
        'bayes_lora_lambda_max': args.bayes_lora_lambda_max if args.full_bayes_lora else None,
        'bayes_lora_mc_samples': args.bayes_lora_mc_samples,
        'bayes_lora_kl_u': bayes_lora_kl_u,
        'bayes_lora_kl_w': bayes_lora_kl_w,
        'bayes_lora_kl_total': bayes_lora_kl_total,
        'bayes_lora_summary': bayes_lora_summary,
        'load_state_missing_key_count': len(load_status.missing_keys),
        'load_state_unexpected_key_count': len(load_status.unexpected_keys),
        'load_state_missing_key_examples': list(load_status.missing_keys[:20]),
        'load_state_unexpected_key_examples': list(load_status.unexpected_keys[:20]),
        **metrics,
        'inference_timing': {
            'timing_warmup_batches': args.timing_warmup,
            'timing_batches': args.timing_batches,
            'timed_images': timing_total_images,
            'total_seconds': timing_total_time,
            'seconds_per_image': inference_seconds_per_image,
            'images_per_second': inference_images_per_second,
        },
        'posthoc_temperature': posthoc_temperature,
        'temperature_protocol': temperature_protocol_details,
        'temperature_eval_raw_metrics': temperature_eval_raw_metrics,
        'posthoc_temperature_scaled': calibrated_metrics,
        'ood_detection': ood_detection,
    }
    save_name = f'{ckpt_name}_{args.dataset}'
    if args.eval_shift != 'clean':
        save_name += f'_{args.eval_shift}_s{args.shift_severity}'
    if args.ood_dataset:
        save_name += f'_OOD_{args.ood_dataset}_{args.ood_score}'
    if mc_mode:
        if mc_mode_name == "dropout":
            save_name += f'_MCDrop{mc_samples}'
        elif mc_mode_name == "pac_pgd":
            save_name += f'_MCPacPGD{mc_samples}'
        elif mc_mode_name == "ivon":
            save_name += f'_MCIVON{mc_samples}'
        else:
            save_name += f'_MC{mc_samples}'
    if args.posthoc_temp_scaling and args.temperature_protocol == 'split':
        save_name += f'_TSSplit{args.temperature_split_seed}'
    save_path = os.path.join(args.save_dir, f'{save_name}.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_path}")
