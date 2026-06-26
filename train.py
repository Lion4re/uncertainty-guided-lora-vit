import os.path
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
import torch
from tqdm import tqdm
from typing import Dict
import functools
import numpy as np
import json
import time

import utils
import pace


@torch.no_grad()
def test(model, dl, device="cpu"):
    model.eval()
    acc_list = []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        acc_list.append(logits.argmax(dim=1) == y)
    return (torch.cat(acc_list) * 1.).mean().item()

def filter_loss(loss_dict: Dict):
    new_dict = {}
    diagnostic_keys = {
        'pac_bayes_kl',
        'pac_bayes_sigma_mean',
        'pac_bayes_sigma_max',
        'pac_bayes_sigma_min',
        'pac_bayes_sigma_std',
        'ce_grad_norm',
        'pace_grad_norm',
        'grad_kl_ce_ratio',
        'blob_kl',
        'bayes_lora_kl_u',
        'bayes_lora_kl_w',
        'bayes_lora_kl_total',
        'mean_uncertainty',
        'selected_fraction',
        'effective_batch_size',
    }
    for k, v in loss_dict.items():
        if ('loss' in k or k.startswith('pac_pgd_')
                or k in diagnostic_keys):
            new_dict[k] = v.item() if hasattr(v, 'item') else v
    return new_dict


def grad_norm_from_grads(grads, device):
    total = torch.zeros((), device=device)
    for grad in grads:
        if grad is not None:
            total = total + grad.detach().pow(2).sum()
    return total.sqrt()


if __name__ == '__main__':
    parser = ArgumentParser()
    group = parser.add_argument_group('tasks arguments')
    group.add_argument('--task',            type=str,   default='vtab', choices=['vtab', 'fs'], help='task name, vtab or fs(few-shot learning)')
    group.add_argument('--dataset',         type=str,   default='caltech101', help='dataset name')
    group.add_argument('--fs_shot',         type=int,   default=1, help='number of shot')
    group.add_argument('--fs_seed',         type=int,   default=0)
    group.add_argument('--hdf5',            action='store_true',    default=False)
    group.add_argument('--vtab_evaluate',   action='store_true',    default=False)

    group = parser.add_argument_group('model arguments and training arguments')
    group.add_argument('--model',           type=str,   default='ViT-B', choices=['ViT-B', 'Swin-B'])
    group.add_argument('--bs',              type=int,   default=64)
    group.add_argument('--lr',              type=float, default=1e-3)
    group.add_argument('--wd',              type=float, default=1e-4)
    group.add_argument('--optimizer',       type=str,   default='adamw', choices=['adamw', 'ivon'],
                       help='Optimizer for trainable PEFT parameters. IVON enables variational LoRA-style posterior sampling.')
    group.add_argument('--ivon_ess',        type=float, default=1e6,
                       help='IVON effective sample size / variational temperature scale.')
    group.add_argument('--ivon_hess_init',  type=float, default=1e-3,
                       help='Initial IVON Hessian/precision estimate.')
    group.add_argument('--ivon_clip_radius', type=float, default=1e-3,
                       help='IVON perturbation clipping radius.')
    group.add_argument('--ivon_beta2',      type=float, default=0.99999,
                       help='IVON Hessian momentum beta2.')
    group.add_argument('--num_workers',     type=int,   default=4)
    group.add_argument('--epoch',           type=int,   default=300)
    group.add_argument('--test_every',      type=int,   default=10, help='testing after specific epochs')

    group = parser.add_argument_group('working settings')
    group.add_argument('--seed',            type=int,   default=42)
    group.add_argument('--out_dir',         type=str,   default='outputs/checkpoints_and_training_logs')
    group.add_argument('--timing_json',     type=str,   default=None,
                       help='Optional path for timing summary JSON. Defaults to save_path/timing.json.')

    group = parser.add_argument_group('adapter settings')
    group.add_argument('--adapter',         type=str,   default='LoRAmul_VPTadd', choices=['LoRAmul_VPTadd', 'LoRAadd', 'VPTadd'])
    group.add_argument('--rank',            type=int,   default=10)
    group.add_argument('--lora_alpha',      type=float, default=None,
                       help='LoRA scaling alpha. Defaults to rank, so alpha/rank keeps existing scale.')

    group = parser.add_argument_group('PACE arguments')
    group.add_argument('--pace_type',      type=str,   default=None, choices=[None, 'pace', 'pace_offset', 'pace_uncert_soft', 'pace_uncert_topk', 'pace_kl', 'pace_kl_uncert_soft', 'pace_kl_uncert_topk', 'pace_kl_learnsigma', 'pace_kl_wp', 'pace_combined', 'pace_kl_margin', 'pace_kl_flat', 'pace_kl_jacobian', 'pace_pacbayes', 'pace_kl_pacbayes', 'pace_kl_margin_pacbayes', 'pace_pacpgd', 'pace_kl_pacpgd', 'full_bayes_lora', 'pace_kl_full_bayes_lora', 'pace_kl_margin_full_bayes_lora', 'lazy', 'fast'])
    group.add_argument('--lbd',            type=float, default=0., help='regularization strength of PACE')
    group.add_argument('--sigma',          type=float, default=0., help='sigma of the Gaussian noise')
    group.add_argument('--adapter_dropout', type=float, default=0.,
                       help='Dropout probability applied to the adapter branch during PACE training.')
    group.add_argument('--temperature',    type=float, default=2.0, help='temperature for KL-based PACE losses')
    group.add_argument('--pace_kl_detach_target', action='store_true',
                       help='Detach the second perturbed prediction in PACE-KL consistency losses.')
    group.add_argument('--uncertainty_score', type=str, default='entropy',
                       choices=['entropy', 'max_conf', 'margin'],
                       help='Predictive uncertainty score for uncertainty-guided PACE/PACE-KL.')
    group.add_argument('--uncertainty_fraction', type=float, default=0.30,
                       help='Fraction of most uncertain samples used by top-k uncertainty-guided PACE/PACE-KL.')
    group.add_argument('--uncertainty_weight', type=float, default=1.0,
                       help='Soft uncertainty weighting strength for uncertainty-guided PACE/PACE-KL.')
    group.add_argument('--jacobian_lbd',   type=float, default=0.1, help='regularization strength for finite-difference Jacobian proxy')
    group.add_argument('--jacobian_eps',   type=float, default=1e-2, help='input perturbation std for finite-difference Jacobian proxy')
    group.add_argument('--pac_lbd',        type=float, default=1e-3, help='regularization strength for PAC-Bayes noise KL')
    group.add_argument('--pac_prior_sigma', type=float, default=1.2, help='prior sigma for PAC-Bayes learnable adapter noise')
    group.add_argument('--pac_pgd_stage1_epochs', type=int, default=100,
                       help='Number of Stage-1 epochs that learn PAC-PGD parameter-noise scales.')
    group.add_argument('--pac_pgd_lbd', type=float, default=1.0,
                       help='Multiplier for the PAC-PGD bound objective during Stage 1.')
    group.add_argument('--pac_pgd_gamma', type=float, default=0.1,
                       help='PAC-PGD bound gamma constant.')
    group.add_argument('--pac_pgd_init_floor', type=float, default=1e-4,
                       help='Minimum initialization scale for PAC-PGD log noise parameters.')
    group.add_argument('--pac_pgd_prior_floor', type=float, default=1e-4,
                       help='Minimum initialization scale for PAC-PGD layer prior parameters.')
    group.add_argument('--lazy_interval',  type=int,   default=1,  help="It only works when pace_type is set to 'lazy'")
    group.add_argument('--blob',           action='store_true',
                       help='Enable BLoB-style Bayesian LoRA A matrices.')
    group.add_argument('--blob_lbd',       type=float, default=1e-3,
                       help='Regularization strength for normalized BLoB posterior KL.')
    group.add_argument('--blob_kl_reduction', type=str, default='mean', choices=['mean', 'sum'],
                       help='Use normalized BLoB KL (mean) or raw summed KL (sum).')
    group.add_argument('--blob_prior_sigma', type=float, default=1.0,
                       help='Gaussian prior sigma for BLoB LoRA A matrices.')
    group.add_argument('--blob_init_sigma', type=float, default=1e-4,
                       help='Initial posterior sigma for BLoB LoRA A matrices.')
    group.add_argument('--full_bayes_lora', action='store_true',
                       help='Enable full Bayesian-LoRA inducing-variable adapter posterior.')
    group.add_argument('--bayes_lora_flow', type=str, default='none', choices=['none', 'maf', 'row_maf'],
                       help='Posterior flow for full Bayesian-LoRA inducing variables.')
    group.add_argument('--bayes_lora_flow_depth', type=int, default=1,
                       help='Number of MAF layers for full Bayesian-LoRA. The paper-faithful default is 1.')
    group.add_argument('--bayes_lora_lbd_u', type=float, default=1e-5,
                       help='Regularization strength for full Bayesian-LoRA KL_U.')
    group.add_argument('--bayes_lora_lbd_w', type=float, default=1e-5,
                       help='Regularization strength for full Bayesian-LoRA KL_W.')
    group.add_argument('--bayes_lora_init_sigma', type=float, default=1e-4,
                       help='Initial posterior sigma for full Bayesian-LoRA inducing variables.')
    group.add_argument('--bayes_lora_prior_sigma', type=float, default=0.1,
                       help='Whitened Gaussian prior sigma for full Bayesian-LoRA.')
    group.add_argument('--bayes_lora_max_sigma_u', type=float, default=0.1,
                       help='Maximum posterior standard deviation for full Bayesian-LoRA inducing variables.')
    group.add_argument('--bayes_lora_lambda_init', type=float, default=1e-3,
                       help='Initial conditional weight noise scale for full Bayesian-LoRA.')
    group.add_argument('--bayes_lora_lambda_max', type=float, default=3e-2,
                       help='Maximum conditional weight noise scale for full Bayesian-LoRA.')
    group.add_argument('--diagnose_grad_norms', action='store_true',
                       help='Log CE and PACE/KL consistency gradient norms for short diagnostic runs. Adds extra autograd cost.')


    args = parser.parse_args()
    if args.lora_alpha is None:
        args.lora_alpha = float(args.rank)
    is_pacbayes = args.pace_type is not None and args.pace_type.endswith('_pacbayes')
    is_pac_pgd = args.pace_type in ['pace_pacpgd', 'pace_kl_pacpgd']
    is_learnable_noise = is_pacbayes or args.pace_type == 'pace_kl_learnsigma'
    is_weight_posterior = args.pace_type == 'pace_kl_wp'
    is_full_bayes_lora = args.full_bayes_lora or args.pace_type in [
        'full_bayes_lora', 'pace_kl_full_bayes_lora', 'pace_kl_margin_full_bayes_lora']
    uses_pace_noise = args.pace_type is not None and args.pace_type != 'full_bayes_lora'
    if is_weight_posterior:
        args.blob = True
    if is_full_bayes_lora:
        args.full_bayes_lora = True
        if args.adapter != 'LoRAadd':
            raise ValueError("Full Bayesian-LoRA follows the paper's additive LoRA form; use --adapter LoRAadd.")

    # set the working directory
    name_configs = []
    if args.pace_type is not None:
        name_configs += [args.pace_type,
                         f'Lbd{args.lbd:g}'         if args.lbd else None,
                         f'S{args.sigma:g}'         if args.sigma else None,
                         f'T{args.temperature:g}'    if args.pace_type in ['pace_kl', 'pace_kl_uncert_soft', 'pace_kl_uncert_topk', 'pace_kl_learnsigma', 'pace_kl_wp', 'pace_combined', 'pace_kl_margin', 'pace_kl_flat', 'pace_kl_jacobian', 'pace_kl_pacbayes', 'pace_kl_margin_pacbayes', 'pace_kl_pacpgd', 'pace_kl_full_bayes_lora', 'pace_kl_margin_full_bayes_lora'] and args.temperature != 2.0 else None,
                         'Detach'                    if args.pace_kl_detach_target and args.pace_type in ['pace_kl', 'pace_kl_uncert_soft', 'pace_kl_uncert_topk', 'pace_kl_learnsigma', 'pace_kl_margin', 'pace_kl_flat', 'pace_kl_jacobian', 'pace_kl_pacbayes', 'pace_kl_margin_pacbayes'] else None,
                         f'Us{args.uncertainty_score}' if args.pace_type in ['pace_uncert_soft', 'pace_uncert_topk', 'pace_kl_uncert_soft', 'pace_kl_uncert_topk'] and args.uncertainty_score != 'entropy' else None,
                         f'Uf{args.uncertainty_fraction:g}' if args.pace_type in ['pace_uncert_topk', 'pace_kl_uncert_topk'] and args.uncertainty_fraction != 0.30 else None,
                         f'Uw{args.uncertainty_weight:g}' if args.pace_type in ['pace_uncert_soft', 'pace_kl_uncert_soft'] and args.uncertainty_weight != 1.0 else None,
                         f'J{args.jacobian_lbd:g}'   if args.pace_type == 'pace_kl_jacobian' and args.jacobian_lbd != 0.1 else None,
                         f'Je{args.jacobian_eps:g}'  if args.pace_type == 'pace_kl_jacobian' and args.jacobian_eps != 1e-2 else None,
                         f'Pac{args.pac_lbd:g}'       if is_pacbayes and args.pac_lbd != 1e-3 else None,
                         f'Ps{args.pac_prior_sigma:g}' if is_learnable_noise and args.pac_prior_sigma != 1.2 else None,
                         'PacPGD'                     if is_pac_pgd else None,
                         f'St1{args.pac_pgd_stage1_epochs:d}' if is_pac_pgd else None,
                         f'Pg{args.pac_pgd_gamma:g}'  if is_pac_pgd and args.pac_pgd_gamma != 0.1 else None,
                         f'Pgl{args.pac_pgd_lbd:g}'   if is_pac_pgd and args.pac_pgd_lbd != 1.0 else None,
                         f'Drop{args.adapter_dropout:g}' if args.adapter_dropout > 0 else None,
                         f'Lz{args.lazy_interval}'   if args.pace_type == 'lazy' and args.lazy_interval != 1 else None]

    name_configs += [    args.adapter,
                         f'R{args.rank:d}',
                         'BLoB'                              if args.blob else None,
                         f'Blob{args.blob_lbd:g}'            if args.blob and args.blob_lbd != 1e-3 else None,
                         f'Bkl{args.blob_kl_reduction}'       if args.blob and args.blob_kl_reduction != 'mean' else None,
                         'FullBayesLoRA'                       if args.full_bayes_lora else None,
                         f'Alpha{args.lora_alpha:g}'            if args.full_bayes_lora and not np.isclose(args.lora_alpha, args.rank) else None,
                         f'BFlow{args.bayes_lora_flow}'         if args.full_bayes_lora and args.bayes_lora_flow != 'none' else None,
                         f'Bfd{args.bayes_lora_flow_depth}'      if args.full_bayes_lora and args.bayes_lora_flow != 'none' and args.bayes_lora_flow_depth != 1 else None,
                         f'Bu{args.bayes_lora_lbd_u:g}'         if args.full_bayes_lora and args.bayes_lora_lbd_u != 1e-5 else None,
                         f'Bw{args.bayes_lora_lbd_w:g}'         if args.full_bayes_lora and args.bayes_lora_lbd_w != 1e-5 else None,
                         'IVON'                                  if args.optimizer == 'ivon' else None,
                         f'Ess{args.ivon_ess:g}'                 if args.optimizer == 'ivon' and args.ivon_ess != 1e6 else None,
                         f'Hi{args.ivon_hess_init:g}'            if args.optimizer == 'ivon' and args.ivon_hess_init != 1e-3 else None,
                         f'Cr{args.ivon_clip_radius:g}'          if args.optimizer == 'ivon' and args.ivon_clip_radius != 1e-3 else None,
                         f'lr{args.lr:g}'                       if args.lr != 1e-3 else None,
                         f'wd{args.wd:.0e}'                     if args.wd != 1e-4 else None,
                         f'Seed{args.seed:d}'                   if args.seed != 42 else None,
                         f'St{args.fs_shot}Sd{args.fs_seed}'    if args.task == 'fs' else None,
                         args.dataset]

    base_sub_path           = '_'.join([nc for nc in name_configs if nc is not None])
    args.save_path          = os.path.join(args.out_dir, base_sub_path)
    utils.ensure_dirs(args.save_path)
    args.save_path_recent   = os.path.join(args.save_path, 'weight.pt')

    print(f"Arguments: {args}")
    utils.set_seed(args.seed)
    args.best_acc = 0

    # set the loggers
    test_logger = utils.MetricsLogger(args.save_path + '_acc', True, 1)
    train_logger = utils.TrainLogger(args.save_path, True, buffer_size=100)

    # load dataset
    if args.task == 'vtab':
        train_dl, val_dl = utils.get_vtab_data(args.dataset, evaluate=args.vtab_evaluate, batch_size=args.bs,
                                               num_workers=args.num_workers, is_hdf5=args.hdf5)
        test_dl = None
        class_dim = utils.get_vtab_classes_num(args.dataset)
    elif args.task == 'fs':
        train_dl, val_dl, test_dl = utils.get_few_shot_data(args.dataset, batch_size=args.bs, num_workers=args.num_workers,
                                                            shot=args.fs_shot, seed=args.fs_seed, is_hdf5=args.hdf5)
        class_dim = utils.get_few_shot_classes_num(args.dataset)
        args.test_acc = 0
    else:
        raise NotImplementedError


    # load the pretrained model
    if args.model == 'ViT-B':
        model = create_model('vit_base_patch16_224_in21k', checkpoint_path='./ViT-B_16.npz', drop_path_rate=0.1)
    elif args.model == 'Swin-B':
        import timm
        timm.models._hub.hf_hub_download = functools.partial(timm.models._hub.hf_hub_download, cache_dir='cache')
        model = create_model('swin_base_patch4_window7_224.ms_in22k', pretrained=True, drop_path_rate=0.1)
    else:
        model = None

    model.reset_classifier(class_dim)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # decide how to compute the loss
    def compute_loss_standard(model, x, y, criterion, **kwargs):
        logits = model(x)
        cls_loss = criterion(logits, y)
        return {'cls_loss': cls_loss, 'total_loss': cls_loss, 'logits': logits}

    compute_loss = compute_loss_standard
    cross_entropy = torch.nn.CrossEntropyLoss()

    # inject residual adapter
    pace.inject_residual_adapter(model, adapter=args.adapter, rank=args.rank)
    if args.blob:
        pace.enable_blob(model, init_sigma=args.blob_init_sigma, prior_sigma=args.blob_prior_sigma)
    if args.full_bayes_lora:
        bayes_lora_config = pace.BayesianLoRAConfig(
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
        pace.enable_full_bayesian_lora(model, bayes_lora_config)

    ############# start of injecting PACE #################################
    if uses_pace_noise:
        # PACE performs two forward passes on the same sample.
        # Ensure DropPath use the same mask for both passes to maintain consistency
        pace.ensure_sharable_drop_path(model)

        ######### STEP 1: inject noise to the adapters
        # get list of [parent_layer, adapter_name, block_id] for each ResidualAdapter
        adapters_and_block_ids = pace.get_adapters_and_block_ids(model)

        # sigma value decreases linearly as block_id increases.
        num_blocks = max(block_id for _, _, block_id in adapters_and_block_ids) + 1
        block_start = 1 if args.model == 'ViT-B' else 0 # ViT-B works better without adding noise to the first block
        sigmas = np.concatenate([np.zeros(block_start), np.linspace(0, args.sigma, num_blocks - block_start + 1)[-1:0:-1]])

        # Inject noise adapter into each residual adapter
        for parent_layer, adapter_name, block_id in adapters_and_block_ids:
            residual_adapter = getattr(parent_layer, adapter_name)
            if is_learnable_noise:
                noise_adapter = pace.LearnableMultiplicativeNoiseAdapter(
                    residual_adapter, init_sigma=sigmas[block_id],
                    prior_sigma=args.pac_prior_sigma,
                    adapter_dropout=args.adapter_dropout)
            else:
                noise_adapter = pace.MultiplicativeNoiseAdapter(
                    residual_adapter, sigma=sigmas[block_id],
                    adapter_dropout=args.adapter_dropout)
            setattr(parent_layer, adapter_name, noise_adapter)

        ######### STEP 2: apply consistency regularization over two perturbations
        if args.pace_type == 'pace':  #PACE: applying consistency regularization at every iteration
            compute_loss = functools.partial(pace.compute_loss_pace, lbd_pace=args.lbd)
        elif args.pace_type == 'pace_offset':
            compute_loss = functools.partial(pace.compute_loss_pace_offset, lbd_pace=args.lbd)
        elif args.pace_type == 'pace_uncert_soft':
            compute_loss = functools.partial(
                pace.compute_loss_pace_uncertainty,
                lbd_pace=args.lbd,
                uncertainty_score=args.uncertainty_score,
                uncertainty_mode='soft',
                uncertainty_fraction=args.uncertainty_fraction,
                uncertainty_weight=args.uncertainty_weight,
            )
        elif args.pace_type == 'pace_uncert_topk':
            compute_loss = functools.partial(
                pace.compute_loss_pace_uncertainty,
                lbd_pace=args.lbd,
                uncertainty_score=args.uncertainty_score,
                uncertainty_mode='topk',
                uncertainty_fraction=args.uncertainty_fraction,
                uncertainty_weight=args.uncertainty_weight,
            )
        elif args.pace_type == 'pace_kl':
            compute_loss = functools.partial(
                pace.compute_loss_pace_kl,
                lbd_pace=args.lbd,
                temperature=args.temperature,
                detach_target=args.pace_kl_detach_target,
            )
        elif args.pace_type == 'pace_kl_uncert_soft':
            compute_loss = functools.partial(
                pace.compute_loss_pace_kl_uncertainty,
                lbd_pace=args.lbd,
                temperature=args.temperature,
                detach_target=args.pace_kl_detach_target,
                uncertainty_score=args.uncertainty_score,
                uncertainty_mode='soft',
                uncertainty_fraction=args.uncertainty_fraction,
                uncertainty_weight=args.uncertainty_weight,
            )
        elif args.pace_type == 'pace_kl_uncert_topk':
            compute_loss = functools.partial(
                pace.compute_loss_pace_kl_uncertainty,
                lbd_pace=args.lbd,
                temperature=args.temperature,
                detach_target=args.pace_kl_detach_target,
                uncertainty_score=args.uncertainty_score,
                uncertainty_mode='topk',
                uncertainty_fraction=args.uncertainty_fraction,
                uncertainty_weight=args.uncertainty_weight,
            )
        elif args.pace_type == 'pace_kl_learnsigma':
            compute_loss = functools.partial(pace.compute_loss_pace_kl_learnsigma, lbd_pace=args.lbd,
                                             temperature=args.temperature,
                                             detach_target=args.pace_kl_detach_target)
        elif args.pace_type == 'pace_kl_wp':
            compute_loss = functools.partial(pace.compute_loss_pace_kl, lbd_pace=args.lbd,
                                             temperature=args.temperature,
                                             detach_target=args.pace_kl_detach_target)
        elif args.pace_type == 'pace_combined':
            compute_loss = functools.partial(pace.compute_loss_pace_combined, lbd_pace=args.lbd, lbd_kl=0.5, temperature=args.temperature)
        elif args.pace_type == 'pace_kl_margin':
            compute_loss = functools.partial(pace.compute_loss_pace_kl_margin, lbd_pace=args.lbd, lbd_margin=0.1, temperature=args.temperature)
        elif args.pace_type == 'pace_kl_flat':
            compute_loss = functools.partial(pace.compute_loss_pace_kl_flat, lbd_pace=args.lbd, temperature=args.temperature)
        elif args.pace_type == 'pace_kl_jacobian':
            compute_loss = functools.partial(pace.compute_loss_pace_kl_jacobian, lbd_pace=args.lbd,
                                             lbd_jacobian=args.jacobian_lbd,
                                             jacobian_eps=args.jacobian_eps,
                                             temperature=args.temperature)
        elif args.pace_type == 'pace_kl_pacbayes':
            compute_loss = functools.partial(pace.compute_loss_pace_kl_pacbayes, lbd_pace=args.lbd,
                                             pac_lbd=args.pac_lbd,
                                             temperature=args.temperature,
                                             detach_target=args.pace_kl_detach_target)
        elif args.pace_type == 'pace_pacbayes':
            compute_loss = functools.partial(pace.compute_loss_pace_pacbayes, lbd_pace=args.lbd,
                                             pac_lbd=args.pac_lbd)
        elif args.pace_type == 'pace_kl_margin_pacbayes':
            compute_loss = functools.partial(pace.compute_loss_pace_kl_margin_pacbayes, lbd_pace=args.lbd,
                                             pac_lbd=args.pac_lbd,
                                             lbd_margin=0.1,
                                             temperature=args.temperature,
                                             detach_target=args.pace_kl_detach_target)
        elif args.pace_type == 'pace_pacpgd':
            compute_loss = functools.partial(pace.compute_loss_pace, lbd_pace=args.lbd)
        elif args.pace_type == 'pace_kl_pacpgd':
            compute_loss = functools.partial(pace.compute_loss_pace_kl, lbd_pace=args.lbd,
                                             temperature=args.temperature)
        elif args.pace_type == 'pace_kl_full_bayes_lora':
            compute_loss = functools.partial(pace.compute_loss_pace_kl, lbd_pace=args.lbd,
                                             temperature=args.temperature)
        elif args.pace_type == 'pace_kl_margin_full_bayes_lora':
            compute_loss = functools.partial(pace.compute_loss_pace_kl_margin, lbd_pace=args.lbd,
                                             lbd_margin=0.1,
                                             temperature=args.temperature)
        elif args.pace_type == 'lazy':  #PACE_lazy_half: apply consistency regularization at every 'lazy_interval' iterations and halve the batch size
            compute_loss = functools.partial(pace.compute_loss_pace_lazy_half, lbd_pace=args.lbd, lazy_interval=args.lazy_interval)
        elif args.pace_type == 'fast':  #PACE_fast
            history_out = torch.zeros(len(train_dl.dataset), class_dim)
            history_out = history_out.to(device)
            compute_loss = functools.partial(pace.compute_loss_pace_fast, lbd_pace=args.lbd, history_logits=history_out)
    ############# end of injecting PACE  #################################

    # set optimizer and scheduler
    model = model.to(device)
    trainable = []
    trainable_names = {}
    for n, p in model.named_parameters():
        if args.full_bayes_lora:
            is_trainable = 'head' in n or 'full_bayes_lora_state' in n
        else:
            is_trainable = any([x in n for x in ['delta', 'head', 'pac_log_sigma', 'blob_rho']])
        if is_trainable:
            trainable.append(p)
            p.requires_grad = True
            trainable_names[n] = p
        else:
            p.requires_grad = False
    pac_pgd_state = None
    pac_pgd_opt = None
    if is_pac_pgd:
        pac_pgd_config = pace.PACPGDConfig(
            stage1_epochs=args.pac_pgd_stage1_epochs,
            lbd=args.pac_pgd_lbd,
            gamma=args.pac_pgd_gamma,
            init_floor=args.pac_pgd_init_floor,
            prior_floor=args.pac_pgd_prior_floor,
        )
        pac_pgd_state = pace.build_pac_pgd_state(model, pac_pgd_config)
        model.pac_pgd_state = pac_pgd_state
        trainable_names['pac_pgd_state.p'] = pac_pgd_state.p
        trainable_names['pac_pgd_state.b'] = pac_pgd_state.b
        pac_pgd_opt = torch.optim.Adam([pac_pgd_state.p, pac_pgd_state.b], lr=0.5, weight_decay=0.0)
    ivon_config = {
        'ess': args.ivon_ess,
        'hess_init': args.ivon_hess_init,
        'clip_radius': args.ivon_clip_radius,
        'beta2': args.ivon_beta2,
    }
    opt = pace.create_optimizer(
        trainable,
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.wd,
        ivon_ess=args.ivon_ess,
        ivon_hess_init=args.ivon_hess_init,
        ivon_clip_radius=args.ivon_clip_radius,
        ivon_beta2=args.ivon_beta2,
    )
    scheduler = CosineLRScheduler(opt, t_initial=args.epoch, warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, cycle_decay=0.1)

    # train the model
    itr = 0
    train_start_time = time.perf_counter()
    epoch_timings = []
    best_val_acc = None
    final_val_results = {}
    for ep in tqdm(range(args.epoch)):
        epoch_start_time = time.perf_counter()
        model.train()
        for i, (x, y, index) in enumerate(train_dl):
            # forward and backward
            x, y, index = x.to(device), y.to(device), index.to(device)
            if is_pac_pgd:
                opt.zero_grad()
                pac_pgd_opt.zero_grad()
                wdecay = pac_pgd_state.weight_decay_mulb()
                noises = pac_pgd_state.inject_noise()
                stage1 = ep < args.pac_pgd_stage1_epochs
                try:
                    out_dict = compute_loss(model, x, y, cross_entropy, itr=itr, index=index)
                    loss = out_dict['total_loss']
                    loss.backward(retain_graph=stage1)
                    pac_kl = pac_pgd_state.kl_term_layer_pb(wdecay)
                    pac_bound = pac_pgd_state.pac_bound(pac_kl, len(train_dl.dataset))
                    if stage1:
                        pac_pgd_state.kl_term_backward(pac_bound, noises)
                    out_dict['pac_pgd_kl'] = pac_kl.detach()
                    out_dict['pac_pgd_bound'] = pac_bound.detach()
                    out_dict['pac_pgd_stage'] = torch.as_tensor(1 if stage1 else 2, device=x.device)
                    out_dict['total_loss'] = loss.detach() + (pac_bound.detach() if stage1 else 0.0)
                finally:
                    pac_pgd_state.remove_noise(noises)
                opt.step()
                if stage1:
                    pac_pgd_opt.step()
                with torch.no_grad():
                    pac_summary = pac_pgd_state.summary()
                    out_dict['pac_pgd_p_mean'] = pac_summary['pac_pgd_p_mean']
                    out_dict['pac_pgd_p_std'] = pac_summary['pac_pgd_p_std']
                    out_dict['pac_pgd_b_mean'] = pac_summary['pac_pgd_b_mean']
            else:
                opt.zero_grad()
                with pace.sampled_params_context(opt, train=True):
                    out_dict = compute_loss(model, x, y, cross_entropy, itr=itr, index=index)
                    if args.blob:
                        blob_kl = pace.blob_kl(model, reduction=args.blob_kl_reduction)
                        if blob_kl is None:
                            blob_kl = torch.zeros((), device=x.device)
                        out_dict['blob_kl'] = blob_kl
                        out_dict['total_loss'] = out_dict['total_loss'] + args.blob_lbd * blob_kl
                    if args.full_bayes_lora:
                        bayes_terms = pace.full_bayes_lora_kl(model, reduction='mean')
                        if bayes_terms is None:
                            zero = torch.zeros((), device=x.device)
                            bayes_terms = {'kl_u': zero, 'kl_w': zero, 'kl_total': zero}
                        out_dict['bayes_lora_kl_u'] = bayes_terms['kl_u']
                        out_dict['bayes_lora_kl_w'] = bayes_terms['kl_w']
                        out_dict['bayes_lora_kl_total'] = bayes_terms['kl_total']
                        out_dict['total_loss'] = (out_dict['total_loss']
                                                  + args.bayes_lora_lbd_u * bayes_terms['kl_u']
                                                  + args.bayes_lora_lbd_w * bayes_terms['kl_w'])
                    if args.pace_type in {'pace_pacbayes', 'pace_kl_pacbayes', 'pace_kl_margin_pacbayes'}:
                        with torch.no_grad():
                            pac_summary = pace.pac_bayes_sigma_summary(model)
                            if pac_summary is not None:
                                out_dict['pac_bayes_sigma_mean'] = pac_summary['mean']
                                out_dict['pac_bayes_sigma_std'] = pac_summary['std']
                                out_dict['pac_bayes_sigma_min'] = pac_summary['min']
                                out_dict['pac_bayes_sigma_max'] = pac_summary['max']
                    if args.diagnose_grad_norms and 'cls_loss' in out_dict and 'pace_loss' in out_dict:
                        diag_params = [p for p in trainable if p.requires_grad]
                        ce_grads = torch.autograd.grad(
                            out_dict['cls_loss'], diag_params, retain_graph=True, allow_unused=True
                        )
                        pace_grads = torch.autograd.grad(
                            out_dict['pace_loss'], diag_params, retain_graph=True, allow_unused=True
                        )
                        ce_norm = grad_norm_from_grads(ce_grads, x.device)
                        pace_norm = grad_norm_from_grads(pace_grads, x.device)
                        out_dict['ce_grad_norm'] = ce_norm
                        out_dict['pace_grad_norm'] = pace_norm
                        out_dict['grad_kl_ce_ratio'] = pace_norm / (ce_norm + 1e-12)
                    loss = out_dict['total_loss']
                    loss.backward()
                opt.step()

            # log training results
            results_dict = filter_loss(out_dict)
            with torch.no_grad():
                out = out_dict['logits']
                results_dict['train_acc'] = torch.mean((out.argmax(dim=1) == y[:out.shape[0]]) * 1.).item()
            train_logger.log(itr, **results_dict)

            itr += 1
            pass  # torch.cuda.empty_cache()

        scheduler.step(ep)
        epoch_seconds = time.perf_counter() - epoch_start_time
        epoch_timings.append(epoch_seconds)

        # on validation
        if (ep+1) % args.test_every == 0 or (ep+1) == args.epoch:
            acc = round(test(model, val_dl, device), 4)
            utils.save_weights(args.save_path_recent, model, trainable_names)
            if args.optimizer == 'ivon':
                pace.save_ivon_state(
                    pace.default_ivon_state_path(args.save_path_recent),
                    opt,
                    ivon_config,
                )
            val_results = {'val_acc': acc}
            best_val_acc = acc if best_val_acc is None else max(best_val_acc, acc)
            if test_dl is not None:
                test_acc = round(test(model, test_dl, device), 4)
                val_results['test_acc'] = test_acc
            final_val_results = val_results
            test_logger.log(E=ep+1, **val_results)

    total_train_seconds = time.perf_counter() - train_start_time
    timing_summary = {
        'save_path': args.save_path,
        'dataset': args.dataset,
        'pace_type': args.pace_type,
        'adapter': args.adapter,
        'rank': args.rank,
        'lora_alpha': args.lora_alpha,
        'lora_scaling': args.lora_alpha / float(args.rank),
        'adapter_dropout': args.adapter_dropout,
        'optimizer': args.optimizer,
        'ivon': args.optimizer == 'ivon',
        'ivon_ess': args.ivon_ess if args.optimizer == 'ivon' else None,
        'ivon_hess_init': args.ivon_hess_init if args.optimizer == 'ivon' else None,
        'ivon_clip_radius': args.ivon_clip_radius if args.optimizer == 'ivon' else None,
        'ivon_beta2': args.ivon_beta2 if args.optimizer == 'ivon' else None,
        'ivon_state_path': pace.default_ivon_state_path(args.save_path_recent) if args.optimizer == 'ivon' else None,
        'learnable_noise': is_learnable_noise,
        'pacbayes': is_pacbayes,
        'weight_posterior': is_weight_posterior,
        'pac_lbd': args.pac_lbd if is_pacbayes else None,
        'pac_prior_sigma': args.pac_prior_sigma if is_learnable_noise else None,
        'pac_pgd': is_pac_pgd,
        'pac_pgd_stage1_epochs': args.pac_pgd_stage1_epochs if is_pac_pgd else None,
        'pac_pgd_lbd': args.pac_pgd_lbd if is_pac_pgd else None,
        'pac_pgd_gamma': args.pac_pgd_gamma if is_pac_pgd else None,
        'pac_pgd_init_floor': args.pac_pgd_init_floor if is_pac_pgd else None,
        'pac_pgd_prior_floor': args.pac_pgd_prior_floor if is_pac_pgd else None,
        'pac_pgd_summary': pac_pgd_state.summary() if pac_pgd_state is not None else None,
        'uncertainty_guided': args.pace_type in ['pace_uncert_soft', 'pace_uncert_topk', 'pace_kl_uncert_soft', 'pace_kl_uncert_topk'],
        'uncertainty_score': args.uncertainty_score if args.pace_type in ['pace_uncert_soft', 'pace_uncert_topk', 'pace_kl_uncert_soft', 'pace_kl_uncert_topk'] else None,
        'uncertainty_fraction': args.uncertainty_fraction if args.pace_type in ['pace_uncert_soft', 'pace_uncert_topk', 'pace_kl_uncert_soft', 'pace_kl_uncert_topk'] else None,
        'uncertainty_weight': args.uncertainty_weight if args.pace_type in ['pace_uncert_soft', 'pace_uncert_topk', 'pace_kl_uncert_soft', 'pace_kl_uncert_topk'] else None,
        'blob': args.blob,
        'blob_lbd': args.blob_lbd if args.blob else None,
        'blob_kl_reduction': args.blob_kl_reduction if args.blob else None,
        'blob_prior_sigma': args.blob_prior_sigma if args.blob else None,
        'blob_init_sigma': args.blob_init_sigma if args.blob else None,
        'full_bayes_lora': args.full_bayes_lora,
        'bayes_lora_paper_faithful_scaling': bool(args.full_bayes_lora),
        'bayes_lora_flow': args.bayes_lora_flow if args.full_bayes_lora else None,
        'bayes_lora_flow_depth': args.bayes_lora_flow_depth if args.full_bayes_lora else None,
        'bayes_lora_lbd_u': args.bayes_lora_lbd_u if args.full_bayes_lora else None,
        'bayes_lora_lbd_w': args.bayes_lora_lbd_w if args.full_bayes_lora else None,
        'bayes_lora_init_sigma': args.bayes_lora_init_sigma if args.full_bayes_lora else None,
        'bayes_lora_prior_sigma': args.bayes_lora_prior_sigma if args.full_bayes_lora else None,
        'bayes_lora_max_sigma_u': args.bayes_lora_max_sigma_u if args.full_bayes_lora else None,
        'bayes_lora_lambda_init': args.bayes_lora_lambda_init if args.full_bayes_lora else None,
        'bayes_lora_lambda_max': args.bayes_lora_lambda_max if args.full_bayes_lora else None,
        'seed': args.seed,
        'epochs': args.epoch,
        'batch_size': args.bs,
        'num_workers': args.num_workers,
        'device': str(device),
        'total_train_seconds': total_train_seconds,
        'mean_epoch_seconds': float(np.mean(epoch_timings)) if epoch_timings else None,
        'std_epoch_seconds': float(np.std(epoch_timings)) if epoch_timings else None,
        'epoch_seconds': epoch_timings,
        'best_val_acc': best_val_acc,
        'final_val_results': final_val_results,
    }
    train_logger.close()
    timing_path = args.timing_json or os.path.join(args.save_path, 'timing.json')
    with open(timing_path, 'w') as f:
        json.dump(timing_summary, f, indent=2)
    print(f"Timing saved to {timing_path}")
