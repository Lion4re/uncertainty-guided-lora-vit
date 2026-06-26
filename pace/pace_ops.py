"""
Implementation of adding noise and applying consistency regularization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .residual_adapters import ResidualAdapter
from .sharable_dropout import set_duplicate

class MultiplicativeNoiseAdapter(nn.Module):
    def __init__(self, adapter:ResidualAdapter, sigma=1.,
                 shape:str='BC', # 'BC' or 'BTC',
                 adapter_dropout=0.,
                 ):
        super(MultiplicativeNoiseAdapter, self).__init__()
        self.adapter = adapter
        self.sigma = sigma
        self.shape = shape
        self.adapter_dropout = float(adapter_dropout)
        self.mc_dropout_enabled = False

    def _get_noise_shape(self, x:torch.Tensor) -> List[int] or torch.Size:
        if len(x.shape) >= 2:
            if self.shape == 'BC':
                return [x.shape[0]] + [1] * (x.ndim - 2) + [x.shape[-1]]
            else:
                return x.shape
        else:
            raise NotImplementedError

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        use_pace_noise = self.training and self.sigma > 0
        use_adapter_dropout = self.adapter_dropout > 0 and (self.training or self.mc_dropout_enabled)
        if use_pace_noise or use_adapter_dropout:
            base_feature = self.adapter.forward_base(x)
            adapter_feature = self.adapter.forward_adapter(x)
            if use_adapter_dropout:
                adapter_feature = torch.nn.functional.dropout(
                    adapter_feature, p=self.adapter_dropout, training=True)
            if use_pace_noise:
                with torch.no_grad():
                    noise_shape = self._get_noise_shape(adapter_feature)
                    noise = torch.randn(*noise_shape, device=adapter_feature.device) * self.sigma + 1
                adapter_feature = adapter_feature * noise
            return base_feature + adapter_feature
        else:
            return self.adapter.forward(x)

    def extra_repr(self) -> str:
        return f'sigma={self.sigma}, adapter_dropout={self.adapter_dropout}, shape={self.shape}'


class LearnableMultiplicativeNoiseAdapter(nn.Module):
    """PACE adapter noise with a learned positive sigma.

    This is a lightweight PAC-Bayes-style extension: each adapter learns the
    posterior perturbation scale while a KL term keeps it near a prior scale.
    """
    def __init__(self, adapter: ResidualAdapter, init_sigma=1., prior_sigma=1.,
                 min_sigma=1e-4, max_sigma=3., shape: str = 'BC',
                 adapter_dropout=0.):
        super().__init__()
        self.adapter = adapter
        self.prior_sigma = float(prior_sigma)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.shape = shape
        self.adapter_dropout = float(adapter_dropout)
        self.mc_dropout_enabled = False
        init = torch.as_tensor(float(init_sigma)).clamp_min(self.min_sigma)
        raw = torch.log(torch.expm1(init - self.min_sigma).clamp_min(1e-12))
        self.pac_log_sigma = nn.Parameter(raw.clone().detach())

    def sigma(self):
        sigma = torch.nn.functional.softplus(self.pac_log_sigma) + self.min_sigma
        return sigma.clamp_max(self.max_sigma)

    def _get_noise_shape(self, x: torch.Tensor) -> List[int] or torch.Size:
        if len(x.shape) >= 2:
            if self.shape == 'BC':
                return [x.shape[0]] + [1] * (x.ndim - 2) + [x.shape[-1]]
            return x.shape
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_pace_noise = self.training and self.sigma().item() > 0
        use_adapter_dropout = self.adapter_dropout > 0 and (self.training or self.mc_dropout_enabled)
        if use_pace_noise or use_adapter_dropout:
            base_feature = self.adapter.forward_base(x)
            adapter_feature = self.adapter.forward_adapter(x)
            if use_adapter_dropout:
                adapter_feature = torch.nn.functional.dropout(
                    adapter_feature, p=self.adapter_dropout, training=True)
            if use_pace_noise:
                noise_shape = self._get_noise_shape(adapter_feature)
                noise = torch.randn(*noise_shape, device=adapter_feature.device) * self.sigma() + 1
                adapter_feature = adapter_feature * noise
            return base_feature + adapter_feature
        return self.adapter.forward(x)

    def pac_bayes_kl(self):
        sigma = self.sigma()
        prior_sigma = torch.as_tensor(self.prior_sigma, device=sigma.device, dtype=sigma.dtype)
        return 0.5 * ((sigma / prior_sigma).square() - 1.0 + 2.0 * torch.log(prior_sigma / sigma))

    def extra_repr(self) -> str:
        return (f'sigma={self.sigma().item():.4f}, prior_sigma={self.prior_sigma:g}, '
                f'adapter_dropout={self.adapter_dropout}, shape={self.shape}')


def pac_bayes_noise_kl(model):
    losses = []
    for module in model.modules():
        if isinstance(module, LearnableMultiplicativeNoiseAdapter):
            losses.append(module.pac_bayes_kl())
    if not losses:
        return None
    return torch.stack(losses).mean()


def set_adapter_mc_dropout(model, enabled):
    for module in model.modules():
        if isinstance(module, (MultiplicativeNoiseAdapter, LearnableMultiplicativeNoiseAdapter)):
            module.mc_dropout_enabled = bool(enabled)


def pac_bayes_sigma_summary(model):
    sigmas = []
    for module in model.modules():
        if isinstance(module, LearnableMultiplicativeNoiseAdapter):
            sigmas.append(module.sigma().detach().reshape(1))
    if not sigmas:
        return None
    values = torch.cat(sigmas).float().cpu()
    return {
        'count': int(values.numel()),
        'mean': values.mean().item(),
        'std': values.std(unbiased=False).item() if values.numel() > 1 else 0.0,
        'min': values.min().item(),
        'max': values.max().item(),
        'p50': values.quantile(0.50).item(),
        'p90': values.quantile(0.90).item(),
        'p99': values.quantile(0.99).item(),
    }


class PACE_MSELoss(nn.Module):
    def forward(self, input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        return (input-target).square().sum(dim=-1).mean()

class PACEOffsetFreeMSELoss(nn.Module):
    """MSE after projecting the logit difference off the all-ones direction."""
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = input - target
        diff = diff - diff.mean(dim=-1, keepdim=True)
        return diff.square().sum(dim=-1).mean()

_mse_criterion = PACE_MSELoss()
_offset_free_mse_criterion = PACEOffsetFreeMSELoss()

def compute_loss_pace(model, x, y, criterion, lbd_pace=1, pace_criterion=_mse_criterion, **kwargs):
    # duplicate the input, e.g [x1, x2, x3] will be [x1, x2, x3, x1, x2, x3]
    set_duplicate(model, 2)
    x_duplicate = torch.cat([x, x])
    logits = model(x_duplicate)
    logits_1, logits_2 = torch.chunk(logits, chunks=2)
    cls_loss = criterion(logits_1, y)
    pace_loss = pace_criterion(logits_1, logits_2)
    total_loss = cls_loss + lbd_pace * pace_loss
    return {'cls_loss': cls_loss, 'pace_loss': pace_loss, 'total_loss': total_loss, 'logits': logits_1}

def compute_loss_pace_offset(model, x, y, criterion, lbd_pace=1, **kwargs):
    return compute_loss_pace(
        model, x, y, criterion,
        lbd_pace=lbd_pace,
        pace_criterion=_offset_free_mse_criterion,
        **kwargs,
    )

def compute_loss_pace_lazy_half(model, x, y, criterion, lbd_pace=1, pace_criterion=_mse_criterion, itr=0, lazy_interval=2, **kwargs):
    bs = x.shape[0]
    results_dict = {}
    if itr % lazy_interval == 0:
        bs_half = bs // 2
        set_duplicate(model, 2)
        x_duplicate = torch.cat([x[:bs_half], x[:bs_half]])
        logits = model(x_duplicate)
        logits_1, logits_2 = torch.chunk(logits, chunks=2)
        cls_loss = criterion(logits_1, y[:bs_half])
        pace_loss = pace_criterion(logits_1, logits_2)
        total_loss = cls_loss + lbd_pace * pace_loss
        results_dict['pace_loss'] = pace_loss
    else:
        set_duplicate(model, 1)
        logits_1 = model(x)
        cls_loss = criterion(logits_1, y)
        total_loss = cls_loss
    results_dict.update({'cls_loss': cls_loss, 'total_loss': total_loss, 'logits': logits_1})
    return results_dict


def compute_loss_pace_fast(model, x, y, criterion, lbd_pace=1, pace_criterion=_mse_criterion, history_logits=None, index=None, **kwargs):
    set_duplicate(model, 1)
    logits = model(x)
    cls_loss = criterion(logits, y)
    with torch.no_grad():
        logits_recent = history_logits[index].to(logits.device)
        history_logits[index] = logits.to(history_logits.device)
    pace_loss = pace_criterion(logits, logits_recent)
    total_loss = cls_loss + lbd_pace * pace_loss
    return {'cls_loss': cls_loss, 'pace_loss': pace_loss, 'total_loss': total_loss, 'logits': logits}



class PACE_KLLoss(nn.Module):
    def __init__(self, temperature=2.0, detach_target=False):
        super().__init__()
        self.temperature = temperature
        self.detach_target = detach_target

    def forward(self, input, target):
        p = torch.nn.functional.log_softmax(input / self.temperature, dim=-1)
        if self.detach_target:
            target = target.detach()
        q = torch.nn.functional.softmax(target / self.temperature, dim=-1)
        return (torch.nn.functional.kl_div(p, q, reduction='batchmean')
                * self.temperature ** 2)

_kl_criterion = PACE_KLLoss(temperature=2.0)


def _get_kl_criterion(temperature, detach_target=False):
    if temperature == 2.0 and not detach_target:
        return _kl_criterion
    return PACE_KLLoss(temperature=temperature, detach_target=detach_target)


def compute_predictive_uncertainty(logits, score="entropy"):
    """Return one detached uncertainty score per sample; larger means harder."""
    probs = F.softmax(logits.detach(), dim=-1)
    if score == "entropy":
        return -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
    if score == "max_conf":
        return 1.0 - probs.max(dim=-1).values
    if score == "margin":
        top2 = probs.topk(k=min(2, probs.shape[-1]), dim=-1).values
        if top2.shape[-1] == 1:
            return torch.ones_like(top2[..., 0])
        return 1.0 - (top2[..., 0] - top2[..., 1])
    raise ValueError(f"Unknown uncertainty score: {score}")


def _normalize_uncertainty_scores(scores):
    scores = scores.detach().float()
    max_score = scores.max().clamp_min(1e-12)
    return (scores / max_score).clamp_min(0.0)


def build_uncertainty_selection_weights(scores, mode="soft", fraction=0.30, weight=1.0):
    """Build detached per-sample weights for uncertainty-guided selective updates."""
    if scores.ndim != 1:
        raise ValueError("Uncertainty scores must be a 1D tensor.")
    if scores.numel() == 0:
        raise ValueError("Uncertainty scores cannot be empty.")
    if mode not in {"soft", "topk"}:
        raise ValueError(f"Unknown uncertainty mode: {mode}")

    scores = scores.detach()
    if mode == "topk":
        k = max(1, int(torch.ceil(torch.as_tensor(float(scores.numel()) * fraction)).item()))
        k = min(k, scores.numel())
        indices = torch.topk(scores, k=k).indices
        weights = torch.zeros_like(scores, dtype=torch.float32)
        weights[indices] = 1.0
    else:
        normalized = _normalize_uncertainty_scores(scores).to(scores.device)
        weights = 1.0 + float(weight) * normalized
        weights = weights / weights.mean().clamp_min(1e-12)

    weights = weights.detach()
    effective_batch_size = weights.sum()
    stats = {
        "mean_uncertainty": scores.float().mean().detach(),
        "selected_fraction": (weights > 0).float().mean().detach(),
        "effective_batch_size": effective_batch_size.detach(),
    }
    return weights, stats


def _weighted_mean(values, weights):
    weights = weights.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


def _kl_per_sample(input, target, temperature=2.0, detach_target=False):
    log_p = F.log_softmax(input / temperature, dim=-1)
    if detach_target:
        target = target.detach()
    q = F.softmax(target / temperature, dim=-1)
    return F.kl_div(log_p, q, reduction="none").sum(dim=-1) * temperature ** 2


def _mse_per_sample(input, target):
    return (input - target).square().sum(dim=-1)

def compute_loss_pace_kl(model, x, y, criterion, lbd_pace=1,
                         pace_criterion=None, temperature=2.0,
                         detach_target=False, **kwargs):
    if pace_criterion is None:
        pace_criterion = _get_kl_criterion(temperature, detach_target=detach_target)
    set_duplicate(model, 2)
    x_duplicate = torch.cat([x, x])
    logits = model(x_duplicate)
    logits_1, logits_2 = torch.chunk(logits, chunks=2)
    cls_loss = criterion(logits_1, y)
    pace_loss = pace_criterion(logits_1, logits_2)
    total_loss = cls_loss + lbd_pace * pace_loss
    return {'cls_loss': cls_loss, 'pace_loss': pace_loss,
            'total_loss': total_loss, 'logits': logits_1}


def compute_loss_pace_kl_uncertainty(model, x, y, criterion, lbd_pace=1,
                                     temperature=2.0,
                                     detach_target=False,
                                     uncertainty_score="entropy",
                                     uncertainty_mode="soft",
                                     uncertainty_fraction=0.30,
                                     uncertainty_weight=1.0,
                                     **kwargs):
    set_duplicate(model, 2)
    x_duplicate = torch.cat([x, x])
    logits = model(x_duplicate)
    logits_1, logits_2 = torch.chunk(logits, chunks=2)
    uncertainty = compute_predictive_uncertainty(logits_1, score=uncertainty_score)
    weights, stats = build_uncertainty_selection_weights(
        uncertainty,
        mode=uncertainty_mode,
        fraction=uncertainty_fraction,
        weight=uncertainty_weight,
    )
    cls_losses = F.cross_entropy(logits_1, y, reduction="none")
    pace_losses = _kl_per_sample(logits_1, logits_2, temperature=temperature,
                                 detach_target=detach_target)
    cls_loss = _weighted_mean(cls_losses, weights)
    pace_loss = _weighted_mean(pace_losses, weights)
    total_loss = cls_loss + lbd_pace * pace_loss
    return {
        "cls_loss": cls_loss,
        "pace_loss": pace_loss,
        "total_loss": total_loss,
        "logits": logits_1,
        "mean_uncertainty": stats["mean_uncertainty"].to(logits_1.device),
        "selected_fraction": stats["selected_fraction"].to(logits_1.device),
        "effective_batch_size": stats["effective_batch_size"].to(logits_1.device),
        "uncertainty_weights": weights.to(logits_1.device),
    }


def compute_loss_pace_uncertainty(model, x, y, criterion, lbd_pace=1,
                                  uncertainty_score="entropy",
                                  uncertainty_mode="soft",
                                  uncertainty_fraction=0.30,
                                  uncertainty_weight=1.0,
                                  **kwargs):
    """PACE-MSE with uncertainty-guided per-sample weighting/selection."""
    set_duplicate(model, 2)
    x_duplicate = torch.cat([x, x])
    logits = model(x_duplicate)
    logits_1, logits_2 = torch.chunk(logits, chunks=2)
    uncertainty = compute_predictive_uncertainty(logits_1, score=uncertainty_score)
    weights, stats = build_uncertainty_selection_weights(
        uncertainty,
        mode=uncertainty_mode,
        fraction=uncertainty_fraction,
        weight=uncertainty_weight,
    )
    cls_losses = F.cross_entropy(logits_1, y, reduction="none")
    pace_losses = _mse_per_sample(logits_1, logits_2)
    cls_loss = _weighted_mean(cls_losses, weights)
    pace_loss = _weighted_mean(pace_losses, weights)
    total_loss = cls_loss + lbd_pace * pace_loss
    return {
        "cls_loss": cls_loss,
        "pace_loss": pace_loss,
        "total_loss": total_loss,
        "logits": logits_1,
        "mean_uncertainty": stats["mean_uncertainty"].to(logits_1.device),
        "selected_fraction": stats["selected_fraction"].to(logits_1.device),
        "effective_batch_size": stats["effective_batch_size"].to(logits_1.device),
        "uncertainty_weights": weights.to(logits_1.device),
    }


def compute_loss_pace_kl_learnsigma(model, x, y, criterion, lbd_pace=1,
                                    pace_criterion=None, temperature=2.0,
                                    detach_target=False, **kwargs):
    """PACE-KL with learned adapter-noise scale but no PAC KL penalty."""
    if pace_criterion is None:
        pace_criterion = _get_kl_criterion(temperature, detach_target=detach_target)
    set_duplicate(model, 2)
    x_duplicate = torch.cat([x, x])
    logits = model(x_duplicate)
    logits_1, logits_2 = torch.chunk(logits, chunks=2)
    cls_loss = criterion(logits_1, y)
    pace_loss = pace_criterion(logits_1, logits_2)
    total_loss = cls_loss + lbd_pace * pace_loss
    pac_kl = pac_bayes_noise_kl(model)
    if pac_kl is None:
        pac_kl = torch.zeros((), device=logits_1.device, dtype=logits_1.dtype)
    return {'cls_loss': cls_loss, 'pace_loss': pace_loss,
            'learned_noise_kl_probe': pac_kl,
            'total_loss': total_loss, 'logits': logits_1}


def compute_loss_pace_combined(model, x, y, criterion, lbd_pace=1,
                               lbd_kl=0.5, temperature=2.0, **kwargs):
    """Combined MSE + KL consistency: accuracy from MSE, calibration from KL."""
    kl_criterion = _get_kl_criterion(temperature)
    set_duplicate(model, 2)
    x_duplicate = torch.cat([x, x])
    logits = model(x_duplicate)
    logits_1, logits_2 = torch.chunk(logits, chunks=2)
    cls_loss = criterion(logits_1, y)
    mse_loss = _mse_criterion(logits_1, logits_2)
    kl_loss = kl_criterion(logits_1, logits_2)
    total_loss = cls_loss + lbd_pace * mse_loss + lbd_kl * kl_loss
    return {'cls_loss': cls_loss, 'pace_loss': mse_loss, 'kl_loss': kl_loss,
            'total_loss': total_loss, 'logits': logits_1}


def compute_loss_pace_kl_margin(model, x, y, criterion, lbd_pace=1,
                                 lbd_margin=0.1, margin_target=1.0,
                                 pace_criterion=None, temperature=2.0, **kwargs):
    """PACE-KL + MaCS-style margin penalty."""
    if pace_criterion is None:
        pace_criterion = _get_kl_criterion(temperature)
    set_duplicate(model, 2)
    x_duplicate = torch.cat([x, x])
    logits = model(x_duplicate)
    logits_1, logits_2 = torch.chunk(logits, chunks=2)
    cls_loss = criterion(logits_1, y)
    pace_loss = pace_criterion(logits_1, logits_2)
    correct_logits = logits_1.gather(1, y.unsqueeze(1)).squeeze(1)
    logits_masked = logits_1.clone()
    logits_masked.scatter_(1, y.unsqueeze(1), float('-inf'))
    max_other = logits_masked.max(dim=1)[0]
    margin = correct_logits - max_other
    margin_loss = (torch.clamp(margin_target - margin, min=0) ** 2).mean()
    total_loss = cls_loss + lbd_pace * pace_loss + lbd_margin * margin_loss
    return {'cls_loss': cls_loss, 'pace_loss': pace_loss,
            'margin_loss': margin_loss, 'total_loss': total_loss, 'logits': logits_1}


def compute_loss_pace_kl_flat(model, x, y, criterion, lbd_pace=1,
                               pace_criterion=None,
                               flat_sigma=0.1, **kwargs):
    """PACE-KL with Flat-LoRA-style random weight perturbation."""
    import torch.nn as nn
    temperature = kwargs.get('temperature', 2.0)
    if pace_criterion is None:
        pace_criterion = _get_kl_criterion(temperature)
    perturbations = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'head' not in name:
                W = module.weight.data
                filter_norms = W.norm(dim=1, keepdim=True)
                n = W.shape[1]
                noise = torch.randn_like(W) * (flat_sigma / (n ** 0.5)) * filter_norms
                perturbations[name] = noise
                module.weight.data = W + noise

    set_duplicate(model, 2)
    x_duplicate = torch.cat([x, x])
    logits = model(x_duplicate)
    logits_1, logits_2 = torch.chunk(logits, chunks=2)
    cls_loss = criterion(logits_1, y)
    pace_loss = pace_criterion(logits_1, logits_2)
    total_loss = cls_loss + lbd_pace * pace_loss

    with torch.no_grad():
        for name, module in model.named_modules():
            if name in perturbations:
                module.weight.data = module.weight.data - perturbations[name]

    return {'cls_loss': cls_loss, 'pace_loss': pace_loss,
            'total_loss': total_loss, 'logits': logits_1}


def compute_loss_pace_kl_jacobian(model, x, y, criterion, lbd_pace=1,
                                  lbd_jacobian=0.1, jacobian_eps=1e-2,
                                  pace_criterion=None, temperature=2.0,
                                  **kwargs):
    """PACE-KL plus a MaCS-style finite-difference input sensitivity penalty.

    The sensitivity proxy avoids building the full input-output Jacobian. It
    measures the average local Lipschitz response of logits to small Gaussian
    input perturbations.
    """
    if pace_criterion is None:
        pace_criterion = _get_kl_criterion(temperature)

    set_duplicate(model, 2)
    x_duplicate = torch.cat([x, x])
    logits = model(x_duplicate)
    logits_1, logits_2 = torch.chunk(logits, chunks=2)
    cls_loss = criterion(logits_1, y)
    pace_loss = pace_criterion(logits_1, logits_2)

    eps = torch.randn_like(x) * jacobian_eps
    x_perturbed = x + eps
    set_duplicate(model, 1)
    logits_perturbed = model(x_perturbed)
    numerator = (logits_perturbed - logits_1).abs().amax(dim=1)
    denominator = eps.flatten(1).norm(dim=1).clamp_min(1e-12)
    jacobian_loss = (numerator / denominator).mean()

    total_loss = cls_loss + lbd_pace * pace_loss + lbd_jacobian * jacobian_loss
    return {'cls_loss': cls_loss, 'pace_loss': pace_loss,
            'jacobian_loss': jacobian_loss, 'total_loss': total_loss,
            'logits': logits_1}


def compute_loss_pace_kl_pacbayes(model, x, y, criterion, lbd_pace=1,
                                  pac_lbd=1e-3, pace_criterion=None,
                                  temperature=2.0, detach_target=False, **kwargs):
    """PACE-KL with a PAC-Bayes KL penalty over learned adapter-noise scales."""
    if pace_criterion is None:
        pace_criterion = _get_kl_criterion(temperature, detach_target=detach_target)
    set_duplicate(model, 2)
    x_duplicate = torch.cat([x, x])
    logits = model(x_duplicate)
    logits_1, logits_2 = torch.chunk(logits, chunks=2)
    cls_loss = criterion(logits_1, y)
    pace_loss = pace_criterion(logits_1, logits_2)
    pac_kl = pac_bayes_noise_kl(model)
    if pac_kl is None:
        pac_kl = torch.zeros((), device=logits_1.device, dtype=logits_1.dtype)
    total_loss = cls_loss + lbd_pace * pace_loss + pac_lbd * pac_kl
    return {'cls_loss': cls_loss, 'pace_loss': pace_loss, 'pac_bayes_kl': pac_kl,
            'total_loss': total_loss, 'logits': logits_1}


def _add_pac_bayes_kl(model, logits, total_loss, pac_lbd):
    pac_kl = pac_bayes_noise_kl(model)
    if pac_kl is None:
        pac_kl = torch.zeros((), device=logits.device, dtype=logits.dtype)
    total_loss = total_loss + pac_lbd * pac_kl
    return total_loss, pac_kl


def compute_loss_pace_pacbayes(model, x, y, criterion, lbd_pace=1,
                               pac_lbd=1e-3, pace_criterion=_mse_criterion,
                               **kwargs):
    """PACE-MSE with a PAC-Bayes KL penalty over learned adapter-noise scales."""
    set_duplicate(model, 2)
    x_duplicate = torch.cat([x, x])
    logits = model(x_duplicate)
    logits_1, logits_2 = torch.chunk(logits, chunks=2)
    cls_loss = criterion(logits_1, y)
    pace_loss = pace_criterion(logits_1, logits_2)
    total_loss = cls_loss + lbd_pace * pace_loss
    total_loss, pac_kl = _add_pac_bayes_kl(model, logits_1, total_loss, pac_lbd)
    return {'cls_loss': cls_loss, 'pace_loss': pace_loss, 'pac_bayes_kl': pac_kl,
            'total_loss': total_loss, 'logits': logits_1}


def compute_loss_pace_kl_margin_pacbayes(model, x, y, criterion, lbd_pace=1,
                                         pac_lbd=1e-3, lbd_margin=0.1,
                                         margin_target=1.0, pace_criterion=None,
                                         temperature=2.0, detach_target=False, **kwargs):
    """PACE-KL + margin with a PAC-Bayes KL penalty over learned noise scales."""
    if pace_criterion is None:
        pace_criterion = _get_kl_criterion(temperature, detach_target=detach_target)
    set_duplicate(model, 2)
    x_duplicate = torch.cat([x, x])
    logits = model(x_duplicate)
    logits_1, logits_2 = torch.chunk(logits, chunks=2)
    cls_loss = criterion(logits_1, y)
    pace_loss = pace_criterion(logits_1, logits_2)
    correct_logits = logits_1.gather(1, y.unsqueeze(1)).squeeze(1)
    logits_masked = logits_1.clone()
    logits_masked.scatter_(1, y.unsqueeze(1), float('-inf'))
    max_other = logits_masked.max(dim=1)[0]
    margin = correct_logits - max_other
    margin_loss = (torch.clamp(margin_target - margin, min=0) ** 2).mean()
    total_loss = cls_loss + lbd_pace * pace_loss + lbd_margin * margin_loss
    total_loss, pac_kl = _add_pac_bayes_kl(model, logits_1, total_loss, pac_lbd)
    return {'cls_loss': cls_loss, 'pace_loss': pace_loss,
            'margin_loss': margin_loss, 'pac_bayes_kl': pac_kl,
            'total_loss': total_loss, 'logits': logits_1}
