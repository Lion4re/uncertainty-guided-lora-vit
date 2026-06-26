"""
Implementation for Residual Adapters
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _blob_sample_enabled(module):
    return bool(getattr(module, "blob_mc_sample", False))


def _blob_delta_w_up(module):
    delta_w_up = module.delta_W_up
    if not hasattr(module, "blob_rho"):
        return delta_w_up
    if module.training or _blob_sample_enabled(module):
        sigma = F.softplus(module.blob_rho) + module.blob_min_sigma
        return delta_w_up + sigma * torch.randn_like(delta_w_up)
    return delta_w_up


def _lora_factors(module):
    delta_w_down = module.delta_W_down
    delta_w_up = _blob_delta_w_up(module)
    state = getattr(module, "full_bayes_lora_state", None)
    if state is not None:
        return state.sample_factors(delta_w_down, delta_w_up)
    return delta_w_down, delta_w_up


def _lora_scaling(module):
    state = getattr(module, "full_bayes_lora_state", None)
    if state is not None:
        return float(state.lora_scaling)
    return 1.0

class ResidualAdapter(nn.Module):
    def __init__(self, base_module, merge_training=True, merge_validation=True):
        super(ResidualAdapter, self).__init__()
        self.base_module = base_module
        self.merge_training = merge_training
        self.merge_validation = merge_validation

    def forward_base(self, x: torch.Tensor, **kwargs):
        return self.base_module(x, **kwargs)

    def forward_adapter(self, x: torch.Tensor, **kwargs):
        raise NotImplementedError

    def forward_merge(self, x: torch.Tensor, **kwargs):
        raise NotImplementedError

    def forward(self, x: torch.Tensor, **kwargs):
        if (self.training and self.merge_training) or (not self.training and self.merge_validation):
            return self.forward_merge(x, **kwargs)
        else:
            return self.forward_base(x, **kwargs) + self.forward_adapter(x, **kwargs)

class LoRAmul_VPTadd_Linear(ResidualAdapter):
    def __init__(self, base_module:nn.Linear, rank=10, merge_training=True, merge_validation=True):
        super(LoRAmul_VPTadd_Linear, self).__init__(base_module, merge_training, merge_validation)
        out_features, in_features = base_module.weight.shape
        self.delta_W_down = nn.Parameter(torch.zeros(out_features, rank))
        self.delta_W_up   = nn.Parameter(torch.zeros(rank, in_features))
        self.delta_b      = nn.Parameter(torch.zeros(out_features))
        self.delta_prompt_down = nn.Parameter(torch.zeros(in_features, rank))
        self.delta_prompt_up   = nn.Parameter(torch.zeros(rank, 1))

        nn.init.xavier_uniform_(self.delta_W_up)
        nn.init.xavier_uniform_(self.delta_prompt_up)

    def forward_adapter(self, x:torch.Tensor, **kwargs):
        delta_w_down, delta_w_up = _lora_factors(self)
        W = self.base_module.weight * (delta_w_down @ delta_w_up)
        b = (self.base_module.weight @ (self.delta_prompt_down @ self.delta_prompt_up)).squeeze()
        if self.base_module.bias is not None: b = b + self.base_module.bias * self.delta_b
        return F.linear(x, W, b)

    def forward_merge(self, x: torch.Tensor, **kwargs):
        delta_w_down, delta_w_up = _lora_factors(self)
        W = self.base_module.weight * (1 + delta_w_down @ delta_w_up)
        b = (self.base_module.weight @ (self.delta_prompt_down @ self.delta_prompt_up)).squeeze()
        if self.base_module.bias is not None: b = b + self.base_module.bias * (1 + self.delta_b)
        return F.linear(x, W, b)

class LoRAadd_Linear(ResidualAdapter):
    def __init__(self, base_module:nn.Linear, rank=4, merge_training=True, merge_validation=True):
        super(LoRAadd_Linear, self).__init__(base_module, merge_training, merge_validation)
        out_features, in_features = base_module.weight.shape
        self.delta_W_down = nn.Parameter(torch.zeros(out_features, rank))
        self.delta_W_up   = nn.Parameter(torch.zeros(rank, in_features))
        self.delta_b      = nn.Parameter(torch.zeros(out_features))

        nn.init.xavier_uniform_(self.delta_W_up)

    def forward_adapter(self, x:torch.Tensor, **kwargs):
        delta_w_down, delta_w_up = _lora_factors(self)
        out = _lora_scaling(self) * F.linear(F.linear(x, delta_w_up, None), delta_w_down, None)
        return out + self.delta_b

    def forward_merge(self, x: torch.Tensor, **kwargs):
        delta_w_down, delta_w_up = _lora_factors(self)
        W = self.base_module.weight + _lora_scaling(self) * (delta_w_down @ delta_w_up)
        b = self.delta_b
        if self.base_module.bias is not None: b = b + self.base_module.bias
        return F.linear(x, W, b)

class VPTadd_Linear(ResidualAdapter):
    def __init__(self, base_module:nn.Linear, rank=4, merge_training=True, merge_validation=True):
        super(VPTadd_Linear, self).__init__(base_module, merge_training, merge_validation)
        out_features, in_features = base_module.weight.shape
        self.delta_prompt_down = nn.Parameter(torch.zeros(in_features, rank))
        self.delta_prompt_up = nn.Parameter(torch.zeros(rank, 1))

        nn.init.xavier_uniform_(self.delta_prompt_up)

    def forward_adapter(self, x:torch.Tensor, **kwargs):
        return (self.base_module.weight @ (self.delta_prompt_down @ self.delta_prompt_up)).view(1, -1).expand(*x.shape[:-1], -1)

    def forward_merge(self, x: torch.Tensor, **kwargs):
        W = self.base_module.weight
        b = (self.base_module.weight @ (self.delta_prompt_down @ self.delta_prompt_up)).squeeze()
        if self.base_module.bias is not None: b = b + self.base_module.bias
        return F.linear(x, W, b)


class HeadLinear(ResidualAdapter):
    def __init__(self, base_module:nn.Linear):
        super(HeadLinear, self).__init__(base_module)

    def forward_adapter(self, x, **kwargs):
        return self.base_module(x, **kwargs)

    def forward_base(self, x, **kwargs):
        return 0

    def forward_merge(self, x, **kwargs):
        return self.base_module(x, **kwargs)


def inject_residual_adapter(model, adapter='LoRAmul_VPTadd', rank=10):
    AdapterClass = LoRAmul_VPTadd_Linear
    if adapter == 'LoRAadd': AdapterClass = LoRAadd_Linear
    elif adapter == 'VPTadd': AdapterClass = VPTadd_Linear

    for name, l in model.named_modules():
        if isinstance(l, nn.Linear):
            parent_layer = model
            tokens = name.strip().split('.')
            for t in tokens[:-1]:
                parent_layer = parent_layer[int(t)] if t.isnumeric() else getattr(parent_layer, t)
            linear = getattr(parent_layer, tokens[-1])
            linear_adapter = HeadLinear(linear) if 'head' in tokens else AdapterClass(linear, rank=rank)
            setattr(parent_layer, tokens[-1], linear_adapter)

def get_adapters_and_block_ids(model):
    """
    Returns a list of [parent_layer, adapter_name, block_id] for each ResidualAdapter.
    The block_id of the classification head is set to max block_id + 1.
    """
    adapters_and_block_ids = []
    for name, l in model.named_modules():
        if isinstance(l, ResidualAdapter):
            parent_layer = model
            tokens = name.strip().split('.')
            block_key = []
            for t in tokens[:-1]:
                parent_layer = parent_layer[int(t)] if t.isnumeric() else getattr(parent_layer, t)
                if t.isnumeric(): block_key.append(int(t))
            if 'head' in name: block_key = [float('inf')]
            adapters_and_block_ids.append([parent_layer, tokens[-1], block_key])

    max_len= max(len(bk) for _, _, bk in adapters_and_block_ids)
    adapters_and_block_ids = [[pl, tk, tuple(bk+[0] * (max_len-len(bk)))] for pl, tk, bk in adapters_and_block_ids]
    block_id_map = {block_key: i for i, block_key in enumerate(sorted(set(bk for _, _, bk in adapters_and_block_ids)))}

    for i, (_, _, block_key) in enumerate(adapters_and_block_ids):
        adapters_and_block_ids[i][2] = block_id_map[block_key]

    return adapters_and_block_ids


def enable_blob(model, init_sigma=1e-4, prior_sigma=1.0, min_sigma=1e-8):
    """Enable BLoB-style mean-field VI over LoRA A matrices.

    Following the C-LoRA paper's BLoB description, B (delta_W_down) stays
    deterministic while A (delta_W_up) receives a diagonal Gaussian posterior.
    The existing delta_W_up parameter is the posterior mean.
    """
    for module in model.modules():
        if hasattr(module, "delta_W_up") and not hasattr(module, "blob_rho"):
            init_sigma_t = torch.as_tensor(float(init_sigma)).clamp_min(float(min_sigma))
            raw = torch.log(torch.expm1(init_sigma_t - float(min_sigma)).clamp_min(1e-12))
            module.blob_rho = nn.Parameter(torch.full_like(module.delta_W_up, raw.item()))
            module.blob_prior_sigma = float(prior_sigma)
            module.blob_min_sigma = float(min_sigma)
            module.blob_mc_sample = False


def set_blob_mc_sample(model, enabled):
    for module in model.modules():
        if hasattr(module, "blob_rho"):
            module.blob_mc_sample = bool(enabled)


def blob_kl(model, reduction="mean"):
    """KL(q(A) || p(A)) for BLoB-style Bayesian LoRA A matrices.

    ``reduction="mean"`` keeps the old normalized scale, which is easier to
    combine with the existing PACE losses. ``reduction="sum"`` exposes the raw
    ELBO-style KL if we want to tune the coefficient against dataset size.
    """
    losses = []
    counts = []
    for module in model.modules():
        if hasattr(module, "blob_rho"):
            mu = module.delta_W_up
            sigma = F.softplus(module.blob_rho) + module.blob_min_sigma
            prior_sigma = torch.as_tensor(module.blob_prior_sigma, device=mu.device, dtype=mu.dtype)
            kl = 0.5 * ((sigma.square() + mu.square()) / prior_sigma.square()
                        - 1.0 + 2.0 * torch.log(prior_sigma / sigma))
            losses.append(kl.sum())
            counts.append(torch.as_tensor(kl.numel(), device=mu.device, dtype=mu.dtype))
    if not losses:
        return None
    total = torch.stack(losses).sum()
    if reduction == "sum":
        return total
    if reduction != "mean":
        raise ValueError(f"Unknown BLoB KL reduction: {reduction}")
    return total / torch.stack(counts).sum().clamp_min(1.0)


def blob_sigma_summary(model):
    sigmas = []
    for module in model.modules():
        if hasattr(module, "blob_rho"):
            sigma = F.softplus(module.blob_rho) + module.blob_min_sigma
            sigmas.append(sigma.detach().flatten())
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
