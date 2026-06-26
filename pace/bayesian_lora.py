from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BayesianLoRAConfig:
    rank: int
    lora_alpha: Optional[float] = None
    inducing_rows: Optional[int] = None
    inducing_cols: Optional[int] = None
    flow: str = "none"
    flow_depth: int = 1
    init_sigma: float = 1e-4
    prior_sigma: float = 0.1
    max_sigma_u: float = 0.1
    lambda_init: float = 1e-3
    lambda_max: float = 3e-2
    min_sigma: float = 1e-8


class MaskedAutoregressiveLayer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        tril = torch.tril(torch.ones(dim, dim), diagonal=-1)
        self.register_buffer("input_mask", tril)
        self.max_log_scale = 3.0

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = z.shape
        flat = z.reshape(-1, self.dim)
        masked_inputs = flat.unsqueeze(1) * self.input_mask.unsqueeze(0)
        params = self.hidden(masked_inputs)
        shift, log_scale = params.unbind(dim=-1)
        log_scale = torch.tanh(log_scale) * self.max_log_scale
        out = flat * torch.exp(log_scale) + shift
        log_det = log_scale.sum(dim=-1)
        return out.reshape(original_shape), log_det.reshape(original_shape[:-1])


class MaskedAutoregressiveFlow(nn.Module):
    def __init__(self, dim: int, depth: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([
            MaskedAutoregressiveLayer(dim, hidden_dim) for _ in range(depth)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        total_log_det = torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype)
        out = z
        for layer in self.layers:
            out, log_det = layer(out)
            total_log_det = total_log_det + log_det
            out = torch.flip(out, dims=[-1])
        return out, total_log_det


class RowWiseMaskedAutoregressiveFlow(nn.Module):
    """Lightweight row-wise MAF over the inducing matrix U.

    Each row of U is transformed autoregressively over the column dimension.
    The same small flow is shared across rows, matching the paper's row-wise
    MAF idea while keeping the parameter overhead modest for ViT adapters.
    """

    def __init__(self, cols: int, depth: int, hidden_dim: int):
        super().__init__()
        self.flow = MaskedAutoregressiveFlow(cols, depth, hidden_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, row_log_det = self.flow(z)
        return out, row_log_det.sum(dim=-1)


class BayesianLoRAState(nn.Module):
    def __init__(self, out_features: int, in_features: int, config: BayesianLoRAConfig):
        super().__init__()
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.rank = int(config.rank)
        self.lora_alpha = float(config.lora_alpha if config.lora_alpha is not None else config.rank)
        self.lora_scaling = self.lora_alpha / float(self.rank)
        self.inducing_rows = int(config.inducing_rows or config.rank)
        self.inducing_cols = int(config.inducing_cols or config.rank)
        self.flow = config.flow
        self.prior_sigma = float(config.prior_sigma)
        self.min_sigma = float(config.min_sigma)
        self.max_sigma_u = float(config.max_sigma_u)
        self.lambda_max = float(config.lambda_max)
        self.mc_sample = False

        shape = (self.inducing_rows, self.inducing_cols)
        self.U_A_mu = nn.Parameter(torch.empty(shape))
        self.U_B_mu = nn.Parameter(torch.zeros(shape))
        raw_sigma = self._inv_softplus(float(config.init_sigma), self.min_sigma)
        self.U_A_rho = nn.Parameter(torch.full(shape, raw_sigma))
        self.U_B_rho = nn.Parameter(torch.full(shape, raw_sigma))

        self.A_row_proj = nn.Parameter(torch.zeros(self.rank, self.inducing_rows))
        self.A_col_proj = nn.Parameter(torch.zeros(self.inducing_cols, self.in_features))
        self.B_row_proj = nn.Parameter(torch.zeros(self.out_features, self.inducing_rows))
        self.B_col_proj = nn.Parameter(torch.zeros(self.inducing_cols, self.rank))
        lambda_raw = self._inv_softplus(float(config.lambda_init), self.min_sigma)
        self.lambda_raw = nn.Parameter(torch.as_tensor(lambda_raw))

        nn.init.normal_(self.U_A_mu, std=0.02)
        self._init_projections()
        if self.flow == "maf":
            dim = self.inducing_rows * self.inducing_cols
            hidden_dim = max(8, 2 * dim)
            self.flow_A = MaskedAutoregressiveFlow(dim, config.flow_depth, hidden_dim)
            self.flow_B = MaskedAutoregressiveFlow(dim, config.flow_depth, hidden_dim)
        elif self.flow == "row_maf":
            hidden_dim = max(8, 2 * self.inducing_cols)
            self.flow_A = RowWiseMaskedAutoregressiveFlow(
                self.inducing_cols, config.flow_depth, hidden_dim)
            self.flow_B = RowWiseMaskedAutoregressiveFlow(
                self.inducing_cols, config.flow_depth, hidden_dim)
        elif self.flow != "none":
            raise ValueError(f"Unknown Bayesian-LoRA flow: {self.flow}")
        else:
            self.flow_A = None
            self.flow_B = None

    @staticmethod
    def _inv_softplus(value: float, min_value: float) -> float:
        v = torch.as_tensor(float(value)).clamp_min(float(min_value) + 1e-12)
        return torch.log(torch.expm1(v - float(min_value)).clamp_min(1e-12)).item()

    def _init_projections(self):
        nn.init.normal_(self.A_row_proj, std=0.02)
        nn.init.normal_(self.A_col_proj, std=0.02)
        nn.init.normal_(self.B_row_proj, std=0.02)
        nn.init.normal_(self.B_col_proj, std=0.02)

    def lambda_sigma(self):
        return (F.softplus(self.lambda_raw) + self.min_sigma).clamp_max(self.lambda_max)

    def _sigma(self, rho):
        return (F.softplus(rho) + self.min_sigma).clamp_max(self.max_sigma_u)

    def _sample_u(self, mu, rho, flow_module, force_sample=False):
        sigma = self._sigma(rho)
        should_sample = force_sample or self.training or self.mc_sample
        if should_sample:
            base = mu + sigma * torch.randn_like(mu)
        else:
            base = mu
        log_q_base = None
        log_det = torch.zeros((), device=mu.device, dtype=mu.dtype)
        if should_sample:
            log_q_base = self._diag_log_prob(base, mu, sigma)
        if flow_module is not None and self.flow == "maf":
            flat = base.flatten().unsqueeze(0)
            transformed, log_det_b = flow_module(flat)
            out = transformed.squeeze(0).reshape_as(base)
            log_det = log_det_b.squeeze(0)
        elif flow_module is not None:
            out, log_det = flow_module(base)
        else:
            out = base
        return out, log_q_base, log_det

    def _deterministic_u(self, mu, flow_module):
        if flow_module is None:
            return mu
        if self.flow == "maf":
            flat = mu.flatten().unsqueeze(0)
            transformed, _ = flow_module(flat)
            return transformed.squeeze(0).reshape_as(mu)
        transformed, _ = flow_module(mu)
        return transformed

    def _project_factors(self, u_a, u_b):
        """Conditional posterior mean M_W(U) = T_r U T_c for LoRA factors.

        The row/column projections are the vision-adapter analogue of the
        paper's inducing-variable projection operators. Conditional Gaussian
        noise is added in ``sample_factors`` through the learned lambda scale.
        """
        a_mean = self.A_row_proj @ u_a @ self.A_col_proj
        b_mean = self.B_row_proj @ u_b @ self.B_col_proj
        return b_mean, a_mean

    def sample_factors(self, base_down, base_up):
        del base_down, base_up
        u_a, _, _ = self._sample_u(self.U_A_mu, self.U_A_rho, self.flow_A)
        u_b, _, _ = self._sample_u(self.U_B_mu, self.U_B_rho, self.flow_B)
        b_mean, a_mean = self._project_factors(u_a, u_b)
        lam = self.lambda_sigma()
        if self.training or self.mc_sample:
            a_mean = a_mean + lam * torch.randn_like(a_mean)
            b_mean = b_mean + lam * torch.randn_like(b_mean)
        return b_mean, a_mean

    @staticmethod
    def _diag_log_prob(value, mu, sigma):
        return (-0.5 * (((value - mu) / sigma).square()
                        + 2.0 * torch.log(sigma)
                        + math.log(2.0 * math.pi))).sum()

    @staticmethod
    def _standard_log_prob(value):
        return (-0.5 * (value.square()
                        + math.log(2.0 * math.pi))).sum()

    def _kl_u_one(self, mu, rho, flow_module):
        sigma = self._sigma(rho)
        if flow_module is None:
            prior_sigma = torch.as_tensor(self.prior_sigma, device=mu.device, dtype=mu.dtype)
            return 0.5 * ((sigma.square() + mu.square()) / prior_sigma.square()
                          - 1.0 + 2.0 * torch.log(prior_sigma / sigma)).sum()
        u, log_q_base, log_det = self._sample_u(mu, rho, flow_module, force_sample=True)
        log_q = log_q_base - log_det
        prior_sigma = torch.as_tensor(self.prior_sigma, device=mu.device, dtype=mu.dtype)
        log_p = self._diag_log_prob(u, torch.zeros_like(u), prior_sigma)
        return log_q - log_p

    def kl_terms(self) -> Dict[str, torch.Tensor]:
        kl_u = self._kl_u_one(self.U_A_mu, self.U_A_rho, self.flow_A)
        kl_u = kl_u + self._kl_u_one(self.U_B_mu, self.U_B_rho, self.flow_B)
        lam = self.lambda_sigma()
        prior_sigma = torch.as_tensor(self.prior_sigma, device=lam.device, dtype=lam.dtype)
        weight_dim = self.rank * self.in_features + self.out_features * self.rank
        kl_w = 0.5 * weight_dim * (
            (lam / prior_sigma).square() - 1.0 + 2.0 * torch.log(prior_sigma / lam)
        )
        return {"kl_u": kl_u, "kl_w": kl_w, "kl_total": kl_u + kl_w}

    def summary(self) -> Dict[str, float]:
        with torch.no_grad():
            sigma_u = torch.cat([
                self._sigma(self.U_A_rho).flatten(),
                self._sigma(self.U_B_rho).flatten(),
            ]).float().cpu()
            lam = self.lambda_sigma().detach().float().cpu()
            return {
                "u_sigma_mean": sigma_u.mean().item(),
                "u_sigma_max": sigma_u.max().item(),
                "w_sigma_mean": lam.item(),
                "w_sigma_max": lam.item(),
                "lambda": lam.item(),
                "lora_alpha": self.lora_alpha,
                "lora_scaling": self.lora_scaling,
                "max_sigma_u": self.max_sigma_u,
                "flow": self.flow,
            }


def _iter_full_bayes_states(model):
    for module in model.modules():
        state = getattr(module, "full_bayes_lora_state", None)
        if isinstance(state, BayesianLoRAState):
            yield state


def enable_full_bayesian_lora(model, config: BayesianLoRAConfig):
    for module in model.modules():
        if hasattr(module, "delta_W_down") and hasattr(module, "delta_W_up"):
            out_features, rank = module.delta_W_down.shape
            rank_up, in_features = module.delta_W_up.shape
            if rank != rank_up:
                raise ValueError("LoRA down/up rank mismatch")
            layer_config = BayesianLoRAConfig(
                rank=rank,
                lora_alpha=config.lora_alpha,
                inducing_rows=config.inducing_rows or rank,
                inducing_cols=config.inducing_cols or rank,
                flow=config.flow,
                flow_depth=config.flow_depth,
                init_sigma=config.init_sigma,
                prior_sigma=config.prior_sigma,
                max_sigma_u=config.max_sigma_u,
                lambda_init=config.lambda_init,
                lambda_max=config.lambda_max,
                min_sigma=config.min_sigma,
            )
            module.full_bayes_lora_state = BayesianLoRAState(
                out_features, in_features, layer_config)


def set_full_bayes_mc_sample(model, enabled: bool):
    for state in _iter_full_bayes_states(model):
        state.mc_sample = bool(enabled)


def full_bayes_lora_kl(model, reduction: str = "mean"):
    states = list(_iter_full_bayes_states(model))
    if not states:
        return None
    totals = {"kl_u": [], "kl_w": [], "kl_total": []}
    counts = []
    for state in states:
        terms = state.kl_terms()
        for key in totals:
            totals[key].append(terms[key])
        counts.append(torch.as_tensor(
            state.U_A_mu.numel() + state.U_B_mu.numel() + state.rank * state.in_features + state.out_features * state.rank,
            device=terms["kl_total"].device,
            dtype=terms["kl_total"].dtype))
    out = {key: torch.stack(values).sum() for key, values in totals.items()}
    if reduction == "mean":
        denom = torch.stack(counts).sum().clamp_min(1.0)
        out = {key: value / denom for key, value in out.items()}
    elif reduction != "sum":
        raise ValueError(f"Unknown full Bayesian-LoRA KL reduction: {reduction}")
    return out


def full_bayes_lora_summary(model):
    summaries = list(state.summary() for state in _iter_full_bayes_states(model))
    if not summaries:
        return None
    keys = ["u_sigma_mean", "u_sigma_max", "w_sigma_mean", "w_sigma_max",
            "lambda", "lora_alpha", "lora_scaling", "max_sigma_u"]
    return {
        "count": len(summaries),
        **{key: float(sum(item[key] for item in summaries) / len(summaries)) for key in keys},
        "flow": summaries[0]["flow"],
    }
