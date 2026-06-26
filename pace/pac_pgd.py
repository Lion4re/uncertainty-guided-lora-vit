"""
PAC-tuning-style perturbed gradient descent for trainable PEFT parameters.

This adapts the official PAC-tuning mechanics to the Vision/PACE setup:
learn a parameter-noise posterior in Stage 1, then keep injecting the learned
parameter perturbation during Stage 2 while updating only model weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


@dataclass
class PACPGDConfig:
    stage1_epochs: int = 100
    lbd: float = 1.0
    gamma: float = 0.1
    init_floor: float = 1e-4
    prior_floor: float = 1e-4


class PACPGDState(nn.Module):
    def __init__(self, named_params: Iterable[Tuple[str, nn.Parameter]],
                 config: PACPGDConfig):
        super().__init__()
        self.config = config
        self.param_names: List[str] = []
        self.param_shapes: List[torch.Size] = []
        self.param_numels: List[int] = []
        self._params: List[nn.Parameter] = []

        p_chunks = []
        w0_chunks = []
        b_init = []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            self.param_names.append(name)
            self.param_shapes.append(param.shape)
            self.param_numels.append(param.numel())
            self._params.append(param)

            flat = param.detach().reshape(-1).clone()
            w0_chunks.append(flat)
            init_scale = flat.abs().mean().clamp_min(config.init_floor)
            p_chunks.append(torch.full_like(flat, torch.log(init_scale)))
            b_init.append(torch.log(init_scale.clamp_min(config.prior_floor)))

        if not p_chunks:
            raise ValueError("PAC-PGD requires at least one trainable parameter.")

        self.p = nn.Parameter(torch.cat(p_chunks))
        self.b = nn.Parameter(torch.stack(b_init).to(self.p.device))
        self.register_buffer("w0", torch.cat(w0_chunks))

    @property
    def num_tensors(self) -> int:
        return len(self.param_numels)

    @property
    def num_params(self) -> int:
        return int(sum(self.param_numels))

    def iter_param_slices(self):
        start = 0
        for idx, (param, numel) in enumerate(zip(self._params, self.param_numels)):
            end = start + numel
            yield idx, param, slice(start, end)
            start = end

    @torch.no_grad()
    def inject_noise(self):
        noises = []
        for _, param, slc in self.iter_param_slices():
            noise = torch.randn_like(param).clamp(min=-2.0, max=2.0)
            scale = torch.exp(self.p[slc]).reshape_as(param)
            param.add_(scale * noise)
            noises.append(noise)
        return noises

    @torch.no_grad()
    def remove_noise(self, noises):
        for (_, param, slc), noise in zip(self.iter_param_slices(), noises):
            scale = torch.exp(self.p[slc]).reshape_as(param)
            param.sub_(scale * noise)

    def weight_decay_mulb(self):
        total = self.p.new_zeros(())
        for idx, param, slc in self.iter_param_slices():
            diff = param.reshape(-1) - self.w0[slc].to(param.device)
            total = total + diff.square().sum() * torch.exp(-2.0 * self.b[idx])
        return total

    def kl_term_layer_pb(self, wdecay_mulb):
        kl1 = self.p.new_zeros((), dtype=torch.float64)
        kl2 = self.p.new_zeros((), dtype=torch.float64)
        p64 = self.p.double()
        b64 = self.b.double()
        for idx, _, slc in self.iter_param_slices():
            kl1 = kl1 + torch.exp(-2.0 * b64[idx]) * torch.exp(2.0 * p64[slc]).sum()
            kl2 = kl2 + 2.0 * b64[idx] * self.param_numels[idx]
        kl = kl1 - (2.0 * p64.sum() - kl2 + float(self.num_params))
        return ((kl + wdecay_mulb.double()) / 2.0).float()

    def pac_bound(self, kl, train_size: int):
        gamma = max(float(self.config.gamma), 1e-12)
        complexity = kl + 10.0 + 3.0 * float(self.num_tensors)
        return self.config.lbd * (1.5 * gamma + complexity / (float(train_size) * gamma))

    def kl_term_backward(self, pac_bound, noises):
        grad_loss = []
        for (_, param, _), noise in zip(self.iter_param_slices(), noises):
            if param.grad is None:
                grad_loss.append(torch.zeros_like(param).reshape(-1))
            else:
                grad_loss.append((noise * param.grad).reshape(-1).detach())

        pac_bound.backward()

        if self.p.grad is None:
            self.p.grad = torch.zeros_like(self.p)
        for grad_chunk, (_, _, slc) in zip(grad_loss, self.iter_param_slices()):
            self.p.grad[slc] = self.p.grad[slc] + grad_chunk.to(self.p.device) * torch.exp(self.p.detach()[slc])

    @torch.no_grad()
    def summary(self) -> Dict[str, float]:
        sigma = torch.exp(self.p.detach()).float().cpu()
        b = self.b.detach().float().cpu()
        return {
            "pac_pgd_num_tensors": int(self.num_tensors),
            "pac_pgd_num_params": int(self.num_params),
            "pac_pgd_p_mean": self.p.detach().float().mean().item(),
            "pac_pgd_p_std": self.p.detach().float().std(unbiased=False).item(),
            "pac_pgd_sigma_mean": sigma.mean().item(),
            "pac_pgd_sigma_std": sigma.std(unbiased=False).item(),
            "pac_pgd_sigma_min": sigma.min().item(),
            "pac_pgd_sigma_max": sigma.max().item(),
            "pac_pgd_b_mean": b.mean().item(),
            "pac_pgd_b_std": b.std(unbiased=False).item(),
        }


def build_pac_pgd_state(model: nn.Module, config: PACPGDConfig,
                        name_filters=("delta", "head")) -> PACPGDState:
    named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if any(token in name for token in name_filters)
    ]
    return PACPGDState(named_params, config)
