import os
from contextlib import nullcontext

import torch


def create_optimizer(
    params,
    optimizer_name,
    lr,
    weight_decay,
    ivon_ess=1e6,
    ivon_hess_init=1e-3,
    ivon_clip_radius=1e-3,
    ivon_beta2=0.99999,
):
    params = list(params)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name != "ivon":
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    try:
        from ivon import IVON
    except ImportError as exc:
        raise ImportError(
            "IVON optimizer requested, but the optional dependency is missing. "
            "Install it in the active environment with: pip install ivon-opt==0.1.3"
        ) from exc

    return IVON(
        params,
        lr=lr,
        weight_decay=weight_decay,
        ess=ivon_ess,
        hess_init=ivon_hess_init,
        clip_radius=ivon_clip_radius,
        beta2=ivon_beta2,
        rescale_lr=False,
    )


def sampled_params_context(optimizer, train):
    if hasattr(optimizer, "sampled_params"):
        return optimizer.sampled_params(train=train)
    return nullcontext()


def default_ivon_state_path(checkpoint_path):
    return os.path.join(os.path.dirname(checkpoint_path), "ivon_state.pt")


def save_ivon_state(path, optimizer, config):
    if not hasattr(optimizer, "sampled_params"):
        return
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dict(config),
        },
        path,
    )


def load_ivon_state(path, optimizer, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return state.get("config", {})
