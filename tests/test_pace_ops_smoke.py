import torch
import torch.nn as nn

import pace


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 3)

    def forward(self, x):
        return self.net(x.view(x.shape[0], -1))


def test_pace_kl_accepts_custom_temperature():
    model = TinyModel()
    x = torch.randn(2, 1, 2, 2)
    y = torch.tensor([0, 1])

    out = pace.compute_loss_pace_kl(
        model,
        x,
        y,
        nn.CrossEntropyLoss(),
        lbd_pace=0.5,
        temperature=0.5,
    )

    assert out["total_loss"].ndim == 0
    assert out["logits"].shape == (2, 3)


def test_pace_kl_jacobian_returns_sensitivity_loss():
    model = TinyModel()
    x = torch.randn(2, 1, 2, 2)
    y = torch.tensor([0, 1])

    out = pace.compute_loss_pace_kl_jacobian(
        model,
        x,
        y,
        nn.CrossEntropyLoss(),
        lbd_pace=0.5,
        lbd_jacobian=0.1,
        jacobian_eps=1e-2,
        temperature=2.0,
    )

    assert out["total_loss"].ndim == 0
    assert out["jacobian_loss"].ndim == 0
    assert out["jacobian_loss"].item() >= 0
