import math

import torch
import torch.nn as nn

import pace


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 3)

    def forward(self, x):
        return self.net(x.view(x.shape[0], -1))


def test_uncertainty_scores_rank_uniform_as_more_uncertain():
    confident = torch.tensor([[8.0, 0.0, -1.0]])
    uniform = torch.tensor([[0.0, 0.0, 0.0]])
    logits = torch.cat([confident, uniform])

    entropy = pace.compute_predictive_uncertainty(logits, score="entropy")
    max_conf = pace.compute_predictive_uncertainty(logits, score="max_conf")
    margin = pace.compute_predictive_uncertainty(logits, score="margin")

    assert entropy[1] > entropy[0]
    assert max_conf[1] > max_conf[0]
    assert margin[1] > margin[0]


def test_topk_uncertainty_weights_select_ceil_fraction():
    scores = torch.tensor([0.1, 0.7, 0.3, 0.9, 0.2])

    weights, stats = pace.build_uncertainty_selection_weights(
        scores,
        mode="topk",
        fraction=0.30,
        weight=1.0,
    )

    assert weights.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]
    assert stats["effective_batch_size"].item() == 2
    assert math.isclose(stats["selected_fraction"].item(), 0.4, abs_tol=1e-6)


def test_soft_uncertainty_weights_are_positive_and_mean_normalized():
    scores = torch.tensor([0.0, 0.5, 1.0])

    weights, stats = pace.build_uncertainty_selection_weights(
        scores,
        mode="soft",
        fraction=0.30,
        weight=1.0,
    )

    assert torch.all(weights > 0)
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
    assert stats["effective_batch_size"].item() == 3
    assert stats["selected_fraction"].item() == 1.0


def test_uncertainty_guided_pace_kl_backpropagates_without_weight_gradients():
    model = TinyModel()
    x = torch.randn(4, 1, 2, 2)
    y = torch.tensor([0, 1, 2, 1])

    out = pace.compute_loss_pace_kl_uncertainty(
        model,
        x,
        y,
        nn.CrossEntropyLoss(),
        lbd_pace=0.5,
        temperature=2.0,
        uncertainty_score="entropy",
        uncertainty_mode="topk",
        uncertainty_fraction=0.30,
        uncertainty_weight=1.0,
    )
    out["total_loss"].backward()

    assert out["total_loss"].ndim == 0
    assert out["effective_batch_size"].item() == 2
    assert out["uncertainty_weights"].requires_grad is False
    assert model.net.weight.grad is not None


def test_uncertainty_guided_pace_mse_backpropagates_without_weight_gradients():
    model = TinyModel()
    x = torch.randn(4, 1, 2, 2)
    y = torch.tensor([0, 1, 2, 1])

    out = pace.compute_loss_pace_uncertainty(
        model,
        x,
        y,
        nn.CrossEntropyLoss(),
        lbd_pace=1.0,
        uncertainty_score="entropy",
        uncertainty_mode="soft",
        uncertainty_fraction=0.30,
        uncertainty_weight=1.0,
    )
    out["total_loss"].backward()

    assert out["total_loss"].ndim == 0
    assert out["effective_batch_size"].item() == 4
    assert out["uncertainty_weights"].requires_grad is False
    assert model.net.weight.grad is not None
