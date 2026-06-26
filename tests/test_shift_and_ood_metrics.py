import torch

import evaluate_all_metrics as eval_metrics
import utils


def test_shift_transform_keeps_shape_and_is_clean_at_severity_zero():
    clean = utils.build_vtab_eval_transform(eval_shift="clean", shift_severity=0)
    shifted = utils.build_vtab_eval_transform(eval_shift="gaussian_noise", shift_severity=0)
    image = torch.rand(3, 224, 224)

    clean_out = clean.transforms[-2](image) if hasattr(clean, "transforms") else image
    assert clean_out.shape == image.shape

    # Test tensor-level helper directly so the check is independent of PIL loading.
    shifted_out = utils.apply_controlled_shift_tensor(image, "gaussian_noise", 0)
    assert torch.allclose(shifted_out, image)
    assert shifted_out.shape == image.shape


def test_shift_transform_outputs_finite_tensor_for_all_shifts():
    image = torch.full((3, 224, 224), 0.5)

    for shift in ["clean", "gaussian_noise", "gaussian_blur", "brightness", "contrast", "cutout"]:
        shifted = utils.apply_controlled_shift_tensor(image, shift, 2)
        assert shifted.shape == image.shape
        assert torch.isfinite(shifted).all()


def test_ood_auroc_is_high_when_ood_scores_are_larger():
    id_scores = torch.tensor([0.05, 0.10, 0.20, 0.25])
    ood_scores = torch.tensor([0.70, 0.80, 0.90, 0.95])

    result = eval_metrics.compute_ood_detection_metrics(id_scores, ood_scores)

    assert result["auroc"] > 0.99
    assert result["fpr95"] == 0.0
    assert result["id_score_mean"] < result["ood_score_mean"]
