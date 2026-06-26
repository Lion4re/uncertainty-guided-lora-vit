import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import pace


class TestIvonUtils(unittest.TestCase):
    def test_create_adamw_optimizer_keeps_existing_default_behavior(self):
        param = torch.nn.Parameter(torch.ones(2))

        opt = pace.create_optimizer(
            [param],
            optimizer_name="adamw",
            lr=1e-3,
            weight_decay=1e-4,
        )

        self.assertIsInstance(opt, torch.optim.AdamW)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 1e-3)
        self.assertAlmostEqual(opt.param_groups[0]["weight_decay"], 1e-4)

    def test_ivon_optimizer_reports_missing_optional_dependency_cleanly(self):
        param = torch.nn.Parameter(torch.ones(2))

        with self.assertRaisesRegex(ImportError, "ivon-opt"):
            pace.create_optimizer(
                [param],
                optimizer_name="ivon",
                lr=0.03,
                weight_decay=0.0,
                ivon_ess=1e6,
                ivon_hess_init=1e-3,
                ivon_clip_radius=1e-3,
                ivon_beta2=0.99999,
            )


if __name__ == "__main__":
    unittest.main()
