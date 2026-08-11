import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from Quadhopper_Planner_Circular.checkpoint_migration import migrate_stable_checkpoint
from Quadhopper_Planner_Circular.direct_collocation_planner import DirectCollocationHopPlanner
from Quadhopper_Planner_Circular.height_schedule import cosine_height


class VariableHeightPlannerTest(unittest.TestCase):
    def test_fixed_height_curriculum_is_clamped_and_monotonic(self):
        commands = [cosine_height(i, 1.30, 0.70, 300.0) for i in (0, 100, 200, 300, 400)]
        self.assertAlmostEqual(commands[0], 1.30)
        self.assertAlmostEqual(commands[3], 0.70)
        self.assertAlmostEqual(commands[4], 0.70)
        self.assertTrue(all(a >= b for a, b in zip(commands, commands[1:])))

    def test_absolute_apex_and_dynamic_duration(self):
        planner = DirectCollocationHopPlanner(2, "cpu", 25)
        ids = torch.arange(2)
        start = torch.tensor([[0.0, 0.0, 0.38], [0.0, 0.0, 0.38]])
        velocity = torch.zeros(2, 3)
        p_t = torch.tensor([[0.22, 0.0], [0.22, 0.0]])
        p_t1 = torch.tensor([[0.44, 0.0], [0.44, 0.0]])
        heights = torch.tensor([0.70, 1.00])
        next_heights = torch.tensor([1.00, 0.70])
        landing = torch.full((2,), 0.38)
        planner.replan(ids, start, velocity, p_t, p_t1, heights, landing, next_heights)

        torch.testing.assert_close(planner.positions_w[:, 0], start, atol=2e-4, rtol=0)
        torch.testing.assert_close(planner.positions_w[:, planner.mid, 2], heights, atol=2e-4, rtol=0)
        torch.testing.assert_close(planner.positions_w[:, -1, :2], p_t, atol=2e-4, rtol=0)
        torch.testing.assert_close(
            planner.next_positions_w[:, planner.mid, 2], next_heights, atol=2e-4, rtol=0
        )
        self.assertLess(planner.flight_duration[0].item(), planner.flight_duration[1].item())
        self.assertAlmostEqual(planner.flight_duration[0].item(), 0.511, delta=0.015)
        self.assertAlmostEqual(planner.flight_duration[1].item(), 0.711, delta=0.015)

    def test_42d_migration_folds_fixed_height_into_bias(self):
        source_weight = torch.randn(8, 42)
        source_bias = torch.randn(8)
        checkpoint = {
            "model_state_dict": {
                "memory_a.rnn.weight_ih_l0": source_weight.clone(),
                "memory_a.rnn.bias_ih_l0": source_bias.clone(),
                "memory_c.rnn.weight_ih_l0": source_weight.clone(),
                "memory_c.rnn.bias_ih_l0": source_bias.clone(),
            },
            "iter": 10,
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            destination = Path(directory) / "destination.pt"
            torch.save(checkpoint, source)
            migrate_stable_checkpoint(source, destination)
            migrated = torch.load(destination, map_location="cpu", weights_only=False)
        for key in ("memory_a.rnn.weight_ih_l0", "memory_c.rnn.weight_ih_l0"):
            weight = migrated["model_state_dict"][key]
            self.assertEqual(weight.shape[1], 43)
            torch.testing.assert_close(weight[:, :41], source_weight[:, :41])
            torch.testing.assert_close(weight[:, 41:], torch.zeros_like(weight[:, 41:]))
            bias_key = key.replace("weight_ih_l0", "bias_ih_l0")
            migrated_bias = migrated["model_state_dict"][bias_key]
            expected_bias = source_bias + source_weight[:, 41] * 0.65
            torch.testing.assert_close(migrated_bias, expected_bias)


if __name__ == "__main__":
    unittest.main()
