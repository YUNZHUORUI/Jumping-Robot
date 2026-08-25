import math
import sys
import unittest
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from Quadhopper_Planner_Random.waypoint_command import TwoHopRandomCommand


class RandomTwoHopCommandTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.num_envs = 4096
        self.origins = torch.zeros(self.num_envs, 3)
        self.command = TwoHopRandomCommand(
            self.num_envs, "cpu", 0.5, 0.8, 0.8, 1.0, 58
        )
        self.env_ids = torch.arange(self.num_envs)
        self.command.reset(self.env_ids, self.origins, random_phase=False)

    def assert_distance_range(self, start, end, lower, upper):
        distance = torch.linalg.norm(end - start, dim=1)
        self.assertTrue(torch.all(distance >= lower - 1.0e-6))
        self.assertTrue(torch.all(distance <= upper + 1.0e-6))

    def test_reset_samples_short_then_long(self):
        p_t, p_t1 = self.command.lookahead()
        self.assertTrue(torch.all(self.command.current_hop_is_short(self.env_ids)))
        self.assert_distance_range(self.origins[:, :2], p_t, 0.5, 0.8)
        self.assert_distance_range(p_t, p_t1, 0.8, 1.0)

    def test_advance_keeps_two_point_lookahead_and_alternates_range(self):
        # Vectorized height training is allowed to randomize its phase without
        # changing the short/long geometric period.
        self.command.cycle_index[:] = torch.randint(0, 2, (self.num_envs,))
        _, original_second = self.command.lookahead()
        original_second = original_second.clone()
        self.command.advance(self.env_ids)
        p_t, p_t1 = self.command.lookahead()
        self.assertTrue(torch.all(~self.command.current_hop_is_short(self.env_ids)))
        torch.testing.assert_close(p_t, original_second)
        self.assert_distance_range(p_t, p_t1, 0.5, 0.8)

        self.command.advance(self.env_ids)
        p_t, p_t1 = self.command.lookahead()
        self.assertTrue(torch.all(self.command.current_hop_is_short(self.env_ids)))
        self.assert_distance_range(p_t, p_t1, 0.8, 1.0)

    def test_headings_cover_all_quadrants(self):
        p_t, _ = self.command.lookahead()
        heading = torch.atan2(p_t[:, 1], p_t[:, 0])
        quadrant = torch.floor((heading + math.pi) / (0.5 * math.pi)).long()
        counts = torch.bincount(quadrant, minlength=4)
        self.assertTrue(torch.all(counts > 800), counts)

    def test_vectorized_reset_balances_short_and_long_start_phases(self):
        self.command.reset(self.env_ids, self.origins, random_phase=True)
        p_t, p_t1 = self.command.lookahead()
        starts_short = self.command.current_hop_is_short(self.env_ids)
        short_count = torch.sum(starts_short).item()
        self.assertGreater(short_count, 1800)
        self.assertLess(short_count, 2300)
        self.assert_distance_range(
            self.origins[starts_short, :2], p_t[starts_short], 0.5, 0.8
        )
        self.assert_distance_range(p_t[starts_short], p_t1[starts_short], 0.8, 1.0)
        self.assert_distance_range(
            self.origins[~starts_short, :2], p_t[~starts_short], 0.8, 1.0
        )
        self.assert_distance_range(p_t[~starts_short], p_t1[~starts_short], 0.5, 0.8)

    def test_invalid_radius_range_is_rejected(self):
        with self.assertRaises(ValueError):
            TwoHopRandomCommand(1, "cpu", 0.8, 0.5, 0.8, 1.0, 58)

    def test_restart_pair_uses_measured_position_and_short_long_order(self):
        self.command.advance(self.env_ids)
        measured = torch.randn(self.num_envs, 2)
        self.command.restart_pair(self.env_ids, measured)
        p_t, p_t1 = self.command.lookahead()
        self.assertTrue(torch.all(self.command.current_hop_is_short(self.env_ids)))
        torch.testing.assert_close(self.command.anchor_w, measured)
        self.assert_distance_range(measured, p_t, 0.5, 0.8)
        self.assert_distance_range(p_t, p_t1, 0.8, 1.0)

    def test_turn_curriculum_bounds_consecutive_heading_change(self):
        max_turn = math.pi / 4.0
        command = TwoHopRandomCommand(
            self.num_envs, "cpu", 0.5, 0.8, 0.8, 1.0, 58, max_turn
        )
        command.reset(self.env_ids, self.origins, random_phase=False)

        def assert_bounded(first, second):
            cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
            dot = torch.sum(first * second, dim=1)
            turn = torch.abs(torch.atan2(cross, dot))
            self.assertTrue(torch.all(turn <= max_turn + 1.0e-6))

        p_t, p_t1 = command.lookahead()
        assert_bounded(p_t - self.origins[:, :2], p_t1 - p_t)
        command.advance(self.env_ids)
        p_t, p_t1 = command.lookahead()
        assert_bounded(p_t - command.anchor_w, p_t1 - p_t)


if __name__ == "__main__":
    unittest.main()
