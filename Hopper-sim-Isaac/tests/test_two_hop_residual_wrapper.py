import sys
import types

import torch


if "rsl_rl.env" not in sys.modules:
    rsl_rl = types.ModuleType("rsl_rl")
    rsl_rl_env = types.ModuleType("rsl_rl.env")
    rsl_rl_env.VecEnv = object
    rsl_rl.env = rsl_rl_env
    sys.modules.setdefault("rsl_rl", rsl_rl)
    sys.modules["rsl_rl.env"] = rsl_rl_env

from Quadhopper_Planner_Random.two_hop_residual_wrapper import (  # noqa: E402
    TeacherTwoHopResidualVecEnv,
)
from Quadhopper_Planner_Random.phase_residual_wrapper import (  # noqa: E402
    TeacherTwoHopPhaseResidualVecEnv,
)
from Quadhopper_Planner_Random.two_hop_state_planner_wrapper import (  # noqa: E402
    TeacherTwoHopStatePlannerVecEnv,
)
from Quadhopper_Planner_Random.pair_objective import (  # noqa: E402
    touchdown_components,
    touchdown_score,
)


def test_high_level_mixer_collective_is_equal_on_all_motors():
    action = torch.tensor([[1.0, 0.0, 0.0]])
    mixed = TeacherTwoHopResidualVecEnv.mix_high_level_actions(action, 0.06, 0.10)
    torch.testing.assert_close(mixed, torch.full((1, 4), 0.06))


def test_high_level_mixer_roll_and_pitch_have_zero_collective():
    action = torch.tensor([[0.0, 0.5, -0.25]])
    mixed = TeacherTwoHopResidualVecEnv.mix_high_level_actions(action, 0.06, 0.10)
    torch.testing.assert_close(mixed.sum(dim=1), torch.zeros(1))
    torch.testing.assert_close(
        mixed, torch.tensor([[0.025, -0.075, -0.025, 0.075]])
    )


def test_residual_observation_width_is_56_for_43d_teacher():
    assert 43 + TeacherTwoHopResidualVecEnv.EXTRA_OBSERVATIONS == 56


def test_state_planner_observation_width_is_69_for_43d_teacher():
    assert 43 + TeacherTwoHopStatePlannerVecEnv.EXTRA_OBSERVATIONS == 69


def test_phase_residual_observation_width_is_67_for_43d_teacher():
    assert 43 + TeacherTwoHopPhaseResidualVecEnv.EXTRA_OBSERVATIONS == 67


def test_phase_residual_zero_action_does_not_change_motors():
    actions = torch.zeros(2, 4)
    scales = torch.tensor([[0.06, 0.05, 0.02], [0.03, 0.04, 0.01]])
    mixed = TeacherTwoHopPhaseResidualVecEnv.mix_high_level_actions(actions, scales)
    torch.testing.assert_close(mixed, torch.zeros(2, 4))


def test_phase_residual_yaw_has_zero_collective():
    actions = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    scales = torch.tensor([[0.06, 0.05, 0.02]])
    mixed = TeacherTwoHopPhaseResidualVecEnv.mix_high_level_actions(actions, scales)
    torch.testing.assert_close(mixed.sum(dim=1), torch.zeros(1))
    torch.testing.assert_close(mixed, torch.tensor([[0.02, -0.02, 0.02, -0.02]]))


def test_phase_residual_safety_gate_keeps_stance_authority():
    wrapper = object.__new__(TeacherTwoHopPhaseResidualVecEnv)
    wrapper.collective_limits = torch.tensor([0.06, 0.04, 0.025, 0.03])
    wrapper.attitude_limits = torch.tensor([0.05, 0.04, 0.03, 0.04])
    wrapper.yaw_limits = torch.tensor([0.02, 0.015, 0.012, 0.015])
    wrapper.safety_gate_error_m = 0.08
    phase = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    projected_error = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    scales = wrapper._phase_scales(phase, projected_error)
    torch.testing.assert_close(scales[0], torch.tensor([0.06, 0.05, 0.02]))
    torch.testing.assert_close(scales[1], torch.tensor([0.0075, 0.0100, 0.00375]))


def test_v42_zero_residual_decodes_to_analytic_expert():
    wrapper = object.__new__(TeacherTwoHopStatePlannerVecEnv)
    wrapper.nominal_forward_speed = 0.08
    wrapper.nominal_forward_tilt_rad = 0.1
    wrapper.max_forward_speed_residual = 0.12
    wrapper.max_lateral_speed = 0.10
    wrapper.max_forward_tilt_residual_rad = 0.07
    wrapper.max_lateral_tilt_rad = 0.06
    decoded = wrapper._decode_command(torch.zeros(3, 4))
    torch.testing.assert_close(decoded[:, 0], torch.full((3,), 0.08))
    torch.testing.assert_close(decoded[:, 1], torch.zeros(3))
    torch.testing.assert_close(decoded[:, 2], torch.full((3,), 0.1))
    torch.testing.assert_close(decoded[:, 3], torch.zeros(3))


def test_touchdown_objective_is_decomposed_and_prefers_nominal_state():
    zeros = torch.zeros(2)
    nominal = touchdown_components(zeros, zeros, zeros, zeros, torch.full((2,), 0.002), zeros)
    bad = touchdown_components(torch.full((2,), 0.1), zeros, zeros, zeros, torch.full((2,), 0.002), zeros)
    assert set(nominal) == {"position", "velocity", "attitude", "spring"}
    assert torch.all(touchdown_score(nominal) > touchdown_score(bad))
