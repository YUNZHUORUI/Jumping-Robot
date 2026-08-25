"""Planner-conditioned arbitrary-direction two-hop waypoint task."""

import gymnasium as gym


gym.register(
    id="Quadhopper-Planner-Random-Two-Hop-Direct-v0",
    entry_point=f"{__name__}.random_two_hop_env:PlannerRandomTwoHopEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.random_two_hop_env:PlannerRandomTwoHopEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "Quadhopper_Planner_Circular.rsl_rl_ppo_cfg:PlannerCircularPPORunnerCfg"
        ),
    },
)
