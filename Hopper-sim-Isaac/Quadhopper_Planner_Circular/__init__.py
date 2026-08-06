"""Planner-conditioned circular hopping task."""

import gymnasium as gym


gym.register(
    id="Quadhopper-Planner-Circular-Direct-v0",
    entry_point=f"{__name__}.planner_circular_env:PlannerCircularEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.planner_circular_env:PlannerCircularEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:PlannerCircularPPORunnerCfg",
    },
)
