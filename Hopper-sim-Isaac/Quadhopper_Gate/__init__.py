"""Gym registration for the Quadhopper gate-curriculum task."""

import gymnasium as gym


gym.register(
    id="Quadhopper-Gate-Direct-v0",
    entry_point=f"{__name__}.gate_env:QuadhopperGateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gate_env:QuadhopperGateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:QuadhopperGatePPORunnerCfg",
    },
)
