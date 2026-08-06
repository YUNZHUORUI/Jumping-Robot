"""Exact stable-jump baseline registration from the user-provided source files."""

import gymnasium as gym


gym.register(
    id="Quadhopper-Stable-Direct-v0",
    entry_point=f"{__name__}.quadhopper_env:QuadhopperEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadhopper_env:QuadhopperEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:QuadhopperPPORunnerCfg",
    },
)
