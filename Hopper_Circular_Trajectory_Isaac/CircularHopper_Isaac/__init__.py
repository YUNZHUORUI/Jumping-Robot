import gymnasium as gym

from . import agents


gym.register(
    id="circular-hopper",
    entry_point=f"{__name__}.circular_hopper_env:CircularHopperEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.circular_hopper_env:CircularHopperEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CircularHopperPPORunnerCfg",
    },
)
