from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from Quadhopper_Stable.rsl_rl_ppo_cfg import QuadhopperPPORunnerCfg


@configclass
class PlannerCircularPPORunnerCfg(QuadhopperPPORunnerCfg):
    max_iterations = 1000
    save_interval = 100
    experiment_name = "quadhopper_planner_circular_v10"

    # Long-horizon landing precision needs controlled exploration. The v7
    # entropy coefficient let action std grow above 1.17 and destabilized
    # touchdown behavior.
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.015,
        max_grad_norm=1.0,
    )
