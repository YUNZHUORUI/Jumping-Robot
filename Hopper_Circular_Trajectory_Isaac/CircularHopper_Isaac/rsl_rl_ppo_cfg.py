from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


QUADHOPPER_NUM_STEPS_PER_ENV = 128
QUADHOPPER_MAX_ITERATIONS = 2000


@configclass
class CircularHopperPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = QUADHOPPER_NUM_STEPS_PER_ENV
    max_iterations = QUADHOPPER_MAX_ITERATIONS
    save_interval = 100
    experiment_name = "circular_hopper_horizon"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.015,
        max_grad_norm=1.0,
    )
