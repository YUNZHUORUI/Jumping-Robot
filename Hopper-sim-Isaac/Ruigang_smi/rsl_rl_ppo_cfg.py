from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class QuadcopterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 128
    max_iterations = 1000
    save_interval = 100
    experiment_name = "myquadcopter"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        # ==============================================================
        # 【核心修复 7】：将初始探索噪声从 0.5 降到 0.2。
        # 121g 的穿梭机太轻了，0.5 会让推力随机跳变几十克，它永远稳不住。
        # ==============================================================
        init_noise_std=0.2,
        actor_hidden_dims=[128, 128, 128, 64],
        critic_hidden_dims=[128, 128, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        # ==============================================================
        # 【核心修复 8】：进一步降低学习率，保障网络更新时动作策略不会震荡退化。
        # ==============================================================
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )