"""Native PyTorch PPO for the fully vectorized CUDA QuadHopper environment."""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal

from .config import TRAINING
from .torch_env import TorchQuadHopperVecEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int, log_std_init: float):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def act(self, obs: torch.Tensor):
        dist = self.distribution(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(1), self.critic(obs).squeeze(1)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        dist = self.distribution(obs)
        return (
            dist.log_prob(action).sum(1),
            dist.entropy().sum(1),
            self.critic(obs).squeeze(1),
        )


def _checkpoint_path(model_path: str) -> Path:
    path = Path(model_path)
    return path if path.suffix == ".pt" else path.with_suffix(".pt")


def load_torch_policy(model_path: str, device: torch.device | str = "cpu") -> ActorCritic:
    """Load a native checkpoint for deterministic rollout or evaluation."""
    device = torch.device(device)
    checkpoint = torch.load(
        _checkpoint_path(model_path), map_location=device, weights_only=False
    )
    model = ActorCritic(
        int(checkpoint["obs_dim"]),
        int(checkpoint["action_dim"]),
        int(checkpoint["hidden_size"]),
        float(checkpoint.get("log_std_init", -1.5)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def train_torch_ppo(
    cfg=TRAINING,
    *,
    target_count: int | None = None,
    model_path: str | None = None,
    resume_model: str | None = None,
    timesteps: int | None = None,
    device: str = "cuda",
    seed: int = 0,
):
    """Train with GPU-resident physics, rollout storage, GAE and PPO updates."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("The native Torch backend is intended for CUDA; use SB3 for CPU training")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    target_count = cfg.target_count if target_count is None else int(target_count)
    total_timesteps = cfg.total_timesteps if timesteps is None else int(timesteps)
    model_path = cfg.model_path if model_path is None else model_path
    env = TorchQuadHopperVecEnv(
        cfg.cuda_n_envs,
        device,
        target_count=target_count,
        seed=seed,
        random_start_probability=0.0,
    )
    model = ActorCritic(
        env.observation_dim,
        env.action_dim,
        cfg.cuda_hidden_size,
        cfg.cuda_log_std_init,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)
    completed_steps = 0
    resuming = resume_model is not None

    if resuming:
        resume_path = _checkpoint_path(resume_model)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        previous_target_count = int(checkpoint.get("target_count", target_count))
        task_changed = previous_target_count != target_count
        if "optimizer" in checkpoint and not task_changed:
            optimizer.load_state_dict(checkpoint["optimizer"])
        elif task_changed:
            with torch.no_grad():
                model.log_std.fill_(cfg.cuda_log_std_init)
            print(
                f"Curriculum transition {previous_target_count} -> {target_count}: "
                "reset optimizer and exploration std"
            )
        completed_steps = int(checkpoint.get("timesteps", 0))
        print(f"Resumed native Torch checkpoint: {resume_path}")

    target_timesteps = completed_steps + total_timesteps if resuming else total_timesteps

    print(
        f"Native CUDA PPO: {cfg.cuda_n_envs} environments, "
        f"rollout={cfg.cuda_rollout_steps}, batch={cfg.cuda_batch_size}"
    )
    print(
        f"Target count: {target_count}; training timesteps: {total_timesteps:,}; "
        f"stop at: {target_timesteps:,}"
    )

    obs = env.observe()
    recent_returns: deque[float] = deque(maxlen=100)
    recent_lengths: deque[float] = deque(maxlen=100)
    total_hits = 0
    started = time.perf_counter()
    run_start_steps = completed_steps
    iteration = 0

    while completed_steps < target_timesteps:
        iteration += 1
        remaining = target_timesteps - completed_steps
        rollout_steps = min(
            cfg.cuda_rollout_steps,
            max(1, math.ceil(remaining / cfg.cuda_n_envs)),
        )
        obs_buf = torch.empty(
            (rollout_steps, cfg.cuda_n_envs, env.observation_dim), device=device
        )
        action_buf = torch.empty(
            (rollout_steps, cfg.cuda_n_envs, env.action_dim), device=device
        )
        logprob_buf = torch.empty((rollout_steps, cfg.cuda_n_envs), device=device)
        reward_buf = torch.empty_like(logprob_buf)
        done_buf = torch.empty_like(logprob_buf)
        value_buf = torch.empty_like(logprob_buf)
        episode_return_chunks = []
        episode_length_chunks = []
        rollout_hits = torch.zeros((), dtype=torch.long, device=device)

        model.eval()
        with torch.no_grad():
            for step in range(rollout_steps):
                action, logprob, value = model.act(obs)
                obs_buf[step] = obs
                action_buf[step] = action
                logprob_buf[step] = logprob
                value_buf[step] = value
                obs, raw_reward, done, info = env.step(action)
                reward_buf[step] = raw_reward * cfg.cuda_reward_scale
                done_buf[step] = done.to(torch.float32)
                rollout_hits += info["target_hits"]
                episode_return_chunks.append(info["episode_return"])
                episode_length_chunks.append(info["episode_length"])

            next_value = model.critic(obs).squeeze(1)

        total_hits += int(rollout_hits.item())
        rollout_returns = torch.cat(episode_return_chunks)
        rollout_lengths = torch.cat(episode_length_chunks)
        finished = torch.isfinite(rollout_returns)
        if finished.any():
            recent_returns.extend(rollout_returns[finished].cpu().tolist())
            recent_lengths.extend(rollout_lengths[finished].cpu().tolist())

        advantage_buf = torch.empty_like(reward_buf)
        last_gae = torch.zeros(cfg.cuda_n_envs, device=device)
        for step in reversed(range(rollout_steps)):
            next_values = next_value if step == rollout_steps - 1 else value_buf[step + 1]
            nonterminal = 1.0 - done_buf[step]
            delta = reward_buf[step] + cfg.gamma * next_values * nonterminal - value_buf[step]
            last_gae = delta + cfg.gamma * cfg.cuda_gae_lambda * nonterminal * last_gae
            advantage_buf[step] = last_gae
        return_buf = advantage_buf + value_buf

        flat_obs = obs_buf.flatten(0, 1)
        flat_actions = action_buf.flatten(0, 1)
        flat_logprobs = logprob_buf.flatten()
        flat_advantages = advantage_buf.flatten()
        flat_returns = return_buf.flatten()
        flat_values = value_buf.flatten()
        flat_advantages = (
            flat_advantages - flat_advantages.mean()
        ) / (flat_advantages.std() + 1e-8)

        model.train()
        sample_count = flat_obs.shape[0]
        losses = []
        policy_losses = []
        value_losses = []
        entropies = []
        kls = []
        clip_fractions = []
        for _ in range(cfg.cuda_update_epochs):
            permutation = torch.randperm(sample_count, device=device)
            for start in range(0, sample_count, cfg.cuda_batch_size):
                index = permutation[start:start + cfg.cuda_batch_size]
                new_logprob, entropy, new_value = model.evaluate(
                    flat_obs[index], flat_actions[index]
                )
                log_ratio = new_logprob - flat_logprobs[index]
                ratio = log_ratio.exp()
                advantage = flat_advantages[index]
                policy_unclipped = -advantage * ratio
                policy_clipped = -advantage * torch.clamp(
                    ratio, 1.0 - cfg.cuda_clip_range, 1.0 + cfg.cuda_clip_range
                )
                policy_loss = torch.maximum(policy_unclipped, policy_clipped).mean()
                value_loss = 0.5 * (new_value - flat_returns[index]).square().mean()
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + cfg.cuda_vf_coef * value_loss
                    - cfg.cuda_ent_coef * entropy_mean
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.cuda_max_grad_norm)
                optimizer.step()
                # Prevent the unbounded exploration observed in the SB3 run.
                with torch.no_grad():
                    model.log_std.clamp_(-5.0, 1.0)

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        torch.abs(ratio - 1.0) > cfg.cuda_clip_range
                    ).float().mean()
                losses.append(loss.detach())
                policy_losses.append(policy_loss.detach())
                value_losses.append(value_loss.detach())
                entropies.append(entropy_mean.detach())
                kls.append(approx_kl.detach())
                clip_fractions.append(clip_fraction.detach())

        completed_steps += sample_count
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        fps = (completed_steps - run_start_steps) / max(elapsed, 1e-6)
        explained_var = 1.0 - torch.var(flat_returns - flat_values) / torch.clamp(
            torch.var(flat_returns), min=1e-8
        )
        mean_return = sum(recent_returns) / len(recent_returns) if recent_returns else float("nan")
        mean_length = sum(recent_lengths) / len(recent_lengths) if recent_lengths else float("nan")
        print(
            f"iter={iteration:03d} steps={completed_steps:,}/{target_timesteps:,} "
            f"fps={fps:,.0f} ep_rew={mean_return:.1f} ep_len={mean_length:.1f} "
            f"loss={torch.stack(losses).mean().item():.4f} "
            f"pg={torch.stack(policy_losses).mean().item():+.5f} "
            f"value={torch.stack(value_losses).mean().item():.4f} "
            f"entropy={torch.stack(entropies).mean().item():.3f} "
            f"std={model.log_std.exp().mean().item():.3f} "
            f"kl={torch.stack(kls).mean().item():.6f} "
            f"clip={torch.stack(clip_fractions).mean().item():.4f} "
            f"ev={explained_var.item():.3f} hits={total_hits}"
        )

    save_path = _checkpoint_path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "timesteps": completed_steps,
            "obs_dim": env.observation_dim,
            "action_dim": env.action_dim,
            "hidden_size": cfg.cuda_hidden_size,
            "log_std_init": cfg.cuda_log_std_init,
            "target_count": target_count,
        },
        save_path,
    )
    print(f"Native Torch model saved: {save_path}")
    return save_path
