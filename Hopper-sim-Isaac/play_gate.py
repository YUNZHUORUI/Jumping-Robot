"""Visualize a trained Quadhopper gate-curriculum checkpoint."""

import argparse
import re
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play a Quadhopper gate policy")
parser.add_argument("--checkpoint", type=str, default=None, help="Defaults to latest synced checkpoint")
parser.add_argument("--stage", type=int, choices=range(7), default=None, help="Inferred from stage_N path")
parser.add_argument("--num_envs", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

PROJECT_DIR = Path(__file__).resolve().parent
SYNCED_LOG_ROOT = PROJECT_DIR / "logs" / "rsl_rl" / "quadhopper_gate_synced"


def find_latest_synced_checkpoint() -> Path | None:
    checkpoints = list(SYNCED_LOG_ROOT.glob("stage_*/*/model_*.pt"))
    return max(checkpoints, key=lambda path: path.stat().st_mtime) if checkpoints else None


def infer_stage(checkpoint: Path) -> int | None:
    for part in checkpoint.parts:
        match = re.fullmatch(r"stage_([0-6])", part)
        if match:
            return int(match.group(1))
    return None


checkpoint_path = Path(args_cli.checkpoint).expanduser() if args_cli.checkpoint else find_latest_synced_checkpoint()
if checkpoint_path is None:
    parser.error(
        f"No synced checkpoint found under {SYNCED_LOG_ROOT}. Train first or pass --checkpoint."
    )
checkpoint_path = checkpoint_path.resolve()
if not checkpoint_path.is_file():
    parser.error(f"Checkpoint does not exist: {checkpoint_path}")

selected_stage = args_cli.stage if args_cli.stage is not None else infer_stage(checkpoint_path)
if selected_stage is None:
    parser.error("Cannot infer stage from checkpoint path; pass --stage 0..6.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys

import gymnasium as gym
import torch

import isaaclab as _il

_ISAACLAB_SOURCE = os.path.join(os.path.dirname(_il.__file__), "source")
_ISAACLAB_RL = os.path.join(_ISAACLAB_SOURCE, "isaaclab_rl")
if _ISAACLAB_RL not in sys.path:
    sys.path.insert(0, _ISAACLAB_RL)

import Quadhopper_Gate  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from Quadhopper_Gate.gate_env import QuadhopperGateEnvCfg
from Quadhopper_Gate.ppo_cfg import QuadhopperGatePPORunnerCfg


def main():
    print(f"[PLAY] checkpoint={checkpoint_path}")
    print(f"[PLAY] stage={selected_stage}, num_envs={args_cli.num_envs}")
    env_cfg = QuadhopperGateEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.curriculum_stage = selected_stage
    env_cfg.episode_length_s = 15.0
    env = RslRlVecEnvWrapper(gym.make("Quadhopper-Gate-Direct-v0", cfg=env_cfg))

    runner_cfg = QuadhopperGatePPORunnerCfg()
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(str(checkpoint_path))
    policy = runner.get_inference_policy(device="cuda:0")
    obs = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
