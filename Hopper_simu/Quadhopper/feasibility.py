"""Feasibility sweeps for QuadHopper target jumping.

This module is intentionally separate from PPO training.  It answers two
questions before spending time on RL:

1. Can the ballistic model land near a target from a planned launch state?
2. Can a simple open-loop thrust pulse produce a ground-start jump to target?
"""

import argparse
import copy
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np

try:
    from .config import ENV
    from .env import QuadhopperTargetEnv
except ImportError:
    from Quadhopper.config import ENV
    from Quadhopper.env import QuadhopperTargetEnv


ActionPolicy = Callable[[int, QuadhopperTargetEnv], np.ndarray]


@dataclass
class EpisodeResult:
    target_x: float
    hit: bool
    landed_x: float
    landing_error: float
    max_height: float
    liftoff_step: int
    touchdown_step: int
    steps: int
    terminated: bool
    truncated: bool
    policy: str


def _make_env(target_x: float) -> QuadhopperTargetEnv:
    env_cfg = copy.copy(ENV)
    env_cfg.targets = np.array([target_x], dtype=np.float64)
    env_cfg.print_hit_events = False
    return QuadhopperTargetEnv(env_cfg=env_cfg)


def _is_touching(env: QuadhopperTargetEnv) -> bool:
    foot_y = float(env.physics.get_foot_pos(env.q)[1])
    return foot_y <= env.pcfg.ground_y + 0.02


def _constant_action(left: float, right: float) -> ActionPolicy:
    action = np.array([left, right], dtype=np.float32)

    def policy(_step: int, _env: QuadhopperTargetEnv) -> np.ndarray:
        return action

    return policy


def _pulse_action(base: float, diff: float, burn_steps: int, coast: float) -> ActionPolicy:
    thrust = np.array(
        [np.clip(base - diff, 0.0, 1.0), np.clip(base + diff, 0.0, 1.0)],
        dtype=np.float32,
    )
    coast_action = np.array([coast, coast], dtype=np.float32)

    def policy(step: int, _env: QuadhopperTargetEnv) -> np.ndarray:
        return thrust if step < burn_steps else coast_action

    return policy


def run_episode(
    *,
    target_x: float,
    reset_mode: str,
    policy: ActionPolicy,
    policy_name: str,
    seed: int,
    max_steps: int,
) -> EpisodeResult:
    env = _make_env(target_x)
    env.reset(seed=seed, options={"mode": reset_mode})

    was_touching = _is_touching(env)
    liftoff_step = -1
    touchdown_step = -1
    landed_x = float(env.physics.get_foot_pos(env.q)[0])
    max_height = float(env.physics.get_com_pos(env.q)[1])
    terminated = False
    truncated = False

    for step in range(max_steps):
        action = policy(step, env)
        _, _, terminated, truncated, _ = env.step(action)

        touching = _is_touching(env)
        foot_x = float(env.physics.get_foot_pos(env.q)[0])
        com_y = float(env.physics.get_com_pos(env.q)[1])
        max_height = max(max_height, com_y)

        if was_touching and not touching and liftoff_step < 0:
            liftoff_step = step
        if (not was_touching) and touching and liftoff_step >= 0:
            touchdown_step = step
            landed_x = foot_x
            break

        landed_x = foot_x
        was_touching = touching

        if terminated or truncated:
            break

    landing_error = abs(landed_x - target_x)
    hit = (env.current_target_idx > 0) or (landing_error <= env.ecfg.target_tolerance)
    return EpisodeResult(
        target_x=target_x,
        hit=hit,
        landed_x=landed_x,
        landing_error=landing_error,
        max_height=max_height,
        liftoff_step=liftoff_step,
        touchdown_step=touchdown_step,
        steps=step + 1,
        terminated=terminated,
        truncated=truncated,
        policy=policy_name,
    )


def ballistic_sweep(targets: Iterable[float], seeds: Iterable[int], max_steps: int) -> List[EpisodeResult]:
    results = []
    for target_x in targets:
        for seed in seeds:
            results.append(
                run_episode(
                    target_x=target_x,
                    reset_mode="ballistic",
                    policy=_constant_action(0.0, 0.0),
                    policy_name="zero_thrust",
                    seed=seed,
                    max_steps=max_steps,
                )
            )
    return results


def ground_pulse_sweep(
    targets: Iterable[float],
    seeds: Iterable[int],
    bases: Iterable[float],
    diffs: Iterable[float],
    burn_steps: Iterable[int],
    coast: float,
    max_steps: int,
) -> List[EpisodeResult]:
    results = []
    for target_x in targets:
        for base in bases:
            for diff in diffs:
                for burn in burn_steps:
                    policy_name = f"pulse(base={base:.2f},diff={diff:+.2f},burn={burn},coast={coast:.2f})"
                    policy = _pulse_action(base, diff, burn, coast)
                    for seed in seeds:
                        results.append(
                            run_episode(
                                target_x=target_x,
                                reset_mode="ground",
                                policy=policy,
                                policy_name=policy_name,
                                seed=seed,
                                max_steps=max_steps,
                            )
                        )
    return results


def summarize(results: List[EpisodeResult]) -> Dict[float, EpisodeResult]:
    best_by_target = {}
    for result in results:
        old = best_by_target.get(result.target_x)
        if old is None or result.landing_error < old.landing_error:
            best_by_target[result.target_x] = result
    return best_by_target


def print_results(title: str, results: List[EpisodeResult]):
    if not results:
        print(f"{title}: no results")
        return

    hits = sum(1 for r in results if r.hit)
    hit_rate = hits / len(results)
    best_by_target = summarize(results)

    print(f"\n{title}")
    print(f"episodes={len(results)}  hit_rate={hit_rate:.1%}")
    print("target  best_err  landed_x  hit  liftoff  touchdown  max_y  policy")
    for target_x in sorted(best_by_target):
        r = best_by_target[target_x]
        print(
            f"{r.target_x:6.2f}  {r.landing_error:8.3f}  {r.landed_x:8.3f}  "
            f"{str(r.hit):>3}  {r.liftoff_step:7d}  {r.touchdown_step:9d}  "
            f"{r.max_height:5.2f}  {r.policy}"
        )


def _parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="QuadHopper target-jump feasibility sweeps")
    parser.add_argument("--mode", choices=["ballistic", "ground", "both"], default="both")
    parser.add_argument("--targets", default="0.5,1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--bases", default="0.55,0.70,0.85,1.00")
    parser.add_argument("--diffs", default="-0.25,-0.10,0.00,0.10,0.25")
    parser.add_argument("--burn-steps", default="20,35,50,70,90")
    parser.add_argument("--coast", type=float, default=0.0)
    args = parser.parse_args()

    targets = _parse_float_list(args.targets)
    seeds = _parse_int_list(args.seeds)

    if args.mode in ("ballistic", "both"):
        print_results(
            "Ballistic planned-launch sweep",
            ballistic_sweep(targets, seeds, args.max_steps),
        )

    if args.mode in ("ground", "both"):
        print_results(
            "Ground-start open-loop pulse sweep",
            ground_pulse_sweep(
                targets=targets,
                seeds=seeds,
                bases=_parse_float_list(args.bases),
                diffs=_parse_float_list(args.diffs),
                burn_steps=_parse_int_list(args.burn_steps),
                coast=args.coast,
                max_steps=args.max_steps,
            ),
        )


if __name__ == "__main__":
    main()
