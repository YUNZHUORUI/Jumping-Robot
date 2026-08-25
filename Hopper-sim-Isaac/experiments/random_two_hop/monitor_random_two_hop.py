"""Summarize the latest random-route TensorBoard run and its stage gate."""

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


parser = argparse.ArgumentParser()
parser.add_argument("log_dir", type=Path)
parser.add_argument(
    "--stage", choices=("direction", "medium", "short", "full"), required=True
)
parser.add_argument("--window", type=int, default=50)
args = parser.parse_args()


GATES = {
    "direction": {"error": 0.10, "hit_rate": 0.70, "streak": 10.0},
    "medium": {"error": 0.12, "hit_rate": 0.60, "streak": 8.0},
    "short": {"error": 0.15, "hit_rate": 0.45, "streak": 5.0},
    "full": {"error": 0.12, "hit_rate": 0.60, "streak": 20.0},
}


def newest_event_file(root: Path) -> Path:
    candidates = list(root.glob("**/events.out.tfevents.*"))
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event file under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


event_file = newest_event_file(args.log_dir.expanduser())
events = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
events.Reload()
available = set(events.Tags()["scalars"])


def rolling_mean(tag: str) -> float:
    if tag not in available:
        return float("nan")
    values = events.Scalars(tag)[-args.window :]
    return sum(item.value for item in values) / max(len(values), 1)


def latest(tag: str) -> float:
    if tag not in available:
        return float("nan")
    return events.Scalars(tag)[-1].value


use_touchdown_ema = "Metrics/target_hit_rate_ema" in available
error = rolling_mean(
    "Metrics/touchdown_error_ema_m"
    if use_touchdown_ema
    else "Metrics/episode_touchdown_error_m"
)
hit_rate = rolling_mean(
    "Metrics/target_hit_rate_ema" if use_touchdown_ema else "Metrics/target_hit_rate"
)
streak = rolling_mean("Metrics/max_consecutive_hits")
live_mean_streak = rolling_mean("Metrics/live_mean_consecutive_hits")
live_p90_streak = rolling_mean("Metrics/live_p90_consecutive_hits")
live_max_streak = rolling_mean("Metrics/live_max_consecutive_hits")
prepared_rate = rolling_mean(
    "Metrics/prepared_landing_rate_ema"
    if "Metrics/prepared_landing_rate_ema" in available
    else "Metrics/prepared_landing_rate"
)
touchdown_attitude_error = rolling_mean("Metrics/touchdown_attitude_error_rad")
touchdown_next_velocity_error = rolling_mean(
    "Metrics/touchdown_next_velocity_error_mps"
)
touchdown_next_velocity_projection = rolling_mean(
    "Metrics/touchdown_next_velocity_projection_mps"
)
touchdown_next_velocity_lateral = rolling_mean(
    "Metrics/touchdown_next_velocity_lateral_abs_mps"
)
route_completion = rolling_mean("Metrics/route_completion")
noise = latest("Policy/mean_noise_std")
current_tolerance = latest("Metrics/current_target_tolerance_m")
short_error = rolling_mean(
    "Metrics/short_touchdown_error_ema_m"
    if use_touchdown_ema
    else "Metrics/short_touchdown_error_m"
)
short_hit_rate = rolling_mean(
    "Metrics/short_target_hit_rate_ema"
    if use_touchdown_ema
    else "Metrics/short_target_hit_rate"
)
long_error = rolling_mean(
    "Metrics/long_touchdown_error_ema_m"
    if use_touchdown_ema
    else "Metrics/long_touchdown_error_m"
)
long_hit_rate = rolling_mean(
    "Metrics/long_target_hit_rate_ema"
    if use_touchdown_ema
    else "Metrics/long_target_hit_rate"
)
high_apex_error = rolling_mean("Metrics/high_command_apex_error_m")
low_apex_error = rolling_mean("Metrics/low_command_apex_error_m")
gate = GATES[args.stage]
alternating_height_metrics = (
    "Metrics/high_command_apex_error_m" in available
    and "Metrics/low_command_apex_error_m" in available
)
apex_ok = (
    high_apex_error <= 0.08 and low_apex_error <= 0.08
    if alternating_height_metrics
    else True
)
passed = (
    error <= gate["error"]
    and hit_rate >= gate["hit_rate"]
    and streak >= gate["streak"]
    and apex_ok
)
if args.stage == "full":
    passed = passed and route_completion > 0.0

# The */time series use elapsed seconds as their x-axis. Read iteration from a
# normal per-update scalar so elapsed time cannot be mistaken for iteration.
last_step = (
    events.Scalars("Policy/mean_noise_std")[-1].step
    if "Policy/mean_noise_std" in available
    else -1
)
print(f"run: {event_file.parent}")
print(f"iteration: {last_step}")
print(f"window: last {args.window} logged iterations")
print(f"touchdown metric source: {'all-touchdown EMA' if use_touchdown_ema else 'episode resets'}")
print(f"episode touchdown error: {error:.4f} m (gate <= {gate['error']:.2f})")
print(f"target hit rate: {hit_rate:.3f} (gate >= {gate['hit_rate']:.2f})")
print(f"short hop: error={short_error:.4f} m, hit_rate={short_hit_rate:.3f}")
print(f"long hop: error={long_error:.4f} m, hit_rate={long_hit_rate:.3f}")
print(f"max consecutive hits: {streak:.2f} (gate >= {gate['streak']:.0f})")
if "Metrics/live_mean_consecutive_hits" in available:
    print(
        "live consecutive hits mean/p90/max: "
        f"{live_mean_streak:.2f} / {live_p90_streak:.2f} / {live_max_streak:.2f}"
    )
if "Metrics/prepared_landing_rate" in available:
    print(f"prepared landing rate: {prepared_rate:.3f}")
    print(
        "touchdown attitude/next-velocity error: "
        f"{touchdown_attitude_error:.3f} rad / "
        f"{touchdown_next_velocity_error:.3f} m/s"
    )
    print(
        "touchdown next-direction/lateral velocity: "
        f"{touchdown_next_velocity_projection:.3f} / "
        f"{touchdown_next_velocity_lateral:.3f} m/s"
    )
print(f"route completion: {route_completion:.4f}")
if alternating_height_metrics:
    print(f"high/low apex error: {high_apex_error:.4f} / {low_apex_error:.4f} m")
else:
    print("apex gate: fixed-height run; verify with deterministic evaluation")
print(f"action noise std: {noise:.4f}")
if current_tolerance == current_tolerance:
    print(f"current target tolerance: {current_tolerance:.3f} m")
print(f"stage gate: {'PASS' if passed else 'HOLD'}")
