from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-quadhopper")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


MOTOR_COLUMNS = ["M1", "M2", "M3", "M4"]


@dataclass(frozen=True)
class ReleaseEpisode:
    label: str
    far_start_time: float
    far_end_time: float
    plot_start_time: float
    plot_end_time: float
    cycle_start_time: float
    cycle_end_time: float
    start_x: float
    start_y: float


@dataclass(frozen=True)
class StableCycleWindow:
    label: str
    start_time: float
    end_time: float
    mean_xy_error: float
    max_xy_error: float
    mean_apex_height: float
    mean_hop_period: float
    mean_current: float
    peak_current: float


def load_active_data(csv_path: Path) -> pd.DataFrame:
    columns = [
        "Time_s",
        "loop_dt",
        "X",
        "Y",
        "Z",
        "Vel_Z",
        "Target_X",
        "Target_Y",
        "Target_Z",
        "M1",
        "M2",
        "M3",
        "M4",
        "Obs_IsContact",
        "VBat_V",
        "Current_A",
    ]
    data = pd.read_csv(csv_path, usecols=columns)
    active = data[MOTOR_COLUMNS].abs().sum(axis=1) > 1e-9
    data = data.loc[active].copy()
    data["radial_error"] = np.hypot(data["X"] - data["Target_X"], data["Y"] - data["Target_Y"])
    data["power_W"] = data["VBat_V"] * data["Current_A"]
    return data.reset_index(drop=True)


def boolean_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(mask.astype(int)) != 0) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(mask)]
    return [(int(start), int(end)) for start, end in zip(starts, ends) if mask[start]]


def contact_start_times(data: pd.DataFrame) -> np.ndarray:
    contact = data["Obs_IsContact"].round().astype(int).to_numpy()
    starts = np.flatnonzero(np.diff(contact) == 1) + 1
    return data["Time_s"].to_numpy()[starts]


def first_return_time(data: pd.DataFrame, start_index: int, near_threshold: float) -> float:
    radial_error = data["radial_error"].to_numpy()
    time = data["Time_s"].to_numpy()
    for index in range(start_index, len(data)):
        if radial_error[index] <= near_threshold:
            return float(time[index])
    return float(time[min(start_index, len(data) - 1)])


def select_cycle_window(
    contact_starts: np.ndarray,
    earliest_start_time: float,
    latest_end_time: float,
    cycles_per_window: int,
) -> tuple[float, float]:
    candidates = contact_starts[contact_starts >= earliest_start_time]
    for index in range(0, max(0, len(candidates) - cycles_per_window)):
        start_time = float(candidates[index])
        end_time = float(candidates[index + cycles_per_window])
        if end_time <= latest_end_time:
            return start_time, end_time

    if len(candidates) > cycles_per_window:
        return float(candidates[0]), float(candidates[cycles_per_window])
    if len(candidates) > 0:
        return float(candidates[0]), min(float(candidates[0] + 6.0), latest_end_time)
    return earliest_start_time, min(float(earliest_start_time + 6.0), latest_end_time)


def find_release_episodes(
    data: pd.DataFrame,
    far_threshold: float = 0.6,
    near_threshold: float = 0.18,
    cycle_near_threshold: float = 0.30,
    min_far_duration: float = 3.0,
    merge_gap_s: float = 2.5,
    cycles_per_window: int = 6,
) -> list[ReleaseEpisode]:
    time = data["Time_s"].to_numpy()
    far_runs = []
    for start, end in boolean_runs(data["radial_error"].to_numpy() > far_threshold):
        duration = time[end - 1] - time[start]
        if duration >= min_far_duration:
            far_runs.append([start, end])

    merged_runs: list[list[int]] = []
    for start, end in far_runs:
        if merged_runs and time[start] - time[merged_runs[-1][1] - 1] <= merge_gap_s:
            merged_runs[-1][1] = end
        else:
            merged_runs.append([start, end])

    contact_starts = contact_start_times(data)
    episodes: list[ReleaseEpisode] = []
    for episode_index, (far_start, far_end) in enumerate(merged_runs, start=1):
        next_far_start_time = (
            float(time[merged_runs[episode_index][0]]) if episode_index < len(merged_runs) else float(time[-1])
        )
        start_window = data.iloc[far_start:far_end]
        start_row = start_window.loc[start_window["radial_error"].idxmax()]
        release_start_time = float(start_row["Time_s"])
        return_time = first_return_time(data, far_end, near_threshold)
        cycle_return_time = first_return_time(data, far_end, cycle_near_threshold)
        plot_start = max(float(time[0]), release_start_time)
        plot_end = min(float(time[-1]), return_time + 1.0, next_far_start_time - 0.75)

        cycle_start, cycle_end = select_cycle_window(
            contact_starts,
            earliest_start_time=cycle_return_time,
            latest_end_time=next_far_start_time - 0.75,
            cycles_per_window=cycles_per_window,
        )

        episodes.append(
            ReleaseEpisode(
                label=f"Release {episode_index}",
                far_start_time=float(time[far_start]),
                far_end_time=float(time[far_end - 1]),
                plot_start_time=plot_start,
                plot_end_time=plot_end,
                cycle_start_time=cycle_start,
                cycle_end_time=cycle_end,
                start_x=float(start_row["X"]),
                start_y=float(start_row["Y"]),
            )
        )

    return episodes


def find_stable_cycle_windows(
    data: pd.DataFrame,
    window_count: int = 3,
    cycles_per_window: int = 6,
    min_start_time: float = 150.0,
    max_end_time: float = 1040.0,
    min_separation_s: float = 220.0,
) -> list[StableCycleWindow]:
    contact = data["Obs_IsContact"].round().astype(int).to_numpy()
    contact_indices = np.flatnonzero(np.diff(contact) == 1) + 1
    time = data["Time_s"].to_numpy()
    contact_times = time[contact_indices]

    candidates = []
    for start_contact in range(0, len(contact_indices) - cycles_per_window):
        start_index = contact_indices[start_contact]
        end_index = contact_indices[start_contact + cycles_per_window]
        start_time = float(time[start_index])
        end_time = float(time[end_index])
        if start_time < min_start_time or end_time > max_end_time:
            continue

        periods = np.diff(contact_times[start_contact : start_contact + cycles_per_window + 1])
        if np.any(periods < 0.75) or np.any(periods > 1.05):
            continue

        segment = data.iloc[start_index:end_index]
        apex_heights = []
        for contact_number in range(start_contact, start_contact + cycles_per_window):
            hop_start = contact_indices[contact_number]
            hop_end = contact_indices[contact_number + 1]
            apex_heights.append(float(data["Z"].iloc[hop_start:hop_end].max()))

        apex_heights_array = np.asarray(apex_heights)
        score = (
            2.0 * float(segment["radial_error"].mean())
            + float(segment["radial_error"].std())
            + 3.0 * float(apex_heights_array.std())
            + float(periods.std())
        )
        candidates.append(
            (
                score,
                StableCycleWindow(
                    label="",
                    start_time=start_time,
                    end_time=end_time,
                    mean_xy_error=float(segment["radial_error"].mean()),
                    max_xy_error=float(segment["radial_error"].max()),
                    mean_apex_height=float(apex_heights_array.mean()),
                    mean_hop_period=float(periods.mean()),
                    mean_current=float(segment["Current_A"].mean()),
                    peak_current=float(segment["Current_A"].max()),
                ),
            )
        )

    selected: list[StableCycleWindow] = []
    for _, candidate in sorted(candidates, key=lambda item: item[0]):
        if all(abs(candidate.start_time - other.start_time) >= min_separation_s for other in selected):
            selected.append(candidate)
        if len(selected) == window_count:
            break

    selected = sorted(selected, key=lambda window: window.start_time)
    labels = ["Stable A", "Stable B", "Stable C", "Stable D", "Stable E"]
    return [
        StableCycleWindow(
            label=labels[index],
            start_time=window.start_time,
            end_time=window.end_time,
            mean_xy_error=window.mean_xy_error,
            max_xy_error=window.max_xy_error,
            mean_apex_height=window.mean_apex_height,
            mean_hop_period=window.mean_hop_period,
            mean_current=window.mean_current,
            peak_current=window.peak_current,
        )
        for index, window in enumerate(selected)
    ]


def plot_filtered_topdown(data: pd.DataFrame, episodes: list[ReleaseEpisode], output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    target_x = float(data["Target_X"].iloc[-1])
    target_y = float(data["Target_Y"].iloc[-1])

    fig, ax = plt.subplots(figsize=(3.45, 3.2), constrained_layout=True)
    plot_start_time = min(episode.plot_start_time for episode in episodes)
    segment = data[
        (data["Time_s"] >= plot_start_time)
        & (data["radial_error"] <= 0.85)
        & (data["Z"] >= 0.15)
    ].copy()

    if segment.empty:
        raise RuntimeError("No top-down points available after filtering.")

    median_dt = float(segment["loop_dt"].median())
    point_stride = max(1, int(round(0.20 / median_dt)))
    points = segment.iloc[::point_stride]
    scatter = ax.scatter(
        points["X"],
        points["Y"],
        c=points["Time_s"],
        s=8,
        cmap="viridis",
        alpha=0.78,
        linewidths=0,
    )
    ax.scatter([target_x], [target_y], marker="x", s=55, color="red", linewidths=1.5, label="target")
    fig.colorbar(scatter, ax=ax, label="time [s]")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Top-down deployed trajectory")
    ax.axis("equal")
    ax.legend(frameon=True, fontsize=7, loc="best")
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_stable_cycle_windows(data: pd.DataFrame, windows: list[StableCycleWindow], output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    phase_handles = [
        Patch(facecolor="#88CCEE", edgecolor="none", alpha=0.16, label="ascent"),
        Patch(facecolor="#EE99AA", edgecolor="none", alpha=0.16, label="descent"),
    ]
    for window, color in zip(windows, colors):
        segment = data[(data["Time_s"] >= window.start_time) & (data["Time_s"] <= window.end_time)].copy()
        relative_time = segment["Time_s"] - segment["Time_s"].iloc[0]
        contact = segment["Obs_IsContact"].round().astype(int).to_numpy()

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(7.1, 3.8),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1.0, 1.0]},
        )

        ax = axes[0]
        ax.plot(relative_time, segment["Z"], color=color, lw=1.1, label="Z")
        ax.plot(relative_time, segment["X"] - segment["Target_X"], color="0.15", lw=0.9, label="X error")
        ax.plot(relative_time, segment["Y"] - segment["Target_Y"], color="0.45", lw=0.9, label="Y error")
        ax.fill_between(
            relative_time,
            0.0,
            contact * 0.12,
            color=color,
            alpha=0.18,
            lw=0,
            label="contact",
        )
        ax.set_ylim(-0.25, 1.12)
        ax.set_ylabel("m")
        ax.set_title(
            f"{window.label}: {window.start_time:.1f}-{window.end_time:.1f} s",
            loc="left",
            fontsize=8,
        )
        ax.legend(frameon=True, fontsize=7, loc="upper right", ncol=4)

        ax = axes[1]
        phase = np.where(segment["Vel_Z"].to_numpy() >= 0.0, 1, -1)
        change_indices = np.r_[0, np.flatnonzero(np.diff(phase) != 0) + 1, len(phase) - 1]
        for start_index, end_index in zip(change_indices[:-1], change_indices[1:]):
            if end_index <= start_index:
                continue
            phase_color = "#88CCEE" if phase[start_index] > 0 else "#EE99AA"
            ax.axvspan(
                float(relative_time.iloc[start_index]),
                float(relative_time.iloc[end_index]),
                color=phase_color,
                alpha=0.16,
                lw=0,
                zorder=0,
            )
        ax.plot(relative_time, segment["Current_A"], color="#D55E00", lw=0.9, zorder=2)
        ax_power = ax.twinx()
        ax_power.plot(relative_time, segment["power_W"], color="0.25", lw=0.85, alpha=0.85, zorder=2)
        electrical_legend_handles = phase_handles
        electrical_legend_labels = [handle.get_label() for handle in phase_handles]
        ax.set_ylim(0, max(14.0, float(segment["Current_A"].quantile(0.995)) + 1.0))
        ax_power.set_ylim(0, max(50.0, float(segment["power_W"].quantile(0.995)) + 2.0))
        ax.set_ylabel("Current [A]")
        ax_power.set_ylabel("Power [W]")
        ax.set_title(
            f"mean I={window.mean_current:.2f} A, peak I={window.peak_current:.1f} A",
            loc="left",
            fontsize=8,
        )
        ax.legend(electrical_legend_handles, electrical_legend_labels, frameon=True, fontsize=7, loc="upper right", ncol=4)
        axes[1].set_xlabel("time within 6 hopping cycles [s]")

        slug = window.label.lower().replace(" ", "_")
        window_output_base = output_base.parent / f"{output_base.name}_{slug}"
        fig.savefig(window_output_base.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(window_output_base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_six_cycle_windows(data: pd.DataFrame, episodes: list[ReleaseEpisode], output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    fig, axes = plt.subplots(len(episodes), 1, figsize=(3.45, 4.8), sharex=False, constrained_layout=True)
    if len(episodes) == 1:
        axes = [axes]

    for ax, color, episode in zip(axes, colors, episodes):
        segment = data[
            (data["Time_s"] >= episode.cycle_start_time) & (data["Time_s"] <= episode.cycle_end_time)
        ].copy()
        relative_time = segment["Time_s"] - segment["Time_s"].iloc[0]
        ax.plot(relative_time, segment["Z"], color=color, lw=1.1, label="height")
        ax.plot(relative_time, segment["radial_error"], color="0.15", lw=0.9, alpha=0.75, label="XY error")
        contact = segment["Obs_IsContact"].round().astype(int).to_numpy()
        contact_scaled = contact * (segment["Z"].max() - segment["Z"].min()) + segment["Z"].min()
        ax.fill_between(relative_time, segment["Z"].min(), contact_scaled, color=color, alpha=0.12, lw=0)
        ax.set_ylabel("m")
        ax.set_title(episode.label, loc="left", fontsize=8)
        ax.set_ylim(0.0, max(1.1, float(segment["Z"].max()) + 0.05))

    axes[-1].set_xlabel("time within 6 hopping cycles [s]")
    axes[0].legend(frameon=True, fontsize=7, loc="upper right", ncol=2)
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_episode_table(data: pd.DataFrame, episodes: list[ReleaseEpisode], output_path: Path) -> None:
    rows = []
    for episode in episodes:
        segment = data[(data["Time_s"] >= episode.plot_start_time) & (data["Time_s"] <= episode.plot_end_time)]
        rows.append(
            {
                "label": episode.label,
                "far_start_time_s": episode.far_start_time,
                "far_end_time_s": episode.far_end_time,
                "plot_start_time_s": episode.plot_start_time,
                "plot_end_time_s": episode.plot_end_time,
                "cycle_start_time_s": episode.cycle_start_time,
                "cycle_end_time_s": episode.cycle_end_time,
                "start_x_m": episode.start_x,
                "start_y_m": episode.start_y,
                "max_radial_error_m": float(segment["radial_error"].max()),
                "final_radial_error_m": float(segment["radial_error"].iloc[-1]),
                "mean_current_A": float(segment["Current_A"].mean()),
                "peak_current_A": float(segment["Current_A"].max()),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_stable_window_table(windows: list[StableCycleWindow], output_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "label": window.label,
                "start_time_s": window.start_time,
                "end_time_s": window.end_time,
                "mean_xy_error_m": window.mean_xy_error,
                "max_xy_error_m": window.max_xy_error,
                "mean_apex_height_m": window.mean_apex_height,
                "mean_hop_period_s": window.mean_hop_period,
                "mean_current_A": window.mean_current,
                "peak_current_A": window.peak_current,
            }
            for window in windows
        ]
    ).to_csv(output_path, index=False)


def release_return_segments(
    data: pd.DataFrame,
    episodes: list[ReleaseEpisode],
    return_threshold: float = 0.18,
    hold_time_s: float = 0.0,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    segments = []
    rows = []
    for index, episode in enumerate(episodes):
        next_release_time = episodes[index + 1].far_start_time if index + 1 < len(episodes) else float(data["Time_s"].iloc[-1])
        start_time = episode.plot_start_time
        candidate = data[(data["Time_s"] >= start_time) & (data["Time_s"] <= next_release_time)].copy()
        if candidate.empty:
            continue

        returned = candidate[candidate["radial_error"] <= return_threshold]
        return_time = float(returned["Time_s"].iloc[0]) if not returned.empty else float(candidate["Time_s"].iloc[-1])
        end_time = min(return_time + hold_time_s, next_release_time - 0.1)
        segment = data[(data["Time_s"] >= start_time) & (data["Time_s"] <= end_time)].copy()
        segment.insert(0, "episode_label", episode.label)
        segment["relative_time_s"] = segment["Time_s"] - start_time
        segment["x_error_m"] = segment["X"] - segment["Target_X"]
        segment["y_error_m"] = segment["Y"] - segment["Target_Y"]

        dx = segment["X"].diff().fillna(0.0)
        dy = segment["Y"].diff().fillna(0.0)
        path_length = float(np.hypot(dx, dy).sum())
        energy = float((segment["power_W"] * segment["loop_dt"]).sum())
        return_duration = return_time - start_time
        rows.append(
            {
                "label": episode.label,
                "start_time_s": start_time,
                "return_time_s": return_time,
                "return_duration_s": return_duration,
                "start_radial_error_m": float(segment["radial_error"].iloc[0]),
                "return_threshold_m": return_threshold,
                "path_length_m": path_length,
                "mean_current_A": float(segment["Current_A"].mean()),
                "peak_current_A": float(segment["Current_A"].max()),
                "energy_J": energy,
                "start_x_m": float(segment["X"].iloc[0]),
                "start_y_m": float(segment["Y"].iloc[0]),
            }
        )
        segments.append(segment)

    return segments, pd.DataFrame(rows)


def plot_release_return_topdown_panels(segments: list[pd.DataFrame], output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.8), constrained_layout=True)
    axes_flat = axes.ravel()
    all_x = pd.concat([segment["X"] for segment in segments])
    all_y = pd.concat([segment["Y"] for segment in segments])
    x_margin = 0.08
    y_margin = 0.08
    target_x = float(segments[0]["Target_X"].iloc[-1])
    target_y = float(segments[0]["Target_Y"].iloc[-1])

    for ax, segment in zip(axes_flat, segments):
        stride = max(1, int(round(0.08 / float(segment["loop_dt"].median()))))
        points = segment.iloc[::stride]
        scatter = ax.scatter(
            points["X"],
            points["Y"],
            c=points["relative_time_s"],
            s=10,
            cmap="viridis",
            alpha=0.82,
            linewidths=0,
        )
        ax.scatter(segment["X"].iloc[0], segment["Y"].iloc[0], marker="o", s=28, color="black", zorder=3)
        ax.scatter(segment["X"].iloc[-1], segment["Y"].iloc[-1], marker="s", s=24, color="0.35", zorder=3)
        ax.scatter([target_x], [target_y], marker="x", s=52, color="red", linewidths=1.4, zorder=4)
        for radius, style in [(0.18, "--"), (0.30, ":")]:
            ax.add_patch(plt.Circle((target_x, target_y), radius, fill=False, color="0.35", lw=0.8, ls=style))
        ax.set_title(str(segment["episode_label"].iloc[0]), loc="left")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_xlim(float(all_x.min()) - x_margin, float(all_x.max()) + x_margin)
        ax.set_ylim(float(all_y.min()) - y_margin, float(all_y.max()) + y_margin)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(scatter, ax=ax, label="time since release [s]")

    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_release_return_errors(segments: list[pd.DataFrame], output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 4.3), sharex=True, constrained_layout=True)

    for color, segment in zip(colors, segments):
        label = str(segment["episode_label"].iloc[0])
        axes[0].plot(segment["relative_time_s"], segment["radial_error"], color=color, lw=1.15, label=label)
        axes[1].plot(segment["relative_time_s"], segment["x_error_m"], color=color, lw=1.0, label=f"{label} X")
        axes[1].plot(segment["relative_time_s"], segment["y_error_m"], color=color, lw=1.0, ls="--", label=f"{label} Y")

    axes[0].axhline(0.18, color="0.25", lw=0.9, ls="--", label="0.18 m")
    axes[0].axhline(0.30, color="0.45", lw=0.9, ls=":", label="0.30 m")
    axes[0].set_ylabel("radial error [m]")
    axes[0].legend(frameon=True, fontsize=7, ncol=3, loc="upper right")
    axes[1].axhline(0.0, color="0.3", lw=0.8)
    axes[1].set_ylabel("component error [m]")
    axes[1].set_xlabel("time since release [s]")
    axes[1].legend(frameon=True, fontsize=6.5, ncol=4, loc="upper right")
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_release_return_metrics(summary: pd.DataFrame, output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    labels = summary["label"].to_list()
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"][: len(labels)]
    metrics = [
        ("return_duration_s", "return time [s]"),
        ("path_length_m", "path length [m]"),
        ("energy_J", "energy [J]"),
        ("peak_current_A", "peak current [A]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.8), constrained_layout=True)
    for ax, (column, ylabel) in zip(axes.ravel(), metrics):
        ax.bar(labels, summary[column], color=colors, alpha=0.86)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAL-style plots from Quadhopper real-world CSV logs.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_active_data(args.csv_path)
    episodes = find_release_episodes(data)
    if not episodes:
        raise RuntimeError("No corner-release episodes found. Try lowering the far_threshold.")
    stable_windows = find_stable_cycle_windows(data)
    if not stable_windows:
        raise RuntimeError("No stable six-cycle windows found. Try relaxing the stability thresholds.")

    plot_filtered_topdown(data, episodes, args.output_dir / "quadhopper_corner_release_topdown_filtered")
    plot_stable_cycle_windows(data, stable_windows, args.output_dir / "quadhopper_stable_six_cycle_windows")
    return_segments, return_summary = release_return_segments(data, episodes)
    plot_release_return_topdown_panels(
        return_segments,
        args.output_dir / "quadhopper_release_return_topdown_panels",
    )
    plot_release_return_errors(return_segments, args.output_dir / "quadhopper_release_return_errors")
    plot_release_return_metrics(return_summary, args.output_dir / "quadhopper_release_return_metrics")
    pd.concat(return_segments, ignore_index=True).to_csv(args.output_dir / "quadhopper_release_return_segments.csv", index=False)
    return_summary.to_csv(args.output_dir / "quadhopper_release_return_summary.csv", index=False)
    write_episode_table(data, episodes, args.output_dir / "quadhopper_release_episode_table.csv")
    write_stable_window_table(stable_windows, args.output_dir / "quadhopper_stable_window_table.csv")

    print(f"Detected {len(episodes)} release episodes")
    for episode in episodes:
        print(
            f"{episode.label}: far {episode.far_start_time:.2f}-{episode.far_end_time:.2f} s, "
            f"six-cycle window {episode.cycle_start_time:.2f}-{episode.cycle_end_time:.2f} s"
        )
    print(f"Selected {len(stable_windows)} stable six-cycle windows")
    for window in stable_windows:
        print(
            f"{window.label}: {window.start_time:.2f}-{window.end_time:.2f} s, "
            f"mean XY error {window.mean_xy_error:.3f} m, mean apex {window.mean_apex_height:.3f} m"
        )


if __name__ == "__main__":
    main()
