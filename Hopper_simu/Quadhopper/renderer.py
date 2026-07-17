"""Visualization and GIF rendering for QuadHopper.

T-shape structure
-----------------
  [L rotor]----[body beam]----[R rotor]   <- horizontal frame at COM
                    |
                   leg
                    |
                  [foot]                   <- foot tip (contact point)

State convention  q = [x_foot, y_foot, theta, l]
  theta = body angle from vertical (positive = lean right/forward)
  The COM is at  (x_foot + l*sin(theta),  y_foot + l*cos(theta))
  The beam is perpendicular to the leg:  direction (cos(theta), -sin(theta))
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
from typing import List, Dict

from .config import RenderConfig


class QuadhopperRenderer:

    def __init__(self, cfg: RenderConfig = RenderConfig()):
        self.cfg = cfg
        self.frames: List[np.ndarray] = []

    def reset(self):
        self.frames = []

    def maybe_render_frame(
        self,
        step: int,
        obs: np.ndarray,   # kept for API compatibility, not used for geometry
        env,               # QuadhopperTargetEnv
        action: np.ndarray = None,
    ):
        if step % self.cfg.render_every_n != 0:
            return

        cfg = self.cfg

        # ── Physical state always from env.q ─────────────────────────────
        x_f   = float(env.q[0])
        y_f   = float(env.q[1])
        theta = float(env.q[2])
        l     = float(env.q[3])

        s_th = math.sin(theta)
        c_th = math.cos(theta)

        # Foot tip
        foot = np.array([x_f, y_f])

        # COM = foot + l*(sin θ, cos θ)
        com = np.array([x_f + l * s_th, y_f + l * c_th])

        # Beam unit vector (perpendicular to leg, pointing "right" in body frame)
        # When θ=0 (upright) this is (1, 0) — horizontal
        beam_unit = np.array([c_th, -s_th])
        beam_half = float(env.pcfg.beam_half_length)

        # Rotor positions (physical, no exaggeration)
        rotor_l = com - beam_half * beam_unit   # left rotor  → action[0]
        rotor_r = com + beam_half * beam_unit   # right rotor → action[1]

        # ── Canvas ───────────────────────────────────────────────────────
        cx = float(com[0])
        fig, ax = plt.subplots(figsize=(cfg.fig_width, cfg.fig_height), dpi=cfg.dpi)
        ax.set_xlim(cx - cfg.view_x_behind, cx + cfg.view_x_ahead)
        ax.set_ylim(cfg.view_y_min, cfg.view_y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', lw=2)

        # ── Target markers ────────────────────────────────────────────────
        for tid, tx in enumerate(env.ecfg.targets):
            color = '#00aa00' if tid == env.current_target_idx else '#aaaaaa'
            ax.plot(tx, 0, 'x', color=color, markersize=12, markeredgewidth=2.5)
            ax.text(tx, -0.15, f'T{tid}', ha='center', fontsize=8, color=color)

        # ── Planned ballistic arc ─────────────────────────────────────────
        if env.planner.valid and env.current_target_idx < len(env.ecfg.targets):
            tx_arr = np.linspace(
                env.planner.x0,
                env.ecfg.targets[env.current_target_idx],
                80,
            )
            dx_arr = tx_arr - env.planner.x0
            ty_arr = (env.planner.a * dx_arr ** 2
                      + env.planner.b_local * dx_arr
                      + env.planner.y0)
            mask = ty_arr > -0.1
            ax.plot(tx_arr[mask], ty_arr[mask], 'r--', alpha=0.45, lw=1.5,
                    label='planned arc')

        # ── Robot: T-shape ────────────────────────────────────────────────
        # 1. Leg  (blue, from COM down to foot)
        ax.plot([com[0], foot[0]], [com[1], foot[1]],
                color='#2255cc', lw=cfg.leg_linewidth, solid_capstyle='round', zorder=4)

        # 2. Horizontal beam  (black, across COM)
        ax.plot([rotor_l[0], rotor_r[0]], [rotor_l[1], rotor_r[1]],
                color='#111111', lw=cfg.body_linewidth, solid_capstyle='round', zorder=5)

        # 3. Rotors (circles colored by thrust: blue=off, red=full)
        thrust_l = 0.0 if action is None else float(np.clip(action[0], 0.0, 1.0))
        thrust_r = 0.0 if action is None else float(np.clip(action[1], 0.0, 1.0))

        ax.scatter(rotor_l[0], rotor_l[1], s=cfg.render_rotor_size,
                   c=[plt.cm.coolwarm(thrust_l)],
                   edgecolors='k', linewidths=0.8, zorder=6)
        ax.scatter(rotor_r[0], rotor_r[1], s=cfg.render_rotor_size,
                   c=[plt.cm.coolwarm(thrust_r)],
                   edgecolors='k', linewidths=0.8, zorder=6)

        # 4. Thrust arrows (pointing in -leg direction, i.e. body "up")
        body_up = np.array([s_th, c_th])       # unit vector: body "up" = foot → COM direction
        arrow_max = 0.35  # metres at full thrust
        for pos, thr in [(rotor_l, thrust_l), (rotor_r, thrust_r)]:
            if thr > 0.02:
                tip = pos + thr * arrow_max * body_up
                ax.annotate('', xy=tip, xytext=pos,
                            arrowprops=dict(arrowstyle='->', color='orange',
                                            lw=1.8, mutation_scale=12),
                            zorder=7)

        # 5. Foot marker
        ax.plot(foot[0], foot[1], 'o', color='#cc2222',
                markersize=cfg.foot_markersize, zorder=8)

        # ── Phase & info text ─────────────────────────────────────────────
        phase = 'STANCE' if env.physics.stance_active else 'FLIGHT'
        com_vel = env.physics.get_com_vel(env.q, env.dq)
        vx = float(com_vel[0])
        vy = float(com_vel[1])
        ax.text(
            cx - cfg.view_x_behind + 0.08,
            cfg.view_y_max - 0.12,
            (f"Step {step}  [{phase}]  "
             f"θ={math.degrees(theta):+.1f}°  "
             f"vx={vx:.2f}  vy={vy:.2f}\n"
             f"L={thrust_l:.2f}  R={thrust_r:.2f}  "
             f"target {env.current_target_idx}/{len(env.ecfg.targets)}"),
            fontsize=9, va='top', family='monospace',
        )

        fig.tight_layout(pad=0.5)
        fig.canvas.draw()
        image = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        self.frames.append(image)
        plt.close(fig)

    def save_gif(self):
        if not self.frames:
            print("No frames to save.")
            return
        Path(self.cfg.gif_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(self.cfg.gif_path, self.frames, fps=self.cfg.fps)
        print(f"GIF saved: {self.cfg.gif_path}")

    @staticmethod
    def save_analysis_plots(history: Dict, path: str):
        fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
        ax1, ax2, ax3 = axes

        ax1.plot(history['step'], history['y'],        'k-',  lw=1.5, label='COM height')
        ax1.plot(history['step'], history['target_y'], 'r--', lw=1.2, alpha=0.7,
                 label='Planned arc height')
        ax1.set_ylabel("Height (m)")
        ax1.set_title("COM Height vs Planned Arc")
        ax1.legend(); ax1.grid(True, alpha=0.4)

        ax2.plot(history['step'], history['theta'], 'g-', lw=1.5, label='θ (deg)')
        ax2.axhline( 60, color='r', ls='--', alpha=0.3, label='sector bounds')
        ax2.axhline(-60, color='r', ls='--', alpha=0.3)
        ax2.set_ylabel("Body Angle (°)")
        ax2.set_title("Body Tilt: touchdown target from reward config, liftoff from planner")
        ax2.legend(); ax2.grid(True, alpha=0.4)

        ax3.plot(history['step'], history['thrust_l'], 'b-', lw=1.2, alpha=0.7,
                 label='Left thrust')
        ax3.plot(history['step'], history['thrust_r'], 'm-', lw=1.2, alpha=0.7,
                 label='Right thrust')
        ax3.set_ylabel("Thrust (0–1)")
        ax3.set_xlabel("Step")
        ax3.set_title("Control Inputs")
        ax3.legend(); ax3.grid(True, alpha=0.4)

        plt.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=120)
        print(f"Analysis saved: {path}")
        plt.close(fig)
