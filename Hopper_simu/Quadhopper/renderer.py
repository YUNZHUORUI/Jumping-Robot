"""Visualization and GIF rendering utilities for QuadHopper."""

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imageio
from typing import List, Dict

from .config import RenderConfig


class QuadhopperRenderer:
    """Renders simulation frames to a GIF and analysis plots."""

    def __init__(self, cfg: RenderConfig = RenderConfig()):
        self.cfg    = cfg
        self.frames: List[np.ndarray] = []

    def reset(self):
        self.frames = []

    def maybe_render_frame(
        self,
        step: int,
        obs: np.ndarray,
        env,              # QuadhopperTargetEnv instance
        action: np.ndarray = None,
    ):
        """
        Render one frame if `step % render_every_n == 0`.
        Appends the frame to the internal buffer.
        """
        if step % self.cfg.render_every_n != 0:
            return

        cfg = self.cfg
        fig = plt.figure(figsize=(cfg.fig_width, cfg.fig_height), dpi=cfg.dpi)
        ax  = fig.add_subplot(111)

        cx = obs[0]
        ax.set_xlim(cx - cfg.view_x_behind, cx + cfg.view_x_ahead)
        ax.set_ylim(cfg.view_y_min, cfg.view_y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Ground line
        ax.axhline(0, color='k', lw=2)

        # Target markers
        for tid, tx in enumerate(env.ecfg.targets):
            color = 'g' if tid == env.current_target_idx else 'gray'
            ax.plot(tx, 0, 'x', color=color, markersize=10, markeredgewidth=3)

        # Ideal trajectory arc
        if env.planner.valid:
            t_idx = env.current_target_idx
            tx_arr = np.linspace(
                env.planner.x0,
                env.ecfg.targets[t_idx],
                50,
            )
            dx_arr = tx_arr - env.planner.x0
            ty_arr = (
                env.planner.a * dx_arr ** 2
                + env.planner.b_local * dx_arr
                + env.planner.y0
            )
            mask = ty_arr > -0.2
            ax.plot(tx_arr[mask], ty_arr[mask], 'r--', alpha=0.5)

        # Robot body & leg
        foot_x, foot_y, theta, l_curr = obs[0], obs[1], obs[2], obs[3]
        foot_pos = np.array([foot_x, foot_y], dtype=np.float64)

        render_scale = cfg.render_geom_scale
        com = env.physics.get_com_pos(obs[:4])
        leg_vec = np.array(foot_pos, dtype=np.float64) - com
        leg_len = float(np.linalg.norm(leg_vec))
        if leg_len > 1e-8:
            leg_dir = leg_vec / leg_len
        else:
            leg_dir = np.array([math.sin(theta), -math.cos(theta)], dtype=np.float64)

        leg_vec_render = leg_vec * render_scale
        render_foot = com + leg_vec_render

        # Prevent visual ground penetration of rendered foot marker.
        # With render_scale > 1, direct scaling can place render_foot below y=0
        # even when the physical foot is exactly on ground.
        ground_y = float(env.pcfg.ground_y)
        if render_foot[1] < ground_y:
            d = render_foot - com
            if abs(d[1]) > 1e-8:
                t = (ground_y - com[1]) / d[1]
                t = float(np.clip(t, 0.0, 1.0))
                render_foot = com + t * d
            render_foot[1] = ground_y

        body_dir = np.array([leg_dir[1], -leg_dir[0]], dtype=np.float64)
        body_half_render = env.pcfg.beam_half_length * render_scale
        body_l = com - body_half_render * body_dir
        body_r = com + body_half_render * body_dir
        ax.plot([body_l[0], body_r[0]], [body_l[1], body_r[1]], 'k-', lw=cfg.body_linewidth)

        thrust_l = 0.0 if action is None else float(np.clip(action[0], 0.0, 1.0))
        thrust_r = 0.0 if action is None else float(np.clip(action[1], 0.0, 1.0))
        rotor_l_color = plt.cm.coolwarm(thrust_l)
        rotor_r_color = plt.cm.coolwarm(thrust_r)
        ax.scatter(body_l[0], body_l[1], s=cfg.render_rotor_size, c=[rotor_l_color], edgecolors='k', linewidths=0.8, zorder=5)
        ax.scatter(body_r[0], body_r[1], s=cfg.render_rotor_size, c=[rotor_r_color], edgecolors='k', linewidths=0.8, zorder=5)

        ax.plot([com[0], render_foot[0]], [com[1], render_foot[1]], 'b-', lw=cfg.leg_linewidth)
        ax.plot(render_foot[0], render_foot[1], 'ro', markersize=cfg.foot_markersize)

        cx = com[0]
        ax.text(
            cx - cfg.view_x_behind + 0.5,
            cfg.view_y_max - 0.5,
            f"Step: {step}  Theta: {math.degrees(theta):.1f} deg\nL: {thrust_l:.2f} R: {thrust_r:.2f}",
            fontsize=12,
        )

        fig.canvas.draw()
        image = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        self.frames.append(image)
        plt.close(fig)

    def save_gif(self):
        """Write buffered frames to GIF file."""
        if not self.frames:
            print("No frames to save.")
            return
        imageio.mimsave(self.cfg.gif_path, self.frames, fps=self.cfg.fps)
        print(f"GIF saved: {self.cfg.gif_path}")

    @staticmethod
    def save_analysis_plots(history: Dict, path: str):
        """
        Generate and save the 3-panel analysis figure.

        Args:
            history: Dict with keys [step, y, target_y, theta, thrust_l, thrust_r]
            path:    Output file path (PNG)
        """
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(10, 12), sharex=True
        )

        ax1.plot(history['step'], history['y'],        'k-',  label='COM Y')
        ax1.plot(history['step'], history['target_y'], 'r--', alpha=0.6,
                 label='Ideal Trajectory Y')
        ax1.set_ylabel("Height (m)")
        ax1.set_title("Trajectory Tracking")
        ax1.legend(); ax1.grid(True)

        ax2.plot(history['step'], history['theta'], 'g-', label='Tilt Angle (deg)')
        ax2.axhline( 40, color='r', ls='--', alpha=0.3)
        ax2.axhline(-40, color='r', ls='--', alpha=0.3)
        ax2.set_ylabel("Theta (degrees)")
        ax2.set_title("Body Tilt Angle (Target < 40 deg)")
        ax2.legend(); ax2.grid(True)

        ax3.plot(history['step'], history['thrust_l'], 'b-', alpha=0.6,
                 label='Left Thrust')
        ax3.plot(history['step'], history['thrust_r'], 'm-', alpha=0.6,
                 label='Right Thrust')
        ax3.set_ylabel("Thrust (0-1)")
        ax3.set_xlabel("Step")
        ax3.set_title("Control Inputs")
        ax3.legend(); ax3.grid(True)

        plt.tight_layout()
        plt.savefig(path)
        print(f"Analysis saved: {path}")
        plt.close(fig)
