# quadhopper/renderer.py
"""
Visualization and GIF rendering utilities for QuadHopper.
"""
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
        x, y, theta = obs[0], obs[1], obs[2]
        foot_pos     = env.get_foot_pos()
        render_foot_y = max(0.0, foot_pos[1])  # cosmetic clamp

        body_x = [x - 0.2 * math.cos(theta), x + 0.2 * math.cos(theta)]
        body_y = [y - 0.2 * math.sin(theta), y + 0.2 * math.sin(theta)]
        ax.plot(body_x, body_y, 'k-', lw=6)
        ax.plot([x, foot_pos[0]], [y, render_foot_y], 'b-', lw=3)
        ax.plot(foot_pos[0], render_foot_y, 'ro', markersize=12)

        ax.text(
            cx - cfg.view_x_behind + 0.5,
            cfg.view_y_max - 0.5,
            f"Step: {step}  Theta: {math.degrees(theta):.1f} deg",
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
