"""
Reward function for QuadHopper.

设计思路
--------
1. Liftoff event (主要训练信号)
   在起飞瞬间，检查实际 (vx_com, vy_com) 是否满足弹道扇形条件：
   - 发射角 alpha = atan2(vy, vx) 在 [alpha_min, alpha_max] 扇区内
   - 速度幅值接近规划器计算的弹道所需速度 v0_nom
   满足条件后关闭控制，系统被动到达目标。

2. Stance phase (倒立摆 + 弹簧)
   - 奖励 dtheta > 0（机体顺时针摆动，从后倾到前倾）
   - 奖励弹簧压缩（theta < 0）和伸展（theta > 0）

3. Touchdown event
   - 奖励正确的落地攻角 phi_td（后倾 theta < 0）

4. Flight phase (小幅姿态调整)
   - 仅在顶点附近调整姿态，保证落地角正确
"""

import math
from dataclasses import dataclass
from typing import Tuple

from .config import RewardConfig


@dataclass
class RewardInfo:
    liftoff_v: float = 0.0
    liftoff_angle: float = 0.0
    stance_pendulum: float = 0.0
    stance_spring: float = 0.0
    stance_theta_pos: float = 0.0
    stance_stall: float = 0.0
    stance_timeout: float = 0.0
    touchdown: float = 0.0
    landing_proximity: float = 0.0   # dense reward: foot lands near next target
    forward_progress: float = 0.0
    flight_height: float = 0.0
    flight_attitude: float = 0.0
    flight_thrust: float = 0.0
    target_hit: float = 0.0
    termination: float = 0.0

    @property
    def total(self) -> float:
        return sum(vars(self).values())


class RewardFunction:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def compute(
        self,
        *,
        # 当前状态
        theta: float,
        dtheta: float,
        vx_com: float,
        vy_com: float,
        l_curr: float,
        l_nominal: float,
        com_y: float,
        dl: float,
        stroke_length: float,
        # 接触事件
        touching: bool,
        touchdown_event: bool,
        liftoff_event: bool,
        # 规划器目标（弹道所需速度）
        traj_valid: bool,
        vx_nom: float,
        vy_nom: float,
        dx_target: float,
        # 控制输入
        u1: float,
        u2: float,
        # 落点信息（用于 landing proximity reward）
        foot_x: float = 0.0,
        target_x: float = 0.0,
        # 离散事件标志
        target_hit: bool = False,
        all_targets_done: bool = False,
        terminated_bad: bool = False,
        out_of_bounds: bool = False,
        stance_timeout: bool = False,
    ) -> Tuple[float, RewardInfo]:
        c = self.cfg
        info = RewardInfo()

        # ── 1. 起飞事件：弹道扇形区奖励（主要信号）─────────────────────
        if liftoff_event and traj_valid and dx_target > 0.1:
            v0_actual = math.hypot(vx_com, vy_com)
            v0_nom    = math.hypot(vx_nom, vy_nom)
            alpha_actual = math.atan2(vy_com, vx_com)   # 实际发射角（相对水平）
            alpha_nom    = math.atan2(vy_nom, vx_nom)   # 规划发射角
            alpha_deg    = math.degrees(alpha_actual)

            in_sector = c.alpha_min_deg <= alpha_deg <= c.alpha_max_deg

            # 速度幅值奖励：实际 v0 与弹道所需 v0 的接近程度
            if v0_nom > 0.1 and in_sector:
                v_err_norm = (v0_actual - v0_nom) / v0_nom
                info.liftoff_v = c.liftoff_v_weight * math.exp(
                    -c.liftoff_v_sharpness * v_err_norm ** 2
                )

            # 发射角奖励：与规划角的误差
            alpha_err = abs(self._wrap(alpha_actual - alpha_nom))
            alpha_center = math.radians(0.5 * (c.alpha_min_deg + c.alpha_max_deg))
            if in_sector:
                info.liftoff_angle = c.liftoff_angle_weight * math.exp(
                    -c.liftoff_angle_sharpness * alpha_err
                )
            else:
                # 扇区外：软惩罚，鼓励往扇区靠拢
                err_from_center = abs(self._wrap(alpha_actual - alpha_center))
                info.liftoff_angle = -c.liftoff_angle_weight * 0.3 * err_from_center

        # ── 2. 支撑相：倒立摆 + 弹簧 ──────────────────────────────────────
        if touching:
            # 奖励顺时针摆动（dtheta > 0：从后倾→直立→前倾）
            info.stance_pendulum = c.stance_pendulum_weight * max(dtheta, 0.0)

            # 摆过竖直（theta > 0）的显式奖励：鼓励机体到达正角区
            if theta > 0.0:
                info.stance_theta_pos = c.stance_theta_pos_weight * math.sin(theta)

            # 弹簧循环奖励
            stroke = max(stroke_length, 1e-6)
            compress_ratio = max((l_nominal - l_curr) / stroke, 0.0)
            if theta < 0.0:
                # 压缩加载阶段（后倾）：奖励腿部压缩储能
                info.stance_spring = c.stance_spring_weight * math.tanh(4.0 * compress_ratio)
            else:
                # 伸展释放阶段（前倾）：奖励快速伸腿
                info.stance_spring = c.stance_spring_weight * 0.5 * max(dl, 0.0)

            info.stance_stall = -c.stance_stall_penalty

        if stance_timeout:
            info.stance_timeout = -c.stance_timeout_penalty

        # Dense shaping: reward moving the COM toward the current target.
        # This keeps early PPO exploration from settling into "stand and twitch".
        if dx_target > 0.05:
            if vx_com >= 0.0:
                info.forward_progress = c.forward_progress_weight * min(vx_com, 2.5)
            else:
                info.forward_progress = c.backward_progress_penalty * vx_com

        # ── 3. 落地事件：攻角奖励 + 落点接近目标奖励 ────────────────────
        if touchdown_event:
            phi_td = math.radians(c.phi_td_target_deg)
            td_err = abs(self._wrap(theta - phi_td))
            if theta < 0.0:
                info.touchdown = c.touchdown_weight * math.exp(
                    -c.touchdown_sharpness * td_err
                )
            else:
                info.touchdown = -c.touchdown_bad_penalty

            # 落点接近奖励：foot_x 越接近 target_x 越高（密集塑形信号）
            dist_foot_to_target = abs(foot_x - target_x)
            info.landing_proximity = c.landing_proximity_weight * math.exp(
                -c.landing_proximity_sharpness * dist_foot_to_target ** 2
            )

        # ── 4. 飞行相：姿态引导（顶点附近调整至目标落地角）──────────────
        if not touching:
            height_err = com_y - c.target_height
            info.flight_height = c.flight_height_weight * math.exp(
                -c.flight_height_sharpness * height_err ** 2
            )
            if height_err > 0.0:
                info.flight_height -= c.overheight_penalty_weight * height_err ** 2
            phi_td = math.radians(c.phi_td_target_deg)
            att_err = abs(self._wrap(theta - phi_td))
            info.flight_attitude = c.flight_attitude_weight * math.exp(
                -c.flight_attitude_sharpness * att_err
            )
            info.flight_thrust = -c.flight_thrust_penalty * (u1 + u2)

        # ── 5. 目标命中 ────────────────────────────────────────────────────
        if target_hit:
            info.target_hit += c.target_hit_reward
        if all_targets_done:
            info.target_hit += c.all_targets_bonus

        # ── 6. 终止惩罚 ────────────────────────────────────────────────────
        if terminated_bad:
            info.termination -= c.termination_penalty
        if out_of_bounds:
            info.termination -= c.out_of_bounds_penalty

        return info.total, info
