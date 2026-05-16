import sys

# 1. Update renderer.py
with open('/Users/yunzhuorui/Jumping-Robot/Hopper_simu/Quadhopper/renderer.py', 'r') as f:
    code = f.read()

# Replace observation x, y meaning
code = code.replace(
    "x, y, theta = obs[0], obs[1], obs[2]\n        foot_pos = env.get_foot_pos()\n\n        render_scale = cfg.render_geom_scale\n        com = np.array([x, y], dtype=np.float64)\n",
    "foot_x, foot_y, theta, l_curr = obs[0], obs[1], obs[2], obs[3]\n        foot_pos = np.array([foot_x, foot_y], dtype=np.float64)\n\n        render_scale = cfg.render_geom_scale\n        com = env.physics.get_com_pos(obs[:4])\n"
)
with open('/Users/yunzhuorui/Jumping-Robot/Hopper_simu/Quadhopper/renderer.py', 'w') as f:
    f.write(code)

# 2. Update env.py
with open('/Users/yunzhuorui/Jumping-Robot/Hopper_simu/Quadhopper/env.py', 'r') as f:
    code = f.read()

code = code.replace(
    "foot_pos = self.get_foot_pos()",
    "foot_pos = self.physics.get_foot_pos(self.q)"
)
code = code.replace(
    "def get_foot_pos(self) -> np.ndarray:\n        return self.physics.get_foot_pos(self.q)",
    ""
)
code = code.replace(
    "self.q[1] > self.reward_fn.cfg.max_height\n            or self.q[1] < self.pcfg.min_com_height",
    "self.physics.get_com_pos(self.q)[1] > self.reward_fn.cfg.max_height\n            or self.physics.get_com_pos(self.q)[1] < self.pcfg.min_com_height"
)

# Replace trajectory state usages which tracked COM instead of foot
code = code.replace("y_ideal, dy_ideal = self.get_trajectory_state(self.q[0])",
                    "y_ideal, dy_ideal = self.get_trajectory_state(self.physics.get_com_pos(self.q)[0])")
code = code.replace("y_error = self.q[1] - y_ideal", "y_error = self.physics.get_com_pos(self.q)[1] - y_ideal")
code = code.replace("dy_error = self.dq[1] - dy_ideal", "dy_error = self.physics.get_com_vel(self.q, self.dq)[1] - dy_ideal")

with open('/Users/yunzhuorui/Jumping-Robot/Hopper_simu/Quadhopper/env.py', 'w') as f:
    f.write(code)

