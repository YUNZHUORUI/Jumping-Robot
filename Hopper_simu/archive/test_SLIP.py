# SLIP simulation: flight -> stance -> flight (takeoff)
# Integrates flight until touchdown (given touchdown leg angle phi_td),
# then integrates stance dynamics with a linear spring (stiffness k)
# until liftoff. Reports takeoff speed and angle and plots results.

import numpy as np
from math import sin, cos, radians, atan2, sqrt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pickle
from pandas import DataFrame

# -------------------------
# Parameters (edit as needed)
# -------------------------
m = 1.0  # mass (kg)
k = 2000.0  # leg spring stiffness (N/m)
L0 = 1.0  # rest length of leg (m)
g = 9.81  # gravity (m/s^2)
phi_td_deg = -10.0  # touchdown leg angle measured from vertical (deg). forward positive.
phi_td = radians(phi_td_deg)

# Initial flight state (before touchdown): (x, y, vx, vy)
x0 = 0.0
y0 = 1.5  # must be above L0*cos(phi_td) for touchdown to occur later
vx0 = 2.0
vy0 = -1.0  # negative means descending

# Simulation time windows
t_flight_max = 2.0  # max time to integrate flight phase (s)
t_stance_max = 1.0  # max time for stance (s)


# -------------------------
# Flight phase dynamics
# -------------------------
def flight_dynamics(t, state):
    x, y, vx, vy = state
    return [vx, vy, 0.0, -g]


# Touchdown height threshold (leg at rest length L0 and angle phi_td)
y_td_threshold = L0 * cos(phi_td)


def touchdown_event(t, state):
    # zero-crossing when mass reaches the touchdown leg geometry height
    x, y, vx, vy = state
    return y - y_td_threshold


touchdown_event.terminal = True
touchdown_event.direction = -1  # detect when crossing downward

# integrate flight until touchdown
state0 = [x0, y0, vx0, vy0]
sol_flight = solve_ivp(
    flight_dynamics, [0, t_flight_max], state0,
    events=touchdown_event, max_step=0.005, rtol=1e-8, atol=1e-10
)

if sol_flight.status != 1 and sol_flight.t_events[0].size == 0:
    raise RuntimeError(
        "Touchdown event not detected in flight integration. Try larger t_flight_max or different initial conditions.")

t_td = float(sol_flight.t_events[0][0])
x_td, y_td, vx_td, vy_td = sol_flight.y_events[0][0]
print(f"Touchdown time: {t_td:.6f} s")
print(f"Touchdown state: x={x_td:.6f}, y={y_td:.6f}, vx={vx_td:.6f}, vy={vy_td:.6f}")

# Compute foot location assuming foot sits on ground y=0 and leg of length L0 at angle phi_td
# x_foot = x_td - L0 * sin(phi_td)  (if phi positive forward, foot is behind mass)
x_foot = x_td - L0 * sin(phi_td)
y_foot = 0.0


# -------------------------
# Stance dynamics (spring leg)
# -------------------------
def stance_dynamics(t, state):
    x, y, vx, vy = state
    dx = x - x_foot
    dy = y - y_foot
    r = np.hypot(dx, dy)  # current leg length
    if r == 0:
        ux, uy = 0.0, 0.0
    else:
        ux, uy = dx / r, dy / r  # unit vector from foot to mass

    # linear spring: force = -k*(r - L0), acts along radial direction
    F_spring = -k * (r - L0)
    ax = (F_spring / m) * ux
    ay = (F_spring / m) * uy - g
    return [vx, vy, ax, ay]


def liftoff_event(t, state):
    # liftoff when leg length reaches L0 from below (extension to rest length)
    x, y, vx, vy = state
    dx = x - x_foot
    dy = y - y_foot
    r = np.hypot(dx, dy)
    return r - L0


liftoff_event.terminal = True
liftoff_event.direction = 1  # detect crossing upward through L0

# initial condition at stance start is the touchdown state
state_td = [x_td, y_td, vx_td, vy_td]
sol_stance = solve_ivp(
    stance_dynamics, [t_td, t_td + t_stance_max], state_td,
    events=liftoff_event, max_step=0.0005, rtol=1e-8, atol=1e-10
)

# If liftoff event not detected, use last integration point as approximation
if sol_stance.status != 1 and sol_stance.t_events[0].size == 0:
    print(
        "Warning: liftoff event not detected within stance integration time. "
        "Using end of stance integration as liftoff approximation.")
    t_lo = sol_stance.t[-1]
    x_lo, y_lo, vx_lo, vy_lo = sol_stance.y[:, -1]
else:
    t_lo = float(sol_stance.t_events[0][0])
    x_lo, y_lo, vx_lo, vy_lo = sol_stance.y_events[0][0]

print(f"Liftoff time: {t_lo:.6f} s")
print(f"Liftoff state: x={x_lo:.6f}, y={y_lo:.6f}, vx={vx_lo:.6f}, vy={vy_lo:.6f}")

# -------------------------
# Take-off results
# -------------------------
v_takeoff = np.hypot(vx_lo, vy_lo)
theta_takeoff = atan2(vy_lo, vx_lo)  # radians

print(f"Take-off speed v^+ = {v_takeoff:.6f} m/s")
print(f"Take-off angle theta^+ = {np.degrees(theta_takeoff):.6f} deg")

# -------------------------
# Plot trajectory and velocities
# -------------------------
plt.figure(figsize=(4, 8))
plt.plot(sol_flight.y[0, :], sol_flight.y[1, :], label='flight')
plt.plot(sol_stance.y[0, :], sol_stance.y[1, :], label='stance')
for i in range(len(sol_stance.y[0, :])):
    if i % 20 == 0:
        plt.plot([sol_stance.y[0, i], x_foot], [sol_stance.y[1, i], y_foot], 'b')
plt.scatter([x_foot], [y_foot], marker='x', label='foot')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.tight_layout()

# velocities vs time (concatenate flight and stance while avoiding duplicate touchdown time)
t_all = np.hstack([sol_flight.t, sol_stance.t[1:]])
vx_all = np.hstack([sol_flight.y[2, :], sol_stance.y[2, 1:]])
vy_all = np.hstack([sol_flight.y[3, :], sol_stance.y[3, 1:]])
x_all = np.hstack([sol_flight.y[0, :], sol_stance.y[0, 1:]])
y_all = np.hstack([sol_flight.y[1, :], sol_stance.y[1, 1:]])

plt.figure(figsize=(8, 9))
plt.subplot(311)
plt.plot(t_all, vx_all, label='vx')
plt.plot(t_all, vy_all, label='vy')
plt.axvline(t_td, linestyle='--', label='touchdown')
plt.axvline(t_lo, linestyle='--', label='liftoff')
plt.title('Velocity components vs time')
plt.xlabel('time (s)')
plt.ylabel('velocity (m/s)')
plt.legend()
plt.subplot(312)
plt.plot(t_all, x_all, label='vx')
plt.plot(t_all, y_all, label='vy')
plt.axvline(t_td, linestyle='--', label='touchdown')
plt.axvline(t_lo, linestyle='--', label='liftoff')
plt.title('Disp components vs time')
plt.xlabel('time (s)')
plt.ylabel('velocity (m/s)')
plt.subplot(313)
plt.plot(t_all, np.ones_like(t_all), 'k--')
plt.plot(t_all, np.sqrt((x_all - x_foot)**2+(y_all - y_foot)**2), 'b')
plt.tight_layout()
