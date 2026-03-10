"""
Problem 3.2 — SCP for obstacle avoidance (drone)

Navigate a drone through spherical obstacles using sequential convex
programming with linearized signed-distance constraints.

CS 5757, Cornell University
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from solution import (
    f_drone,
    signed_distance,
    affinize_dynamics,
    affinize_constraint,
    rollout,
    build_drone_scp,
    solve_scp,
    OBSTACLE_CENTERS,
    OBSTACLE_RADII,
    DT_DRONE,
)

import cvxpy as cvx

jax.config.update("jax_enable_x64", True)

nx = 6
nu = 3

# Cost matrices and goal
DRONE_GOAL = jnp.array([2.0, 2.0, 1.0, 0.0, 0.0, 0.0])
Q_DRONE = jnp.diag(jnp.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]))
Q_TERMINAL_DRONE = jnp.diag(jnp.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]))
R_DRONE = jnp.diag(jnp.array([0.1, 0.1, 0.1]))

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

# Problem setup
T = 20
N_scp = 10
s0 = jnp.array([-2.0, -2.0, 0.5, 0.0, 0.0, 0.0])

# Initialize with zero controls
u_init = jnp.zeros((T, nu))
s_init = rollout(f_drone, s0, u_init)

# Build and solve
print("Running SCP for drone obstacle avoidance ...")
prob, s_var, u_var, params = build_drone_scp(
    T=T,
    nx=nx,
    nu=nu,
    Q=Q_DRONE,
    R=R_DRONE,
    Q_T=Q_TERMINAL_DRONE,
    s_goal=DRONE_GOAL,
)
s_sol, u_sol = solve_scp(
    prob,
    s_var,
    u_var,
    params,
    f_drone,
    signed_distance,
    s0,
    s_init,
    u_init,
)

# Evaluate
sd = jax.vmap(signed_distance)(s_sol)
print(f"Min signed distance to obstacle: {sd.min():.4f} m")

# ── Plot 3D trajectory ──

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

u_sphere = np.linspace(0, 2 * np.pi, 20)
v_sphere = np.linspace(0, np.pi, 15)
for center, radius in zip(np.array(OBSTACLE_CENTERS), np.array(OBSTACLE_RADII)):
    x = center[0] + radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y = center[1] + radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z = center[2] + radius * np.outer(np.ones_like(u_sphere), np.cos(v_sphere))
    ax.plot_surface(x, y, z, alpha=0.15, color="red")

s_np = np.array(s_sol)
ax.plot(s_np[:, 0], s_np[:, 1], s_np[:, 2], "b.-", linewidth=2, label="SCP trajectory")
ax.scatter(*s_np[0, :3], color="green", s=100, zorder=5, label="Start")
ax.scatter(
    *np.array(DRONE_GOAL[:3]), color="gold", s=100, marker="*", zorder=5, label="Goal"
)
ax.set(xlabel="x", ylabel="y", zlabel="z", title="SCP Obstacle Avoidance")
ax.legend()
plt.tight_layout()
plt.savefig("problem2_scp_trajectory.pdf", bbox_inches="tight")
plt.show()
