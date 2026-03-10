"""
Problem 4 — MPC under model mismatch (acrobot)

Plan with an RK4 acrobot model using iLQR, then track the plan on a
MuJoCo acrobot using receding-horizon SCP.

CS 5757, Cornell University
"""

import time
from typing import Callable

import cvxpy as cvx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from solution import (
    acrobot,
    discretize,
    rollout,
    ilqr,
    affinize_dynamics,
    quadratic_cost,
    quadratic_terminal_cost,
    build_acrobot_scp,
)


# ──────────────────────────────────────────────────────────────────────────────
# MuJoCo "reality"
# ──────────────────────────────────────────────────────────────────────────────

ACROBOT_XML = """
<mujoco>
  <option timestep="0.01" integrator="RK4"/>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <body name="link1" pos="0 0 0">
      <joint name="q1" type="hinge" axis="0 1 0" damping="0.1"/>
      <inertial pos="0 0 0.5" mass="1.0" diaginertia="0.75 0.75 0.01"/>
      <geom type="capsule" fromto="0 0 0 0 0 1.0" size="0.05" rgba="0.8 0.2 0.2 1"/>
      <body name="link2" pos="0 0 1.0">
        <joint name="q2" type="hinge" axis="0 1 0" damping="0.1"/>
        <inertial pos="0 0 0.5" mass="1.0" diaginertia="0.75 0.75 0.01"/>
        <geom type="capsule" fromto="0 0 0 0 0 1.0" size="0.05" rgba="0.2 0.2 0.8 1"/>
        <site name="tip" pos="0 0 1.0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="q2" ctrlrange="-20 20"/>
  </actuator>
</mujoco>
"""


def make_mujoco_env():
    model = mujoco.MjModel.from_xml_string(ACROBOT_XML)
    return model, mujoco.MjData(model)


def mujoco_step(model, data, state, control, n_substeps=5):
    """Step MuJoCo forward by n_substeps * model.opt.timestep."""
    data.qpos[:] = state[:2]
    data.qvel[:] = state[2:]
    data.ctrl[:] = np.clip(np.atleast_1d(control), -20, 20)
    for _ in range(n_substeps):
        mujoco.mj_step(model, data)
    return np.concatenate([data.qpos.copy(), data.qvel.copy()])


def mujoco_rollout(model, data, s0, controls):
    """Roll out open-loop controls on MuJoCo."""
    states = np.zeros((len(controls) + 1, 4))
    states[0] = s0
    for k in range(len(controls)):
        states[k + 1] = mujoco_step(model, data, states[k], controls[k])
    return states


# ──────────────────────────────────────────────────────────────────────────────
# Receding-horizon MPC via SCP
# ──────────────────────────────────────────────────────────────────────────────


def mpc_loop(
    fd: Callable,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    s0: np.ndarray,
    u_plan: np.ndarray,
    T: int,
    replan_steps: int,
    Q: np.ndarray,
    R: np.ndarray,
    Q_T: np.ndarray,
    N_scp: int = 5,
):
    """Run receding-horizon SCP-based MPC on MuJoCo.

    Arguments
    ---------
    fd           : JAX discrete-time dynamics (planning model)
    model, data  : MuJoCo model/data (execution model)
    s0           : initial state
    u_plan       : warm-start control sequence, shape (T_total, nu)
    T            : MPC horizon
    replan_steps : steps to execute before replanning
    Q, R, Q_T    : cost matrices
    N_scp        : inner SCP iterations per replan
    """
    nx, nu = 4, 1
    total_steps = len(u_plan)
    s_goal = np.array([0, 0, 0, 0])

    # Compile the CVXPY problem once
    prob, s_vars, u_vars, params = build_acrobot_scp(
        T, nx, nu, np.array(Q), np.array(R), np.array(Q_T), trust_region=1e1
    )
    s0_p, A_p, B_p, c_p, s_bar_p, u_bar_p = params

    s_cur = np.copy(s0)
    states = [s_cur]
    controls = []
    u_bar = np.array(u_plan[:T])

    for t in range(0, total_steps, replan_steps):
        # Warm-start: shift previous solution forward
        if t > 0:
            u_bar = np.roll(u_bar, -replan_steps, axis=0)
            u_bar[-replan_steps:] = 0.0

        # Initialize s_bar
        if t == 0:
            s_bar = np.linspace(s_cur, s_goal, T + 1)
        else:
            s_bar = np.roll(s_vars.value, -replan_steps, axis=0)
            s_bar[-replan_steps:] = s_goal

        s0_p.value = s_cur

        # Inner SCP iterations
        for _ in range(N_scp):
            A, B, c = jax.vmap(affinize_dynamics, in_axes=(None, 0, 0))(
                fd, s_bar[:-1], u_bar
            )
            A_p.value = np.array(A)
            B_p.value = np.array(B)
            c_p.value = np.array(c)
            s_bar_p.value = np.array(s_bar)
            u_bar_p.value = np.array(u_bar)

            prob.solve(solver=cvx.CLARABEL, warm_start=True)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                break

            s_bar = s_vars.value
            u_bar = u_vars.value

        # Execute on MuJoCo
        for k in range(replan_steps):
            s_cur = mujoco_step(model, data, s_cur, u_bar[k])
            states.append(s_cur)
            controls.append(u_bar[k])

    return np.array(states), np.array(controls)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

# Problem setup
dt = 0.05
T_total = 600  # 30 s total
s0 = np.array([np.pi, 0.0, 0.0, 0.0])

Q = jnp.diag(jnp.array([0.1, 0.01, 0.01, 0.01]))
R = jnp.diag(jnp.array([0.01]))
Q_T = jnp.diag(jnp.array([100.0, 100.0, 1.0, 1.0]))

cost = lambda s, a: quadratic_cost(s, a, Q, R)
cost_T = lambda s: quadratic_terminal_cost(s, Q_T)

fd = jax.jit(discretize(acrobot, dt))
model, data = make_mujoco_env()

# ── (a) Plan with iLQR ──

print("Planning global trajectory with iLQR ... ", end="", flush=True)
start = time.time()
u_init = jnp.zeros((T_total, 1))
s_init = rollout(fd, jnp.array(s0), u_init)
s_ilqr, u_ilqr, _, _ = ilqr(fd, cost, cost_T, jnp.array(s0), s_init, u_init)
print(f"done! ({time.time() - start:.2f} s)")

# ── (b) Open-loop execution on MuJoCo ──

print("Executing open-loop on MuJoCo ... ", end="", flush=True)
s_ol = mujoco_rollout(model, data, s0, np.array(u_ilqr))
print("done!")

# ── (c) Receding-horizon SCP MPC on MuJoCo ──
print("Running SCP MPC ... ", end="", flush=True)
start = time.time()

# Run MPC from "cold" initialization u_init.
s_mpc, u_mpc = mpc_loop(
    fd, model, data, s0, u_init, T=25, replan_steps=1, Q=Q, R=R, Q_T=Q_T, N_scp=1
)
print(f"done! ({time.time() - start:.2f} s)")

# ── Evaluate ──

final_cost_ilqr = cost_T(s_ilqr[-1]) + sum(
    cost(s_ilqr[k], u_ilqr[k]) for k in range(T_total)
)
final_cost_ol = cost_T(s_ol[-1]) + sum(cost(s_ol[k], u_ilqr[k]) for k in range(T_total))
final_cost_mpc = cost_T(s_mpc[-1]) + sum(
    cost(s_mpc[k], u_mpc[k]) for k in range(len(u_mpc))
)
print(f"Final cost (iLQR plan): {final_cost_ilqr:.2f}")
print(f"Final cost (open-loop):    {final_cost_ol:.2f}")
print(f"Final cost (MPC):          {final_cost_mpc:.2f}")

# ── Plot ──

labels = (r"$\theta_1$", r"$\theta_2$", r"$\dot\theta_1$", r"$\dot\theta_2$")
t_plan = np.arange(T_total + 1) * dt
t_mpc = np.arange(len(s_mpc)) * dt

fig, axes = plt.subplots(1, 4, figsize=(16, 3), dpi=150)
plt.subplots_adjust(wspace=0.4)
for i in range(4):
    axes[i].plot(t_plan, s_ilqr[:, i], "k--", alpha=0.5, label="iLQR plan")
    axes[i].plot(t_plan, s_ol[:, i], "r", alpha=0.7, label="Open-loop (MuJoCo)")
    axes[i].plot(t_mpc, s_mpc[:, i], "b", alpha=0.7, label="SCP MPC (MuJoCo)")
    axes[i].set(xlabel="Time [s]", ylabel=labels[i])
    if i == 0:
        axes[i].legend(fontsize=8)
plt.savefig("problem4_mpc_comparison.pdf", bbox_inches="tight")
plt.show()
