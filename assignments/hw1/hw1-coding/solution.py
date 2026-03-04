from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxlie import SO3


# ---- Question 2: Spline-based Trajectory Generation ---- #


def build_Q(dt: jnp.ndarray) -> jnp.ndarray:
    """Builds the cost matrix Q for the spline trajectory optimization."""

    def Q_block(dt: jnp.ndarray) -> jnp.ndarray:
        Q_t = jnp.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 4 * dt, 6 * dt**2],
                [0, 0, 6 * dt**2, 12 * dt**3],
            ]
        )
        return Q_t

    Q1 = Q_block(dt[0])
    Q2 = Q_block(dt[1])

    return jax.scipy.linalg.block_diag(Q1, Q2)

def phi(t: float, dt: Array) -> Array:
    """Builds the basis functions for the spline."""
    return jnp.array([1, t, t**2, t**3])

def phi_dot(t: float, dt: Array) -> Array:
    """Builds the time derivative of the basis functions for the spline."""
    return jnp.array([0, 1, 2 * t, 3 * t**2])

def phi_dot_dot(t: float, dt: Array) -> Array:
    """Builds the second time derivative of the basis functions for the spline."""
    return jnp.array([0, 0, 2, 6 * t])

def build_constraints(x0: ArrayLike, gates: Array, dt: Array) -> tuple[Array, Array]:
    """Builds constraint matrices for 1D spline problem. """

    zeros = jnp.zeros(4)
    dt0, dt1 = dt[0], dt[1]

    a_init = jnp.concatenate([phi(0.0, dt), zeros])
    a_gate0 = jnp.concatenate([phi(dt0, dt), zeros])
    a_gate1 = jnp.concatenate([zeros, phi(dt1, dt)])
    a_cont = jnp.concatenate([phi(dt0, dt), -phi(0.0, dt)])
    a_smooth = jnp.concatenate([phi_dot(dt0, dt), -phi_dot(0.0, dt)])
    a_beg = jnp.concatenate([phi_dot(0.0, dt), zeros])
    a_end = jnp.concatenate([zeros, phi_dot(dt1, dt)])

    A = jnp.stack([a_init, a_gate0, a_gate1, a_cont, a_smooth, a_beg, a_end])
    b = jnp.array([x0, gates[0], gates[1], 0.0, 0.0, 0.0, 0.0])

    return A, b


def solve_spline_1d(x0: ArrayLike, gates: Array, dt: Array) -> Array:
    """Solves for the spline coefficients in 1D.    """
    Q = build_Q(dt)
    A, b = build_constraints(x0, gates, dt)

    zeros_cc = jnp.zeros((A.shape[0], A.shape[0]))
    KKT = jnp.block([[2.0 * Q, A.T], [A, zeros_cc]])
    rhs = jnp.concatenate([jnp.zeros(8), b])

    sol = jnp.linalg.solve(KKT, rhs)
    return sol[:8]


def evaluate_spline_1d(
    coeffs: Array, t: float, dt: Array
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Evaluates the spline at time t."""
    dt0 = dt[0]
    c0 = coeffs[:4]
    c1 = coeffs[4:]

    seg = (t > dt0).astype(jnp.int32)

    pos = jnp.where(seg == 0, phi(t, dt) @ c0, phi(t - dt0, dt) @ c1)
    vel = jnp.where(seg == 0, phi_dot(t, dt) @ c0, phi_dot(t - dt0, dt) @ c1)
    acc = jnp.where(seg == 0, phi_dot_dot(t, dt) @ c0, phi_dot_dot(t - dt0, dt) @ c1)

    return pos, vel, acc


# ---- Question 3: Drone Dynamics and Simulation ---- #

# Define physical constants.
M_DRONE = 1.0  # Mass of the drone (kg)
GRAVITY = 9.81  # Gravitational acceleration (m/s^2)
INERTIA = jnp.diag(jnp.array([0.02, 0.02, 0.04]))  # Inertia matrix (kg*m^2)

# Controller parameters.
KP = 10.0
KV = 10.0
KR = 4.0
KW = 1.0


class DroneState(NamedTuple):
    posW: Array
    R: Array
    velW: Array
    omegaB: Array


def f_drone(state: DroneState, control: Array) -> tuple[Array, Array, Array, Array]:
    """Computes the time derivative of the drone state."""
    p, R, dp, w = state
    f = control[0]
    tau = control[1:]

    e3 = jnp.array([0.0, 0.0, 1.0])

    dvelW = (-GRAVITY * e3 + (R @ e3) * f) / M_DRONE

    Jw = INERTIA @ w
    dw = jnp.linalg.solve(INERTIA, -jnp.cross(w, Jw) + tau)

    return dp, w, dvelW, dw


def state_add(
    state: DroneState, delta: tuple[Array, Array, Array, Array]
) -> DroneState:
    """Adds a delta to the drone state."""
    p, R, dp, w = state
    dp_pos, dR, dvel, dw = delta

    R_next = R @ SO3.exp(dR).as_matrix()

    return DroneState(
        posW=p + dp_pos,
        R=R_next,
        velW=dp + dvel,
        omegaB=w + dw,
    )

def scale(delta: tuple[Array, Array, Array, Array], s: float) -> tuple[Array, Array, Array, Array]:
    return tuple(s * d for d in delta)

def rk4_step(state: DroneState, control: Array, dt: float) -> DroneState:
    """Performs a single RK4 integration step on the manifold."""

    k1 = f_drone(state, control)
    k2 = f_drone(state_add(state, scale(k1, dt / 2.0)), control)
    k3 = f_drone(state_add(state, scale(k2, dt / 2.0)), control)
    k4 = f_drone(state_add(state, scale(k3, dt)), control)

    incr = tuple(
        (dt / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
        for d1, d2, d3, d4 in zip(k1, k2, k3, k4)
    )
    return state_add(state, incr)


def compute_attitude_control(R: Array, R_d: Array, w: Array) -> Array:
    """Computes attitude control torques."""
    eR = SO3.from_matrix(R_d.T @ R).log()
    tau = -KR * eR - KW * w + jnp.cross(w, INERTIA @ w)
    return tau


def compute_outer_loop(
    state: DroneState, p_des: Array, dp_des: Array, ddp_des: Array
) -> tuple[Array, Array]:
    """Computes thrust and desired orientation from position error. PROVIDED."""
    p, dp = state.posW, state.velW

    ddp_cmd = ddp_des + KV * (dp_des - dp) + KP * (p_des - p)
    f_des = M_DRONE * (ddp_cmd + GRAVITY * jnp.array([0.0, 0.0, 1.0]))
    f = jnp.linalg.norm(f_des)
    z_d = f_des / f

    x_c = jnp.array([1.0, 0.0, 0.0])
    y_d = jnp.cross(z_d, x_c)
    y_d = y_d / jnp.linalg.norm(y_d)
    x_d = jnp.cross(y_d, z_d)
    R_d = jnp.column_stack([x_d, y_d, z_d])

    return f, R_d


def simulate_trajectory(
    coeffs: Array,
    spline_dt: Array,
    sim_dt: float,
    initial_state: DroneState,
) -> DroneState:
    """Simulates drone tracking a spline trajectory.

    Args:
        coeffs: Spline coefficients, shape (3, 8)
        sim_dt: Simulation timestep
        initial_state: Initial drone state

    Returns:
        states: Stacked drone states, each field has shape (n_steps, ...)
    """

    # Time grid
    T = jnp.sum(spline_dt)
    n_steps = jnp.int32(T / sim_dt)
    ts = jnp.linspace(0.0, T, n_steps)

    def step(state: DroneState, t: ArrayLike) -> tuple[DroneState, DroneState]:
        # Evaluate desired trajectory at time t (vmap over 3 spatial dims)
        p_des, dp_des, ddp_des = jax.vmap(evaluate_spline_1d, in_axes=(0, None, None))(
            coeffs, t, spline_dt
        )

        # Compute control
        f, R_d = compute_outer_loop(state, p_des, dp_des, ddp_des)
        tau = compute_attitude_control(state.R, R_d, state.omegaB)
        u = jnp.concatenate([jnp.array([f]), tau])

        # Step dynamics
        next_state = rk4_step(state, u, sim_dt)

        return next_state, state

    _, states = jax.lax.scan(step, initial_state, ts)
    return states
