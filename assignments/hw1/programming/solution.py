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


def build_constraints(x0: ArrayLike, gates: Array, dt: Array) -> tuple[Array, Array]:
    """Builds constraint matrices for 1D spline problem. """
    raise NotImplementedError("Not implemented")


def solve_spline_1d(x0: ArrayLike, gates: Array, dt: Array) -> Array:
    """Solves for the spline coefficients in 1D.    """
    raise NotImplementedError("Not implemented")


def evaluate_spline_1d(
    coeffs: Array, t: float, dt: Array
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Evaluates the spline at time t."""
    raise NotImplementedError("Not implemented")


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
    raise NotImplementedError("Not implemented")


def state_add(
    state: DroneState, delta: tuple[Array, Array, Array, Array]
) -> DroneState:
    """Adds a delta to the drone state."""
    raise NotImplementedError("Not implemented")


def rk4_step(state: DroneState, control: Array, dt: float) -> DroneState:
    """Performs a single RK4 integration step on the manifold."""
    raise NotImplementedError("Not implemented")


def compute_attitude_control(R: Array, R_d: Array, w: Array) -> Array:
    """Computes attitude control torques."""
    raise NotImplementedError("Not implemented")


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
