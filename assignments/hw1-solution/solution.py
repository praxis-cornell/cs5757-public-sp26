from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxlie import SO3


# ---- Question 2: Spline-based Trajectory Generation ---- #
def phi(tau: ArrayLike) -> Array:
    """Basis vector for cubic polynomial at time tau."""
    return jnp.array([1.0, tau, tau**2, tau**3])


def dphi(tau: ArrayLike) -> Array:
    """Derivative of basis vector at time tau."""
    return jnp.array([0.0, 1.0, 2 * tau, 3 * tau**2])


def ddphi(tau: ArrayLike) -> Array:
    """Second derivative of basis vector at time tau."""
    return jnp.array([0.0, 0.0, 2.0, 6 * tau])


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
    """Builds constraint matrices for 1D spline problem.

    Args:
        x0: Initial position (scalar)
        gates: Gate positions, shape (2,)
        dt: Segment durations, shape (2,)

    Returns:
        A: Constraint matrix, shape (7, 8)
        b: Constraint vector, shape (7,)
    """
    A = jnp.stack(
        [
            jnp.concatenate([phi(0.0), jnp.zeros(4)]),  # x_0(0) = x0
            jnp.concatenate([phi(dt[0]), jnp.zeros(4)]),  # x_0(dt_0) = g0
            jnp.concatenate([jnp.zeros(4), phi(dt[1])]),  # x_1(dt_1) = g1
            jnp.concatenate([phi(dt[0]), -phi(0.0)]),  # continuity
            jnp.concatenate([dphi(dt[0]), -dphi(0.0)]),  # smoothness
            jnp.concatenate([dphi(0.0), jnp.zeros(4)]),  # dx_0(0) = 0
            jnp.concatenate([jnp.zeros(4), dphi(dt[1])]),  # dx_1(dt_1) = 0
        ]
    )

    b = jnp.array([x0, gates[0], gates[1], 0.0, 0.0, 0.0, 0.0])

    return A, b


def solve_spline_1d(x0: Array, gates: Array, dt: Array) -> Array:
    """Solves for the spline coefficients in 1D.

    Args:
        x0: Initial position (scalar)
        gates: Gate positions, shape (2,)
        dt: Segment durations, shape (2,)

    Returns:
        coeffs: Spline coefficients, shape (8,)
    """
    A, b = build_constraints(x0, gates, dt)
    Q = build_Q(dt)

    kkt_lhs = jnp.block([[2 * Q, A.T], [A, jnp.zeros((7, 7))]])
    kkt_rhs = jnp.concatenate([jnp.zeros(8), b])

    coeffs = jnp.linalg.solve(kkt_lhs, kkt_rhs)[:8]
    return coeffs


def evaluate_spline_1d(
    coeffs: Array, t: ArrayLike, dt: Array
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Evaluates the spline at time t."""
    t1 = dt[0]
    in_seg0 = t < t1
    tau = jnp.where(in_seg0, t, t - t1)
    c = jnp.where(in_seg0, coeffs[:4], coeffs[4:])
    return phi(tau) @ c, dphi(tau) @ c, ddphi(tau) @ c


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
    fW, tauB = control[0], control[1:]

    ddposW = -jnp.array([0.0, 0.0, GRAVITY]) + (fW / M_DRONE) * (
        state.R @ jnp.array([0.0, 0.0, 1.0])
    )

    domegaB = jnp.linalg.inv(INERTIA) @ (
        tauB - jnp.cross(state.omegaB, INERTIA @ state.omegaB)
    )
    return state.velW, state.omegaB, ddposW, domegaB


def state_add(
    state: DroneState, delta: tuple[Array, Array, Array, Array]
) -> DroneState:
    """Adds a delta to the drone state."""
    d_pos, omega, d_vel, d_omega = delta

    return DroneState(
        posW=state.posW + d_pos,
        R=state.R @ SO3.exp(omega).as_matrix(),
        velW=state.velW + d_vel,
        omegaB=state.omegaB + d_omega,
    )


def rk4_step(state: DroneState, control: Array, dt: float) -> DroneState:
    """Performs a single RK4 integration step on the manifold."""

    # 1. k1 calculation
    k1 = f_drone(state, control)

    # 2. k2 calculation
    k1_half = jax.tree.map(lambda x: x * (dt / 2), k1)
    k2 = f_drone(state_add(state, k1_half), control)

    # 3. k3 calculation
    k2_half = jax.tree.map(lambda x: x * (dt / 2), k2)
    k3 = f_drone(state_add(state, k2_half), control)

    # 4. k4 calculation
    k3_full = jax.tree.map(lambda x: x * dt, k3)
    k4 = f_drone(state_add(state, k3_full), control)

    # 5. Full RK4 step.
    delta = jax.tree.map(
        lambda a, b, c, d: (dt / 6) * (a + 2 * b + 2 * c + d), k1, k2, k3, k4
    )

    return state_add(state, delta)


def compute_attitude_control(R: Array, R_d: Array, w: Array) -> Array:
    """Computes attitude control torques."""
    e_R = SO3.from_matrix(R_d.T @ R).log()
    tau = -KR * e_R - KW * w + jnp.cross(w, INERTIA @ w)
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
