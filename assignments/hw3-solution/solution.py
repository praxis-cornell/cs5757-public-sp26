"""
Homework 3: Optimal Control — Solution File

    Problem 3.1(b) — linearize
    Problem 3.1(c) — iLQR backward step
    Problem 3.1(d) — simulate
    Problem 3.2(a) — signed_distance
    Problem 3.2(d) — affinize (dynamics + constraints)
    Problem 3.2(e) — build_drone_scp (obstacle constraints)
    Problem 3.2(f) — solve_scp
    Problem 3.4(c) — build_acrobot_scp (trust regions + control bounds)

CS 5757, Cornell University
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cvx


# ══════════════════════════════════════════════════════════════════════════════
# Dynamics
# ══════════════════════════════════════════════════════════════════════════════

# Acrobot parameters
_m1, _m2 = 1.0, 1.0
_l1, _l2 = 1.0, 1.0
_lc1, _lc2 = 0.5, 0.5
_I1, _I2 = 1.0, 1.0
_g = 9.81


def acrobot(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Acrobot continuous-time dynamics.

    State: x = [θ₁, θ₂, dθ₁, dθ₂].  Torque applied at the elbow.
    """
    θ1, θ2, dθ1, dθ2 = x
    sinθ2, cosθ2 = jnp.sin(θ2), jnp.cos(θ2)
    sinθ1, sinθ12 = jnp.sin(θ1), jnp.sin(θ1 + θ2)
    u = jnp.squeeze(u)

    h = _m2 * _l1 * _lc2 * cosθ2
    M = jnp.array([[_I1 + _I2 + _m2 * _l1**2 + 2 * h, _I2 + h], [_I2 + h, _I2]])
    C = _m2 * _l1 * _lc2 * sinθ2 * jnp.array([-2 * dθ1 * dθ2 - dθ2**2, dθ1**2])
    G = -_g * jnp.array(
        [_m1 * _lc1 * sinθ1 + _m2 * (_l1 * sinθ1 + _lc2 * sinθ12), _m2 * _lc2 * sinθ12]
    )
    τ = jnp.array([0.0, u])
    qdd = jnp.linalg.solve(M, τ - C - G)
    return jnp.concatenate([jnp.array([dθ1, dθ2]), qdd])


# Drone parameters
DRONE_RADIUS = 0.1
B_DRAG = 0.1
DT_DRONE = 0.1

OBSTACLE_CENTERS = jnp.array(
    [
        [0.0, 0.0, 1.0],
        [-1.5, -1.5, 1.25],
        [-1.5, -0.75, 0.5],
        [1.5, 1.0, 0.75],
        [1.0, 1.75, 1.0],
    ]
)
OBSTACLE_RADII = jnp.array([1.0, 0.5, 0.5, 0.5, 0.5])
N_OBS = OBSTACLE_CENTERS.shape[0]


def f_drone(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    """Discrete-time drone dynamics: double integrator with quadratic drag."""
    p, v = state[:3], state[3:]
    a = control - B_DRAG * v * jnp.abs(v)
    return jnp.concatenate([p + v * DT_DRONE, v + a * DT_DRONE])


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities (do not change)
# ══════════════════════════════════════════════════════════════════════════════


def discretize(f: Callable, dt: float):
    """Discretize continuous-time dynamics `f` via RK4 integration."""

    def integrator(s: jnp.ndarray, u: jnp.ndarray, dt: float = dt):
        k1 = dt * f(s, u)
        k2 = dt * f(s + k1 / 2, u)
        k3 = dt * f(s + k2 / 2, u)
        k4 = dt * f(s + k3, u)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return integrator


def rollout(fd: Callable, x0: jnp.ndarray, us: jnp.ndarray):
    """Roll out discrete dynamics.  Returns states of shape (T+1, nx)."""

    def step(x: jnp.ndarray, u: jnp.ndarray):
        x_next = fd(x, u)
        return x_next, x_next

    _, xs = jax.lax.scan(step, x0, us)
    return jnp.concatenate([x0[None], xs])


def quadratic_cost(s: jnp.ndarray, a: jnp.ndarray, Q: jnp.ndarray, R: jnp.ndarray):
    """Running cost  ½ s'Q s + ½ a'R a."""
    return 0.5 * s @ Q @ s + 0.5 * a @ R @ a


def quadratic_terminal_cost(s: jnp.ndarray, Q_T: jnp.ndarray):
    """Terminal cost  ½ s'Q_T s."""
    return 0.5 * s @ Q_T @ s


# ══════════════════════════════════════════════════════════════════════════════
# Problem 3.1: LQR with Stochastic Noise
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# Problem 3.1(b): Linearize dynamics and quadraticize cost
# ──────────────────────────────────────────────────────────────────────────────


def linearize(f: Callable, cost: Callable, s: jnp.ndarray, a: jnp.ndarray):
    """Linearize dynamics and quadraticize cost around (s, a).

    Returns
    -------
    Q, R, S : cost Hessians (state, control, cross-term ∂²c/∂x∂u)
    q, r    : cost gradients (state, control)
    A, B    : dynamics Jacobians
    """
    # Problem 3.1(b) ##########################################################
    # SOLUTION — STRIP FOR RELEASE
    q = jax.grad(cost, 0)(s, a)
    r = jax.grad(cost, 1)(s, a)
    Q = jax.hessian(cost, 0)(s, a)
    R = jax.hessian(cost, 1)(s, a)
    S = jax.jacfwd(jax.grad(cost, 0), 1)(s, a)
    A = jax.jacfwd(f, 0)(s, a)
    B = jax.jacfwd(f, 1)(s, a)
    # END SOLUTION
    return Q, R, S, q, r, A, B


# ──────────────────────────────────────────────────────────────────────────────
# Problem 3.1(c): iLQR backward step
# ──────────────────────────────────────────────────────────────────────────────


def ilqr_backward_step(
    P: jnp.ndarray,
    p: jnp.ndarray,
    Q: jnp.ndarray,
    R: jnp.ndarray,
    S: jnp.ndarray,
    q: jnp.ndarray,
    r: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    reg: float = 1e-3,
):
    """Single backward Riccati step of iLQR.

    Given the value function at time t+1, V(x) ≈ ½ x'Px + p'x, compute the
    optimal feedback gains (K, k) and the updated value function at time t.

    Returns
    -------
    P_new, p_new : updated value function parameters
    K            : feedback gain, shape (nu, nx)
    k            : feedforward term, shape (nu,)
    """
    # Problem 3.1(c) ##########################################################
    # SOLUTION — STRIP FOR RELEASE
    h_x = q + A.T @ p
    h_u = r + B.T @ p
    H_xx = Q + A.T @ P @ A
    H_xu = S + A.T @ P @ B
    H_uu = R + B.T @ P @ B + reg * jnp.eye(B.shape[1])

    K = jnp.linalg.solve(H_uu, H_xu.T)
    k = jnp.linalg.solve(H_uu, h_u)

    P_new = H_xx - H_xu @ K
    p_new = h_x - H_xu @ k
    # END SOLUTION
    return P_new, p_new, K, k


# ──────────────────────────────────────────────────────────────────────────────
# Problem 3.1(d): Simulate with open-loop or closed-loop control
# ──────────────────────────────────────────────────────────────────────────────


def simulate(
    fd: Callable,
    x0: jnp.ndarray,
    x_nom: jnp.ndarray,
    u_nom: jnp.ndarray,
    K_seq: jnp.ndarray,
    k_seq: jnp.ndarray,
    noise_scale: float,
    key: jnp.ndarray,
    use_feedback: bool = False,
):
    """Roll out the system under stochastic noise."""
    T, nu = u_nom.shape
    nx = x0.shape[0]
    keys = jax.random.split(key, T)

    xs = [x0]
    us = []
    x = x0
    for t in range(T):
        # Problem 3.1(d) ######################################################
        # SOLUTION — STRIP FOR RELEASE
        if use_feedback:
            u = u_nom[t] - k_seq[t] - K_seq[t] @ (x - x_nom[t])
        else:
            u = u_nom[t]
        # END SOLUTION

        # Apply dynamics + noise
        x_next = fd(x, u)
        w = noise_scale * jax.random.normal(keys[t], shape=(nx,))
        x = x_next + w

        xs.append(x)
        us.append(u)

    return jnp.stack(xs), jnp.stack(us)


def ilqr_forward(
    f: Callable,
    s0: jnp.ndarray,
    s_nom: jnp.ndarray,
    a_nom: jnp.ndarray,
    k_seq: jnp.ndarray,
    K_seq: jnp.ndarray,
    alpha: float,
):
    """Forward rollout with feedback gains and line-search step alpha."""

    def step(
        s: jnp.ndarray, args: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ):
        s_bar, a_bar, k, K = args
        a = a_bar - alpha * k - K @ (s - s_bar)
        s_next = f(s, a)
        return s_next, (s_next, a)

    _, (s_traj, a_traj) = jax.lax.scan(step, s0, (s_nom[:-1], a_nom, k_seq, K_seq))
    return jnp.concatenate([s0[None], s_traj]), a_traj


def ilqr_backward(
    f: Callable,
    cost: Callable,
    cost_term: Callable,
    s_nom: jnp.ndarray,
    a_nom: jnp.ndarray,
):
    """Full backward pass: linearize along trajectory and solve Riccati."""
    Q, R, S, q, r, A, B = jax.vmap(linearize, in_axes=(None, None, 0, 0))(  # type: ignore
        f, cost, s_nom[:-1], a_nom
    )
    Q_T = jax.hessian(cost_term, 0)(s_nom[-1])
    q_T = jax.grad(cost_term, 0)(s_nom[-1])

    def step(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        args: tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
    ):
        P, p = carry
        P, p, K, k = ilqr_backward_step(P, p, *args)  # type: ignore
        return (P, p), (K, k)

    _, (K_seq, k_seq) = jax.lax.scan(
        step, (Q_T, q_T), (Q, R, S, q, r, A, B), reverse=True
    )
    return K_seq, k_seq


def ilqr(
    f: Callable,
    cost: Callable,
    cost_term: Callable,
    s0: jnp.ndarray,
    s_nom: jnp.ndarray,
    a_nom: jnp.ndarray,
    max_iters: int = 50,
    eps: float = 1e-3,
    verbose: bool = True,
):
    """Iterative LQR with backtracking line search."""
    alphas = jnp.logspace(0, -3, num=11)

    # JIT-compile the backward and forward passes, and the cost function.
    backward_jit = jax.jit(ilqr_backward, static_argnums=(0, 1, 2))
    forward_jit = jax.jit(
        jax.vmap(ilqr_forward, in_axes=(None, None, None, None, None, None, 0)),
        static_argnums=(0,),
    )
    cost_jit = jax.jit(cost)

    prev_cost = jnp.inf
    for it in range(max_iters):
        # Backward pass to compute gains
        K_seq, k_seq = backward_jit(f, cost, cost_term, s_nom, a_nom)

        # Forward pass with line search to find best trajectory
        all_s, all_a = forward_jit(f, s0, s_nom, a_nom, k_seq, K_seq, alphas)

        # Line search: pick the trajectory with the lowest cost
        costs = jax.vmap(
            lambda s, a: jax.vmap(cost_jit)(s[:-1], a).sum() + cost_term(s[-1])
        )(all_s, all_a)

        best = jnp.argmin(costs)
        best_cost = costs[best]
        s_nom, a_nom = all_s[best], all_a[best]

        if verbose:
            print(f"  iLQR iter {it:3d}:  cost = {best_cost:.4f}")
        if jnp.abs(prev_cost - best_cost) < eps:
            break
        prev_cost = best_cost

    return s_nom, a_nom, K_seq, k_seq


# ══════════════════════════════════════════════════════════════════════════════
# Problem 3.2: Sequential Convex Programming
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# Problem 3.2(a): Signed distance
# ──────────────────────────────────────────────────────────────────────────────


def signed_distance(state: jnp.ndarray) -> jnp.ndarray:
    """Signed distance from drone center to each obstacle.

    Returns a vector of length N_OBS.  Positive means outside the obstacle.
    """
    # Problem 3.2(a) ##########################################################
    # SOLUTION — STRIP FOR RELEASE
    return (
        jnp.linalg.norm(state[:3] - OBSTACLE_CENTERS, axis=1)
        - OBSTACLE_RADII
        - DRONE_RADIUS
    )
    # END SOLUTION


# ──────────────────────────────────────────────────────────────────────────────
# Problem 3.2(d): Affinize dynamics
# ──────────────────────────────────────────────────────────────────────────────


def affinize_dynamics(
    f: Callable, s_bar: jnp.ndarray, u_bar: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Affinize f(s, u) ≈ A s + B u + c around (s_bar, u_bar).

    Returns
    -------
    A : Jacobian of f w.r.t. s
    B : Jacobian of f w.r.t. u
    c : affine offset, f(s̄, ū) − A s̄ − B ū
    """
    # Problem 3.2(d) ##########################################################
    # SOLUTION — STRIP FOR RELEASE
    A = jax.jacfwd(f, 0)(s_bar, u_bar)
    B = jax.jacfwd(f, 1)(s_bar, u_bar)
    c = f(s_bar, u_bar) - A @ s_bar - B @ u_bar
    # END SOLUTION
    return A, B, c


# ──────────────────────────────────────────────────────────────────────────────
# Problem 3.2(d): Affinize constraints
# ──────────────────────────────────────────────────────────────────────────────


def affinize_constraint(
    g: Callable, s_bar: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Affinize a constraint g(s) ≈ G s + h around s_bar.

    Returns
    -------
    G : Jacobian of g w.r.t. s
    h : offset, g(s̄) − G s̄
    """
    # Problem 3.2(d) ##########################################################
    # SOLUTION — STRIP FOR RELEASE
    G = jax.jacobian(g)(s_bar)
    h = g(s_bar) - G @ s_bar
    # END SOLUTION
    return G, h


# ──────────────────────────────────────────────────────────────────────────────
# Problem 3.2(e): Drone SCP — obstacle constraints
# ──────────────────────────────────────────────────────────────────────────────


def build_drone_scp(
    T: int,
    nx: int,
    nu: int,
    Q: jnp.ndarray,
    R: jnp.ndarray,
    Q_T: jnp.ndarray,
    s_goal: jnp.ndarray,
):
    """Build the CVXPY problem for drone SCP with obstacle constraints.

    Returns
    -------
    prob   : cvx.Problem
    s, u   : cvx.Variable decision variables
    params : tuple of all cvx.Parameter objects
    """
    n_obs = OBSTACLE_CENTERS.shape[0]
    s = cvx.Variable((T + 1, nx))
    u = cvx.Variable((T, nu))

    # Parameters for affinized dynamics
    s0_p = cvx.Parameter(nx)
    A_p = cvx.Parameter((T, nx, nx))
    B_p = cvx.Parameter((T, nx, nu))
    c_p = cvx.Parameter((T, nx))

    # Parameters for affinized obstacle constraints
    G_p = cvx.Parameter((T + 1, n_obs, nx))
    h_p = cvx.Parameter((T + 1, n_obs))

    # Objective + dynamics constraints
    s_ref = s_goal
    cost = 0.5 * cvx.quad_form(s[T] - s_ref, Q_T)
    constraints = [s[0] == s0_p]
    for t in range(T):
        cost += 0.5 * cvx.quad_form(s[t] - s_ref, Q) + 0.5 * cvx.quad_form(u[t], R)
        constraints += [s[t + 1] == A_p[t] @ s[t] + B_p[t] @ u[t] + c_p[t]]

    # Problem 3.2(e) ##########################################################
    # SOLUTION — STRIP FOR RELEASE
    for t in range(T + 1):
        constraints += [G_p[t] @ s[t] + h_p[t] >= 0]
    # END SOLUTION

    prob = cvx.Problem(cvx.Minimize(cost), constraints)
    return prob, s, u, (s0_p, A_p, B_p, c_p, G_p, h_p)


# ──────────────────────────────────────────────────────────────────────────────
# Problem 3.2(f): SCP solve loop
# ──────────────────────────────────────────────────────────────────────────────


def solve_scp(
    prob: cvx.Problem,
    s_var: cvx.Variable,
    u_var: cvx.Variable,
    params: tuple,
    f: Callable,
    constraint_fn: Callable,
    s0: jnp.ndarray,
    s_nom: jnp.ndarray,
    u_nom: jnp.ndarray,
    max_iters: int = 20,
    tol: float = 1e-3,
    verbose: bool = True,
):
    """Run the SCP loop: linearize, update parameters, solve."""
    s0_p, A_p, B_p, c_p, G_p, h_p = params
    T = u_nom.shape[0]

    prev_cost = np.inf
    for it in range(max_iters):
        # Problem 3.2(f) ######################################################
        # SOLUTION — STRIP FOR RELEASE
        s0_p.value = np.array(s0)

        A, B, c = jax.vmap(affinize_dynamics, in_axes=(None, 0, 0))(
            f, s_nom[:-1], u_nom
        )
        A_p.value = np.array(A)
        B_p.value = np.array(B)
        c_p.value = np.array(c)

        G, h = jax.vmap(affinize_constraint, in_axes=(None, 0))(constraint_fn, s_nom)
        G_p.value = np.array(G)
        h_p.value = np.array(h)
        # END SOLUTION

        prob.solve(solver=cvx.OSQP, warm_start=True, verbose=False)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            if verbose:
                print(f"  SCP iter {it}: solver status = {prob.status}")
            break

        s_nom = np.array(s_var.value)
        u_nom = np.array(u_var.value)

        cur_cost = prob.value
        if verbose:
            print(f"  SCP iter {it:3d}:  cost = {cur_cost:.4f}")
        if abs(prev_cost - cur_cost) < tol:
            break
        prev_cost = cur_cost

    return jnp.array(s_nom), jnp.array(u_nom)


# ══════════════════════════════════════════════════════════════════════════════
# Problem 3.4: Acrobot MPC via SCP
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# Problem 3.4(c): Acrobot SCP — trust regions and control bounds
# ──────────────────────────────────────────────────────────────────────────────


def build_acrobot_scp(
    T: int,
    nx: int,
    nu: int,
    Q: jnp.ndarray,
    R: jnp.ndarray,
    Q_T: jnp.ndarray,
    trust_region: float = 3.0,
    u_bound: float = 20.0,
):
    """Build the CVXPY problem for acrobot SCP with trust regions.

    Returns
    -------
    prob   : cvx.Problem
    s, u   : cvx.Variable decision variables
    params : tuple of all cvx.Parameter objects
    """
    s = cvx.Variable((T + 1, nx))
    u = cvx.Variable((T, nu))

    # Parameters for affinized dynamics
    s0_p = cvx.Parameter(nx)
    A_p = cvx.Parameter((T, nx, nx))
    B_p = cvx.Parameter((T, nx, nu))
    c_p = cvx.Parameter((T, nx))

    # Parameters for trust region centers
    s_bar_p = cvx.Parameter((T + 1, nx))
    u_bar_p = cvx.Parameter((T, nu))

    # Objective + dynamics constraints
    cost = 0.5 * cvx.quad_form(s[T], Q_T)
    constraints = [s[0] == s0_p]
    for t in range(T):
        cost += 0.5 * cvx.quad_form(s[t], Q) + 0.5 * cvx.quad_form(u[t], R)
        constraints += [s[t + 1] == A_p[t] @ s[t] + B_p[t] @ u[t] + c_p[t]]

    # Problem 3.4(c) ##########################################################
    # SOLUTION — STRIP FOR RELEASE
    for t in range(T):
        constraints += [u[t] >= -u_bound, u[t] <= u_bound]
        constraints += [cvx.norm(u[t] - u_bar_p[t], "inf") <= trust_region]
        if t > 0:
            constraints += [cvx.norm(s[t] - s_bar_p[t], "inf") <= trust_region]

    constraints += [cvx.norm(s[T] - s_bar_p[T], "inf") <= trust_region]
    # END SOLUTION

    prob = cvx.Problem(cvx.Minimize(cost), constraints)
    return prob, s, u, (s0_p, A_p, B_p, c_p, s_bar_p, u_bar_p)
