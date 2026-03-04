"""CS 5757 – Homework 2: Humanoid Motion Retargeting via Unconstrained Optimization"""

import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
from typing import Callable, NamedTuple, Optional

# ============================================================
# Provided utilities – do not modify
# ============================================================


class RetargetingProblem(NamedTuple):
    """Data structure encapsulating all information about a retargeting problem."""

    robot: pk.Robot
    target_kps: jnp.ndarray  # [T, n_retarget, 3]
    g1_indices: jnp.ndarray  # [n_retarget]
    rest_joint_angles: jnp.ndarray  # [n_dof]
    lower_limits: jnp.ndarray  # [n_dof]
    upper_limits: jnp.ndarray  # [n_dof]


def pack_trajectory(joint_angles: jnp.ndarray, pose_vecs: jnp.ndarray) -> jnp.ndarray:
    """Pack (joint_angles [T, n_dof], pose_vecs [T, 6]) into a flat vector x."""
    return jnp.concatenate([joint_angles.ravel(), pose_vecs.ravel()])


def unpack_trajectory(
    x: jnp.ndarray, T: int, n_dof: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Unpack flat vector x into (joint_angles [T, n_dof], pose_vecs [T, 6])."""
    joint_angles = x[: T * n_dof].reshape(T, n_dof)
    pose_vecs = x[T * n_dof :].reshape(T, 6)

    return joint_angles, pose_vecs


def robot_keypoints_single(
    joint_angles: jnp.ndarray,
    pose_vec: jnp.ndarray,
    robot: pk.Robot,
    g1_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Compute retarget keypoint positions for a single frame."""
    # Exponentiate the root pose, apply FK to get all link poses, then extract keypoints.
    T_root = jaxlie.SE3.exp(pose_vec)
    T_links = T_root @ jaxlie.SE3(robot.forward_kinematics(cfg=joint_angles))

    return T_links.translation()[g1_indices]


def build_objective(
    problem: RetargetingProblem,
    w_match: float = 1.0,
    w_smooth: float = 0.5,
    w_rest: float = 0.01,
    w_limit: float = 10.0,
) -> tuple[Callable, Callable]:
    """Assemble cost and residual functions from components."""
    T = problem.target_kps.shape[0]
    n_dof = problem.rest_joint_angles.shape[0]

    # Pair each residual function with its weight, filter out zero-weighted ones.
    all_pairs = [
        (residuals_matching, w_match),
        (residuals_smoothness, w_smooth),
        (residuals_rest, w_rest),
        (residuals_limits, w_limit),
    ]

    residual_fns = []
    weights = []
    for fn, w in all_pairs:
        if w > 0:
            residual_fns.append(fn)
            weights.append(w)

    sqrt_weights = [jnp.sqrt(w) for w in weights]

    # Convenience function to evaluate each residual block separately, for use in both cost and residual functions.
    def _eval_blocks(x: jnp.ndarray) -> list[jnp.ndarray]:
        """Evaluate each residual function separately, returning a list of blocks."""
        joint_angles, pose_vecs = unpack_trajectory(x, T, n_dof)
        return [fn(joint_angles, pose_vecs, problem) for fn in residual_fns]

    def residual_fn(x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the full residual vector r(x), applying weights to each block."""
        blocks = _eval_blocks(x)
        return jnp.concatenate([sw * b for sw, b in zip(sqrt_weights, blocks)])

    def cost_fn(x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the scalar cost f(x) = 0.5 * ||r(x)||^2, using the separate blocks."""
        blocks = _eval_blocks(x)
        return 0.5 * sum(jnp.array([w * jnp.sum(b**2) for w, b in zip(weights, blocks)]))  # type: ignore

    return cost_fn, residual_fn


def run_optimizer(
    step_fn: Callable,
    x0: jnp.ndarray,
    state0: dict,
    n_iters: int,
) -> jnp.ndarray:
    """Convenience function to run an optimizer for a fixed number of iterations.

    Args:
        step_fn: A function that takes (x, state) and returns (x_new, state_new). Often this is a "closure" that captures the cost_fn and/or residual_fn.
        x0: Initial trajectory, as a flat vector.
        state0: Initial optimizer state (e.g. for momentum, adaptive methods, etc.) as a dictionary. Can be empty if not needed.
        n_iters: Number of iterations to run the optimizer for.

    Returns:
        The final trajectory after optimization, as a flat vector.
    """
    # Compile the step function with JIT for efficiency, then run it in a loop for n_iters iterations.
    step_jit = jax.jit(step_fn)
    x, state = x0, state0
    for _ in range(n_iters):
        x, state = step_jit(x, state)
    return x


# ============================================================
# Question 2.2a — Keypoint matching residuals
# ============================================================


def residuals_matching(
    joint_angles: jnp.ndarray,
    pose_vecs: jnp.ndarray,
    problem: RetargetingProblem,
) -> jnp.ndarray:
    """Keypoint matching residuals."""
    # Vectorize robot_keypoints_single over T frames -> [T, n_retarget, 3]
    robot_kps = jax.vmap(
        robot_keypoints_single, in_axes=(0, 0, None, None)
    )(joint_angles, pose_vecs, problem.robot, problem.g1_indices)

    # Residual: predicted - target, flattened to 1D
    return (robot_kps - problem.target_kps).ravel()


# ============================================================
# Question 2.2b — Temporal smoothness residuals
# ============================================================


def residuals_smoothness(
    joint_angles: jnp.ndarray,
    pose_vecs: jnp.ndarray,
    problem: RetargetingProblem,
) -> jnp.ndarray:
    """Temporal smoothness residuals."""
    # Joint angle velocity: q_{t+1} - q_t, shape [T-1, n_dof]
    delta_ja = joint_angles[1:] - joint_angles[:-1]

    # SE(3) relative twist: log((T_{t+1})^{-1} @ T_t), shape [T-1, 6]
    T_curr = jax.vmap(jaxlie.SE3.exp)(pose_vecs[:-1])
    T_next = jax.vmap(jaxlie.SE3.exp)(pose_vecs[1:])
    delta_pv = jax.vmap(lambda a, b: (b.inverse() @ a).log())(T_curr, T_next)

    return jnp.concatenate([delta_ja, delta_pv], axis=-1).ravel()  # [(T-1)*(n_dof+6)]


# ============================================================
# Question 2.2c — Rest-pose regularisation residuals
# ============================================================


def residuals_rest(
    joint_angles: jnp.ndarray,
    pose_vecs: jnp.ndarray,
    problem: RetargetingProblem,
) -> jnp.ndarray:
    """Rest-pose regularisation residuals."""
    # Penalize deviation from rest joint angles across all frames
    # joint_angles: [T, n_dof], rest_joint_angles: [n_dof]
    return (joint_angles - problem.rest_joint_angles).ravel()  # [T * n_dof]


# ============================================================
# Question 2.2d — Joint limits residuals
# ============================================================


def residuals_limits(
    joint_angles: jnp.ndarray,
    pose_vecs: jnp.ndarray,
    problem: RetargetingProblem,
) -> jnp.ndarray:
    """Soft joint-limit residuals (one-sided)."""
    # Hinge penalties: positive only when limits are violated
    # joint_angles: [T, n_dof], lower/upper_limits: [n_dof]
    lower_violation = jnp.maximum(0.0, problem.lower_limits - joint_angles)  # [T, n_dof]
    upper_violation = jnp.maximum(0.0, joint_angles - problem.upper_limits)  # [T, n_dof]

    return jnp.concatenate([lower_violation.ravel(), upper_violation.ravel()])  # [2 * T * n_dof]


# ============================================================
# Question 2.3a — Gradient descent step
# ============================================================


def gradient_descent_step(
    cost_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    state: dict,
    lr: float,
) -> tuple[jnp.ndarray, dict]:
    """Applies one step of gradient descent."""
    grad = jax.grad(cost_fn)(x)
    return x - lr * grad, state


# ============================================================
# Question 2.3b — Gauss-Newton step
# ============================================================


def gauss_newton_step(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    state: dict,
    damping: float = 1e-6,
) -> tuple[jnp.ndarray, dict]:
    """One step of damped Gauss-Newton (Levenberg-Marquardt)."""
    r = residual_fn(x)                      # [m]
    J = jax.jacfwd(residual_fn)(x)          # [m, n]
    JtJ = J.T @ J                           # [n, n]
    Jtr = J.T @ r                           # [n]
    delta = jnp.linalg.solve(JtJ + damping * jnp.eye(JtJ.shape[0]), -Jtr)
    return x + delta, state


# ============================================================
# Question 2.3c — Gradient descent with parallel line search step
# ============================================================


def gradient_descent_line_search_step(
    cost_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    state: dict,
    candidate_alphas: jnp.ndarray,
) -> tuple[jnp.ndarray, dict]:
    """One step of gradient descent with parallel line search."""
    g = jax.grad(cost_fn)(x)
    # Evaluate cost for each candidate step size in parallel
    costs = jax.vmap(lambda alpha: cost_fn(x - alpha * g))(candidate_alphas)
    best_alpha = candidate_alphas[jnp.argmin(costs)]
    return x - best_alpha * g, state


# ============================================================
# Question 2.3d — Matrix-free Gauss-Newton step
# ============================================================


def gauss_newton_matrix_free_step(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    state: dict,
    damping: float = 1e-6,
    cg_tol: float = 1e-5,
    cg_maxiter: int = 10,
) -> tuple[jnp.ndarray, dict]:
    """One step of Gauss-Newton, using the implicit form, and solving the normal equations with CG."""
    r = residual_fn(x)
    _, vjp_fn = jax.vjp(residual_fn, x)
    Jtr = vjp_fn(r)[0]  # J^T r

    def matvec(v):
        # (J^T J + λI) v = J^T (J v) + λv
        _, Jv = jax.jvp(residual_fn, (x,), (v,))
        JtJv = vjp_fn(Jv)[0]
        return JtJv + damping * v

    delta, _ = jax.scipy.sparse.linalg.cg(matvec, -Jtr, tol=cg_tol, maxiter=cg_maxiter)
    return x + delta, state


# ============================================================
# Question 2.4 — Custom optimizer (for leaderboard)
# ============================================================


_ADAM_LR = 5e-2
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_EPS = 1e-8


def custom_optimizer_init(
    cost_fn: Callable[[jnp.ndarray], jnp.ndarray],
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    problem: Optional[RetargetingProblem] = None,
) -> tuple[jnp.ndarray, dict]:
    """Creates an initial state for your custom optimizer (Adam)."""
    return x0, {
        "m": jnp.zeros_like(x0),   # first moment
        "v": jnp.zeros_like(x0),   # second moment
        "t": jnp.array(0),         # step count
    }


def custom_optimizer_step(
    cost_fn: Callable[[jnp.ndarray], jnp.ndarray],
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    state: dict,
) -> tuple[jnp.ndarray, dict]:
    """Applies one step of Adam."""
    m, v, t = state["m"], state["v"], state["t"]
    t = t + 1

    g = jax.grad(cost_fn)(x)
    m = _ADAM_BETA1 * m + (1 - _ADAM_BETA1) * g
    v = _ADAM_BETA2 * v + (1 - _ADAM_BETA2) * g ** 2

    # Bias correction
    m_hat = m / (1 - _ADAM_BETA1 ** t)
    v_hat = v / (1 - _ADAM_BETA2 ** t)

    x = x - _ADAM_LR * m_hat / (jnp.sqrt(v_hat) + _ADAM_EPS)
    return x, {"m": m, "v": v, "t": t}

N_ITERS = 200
# Modify the iterations for custom optimizer
