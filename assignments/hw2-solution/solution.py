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

    def _frame_residuals(joint_angles, pose_vec, target):
        robot_kps = robot_keypoints_single(
            joint_angles, pose_vec, problem.robot, problem.g1_indices
        )
        return (robot_kps - target).ravel()

    per_frame = jax.vmap(_frame_residuals)(joint_angles, pose_vecs, problem.target_kps)
    return per_frame.ravel()


# ============================================================
# Question 2.2b — Temporal smoothness residuals
# ============================================================


def residuals_smoothness(
    joint_angles: jnp.ndarray,
    pose_vecs: jnp.ndarray,
    problem: RetargetingProblem,
) -> jnp.ndarray:
    """Temporal smoothness residuals."""
    djoint_angles = (joint_angles[1:] - joint_angles[:-1]).ravel()
    dpose = jaxlie.SE3.log(
        jaxlie.SE3.exp(pose_vecs[:-1]).inverse() @ jaxlie.SE3.exp(pose_vecs[1:])
    ).ravel()
    return jnp.concatenate([djoint_angles, dpose])


# ============================================================
# Question 2.2c — Rest-pose regularisation residuals
# ============================================================


def residuals_rest(
    joint_angles: jnp.ndarray,
    pose_vecs: jnp.ndarray,
    problem: RetargetingProblem,
) -> jnp.ndarray:
    """Rest-pose regularisation residuals."""
    joint_angles_res = (joint_angles - problem.rest_joint_angles[None, :]).ravel()
    return joint_angles_res


# ============================================================
# Question 2.2d — Joint limits residuals
# ============================================================


def residuals_limits(
    joint_angles: jnp.ndarray,
    pose_vecs: jnp.ndarray,
    problem: RetargetingProblem,
) -> jnp.ndarray:
    """Soft joint-limit residuals (one-sided)."""
    over = jnp.maximum(joint_angles - problem.upper_limits[None, :], 0.0).ravel()
    under = jnp.maximum(problem.lower_limits[None, :] - joint_angles, 0.0).ravel()
    return jnp.concatenate([over, under])


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
    g = jax.grad(cost_fn)(x)
    return x - lr * g, state


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
    # Compute the residual and Jacobian at the current point.
    r = residual_fn(x)
    J = jax.jacrev(residual_fn)(x)
    JtJ = J.T @ J
    Jtr = J.T @ r
    n = JtJ.shape[0]

    # Solve for the update step.
    dx = jnp.linalg.solve(JtJ + damping * jnp.eye(n), -Jtr)

    # Return full Gauss-Newton step.
    return x + dx, state


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
    # Compute the gradient at the current point.
    g = jax.grad(cost_fn)(x)

    # Generate candidate points along the negative gradient direction for each alpha.
    candidates = x[None, :] - candidate_alphas[:, None] * g[None, :]

    # Evaluate the cost function at each candidate point in parallel.
    costs = jax.vmap(cost_fn)(candidates)

    # Select the candidate with the lowest cost and return it.
    return candidates[jnp.argmin(costs)], state


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
    # VJP gives us r and a function for J^T multiplication.
    r, vjp_fn = jax.vjp(residual_fn, x)
    rhs = -vjp_fn(r)[0]

    # JVP (linearize) gives us a function for J multiplication.
    _, jvp_fn = jax.linearize(residual_fn, x)

    # Create the matvec function for the CG solver, which applies J.T @ J + damping to a vector.
    def matvec(v: jnp.ndarray) -> jnp.ndarray:
        """Matrix-vector product for the normal equations: J.T @ (J @ v) + damping * v."""
        return vjp_fn(jvp_fn(v))[0] + damping * v

    # Solve normal equations using Conjugate Gradient.
    dx, _ = jax.scipy.sparse.linalg.cg(matvec, rhs, tol=cg_tol, maxiter=cg_maxiter)
    return x + dx, state


# ============================================================
# Question 2.4 — Custom optimizer (for leaderboard)
# ============================================================

def get_orientation_initialization(problem: RetargetingProblem) -> jaxlie.SO3:
    """Clever initialization of root orientation by aligning torso keypoints via point cloud alignment.

    Here we will use the torso keypoints (which are rigidly connected to the root) to compute an initial rotation that
    roughly aligns the robot's rest pose with the target pose. We can do this by solving Procrustes' problem,
    which uses the singular value decomposition (SVD) to find the optimal rotation between two (matched) point clouds.
    """

    # Indices into the 13 retarget pairs for torso-adjacent keypoints:
    # 0=pelvis, 1=left_hip, 2=right_hip, 7=left_shoulder, 8=right_shoulder
    torso_subset = jnp.array([0, 1, 2, 7, 8])

    pelvis_positions = problem.target_kps[:, 0]

    # Extract positions of world-frame torso keypoints.
    target_torso_keypoints = (
        problem.target_kps[:, torso_subset] - pelvis_positions[:, None, :]
    )

    # Extract positions of body-frame torso keypoints.
    fk_results = jaxlie.SE3(
        problem.robot.forward_kinematics(cfg=problem.rest_joint_angles)
    ).translation()

    # Use g1_indices so FK is aligned with target_kps, then take the same torso subset
    rest_retarget_keypoints = fk_results[problem.g1_indices]
    rest_torso_keypoints = (
        rest_retarget_keypoints[torso_subset] - rest_retarget_keypoints[0]
    )

    def procrustes(A: jnp.ndarray, B: jnp.ndarray) -> jaxlie.SO3:
        """Solve Procrustes' problem to find optimal rotation from A to B."""

        # Center both point clouds (already centered around pelvis, but do it again just in case).
        A_centered = A - A.mean(axis=0)
        B_centered = B - B.mean(axis=0)

        # Compute the SVD of the covariance matrix B^T A.
        U, _, Vt = jnp.linalg.svd(B_centered.T @ A_centered)

        # Compute rotation matrix R = U S V^T, where S corrects for reflection if needed.
        d = jnp.linalg.det(U @ Vt)
        S = jnp.diag(jnp.array([1.0, 1.0, d]))
        R = U @ S @ Vt

        return jaxlie.SO3.from_matrix(R)

    # vmap Procrustes over all frames to get a T-length sequence of initial orientations.
    return jax.vmap(procrustes, in_axes=(None, 0))(
        rest_torso_keypoints, target_torso_keypoints
    )

def custom_optimizer_init(
    cost_fn: Callable[[jnp.ndarray], jnp.ndarray],
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    problem: Optional[RetargetingProblem] = None,
) -> tuple[jnp.ndarray, dict]:
    """Creates an initial state for your custom optimizer."""
    if problem is not None:
        orientation_init = get_orientation_initialization(problem)
        pelvis = problem.target_kps[:, 0]
        x0 = pack_trajectory(
            jnp.tile(problem.rest_joint_angles, (problem.target_kps.shape[0], 1)),
            jaxlie.SE3.from_rotation_and_translation(orientation_init, pelvis).log(),
        )
    return x0, {
        "m": jnp.zeros_like(x0),  # First moment estimate
        "v": jnp.zeros_like(x0),  # Second moment estimate
        "t": jnp.array(0.0, dtype=jnp.float32),  # Time step counter (float for bias correction)
    }


def custom_optimizer_step(
    cost_fn: Callable[[jnp.ndarray], jnp.ndarray],
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    state: dict,
    lr: float = 0.075,
    beta1: float = 0.9,
    beta2: float = 0.98,
) -> tuple[jnp.ndarray, dict]:
    """Applies one step of a RAdam-style optimizer."""
    epsilon = 1e-8
    

    # Extract and update step counter (kept as float for consistency with existing state)
    m = state["m"]
    v = state["v"]
    t = state["t"] + 1.0

    # Gradient of the cost
    g = jax.grad(cost_fn)(x)

    # Exponential moving averages of gradient and its square
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * (g ** 2)

    x_new = x - lr * m / (jnp.sqrt(v) + epsilon)

    new_state = {"m": m, "v": v, "t": t}

    return x_new, new_state

N_ITERS = 70
# ============================================================
# Additional residuals for qualitative improvement (optional)
# ============================================================
