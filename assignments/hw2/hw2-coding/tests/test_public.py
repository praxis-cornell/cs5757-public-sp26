import unittest
import os
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp
import pyroki as pk
from jax import Array
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from robot_descriptions.loaders.yourdfpy import load_robot_description
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False

from solution import (
        RetargetingProblem,
        residuals_smoothness,
        residuals_rest,
        residuals_limits,
        gradient_descent_step,
        gauss_newton_step,
        gradient_descent_line_search_step,
        gauss_newton_matrix_free_step,
        custom_optimizer_init,
        custom_optimizer_step,
        build_objective,
        pack_trajectory,
        run_optimizer,
        N_ITERS,
    )

# Constants for retargeting (from run_retargeting.py)
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine_1", "left_knee", "right_knee",
    "spine_2", "left_ankle", "right_ankle", "spine_3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
    "nose", "right_eye", "left_eye", "right_ear", "left_ear", "left_big_toe",
    "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel",
    "left_thumb", "left_index", "left_middle", "left_ring", "left_pinky",
    "right_thumb", "right_index", "right_middle", "right_ring", "right_pinky",
]

G1_LINK_NAMES = [
    "pelvis", "pelvis_contour_link", "left_hip_pitch_link", "left_hip_roll_link",
    "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link",
    "right_ankle_pitch_link", "right_ankle_roll_link", "waist_yaw_link", "waist_roll_link",
    "torso_link", "logo_link", "head_link", "waist_support_link", "imu_in_torso",
    "imu_in_pelvis", "d435_link", "mid360_link", "left_shoulder_pitch_link",
    "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
    "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link", "left_rubber_hand",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
    "right_wrist_yaw_link", "right_rubber_hand",
]

RETARGET_PAIRS = [
    ("pelvis", "pelvis_contour_link"),
    ("left_hip", "left_hip_pitch_link"),
    ("right_hip", "right_hip_pitch_link"),
    ("left_knee", "left_knee_link"),
    ("right_knee", "right_knee_link"),
    ("left_ankle", "left_ankle_roll_link"),
    ("right_ankle", "right_ankle_roll_link"),
    ("left_shoulder", "left_shoulder_roll_link"),
    ("right_shoulder", "right_shoulder_roll_link"),
    ("left_elbow", "left_elbow_link"),
    ("right_elbow", "right_elbow_link"),
    ("left_wrist", "left_wrist_roll_link"),
    ("right_wrist", "right_wrist_roll_link"),
]


def get_retarget_indices():
    """Get SMPL and G1 indices for retargeting pairs."""
    smpl_idx = [SMPL_JOINT_NAMES.index(s) for s, _ in RETARGET_PAIRS]
    g1_idx = [G1_LINK_NAMES.index(g) for _, g in RETARGET_PAIRS]
    return jnp.array(smpl_idx), jnp.array(g1_idx)


def compute_global_scale(robot, rest_joint_angles, smpl_keypoints, smpl_indices, g1_indices):
    """Scale from average bone-length ratio (pose-invariant)."""
    smpl_pos = smpl_keypoints[smpl_indices]
    robot_pos = jaxlie.SE3(
        robot.forward_kinematics(cfg=rest_joint_angles)
    ).translation()[g1_indices]

    bone_pairs = [
        (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
        (7, 9), (8, 10), (9, 11), (10, 12),
    ]

    smpl_lengths = []
    robot_lengths = []
    for i, j in bone_pairs:
        smpl_lengths.append(float(jnp.linalg.norm(smpl_pos[i] - smpl_pos[j])))
        robot_lengths.append(float(jnp.linalg.norm(robot_pos[i] - robot_pos[j])))

    return sum(robot_lengths) / sum(smpl_lengths)


class TestRetargeting(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(42)

    # =========================================================================
    # Helpers
    # =========================================================================

    def create_random_problem_data(
        self, T: int = 4, n_dof: int = 5, n_retarget: int = 3
    ):
        """Creates randomized data for RetargetingProblem."""
        k1, k2, k3, k4 = jax.random.split(self.key, 4)

        target_kps = jax.random.normal(k1, (T, n_retarget, 3))
        g1_indices = jnp.arange(n_retarget)
        rest_joint_angles = jnp.zeros(n_dof)

        lower_limits = -2.0 * jax.random.uniform(k2, (n_dof,))
        upper_limits = 2.0 * jax.random.uniform(k3, (n_dof,))

        # Random inputs
        ja = jax.random.normal(k4, (T, n_dof))
        pv = jax.random.normal(k4, (T, 6))

        return (
            (target_kps, g1_indices, rest_joint_angles, lower_limits, upper_limits),
            ja,
            pv,
        )

    def create_quadratic_problem(self, n: int = 20):
        """Creates a synthetic linear least-squares problem: min 0.5 ||Ax - b||^2."""
        k1, k2 = jax.random.split(self.key)
        m = n * 3
        A = jax.random.normal(k1, (m, n)) / jnp.sqrt(m)
        b = jax.random.normal(k2, (m,))
        x_star = jnp.linalg.lstsq(A, b)[0]

        def residual_fn(x):
            return A @ x - b

        def cost_fn(x):
            r = residual_fn(x)
            return 0.5 * jnp.sum(r**2)

        return cost_fn, residual_fn, jnp.zeros(n), x_star

    def run_retargeting_trajectory(
        self,
        problem: RetargetingProblem,
        n_iters: int = 100,
        w_match: float = 5.0,
        w_smooth: float = 5.0,
        w_rest: float = 0.01,
        w_limit: float = 10.0,
    ) -> float:
        """
        Helper function to run a retargeting trajectory with custom optimizer.
        
        Follows the logic from run_retargeting.py:
        1. Build objective functions (cost and residual)
        2. Initialize trajectory x0
        3. Initialize custom optimizer state
        4. Run optimizer for n_iters
        5. Return final cost
        
        Args:
            problem: RetargetingProblem instance
            n_iters: Number of optimization iterations
            w_match: Weight for matching residual
            w_smooth: Weight for smoothness residual
            w_rest: Weight for rest-pose residual
            w_limit: Weight for joint limits residual
            
        Returns:
            Final cost value (scalar)
        """
        T = problem.target_kps.shape[0]
        n_dof = problem.rest_joint_angles.shape[0]
        
        # Build objective functions with specified weights
        cost_fn, residual_fn = build_objective(
            problem,
            w_match=w_match,
            w_smooth=w_smooth,
            w_rest=w_rest,
            w_limit=w_limit,
        )
        
        # Initialize trajectory x0
        # Start with rest joint angles for all frames
        joint_angles_0 = jnp.tile(problem.rest_joint_angles, (T, 1))
        
        # Initialize root poses from first target keypoint (pelvis position)
        # Use the first retarget keypoint as the root position
        pelvis_positions = problem.target_kps[:, 0, :]  # [T, 3]
        pose_vecs_0 = jaxlie.SE3.from_translation(pelvis_positions).log()
        
        # Pack into flat vector
        x0 = pack_trajectory(joint_angles_0, pose_vecs_0)
        
        # Initialize custom optimizer
        x0, state0 = custom_optimizer_init(cost_fn, residual_fn, x0, problem)
        
        # Create step function
        def step_fn(x, state):
            return custom_optimizer_step(cost_fn, residual_fn, x, state)
        
        # Run optimizer
        x_opt = run_optimizer(step_fn, x0, state0, n_iters)
        x_opt.block_until_ready()
        
        # Compute and return final cost
        final_cost = float(cost_fn(x_opt))
        return final_cost

    def create_retargeting_problem(
        self,
        npz_file: str | Path | None = None,
        T: int = 10,
        n_retarget: int = 13,
        seed: int = 42,
    ) -> RetargetingProblem:
        """
        Creates a RetargetingProblem for testing retargeting.
        
        Args:
            npz_file: Optional path to .npz file containing SMPL motion data.
                     If provided, loads real data from file. If None, generates synthetic data.
            T: Number of time steps (frames) - only used if npz_file is None
            n_retarget: Number of retarget keypoints - only used if npz_file is None
            seed: Random seed for reproducibility - only used if npz_file is None
            
        Returns:
            RetargetingProblem instance
        """
        if not ROBOT_AVAILABLE:
            raise ImportError(
                "robot_descriptions not available. Cannot create RetargetingProblem."
            )
        
        # Load robot
        urdf = load_robot_description("g1_description")
        robot = pk.Robot.from_urdf(urdf)
        
        # Get robot properties
        n_dof = len(robot.joints.lower_limits)
        rest_joint_angles = jnp.zeros(n_dof)
        lower_limits = robot.joints.lower_limits
        upper_limits = robot.joints.upper_limits
        
        # Get retarget indices
        smpl_indices, g1_indices = get_retarget_indices()
        n_retarget_actual = len(g1_indices)
        
        if npz_file is not None:
            # Load data from npz file
            npz_path = Path(npz_file)
            if not npz_path.exists():
                # Try relative to test file directory
                test_dir = Path(__file__).parent.parent
                npz_path = test_dir / "data" / npz_path.name
                if not npz_path.exists():
                    raise FileNotFoundError(f"Could not find npz file: {npz_file}")
            
            smpl_data = onp.load(npz_path)
            smpl_keypoints = smpl_data["joints"]  # [T, 45, 3]
            
            # Compute scale
            scale = compute_global_scale(
                robot, rest_joint_angles, smpl_keypoints[0], smpl_indices, g1_indices
            )
            
            # Extract target keypoints for retargeting pairs
            target_kps = smpl_keypoints[:, smpl_indices] * scale  # [T, n_retarget, 3]
        else:
            # Create synthetic target keypoints
            # Use a fixed seed for reproducibility
            key = jax.random.PRNGKey(seed)
            k1, k2 = jax.random.split(key, 2)
            
            # Generate target keypoints: create a simple trajectory
            # Start with some base positions and add smooth motion
            base_positions = jax.random.normal(k1, (n_retarget_actual, 3)) * 0.5  # [n_retarget, 3]
            time_variation = jnp.linspace(0, 2 * jnp.pi, T)  # [T]
            # Create 3D motion that varies over time
            motion_scale = 0.2
            motion_x = jnp.sin(time_variation)[:, None, None] * motion_scale  # [T, 1, 1]
            motion_y = jnp.cos(time_variation)[:, None, None] * motion_scale  # [T, 1, 1]
            motion_z = jnp.sin(time_variation * 0.5)[:, None, None] * motion_scale  # [T, 1, 1]
            motion = jnp.concatenate([motion_x, motion_y, motion_z], axis=2)  # [T, 1, 3]
            target_kps = base_positions[None, :, :] + motion  # [T, n_retarget, 3]
        
        return RetargetingProblem(
            robot=robot,
            target_kps=jnp.array(target_kps),
            g1_indices=g1_indices,
            rest_joint_angles=rest_joint_angles,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
        )

    # =========================================================================
    # Question 2.2: Residuals
    # =========================================================================

    def test_residuals_smoothness_shape(self) -> None:
        """Checks return type and shape of residuals_smoothness."""
        (params), ja, pv = self.create_random_problem_data()
        problem = RetargetingProblem(None, *params)
        r = residuals_smoothness(ja, pv, problem)

        assert isinstance(r, Array), "Returned residual is not a JAX Array"
        assert r.ndim == 1, f"Residuals must be 1D, got ndim={r.ndim}"

        T, n_dof = ja.shape
        expected_len = (T - 1) * (n_dof + 6)
        assert (
            r.shape[0] == expected_len
        ), f"Incorrect shape, got {r.shape}, expected ({expected_len},)"

    def test_residuals_smoothness_logic_zero(self) -> None:
        """Property Check: Constant trajectory should have ZERO smoothness cost."""
        (params), ja, pv = self.create_random_problem_data()
        problem = RetargetingProblem(None, *params)

        # Create constant trajectory
        ja_const = jnp.ones_like(ja) * 0.5
        pv_const = jnp.ones_like(pv) * 0.1

        r = residuals_smoothness(ja_const, pv_const, problem)

        # Should be exactly zero (or very close due to float precision)
        assert jnp.allclose(
            r, 0.0, atol=1e-6
        ), "Constant trajectory produced non-zero smoothness residual"

    def test_residuals_rest_shape(self) -> None:
        """Checks return type and shape of residuals_rest."""
        (params), ja, pv = self.create_random_problem_data()
        problem = RetargetingProblem(None, *params)
        r = residuals_rest(ja, pv, problem)

        assert isinstance(r, Array), "Returned residual is not a JAX Array"
        assert r.shape == (
            ja.size,
        ), f"Incorrect shape, got {r.shape}, expected ({ja.size},)"

    def test_residuals_rest_logic(self) -> None:
        """Property Check: Residual should be zero exactly at rest config."""
        (params), ja, pv = self.create_random_problem_data()
        problem = RetargetingProblem(None, *params)

        # Set trajectory exactly to rest configuration
        ja_rest = jnp.tile(problem.rest_joint_angles, (ja.shape[0], 1))

        r = residuals_rest(ja_rest, pv, problem)
        assert jnp.allclose(
            r, 0.0, atol=1e-6
        ), "Rest residual is non-zero at rest configuration"

    def test_residuals_limits_shape(self) -> None:
        """Checks return type and shape of residuals_limits."""
        (params), ja, pv = self.create_random_problem_data()
        problem = RetargetingProblem(None, *params)
        r = residuals_limits(ja, pv, problem)

        assert isinstance(r, Array), "Returned residual is not a JAX Array"
        expected_len = 2 * ja.size
        assert r.shape == (
            expected_len,
        ), f"Incorrect shape, got {r.shape}, expected ({expected_len},)"

    def test_residuals_limits_logic(self) -> None:
        """Property Check: Residuals should be non-negative and zero when inside limits."""
        (params), ja, pv = self.create_random_problem_data()
        problem = RetargetingProblem(None, *params)

        # 1. Inside limits -> Zero residual
        ja_safe = jnp.zeros_like(ja)  # limits are -2 to 2, so 0 is safe
        r_safe = residuals_limits(ja_safe, pv, problem)
        assert jnp.allclose(
            r_safe, 0.0, atol=1e-6
        ), "Limit residual is non-zero inside valid range"

        # 2. Outside limits -> Positive residual
        ja_unsafe = jnp.ones_like(ja) * 10.0  # Way outside
        r_unsafe = residuals_limits(ja_unsafe, pv, problem)
        assert jnp.all(r_unsafe >= -1e-7), "Limit residuals must be non-negative"
        assert jnp.any(r_unsafe > 1.0), "Limit residual did not detect violation"

    # =========================================================================
    # Question 2.3a: Gradient Descent
    # =========================================================================
    def test_gradient_descent_step_shape(self) -> None:
        """Checks return type and shape of gradient_descent_step."""
        cost_fn, _, x0, _ = self.create_quadratic_problem()
        state = {}
        lr = 0.1

        x_new, state_new = gradient_descent_step(cost_fn, x0, state, lr)

        assert isinstance(x_new, Array), "x_new is not a JAX Array"
        assert x_new.shape == x0.shape, f"x_new shape mismatch"
        assert isinstance(state_new, dict), "state_new must be a dictionary"

    
    def test_gradient_descent_step_descent(self) -> None:
        """Property Check: A single step should strictly reduce cost on a convex problem."""
        cost_fn, _, x0, _ = self.create_quadratic_problem()
        lr = 0.1
        state = {}

        x_new, _ = gradient_descent_step(cost_fn, x0, state, lr)

        cost_before = cost_fn(x0)
        cost_after = cost_fn(x_new)

        assert (
            cost_after < cost_before
        ), f"Gradient descent increased cost: {cost_before} -> {cost_after}"

    # =========================================================================
    # Question 2.3b: Gauss-Newton
    # =========================================================================

    def test_gauss_newton_step_shape(self) -> None:
        """Checks return type and shape of gauss_newton_step."""
        _, residual_fn, x0, _ = self.create_quadratic_problem()
        state = {}
        damping = 1e-4

        x_new, state_new = gauss_newton_step(residual_fn, x0, state, damping)

        assert isinstance(x_new, Array), "x_new is not a JAX Array"
        assert x_new.shape == x0.shape, f"x_new shape mismatch"

    def test_gauss_newton_step_exact_convergence(self) -> None:
        """Property Check: GN should solve linear least-squares in EXACTLY one step."""
        _, residual_fn, x0, x_star = self.create_quadratic_problem()

        # With zero/tiny damping, GN = Least Squares
        x_new, _ = gauss_newton_step(residual_fn, x0, {}, damping=1e-9)

        assert jnp.allclose(
            x_new, x_star, atol=1e-4
        ), "Gauss-Newton failed to solve linear LS in one step"

    # =========================================================================
    # Question 2.3c: Line Search
    # =========================================================================

    def test_line_search_selection_logic(self) -> None:
        """Property Check: Line search must pick the alpha that minimizes cost."""
        cost_fn, _, x0, _ = self.create_quadratic_problem()

        # Candidates: 0 (no move), 1000 (bad move), 0.1 (good move)
        alphas = jnp.array([0.0, 1000.0, 0.1])

        x_new, _ = gradient_descent_line_search_step(cost_fn, x0, {}, alphas)

        # Calculate what the step should be for alpha=0.1
        g = jax.grad(cost_fn)(x0)
        expected = x0 - 0.1 * g

        assert jnp.allclose(
            x_new, expected, atol=1e-5
        ), "Line search did not select the optimal alpha"


    # =========================================================================
    # Question 4: Custom Optimizer
    # =========================================================================

    def test_custom_optimizer_descent(self) -> None:
        """Creativity Check: Custom optimizer must strictly reduce cost over iterations + Different from Gauss-Newton."""
        cost_fn, residual_fn, x0, _ = self.create_quadratic_problem()

        x0, state = custom_optimizer_init(cost_fn, residual_fn, x0)

        # Run 5 steps
        x = x0
        costs = []
        gn_costs = []
        for _ in range(5):
            gn_x, _ = gauss_newton_step(residual_fn, x, {}, damping=1e-4)
            x, state = custom_optimizer_step(cost_fn, residual_fn, x, state)

            costs.append(cost_fn(x))
            gn_costs.append(cost_fn(gn_x))
        assert (
            costs[-1] < costs[0]
        ), "Custom optimizer failed to reduce cost after 5 iterations"
        costs = jnp.array(costs)
        gn_costs = jnp.array(gn_costs)
        assert not jnp.allclose(
            costs, gn_costs, atol=1e-5
        ), "Custom optimizer and Gauss-Newton produced the same cost after 5 iterations"

    def test_custom_optimizer_cost(self, set_leaderboard_value=None) -> None:
        """Leaderboard: Cost of the custom optimizer."""
        # Create a retargeting problem from npz file or synthetic data
        # You can specify an npz file like: "walking1.npz" or None for synthetic data
        import time
        import sys
        npz_files = ["data/walking1.npz", "data/cartwheel.npz", "data/alaska.npz"]  # Change this to use different motion files
        final_cost = 0.0
        for npz_file in npz_files:
            problem = self.create_retargeting_problem(npz_file=npz_file, T=20, n_retarget=13, seed=42)
            start_time = time.time()
            cost = self.run_retargeting_trajectory(problem, n_iters=N_ITERS)
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds", file=sys.stderr)
            print(f"Cost: {cost}", file=sys.stderr)   
            final_cost += cost + (end_time - start_time)*5
        # set_leaderboard_value(final_cost)
        print(f"Final cost: {final_cost}", file=sys.stderr)
        assert final_cost > 0.0, "Final cost is too low"



        

if __name__ == "__main__":
    unittest.main()