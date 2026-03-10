"""CS 5757 – Retargeting Visualiser

Loads SMPL motion-capture data, builds a RetargetingProblem, and provides
a viser GUI for running your optimisers and visualising the result on the
Unitree G1 humanoid.

Usage:
    python run_retargeting.py

Then open http://localhost:8080 in your browser.
"""

import argparse
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from solution import (
    RetargetingProblem,
    build_objective,
    custom_optimizer_init,
    custom_optimizer_step,
    gauss_newton_matrix_free_step,
    gauss_newton_step,
    gradient_descent_line_search_step,
    gradient_descent_step,
    pack_trajectory,
    run_optimizer,
    unpack_trajectory,
)

# ============================================================
# Constants
# ============================================================

SMPL_BODY_SKELETON = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (9, 12),
    (9, 13),
    (9, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
    (20, 22),
    (21, 23),
]

SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine_1",
    "left_knee",
    "right_knee",
    "spine_2",
    "left_ankle",
    "right_ankle",
    "spine_3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]

G1_LINK_NAMES = [
    "pelvis",
    "pelvis_contour_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "logo_link",
    "head_link",
    "waist_support_link",
    "imu_in_torso",
    "imu_in_pelvis",
    "d435_link",
    "mid360_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_rubber_hand",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
]

G1_TORSO_LINK_NAMES = [
    "pelvis",
    "pelvis_contour_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "logo_link",
    "head_link",
    "waist_support_link",
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


# ============================================================
# Helpers
# ============================================================


def get_retarget_indices():
    smpl_idx = [SMPL_JOINT_NAMES.index(s) for s, _ in RETARGET_PAIRS]
    g1_idx = [G1_LINK_NAMES.index(g) for _, g in RETARGET_PAIRS]
    return jnp.array(smpl_idx), jnp.array(g1_idx)


def compute_global_scale(
    robot, rest_joint_angles, smpl_keypoints, smpl_indices, g1_indices
):
    """Scale from average bone-length ratio (pose-invariant)."""
    smpl_pos = smpl_keypoints[smpl_indices]
    robot_pos = jaxlie.SE3(
        robot.forward_kinematics(cfg=rest_joint_angles)
    ).translation()[g1_indices]

    bone_pairs = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (7, 9),
        (8, 10),
        (9, 11),
        (10, 12),
    ]

    smpl_lengths = []
    robot_lengths = []
    for i, j in bone_pairs:
        smpl_lengths.append(float(jnp.linalg.norm(smpl_pos[i] - smpl_pos[j])))
        robot_lengths.append(float(jnp.linalg.norm(robot_pos[i] - robot_pos[j])))

    return sum(robot_lengths) / sum(smpl_lengths)


def build_problem(
    robot, smpl_keypoints, smpl_indices, g1_indices, rest_joint_angles, scale
):
    """Package everything into a RetargetingProblem."""
    target_kps = smpl_keypoints[:, smpl_indices] * scale

    return RetargetingProblem(
        robot=robot,
        target_kps=jnp.array(target_kps),
        g1_indices=g1_indices,
        rest_joint_angles=rest_joint_angles,
        lower_limits=robot.joints.lower_limits,
        upper_limits=robot.joints.upper_limits,
    )


def scan_npz_files(directory: Path) -> list[str]:
    """Return sorted list of .npz filenames in the given directory."""
    return sorted(p.name for p in directory.glob("*.npz"))


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Directory containing .npz motion files",
    )
    args = parser.parse_args()

    # Load robot.
    urdf = load_robot_description("g1_description")
    robot = pk.Robot.from_urdf(urdf)

    # Load mocap data.
    asset_dir = Path(__file__).parent / "data"

    smpl_indices, g1_indices = get_retarget_indices()
    rest_joint_angles = jnp.zeros_like(robot.joints.lower_limits)

    # Create app state (shared across callbacks).
    smpl_keypoints: onp.ndarray | None = None
    num_timesteps: int = 0
    scale: float = 1.0
    root_positions: jnp.ndarray | None = None
    root_wxyz: jnp.ndarray | None = None
    joint_angles: jnp.ndarray | None = None
    fps: float = 30.0

    # Viser setup.
    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    playing = server.gui.add_checkbox("playing", True)
    timestep_slider = server.gui.add_slider("timestep", 0, 1, 1, 0)
    server.scene.add_grid("grid")

    # Load trajectory files for dropdown menu.
    npz_files = scan_npz_files(asset_dir)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {asset_dir}")

    with server.gui.add_folder("Trajectory"):
        data_dropdown = server.gui.add_dropdown(
            "file", npz_files, initial_value="walking1.npz"
        )
        load_button = server.gui.add_button("Load trajectory")
        refresh_button = server.gui.add_button("Refresh file list")

    # Optimizer selection.
    with server.gui.add_folder("Optimizer"):
        opt_dropdown = server.gui.add_dropdown(
            "method",
            [
                "gradient_descent",
                "gauss_newton",
                "line_search",
                "gauss_newton_matrix_free",
                "custom",
            ],
            initial_value="gradient_descent",
        )
        sl_n_iters = server.gui.add_slider("iterations", 0, 5000, 100, 500)
        sl_lr = server.gui.add_slider("learning rate (GD)", 1e-4, 1e-1, 1e-4, 1e-3)
        sl_damping = server.gui.add_slider("damping (GN)", 1e-8, 1e-1, 1e-8, 1e-4)

    # Cost weight sliders.
    with server.gui.add_folder("Weights"):
        sl_w_match = server.gui.add_slider("match", 0.0, 10.0, 0.1, 5.0)
        sl_w_smooth = server.gui.add_slider("smooth", 0.0, 100.0, 0.1, 5.0)
        sl_w_rest = server.gui.add_slider("rest", 0.0, 1.0, 0.001, 0.01)
        sl_w_limit = server.gui.add_slider("limit", 0.0, 50.0, 0.5, 10.0)


    # -- UNCOMMENT AND ADD YOUR CUSTOM OPTIMIZER HYPERPARAMETERS HERE --
    # with server.gui.add_folder("Custom Optimizer"):
    # sl_custom_1 = server.gui.add_slider("param_1", 1e-8, 1.0, 1e-8, 1e-3)
    # sl_custom_2 = server.gui.add_slider("param_2", 1e-8, 1.0, 1e-8, 1e-1)

    cost_label = server.gui.add_text("cost", initial_value="—", disabled=True)

    # -- Load trajectory --
    def load_trajectory(filename: str) -> None:
        nonlocal smpl_keypoints, num_timesteps, scale, fps
        nonlocal root_positions, root_wxyz, joint_angles

        filepath = asset_dir / filename
        print(f"Loading {filepath} ...")
        smpl_data = onp.load(filepath)
        smpl_keypoints = smpl_data["joints"]
        num_timesteps = smpl_keypoints.shape[0]
        assert smpl_keypoints.shape == (num_timesteps, 45, 3)
        fps = float(smpl_data.get("fps", 30.0))

        scale = compute_global_scale(
            robot, rest_joint_angles, smpl_keypoints[0], smpl_indices, g1_indices
        )
        print(f"  {num_timesteps} frames, scale={scale:.4f}")

        # Reset slider range and position.
        timestep_slider.max = num_timesteps - 1
        timestep_slider.value = 0

        # Clear stale retargeting results.
        root_positions = None
        root_wxyz = None
        joint_angles = None

    @refresh_button.on_click
    def refresh_files(_) -> None:
        updated = scan_npz_files(asset_dir)
        data_dropdown.options = updated
        print(f"Refreshed file list: {len(updated)} files found")

    @load_button.on_click
    def on_load_click(_: None) -> None:
        load_trajectory(data_dropdown.value)
        do_retarget(None)

    # Create button
    gen_button = server.gui.add_button("Retarget!")

    # Main retargeting loop. Called when "Retarget!" button is pressed, and also after loading a new trajectory.
    @gen_button.on_click
    def do_retarget(_: None) -> None:
        nonlocal root_positions, root_wxyz, joint_angles
        if smpl_keypoints is None:
            return
        gen_button.disabled = True

        # Load objective weights from current slider values.
        weights = dict(
            w_match=sl_w_match.value,
            w_smooth=sl_w_smooth.value,
            w_rest=sl_w_rest.value,
            w_limit=sl_w_limit.value,
        )

        # Construct retargeting problem and objective functions.
        problem = build_problem(
            robot, smpl_keypoints, smpl_indices, g1_indices, rest_joint_angles, scale
        )
        T = problem.target_kps.shape[0]
        n_dof = problem.rest_joint_angles.shape[0]

        cost_fn, residual_fn = build_objective(problem, **weights)

        # Compute initialization via smart heuristic, then run optimizer.
        pelvis_positions = smpl_keypoints[:, int(smpl_indices[0]), :] * scale
        x0 = pack_trajectory(
            jnp.tile(rest_joint_angles, (T, 1)),
            jaxlie.SE3.from_translation(pelvis_positions).log(),
        )

        method = opt_dropdown.value
        n_iters = sl_n_iters.value
        print(f"\n[{method}] Running {n_iters} iterations on {T} frames ...")
        t0 = time.time()

        # Create a step_fn using a "closure" over the cost_fn and residual_fn, and any hyperparameters.
        # Since our run_optimizer function only takes a step_fn, we need to use partial application to "bake in" the cost_fn, residual_fn, and hyperparameters for the selected optimizer.

        if method == "gradient_descent":
            step_fn = partial(gradient_descent_step, cost_fn, lr=sl_lr.value)
            state0 = {}

        elif method == "gauss_newton":
            step_fn = partial(gauss_newton_step, residual_fn, damping=sl_damping.value)
            state0 = {}

        elif method == "line_search":
            step_fn = partial(
                gradient_descent_line_search_step,
                cost_fn,
                candidate_alphas=jnp.logspace(-4, 0, 13),
            )
            state0 = {}

        elif method == "gauss_newton_matrix_free":
            step_fn = partial(
                gauss_newton_matrix_free_step,
                residual_fn,
                damping=sl_damping.value,
            )
            state0 = {}

        elif method == "custom":
            x0, state0 = custom_optimizer_init(cost_fn, residual_fn, x0, problem)
            # If you have hyperparameters for your custom optimizer, you can pass them via partial here, and add sliders for them in the GUI section above.
            step_fn = partial(custom_optimizer_step, cost_fn, residual_fn)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Run optimizer.
        x_opt = run_optimizer(step_fn, x0, state0, n_iters)
        x_opt.block_until_ready()
        elapsed = time.time() - t0

        # Print final cost and update label.
        final_cost = float(cost_fn(x_opt))
        print(f"  Done in {elapsed:.2f}s — final cost: {final_cost:.6f}")
        cost_label.value = f"{final_cost:.6f}  ({elapsed:.1f}s)"

        # Unpack solution.
        joint_angles, pose_vecs = unpack_trajectory(x_opt, T, n_dof)
        T_root = jaxlie.SE3.exp(pose_vecs)
        root_wxyz = T_root.rotation().wxyz
        root_positions = T_root.translation()

        gen_button.disabled = False

    # -- Load initial trajectory and retarget --
    load_trajectory("walking1.npz")
    do_retarget(None)
    assert root_positions is not None and joint_angles is not None

    # -- SMPL skeleton overlay --
    keypoints_handle = server.scene.add_point_cloud(
        "/smpl_keypoints",
        points=onp.array(smpl_keypoints[0, :24] * scale),
        colors=onp.array([255, 0, 0], dtype=onp.uint8),
        point_size=0.025,
    )
    bones_handle = server.scene.add_line_segments(
        "/human/bones",
        points=onp.array(
            [
                [smpl_keypoints[0, j1] * scale, smpl_keypoints[0, j2] * scale]
                for j1, j2 in SMPL_BODY_SKELETON
            ]
        ),
        colors=onp.array([0, 0, 255], dtype=onp.uint8),
        line_width=0.5,
    )

    # -- Playback loop --
    while True:
        with server.atomic():
            # Update timestep if playing.
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % max(
                    num_timesteps, 1
                )
            t = timestep_slider.value

            # Update robot pose from retargeting result if available, otherwise just show the SMPL keypoints.
            if root_positions is not None and joint_angles is not None:
                base_frame.wxyz = onp.array(root_wxyz[t])
                base_frame.position = onp.array(root_positions[t])
                urdf_vis.update_cfg(onp.array(joint_angles[t]))

            # Visualize SMPL keypoints and skeleton.
            if smpl_keypoints is not None:
                keypoints_handle.points = onp.array(smpl_keypoints[t, :24] * scale)
                bones_handle.points = onp.array(
                    [
                        [
                            smpl_keypoints[t, j1] * scale,
                            smpl_keypoints[t, j2] * scale,
                        ]
                        for j1, j2 in SMPL_BODY_SKELETON
                    ]
                )
        time.sleep(1 / fps)


if __name__ == "__main__":
    main()