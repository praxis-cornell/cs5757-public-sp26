"""run_slam.py — Estimate a warehouse drone trajectory via Gauss-Newton SLAM."""

# -- Path / JAX config (must precede jax imports) --------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Imports ----------------------------------------------------------------------
import jax
import jax.numpy as jnp
import jaxlie
import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.patches import Rectangle

from solution import SlamState, build_stacked_residual_fn, gauss_newton_step

# -- Constants --------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"


# ── Helpers ──────────────────────────────────────────────────────────────────────


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    """Load the .npz dataset into a plain dict."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


def build_se3(quaternions_wxyz: np.ndarray, positions: np.ndarray) -> jaxlie.SE3:
    """Construct a batched SE3 from (N,4) wxyz quaternions and (N,3) positions."""
    wxyz_xyz = jnp.concatenate(
        [jnp.asarray(quaternions_wxyz), jnp.asarray(positions)], axis=-1
    )
    return jaxlie.SE3(wxyz_xyz=wxyz_xyz)


def diagonal_sqrt_info(trans_std: float, rot_std: float) -> jnp.ndarray:
    """Diagonal 6×6 sqrt-information matrix for SE3 residuals.

    Layout: [translation (3) | rotation (3)], matching jaxlie SE3.log().
    """
    return jnp.diag(jnp.array([1.0 / trans_std] * 3 + [1.0 / rot_std] * 3))


def position_rmse(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    """Root-mean-square position error across timesteps."""
    return float(np.sqrt(np.mean(np.sum((estimated - ground_truth) ** 2, axis=-1))))


# ── Plotting ─────────────────────────────────────────────────────────────────────


def plot_results(
    gt_positions: np.ndarray,
    init_positions: np.ndarray,
    est_positions: np.ndarray,
    gt_landmarks: np.ndarray,
    prior_landmarks: np.ndarray,
    est_landmarks: np.ndarray,
    shelf_row_y: np.ndarray,
    shelf_row_depth: float,
    shelf_length: float,
    rmse_final: float,
    save_path: Path,
    suptitle: str = "Warehouse SLAM: Trajectory Estimation",
) -> None:
    """Three-panel comparison: ground truth, odometry init, and GN estimate."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(suptitle, fontweight="bold")

    panels = [
        ("Ground Truth", gt_positions, None),
        ("Initial (Odometry)", init_positions, ("Prior tags", prior_landmarks)),
        (
            f"Gauss-Newton  (RMSE {rmse_final:.2f} m)",
            est_positions,
            ("Est. tags", est_landmarks),
        ),
    ]

    for ax, (title, traj, extra_lm) in zip(axes, panels):
        # Shelf rows
        for ry in shelf_row_y:
            ax.add_patch(
                Rectangle(
                    (0, ry - shelf_row_depth / 2),
                    shelf_length,
                    shelf_row_depth,
                    fc="#d4a574",
                    ec="#8b6914",
                    alpha=0.35,
                    lw=0.8,
                )
            )

        # Ground-truth tags, trajectory, start marker
        ax.scatter(
            gt_landmarks[:, 0],
            gt_landmarks[:, 1],
            c="red",
            s=6,
            alpha=0.4,
            label="Tags (GT)",
        )
        ax.plot(traj[:, 0], traj[:, 1], "b-", lw=0.8, alpha=0.7, label="Trajectory")
        ax.scatter(*traj[0, :2], c="green", s=60, marker="^", zorder=5, label="Start")

        # Per-panel extra landmarks (prior or estimated)
        if extra_lm is not None:
            label, pts = extra_lm
            ax.scatter(pts[:, 0], pts[:, 1], c="orange", s=6, alpha=0.4, label=label)

        ax.set(title=title, xlabel="x (m)", ylabel="y (m)", aspect="equal")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {save_path}")
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────────


def plot_jtj_spy(
    residual_fn,
    state: SlamState,
    save_path: Path,
) -> None:
    """Compute J^T J and save a spy plot of its sparsity structure."""
    T = state.drone_state.wxyz_xyz.shape[0]
    N = state.landmark_pos.shape[0]
    n_params = T * 6 + N * 3
    delta0 = jnp.zeros(n_params)

    J = jax.jit(jax.jacfwd(lambda delta: residual_fn(state, delta)))(
        delta0
    )  # (n_residuals, n_params)
    JtJ = np.asarray(J.T @ J)

    threshold = 1e-6 * np.abs(JtJ).max()

    _, ax = plt.subplots(figsize=(8, 8))
    ax.spy(np.abs(JtJ) > threshold, markersize=0.5, color="steelblue")
    ax.set_title(r"Sparsity of $J^\top J$ (Hessian approximation)", fontsize=12)
    ax.set_xlabel("Parameter index")
    ax.set_ylabel("Parameter index")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved spy plot → {save_path}")
    plt.show()


def main(
    dataset: Path = DATA_DIR / "dataset_clean.npz",
    n_iters: int = 5,
    huber_delta: float | None = 1.0,
    spy_hessian: bool = False,
) -> None:
    # ╭─────────────────────────────────────────╮
    # │  1. Load dataset                        │
    # ╰─────────────────────────────────────────╯
    print(f"Loading dataset from {dataset} ...")
    d = load_dataset(dataset)

    use_outliers = "outlier" in dataset.name
    suptitle = (
        "Warehouse SLAM: Trajectory Estimation (with outliers)"
        if use_outliers
        else "Warehouse SLAM: Trajectory Estimation"
    )

    # ╭─────────────────────────────────────────╮
    # │  2. Build initial state                 │
    # ╰─────────────────────────────────────────╯
    # Poses  : noisy odometry integration (from data generator).
    # Landmarks: coarse prior positions.
    init_poses = build_se3(d["init_quaternions"], d["init_positions"])
    prior_landmarks = jnp.asarray(d["prior_landmarks"])

    state = SlamState(drone_state=init_poses, landmark_pos=prior_landmarks)

    # ╭─────────────────────────────────────────╮
    # │  3. Assemble measurements & noise       │
    # ╰─────────────────────────────────────────╯
    # Odometry (relative SE3 between consecutive poses)
    odom_meas = build_se3(d["odom_quaternions"], d["odom_translations"])
    odom_i = jnp.asarray(d["odom_i"])
    odom_j = jnp.asarray(d["odom_j"])
    sqrt_info_odom = diagonal_sqrt_info(
        float(d["odom_trans_noise_std"]),
        float(d["odom_rot_noise_std"]),
    )

    # Landmark observations (body-frame positions)
    obs_pose_idx = jnp.asarray(d["obs_pose_indices"])
    obs_lm_idx = jnp.asarray(d["obs_landmark_indices"])
    obs_z_body = jnp.asarray(d["obs_body_positions"])
    sqrt_info_obs = (1.0 / float(d["obs_noise_std"])) * jnp.eye(3)

    # Landmark prior
    sqrt_info_lm_prior = (1.0 / float(d["landmark_prior_noise_std"])) * jnp.eye(3)

    # ╭─────────────────────────────────────────╮
    # │  4. Build residual & run Gauss-Newton   │
    # ╰─────────────────────────────────────────╯
    residual_fn = build_stacked_residual_fn(
        odom_i,
        odom_j,
        odom_meas,
        sqrt_info_odom,
        obs_pose_idx,
        obs_lm_idx,
        obs_z_body,
        sqrt_info_obs,
        lm_prior_pos=prior_landmarks,
        sqrt_info_lm_prior=sqrt_info_lm_prior,
        huber_delta=huber_delta,
    )

    if spy_hessian:
        plot_jtj_spy(
            residual_fn,
            state,
            save_path=Path("plots") / "jtj_spy.pdf",
        )
        return

    step = jax.jit(gauss_newton_step, static_argnames=("f",))

    huber_str = f"δ = {huber_delta}" if huber_delta is not None else "disabled"
    print(f"\nRunning {n_iters} Gauss-Newton iterations (Huber {huber_str}) ...")
    print(f"  {'Iter':>4}  {'Cost':>14}")
    print(f"  {'----':>4}  {'-'*14}")
    for i in range(n_iters):
        state, cost = step(state, residual_fn)
        print(f"  {i:>4}  {float(cost):>14.4f}")

    # ╭─────────────────────────────────────────╮
    # │  5. Evaluate & visualize                │
    # ╰─────────────────────────────────────────╯
    est_positions = np.asarray(state.drone_state.wxyz_xyz[:, 4:])
    est_landmarks = np.asarray(state.landmark_pos)
    gt_positions = d["gt_positions"]

    rmse_init = position_rmse(d["init_positions"], gt_positions)
    rmse_final = position_rmse(est_positions, gt_positions)
    print(f"\nPosition RMSE:  init = {rmse_init:.3f} m  →  final = {rmse_final:.3f} m")

    plot_results(
        gt_positions=gt_positions,
        init_positions=d["init_positions"],
        est_positions=est_positions,
        gt_landmarks=d["gt_landmarks"],
        prior_landmarks=np.asarray(prior_landmarks),
        est_landmarks=est_landmarks,
        shelf_row_y=d["shelf_row_y"],
        shelf_row_depth=float(d["shelf_row_depth"]),
        shelf_length=float(d["shelf_length"]),
        rmse_final=rmse_final,
        save_path=Path("plots") / "slam_result.pdf",
        suptitle=suptitle,
    )


if __name__ == "__main__":
    tyro.cli(main)
