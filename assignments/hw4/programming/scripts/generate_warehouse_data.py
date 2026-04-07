"""
Synthetic 3D SLAM Dataset Generator — Warehouse Drone Inspection
================================================================
Drone inspects an Amazon-style warehouse with AprilTags on shelving units.
Per-aisle spline segments connected by straight-line transitions that route
around shelf endcaps.
"""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import tyro
from scipy.interpolate import CubicSpline


@dataclass
class WarehouseDataConfig:
    # Warehouse geometry
    length: float = 40.0
    width: float = 24.0
    height: float = 5.0
    num_rows: int = 5
    row_depth: float = 1.5
    row_height: float = 4.0
    # Trajectory
    num_aisle_visits: int = 14
    pose_spacing: float = 0.8
    transition_spacing: float = 0.8
    endcap_x_margin: float = 2.0
    # Landmarks
    landmark_z_range: tuple[float, float] = (0.5, 4.0)
    landmark_y_jitter: float = 0.05
    landmark_prior_noise_std: float = 0.25
    # Noise
    odom_trans_noise_std: float = 0.05
    odom_rot_noise_std: float = 0.025
    odom_trans_bias: tuple[float, float, float] = (0.002, 0.001, 0.0005)
    max_observation_range: float = 6.0
    obs_noise_std: float = 0.1
    outlier_rate: float = 0.12
    # Output
    output_path: Path = Path("warehouse_slam_dataset.npz")
    seed: int = 42
    visualize: bool = True


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def compute_aisle_centers(row_y, row_depth):
    """Aisle y-centers: one before first row, between each pair, after last."""
    centers = [row_y[0] - 1.5 * row_depth]
    for i in range(len(row_y) - 1):
        centers.append((row_y[i] + row_y[i + 1]) / 2)
    centers.append(row_y[-1] + 1.5 * row_depth)
    return np.array(centers)


def compute_aisle_half_widths(aisle_centers, row_y, row_depth, warehouse_width):
    """Max lateral deviation per aisle (with 0.3 m clearance)."""
    faces = np.concatenate([row_y - row_depth / 2, row_y + row_depth / 2])
    hw = np.empty(len(aisle_centers))
    for i, ay in enumerate(aisle_centers):
        hw[i] = min(np.min(np.abs(ay - faces)), ay, warehouse_width - ay) - 0.3
    return hw


# ---------------------------------------------------------------------------
# Occlusion
# ---------------------------------------------------------------------------


def is_occluded_batch(p_drone, landmarks, row_y, row_depth, row_height, face_tol=0.1):
    """Vectorised shelf-occlusion test for one drone pose vs all landmarks.

    For each shelf row, checks whether the drone→landmark ray passes through
    the shelf slab (y-band × z < row_height).  A landmark mounted on a shelf
    face is exempt only when the drone is on the *same side* as that face
    (i.e. can actually see it without looking through the shelf).

    Returns bool array of shape (num_landmarks,).
    """
    n = len(landmarks)
    occluded = np.zeros(n, dtype=bool)
    dy = landmarks[:, 1] - p_drone[1]
    dz = landmarks[:, 2] - p_drone[2]
    parallel = np.abs(dy) < 1e-9

    for ry in row_y:
        y_lo = ry - row_depth / 2
        y_hi = ry + row_depth / 2

        # Both endpoints on the same side → ray doesn't cross this shelf
        same_side = ((p_drone[1] < y_lo) & (landmarks[:, 1] < y_lo)) | (
            (p_drone[1] > y_hi) & (landmarks[:, 1] > y_hi)
        )

        # Drone physically inside the shelf band (shouldn't happen, but safe)
        drone_inside = y_lo <= p_drone[1] <= y_hi

        # Own-face exemption: tag is on a face AND drone is on that face's
        # side of the shelf centre.  A tag on the far face is NOT exempt.
        on_lo = np.abs(landmarks[:, 1] - y_lo) < face_tol
        on_hi = np.abs(landmarks[:, 1] - y_hi) < face_tol
        own_face = (on_lo & (p_drone[1] < ry)) | (on_hi & (p_drone[1] > ry))

        eligible = ~same_side & ~parallel & ~drone_inside & ~own_face

        if not np.any(eligible):
            continue

        idx = np.where(eligible)[0]
        dy_e = dy[idx]
        dz_e = dz[idx]

        t_lo = (y_lo - p_drone[1]) / dy_e
        t_hi = (y_hi - p_drone[1]) / dy_e
        t_enter = np.maximum(np.minimum(t_lo, t_hi), 0.0)
        t_exit = np.minimum(np.maximum(t_lo, t_hi), 1.0)

        valid = t_enter < t_exit
        t_mid = (t_enter + t_exit) / 2
        z_cross = p_drone[2] + dz_e * t_mid

        occluded[idx] |= valid & (z_cross < row_height)

    return occluded


# ---------------------------------------------------------------------------
# Trajectory sampling
# ---------------------------------------------------------------------------


def sample_arc_length(waypoints, spacing):
    """Resample waypoints at uniform arc-length intervals (spline or linear)."""
    pts = np.asarray(waypoints)
    if len(pts) < 2:
        return pts
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total = cum[-1]
    if total < spacing:
        return pts
    s = np.linspace(0, total, max(2, int(total / spacing)))
    if len(pts) >= 4:
        interp = [CubicSpline(cum, pts[:, d], bc_type="clamped")(s) for d in range(3)]
    else:
        interp = [np.interp(s, cum, pts[:, d]) for d in range(3)]
    return np.column_stack(interp)


def sample_straight(p0, p1, spacing):
    dist = np.linalg.norm(p1 - p0)
    n = max(2, int(dist / spacing))
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) * p0 + t * p1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: WarehouseDataConfig) -> None:
    rng = np.random.default_rng(cfg.seed)

    # -- Warehouse geometry --
    row_y = np.linspace(4.0, 20.0, cfg.num_rows)
    aisle_y = compute_aisle_centers(row_y, cfg.row_depth)
    aisle_hw = compute_aisle_half_widths(aisle_y, row_y, cfg.row_depth, cfg.width)

    x_lo_endcap = -cfg.endcap_x_margin
    x_hi_endcap = cfg.length + cfg.endcap_x_margin

    print(
        f"Warehouse: {cfg.length}×{cfg.width}×{cfg.height} m, "
        f"{cfg.num_rows} rows, {len(aisle_y)} aisles"
    )

    # -- AprilTag placement --
    landmarks = []
    for ry in row_y:
        for sign in (-1, 1):
            face = ry + sign * cfg.row_depth / 2
            n = rng.integers(8, 15)
            x = rng.uniform(1.0, cfg.length - 1.0, n)
            y = face + rng.uniform(-cfg.landmark_y_jitter, cfg.landmark_y_jitter, n)
            z = rng.uniform(*cfg.landmark_z_range, size=n)
            landmarks.append(np.column_stack([x, y, z]))
        for x_end in (0.3, cfg.length - 0.3):
            n = rng.integers(1, 4)
            y = ry + rng.uniform(-cfg.row_depth / 2, cfg.row_depth / 2, n)
            z = rng.uniform(*cfg.landmark_z_range, size=n)
            landmarks.append(np.column_stack([np.full(n, x_end), y, z]))

    landmarks_gt = np.vstack(landmarks)
    n_lm = len(landmarks_gt)
    print(f"Placed {n_lm} AprilTags")

    landmarks_prior = landmarks_gt + rng.normal(
        0, cfg.landmark_prior_noise_std, landmarks_gt.shape
    )

    # -- Drone trajectory --
    n_aisles = len(aisle_y)
    base_visits = cfg.num_aisle_visits // n_aisles
    extra = cfg.num_aisle_visits % n_aisles
    extra_aisles = rng.choice(n_aisles, size=extra, replace=False)
    counts = np.full(n_aisles, base_visits)
    counts[extra_aisles] += 1
    aisle_order = np.repeat(np.arange(n_aisles), counts)
    rng.shuffle(aisle_order)
    segments = []
    prev_exit = None

    for ai in aisle_order:
        ay, hw = aisle_y[ai], aisle_hw[ai]
        left = rng.random() < 0.5
        nw = rng.integers(5, 8)
        xs = np.sort(
            rng.uniform(
                rng.uniform(2, 8), rng.uniform(cfg.length - 8, cfg.length - 2), nw
            )
        )
        if not left:
            xs = xs[::-1]
        ys = ay + rng.uniform(-hw * 0.6, hw * 0.6, nw)
        zs = rng.uniform(1.0, 3.5, nw)
        wp = np.column_stack([xs, ys, zs])
        entry = wp[0]

        if prev_exit is not None:
            ex = x_lo_endcap if prev_exit[0] < cfg.length / 2 else x_hi_endcap
            tz = (prev_exit[2] + entry[2]) / 2
            p1 = np.array([ex, prev_exit[1], tz])
            p2 = np.array([ex, entry[1], tz])
            for a, b in [(prev_exit, p1), (p1, p2), (p2, entry)]:
                segments.append(sample_straight(a, b, cfg.transition_spacing)[:-1])

        aisle_pts = sample_arc_length(wp, cfg.pose_spacing)
        aisle_pts[:, 1] = np.clip(aisle_pts[:, 1], ay - hw, ay + hw)
        segments.append(aisle_pts)
        prev_exit = aisle_pts[-1].copy()

    positions_gt = np.vstack(segments)
    positions_gt[:, 0] = np.clip(positions_gt[:, 0], x_lo_endcap, x_hi_endcap)
    positions_gt[:, 1] = np.clip(positions_gt[:, 1], 0.3, cfg.width - 0.3)
    positions_gt[:, 2] = np.clip(positions_gt[:, 2], 0.5, cfg.height - 0.5)
    n_poses = len(positions_gt)

    # Yaw from trajectory tangent
    tangents = np.diff(positions_gt, axis=0, prepend=positions_gt[:1])
    tangents[0] = tangents[1]  # first pose copies second
    yaw = np.arctan2(tangents[:, 1], tangents[:, 0])

    # Rotation matrices (batch) — jaxlie SO3.from_rpy_radians matches scipy ZYX convention
    roll = rng.normal(0, 0.02, n_poses)
    pitch = rng.normal(0, 0.02, n_poses)
    rotations_gt_so3 = jax.vmap(jaxlie.SO3.from_rpy_radians)(
        jnp.array(roll), jnp.array(pitch), jnp.array(yaw)
    )
    R_gt = np.array(rotations_gt_so3.as_matrix())  # (n_poses, 3, 3)

    print(f"Sampled {n_poses} poses")

    # -- Odometry --
    bias = np.array(cfg.odom_trans_bias)
    T_gt = jax.vmap(jaxlie.SE3.from_rotation_and_translation)(
        rotations_gt_so3, jnp.array(positions_gt)
    )

    # Relative transforms via SE3 inverse composition
    T_rel_gt = jax.vmap(lambda Ti, Tj: Ti.inverse() @ Tj)(
        jaxlie.SE3(wxyz_xyz=T_gt.wxyz_xyz[:-1]),
        jaxlie.SE3(wxyz_xyz=T_gt.wxyz_xyz[1:]),
    )
    t_gt = np.array(T_rel_gt.translation())  # body-frame translation (n-1, 3)

    t_noisy = t_gt + rng.normal(0, cfg.odom_trans_noise_std, t_gt.shape) + bias
    rot_noise_so3 = jax.vmap(jaxlie.SO3.exp)(
        jnp.array(rng.normal(0, cfg.odom_rot_noise_std, (n_poses - 1, 3)))
    )
    R_rel_noisy_so3 = jax.vmap(lambda R, N: R @ N)(T_rel_gt.rotation(), rot_noise_so3)
    odom_meas = jax.vmap(jaxlie.SE3.from_rotation_and_translation)(
        R_rel_noisy_so3, jnp.array(t_noisy)
    )

    print(f"{n_poses - 1} odometry edges")

    # -- Observations (vectorised per pose) --
    obs_pose, obs_lm, obs_body = [], [], []
    obs_outlier = []

    for i in range(n_poses):
        p_i, R_i = positions_gt[i], R_gt[i]
        RiT = R_i.T

        # Body-frame offsets for all landmarks
        delta = landmarks_gt - p_i
        dists = np.linalg.norm(delta, axis=1)
        in_range = dists <= cfg.max_observation_range

        if not np.any(in_range):
            continue

        # Occlusion check (only on in-range landmarks)
        candidate_idx = np.where(in_range)[0]
        occ = is_occluded_batch(
            p_i, landmarks_gt[candidate_idx], row_y, cfg.row_depth, cfg.row_height
        )
        visible_idx = candidate_idx[~occ]

        if len(visible_idx) == 0:
            continue

        p_body = (RiT @ delta[visible_idx].T).T
        noise = rng.normal(0, cfg.obs_noise_std, p_body.shape)
        p_body_noisy = p_body + noise

        # Outlier injection
        is_out = rng.random(len(visible_idx)) < cfg.outlier_rate
        lm_ids = visible_idx.copy()
        if np.any(is_out):
            n_out = int(np.sum(is_out))
            lm_ids[is_out] = rng.integers(0, n_lm, n_out)
            # Re-draw if same as true id (rare, accept the tiny bias)

        obs_pose.append(np.full(len(visible_idx), i, dtype=np.int32))
        obs_lm.append(lm_ids.astype(np.int32))
        obs_body.append(p_body_noisy)
        obs_outlier.append(is_out)

    obs_pose = np.concatenate(obs_pose)
    obs_lm = np.concatenate(obs_lm)
    obs_body = np.concatenate(obs_body)
    obs_outlier = np.concatenate(obs_outlier)
    n_obs = len(obs_pose)
    n_outliers = int(obs_outlier.sum())

    obs_per_pose = np.bincount(obs_pose, minlength=n_poses)
    obs_per_lm = np.bincount(obs_lm, minlength=n_lm)
    n_observed = int(np.sum(obs_per_lm > 0))

    print(
        f"{n_obs} observations ({n_outliers} outliers = "
        f"{100 * n_outliers / max(n_obs, 1):.1f}%)"
    )
    print(
        f"  Per pose: mean={obs_per_pose.mean():.1f}, "
        f"min={obs_per_pose.min()}, max={obs_per_pose.max()}"
    )
    print(f"  Landmarks observed: {n_observed}/{n_lm}")

    # -- Odometry integration (initial estimate) --
    def integrate_step(T_curr: jaxlie.SE3, T_rel_i: jaxlie.SE3):
        T_next = T_curr @ T_rel_i
        return T_next, T_next

    T_0 = jaxlie.SE3(wxyz_xyz=T_gt.wxyz_xyz[0])
    _, T_rest = jax.lax.scan(integrate_step, T_0, odom_meas)
    T_init = jaxlie.SE3(
        wxyz_xyz=jnp.concatenate([T_0.wxyz_xyz[None], T_rest.wxyz_xyz], axis=0)
    )
    positions_init = np.array(T_init.translation())

    landmarks_init = landmarks_prior.copy()

    drift_final = np.linalg.norm(positions_init[-1] - positions_gt[-1])
    drift_max = np.max(np.linalg.norm(positions_init - positions_gt, axis=1))
    print(f"Odometry drift: final={drift_final:.2f} m, max={drift_max:.2f} m")

    # -- Save --
    odom_i = np.arange(n_poses - 1, dtype=np.int32)
    odom_j = odom_i + 1

    np.savez(
        cfg.output_path,
        shelf_row_y=row_y,
        shelf_row_depth=np.float64(cfg.row_depth),
        shelf_length=np.float64(cfg.length),
        gt_positions=positions_gt,
        gt_quaternions=np.array(T_gt.rotation().wxyz),
        gt_landmarks=landmarks_gt,
        init_positions=positions_init,
        init_quaternions=np.array(T_init.rotation().wxyz),
        init_landmarks=landmarks_init,
        odom_i=odom_i,
        odom_j=odom_j,
        odom_translations=t_noisy,
        odom_quaternions=np.array(odom_meas.rotation().wxyz),
        obs_pose_indices=obs_pose,
        obs_landmark_indices=obs_lm,
        obs_body_positions=obs_body,
        obs_is_outlier=obs_outlier,
        odom_trans_noise_std=cfg.odom_trans_noise_std,
        odom_rot_noise_std=cfg.odom_rot_noise_std,
        obs_noise_std=cfg.obs_noise_std,
        prior_landmarks=landmarks_prior,
        landmark_prior_noise_std=cfg.landmark_prior_noise_std,
    )
    print(f"Saved to {cfg.output_path}")

    # -- Visualization (opt-in) --
    if cfg.visualize:
        _plot(cfg, positions_gt, positions_init, landmarks_gt, landmarks_prior, row_y)

    # -- Summary --
    print(f"\n{'=' * 55}")
    print(f"  Poses: {n_poses}  |  Landmarks: {n_lm} ({n_observed} observed)")
    print(f"  Odometry edges: {n_poses - 1}")
    print(f"  Observations: {n_obs} ({n_outliers} outliers)")
    print(f"  DOF: {n_poses * 6 + n_observed * 3}")
    print(f"  Drift: final={drift_final:.2f} m, max={drift_max:.2f} m")
    print(f"{'=' * 55}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot(cfg, positions_gt, positions_init, landmarks_gt, landmarks_prior, row_y):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Top view
    ax = axes[0]
    ax.set_title("Top View", fontweight="bold")
    for ry in row_y:
        ax.add_patch(
            Rectangle(
                (0, ry - cfg.row_depth / 2),
                cfg.length,
                cfg.row_depth,
                fc="#d4a574",
                ec="#8b6914",
                alpha=0.35,
                lw=0.8,
            )
        )
    ax.scatter(
        landmarks_prior[:, 0],
        landmarks_prior[:, 1],
        c="orange",
        s=8,
        alpha=0.5,
        zorder=2,
        label="Prior",
    )
    ax.scatter(
        landmarks_gt[:, 0],
        landmarks_gt[:, 1],
        c="red",
        s=8,
        alpha=0.6,
        zorder=3,
        label="Tags (GT)",
    )
    ax.plot(positions_gt[:, 0], positions_gt[:, 1], "b-", alpha=0.5, lw=0.6, label="GT")
    ax.plot(
        positions_init[:, 0],
        positions_init[:, 1],
        "orange",
        ls="--",
        alpha=0.4,
        lw=0.6,
        label="Odom",
    )
    ax.scatter(*positions_gt[0, :2], c="green", s=50, marker="^", zorder=5)
    ax.set(xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax.set_xlim(-4, cfg.length + 4)
    ax.set_ylim(-2, cfg.width + 2)
    ax.legend(fontsize=7)

    # Side view
    ax = axes[1]
    ax.set_title("Side View", fontweight="bold")
    ax.axhspan(0, cfg.row_height, alpha=0.08, color="brown")
    ax.scatter(
        landmarks_prior[:, 0],
        landmarks_prior[:, 2],
        c="orange",
        s=5,
        alpha=0.4,
        label="Prior",
    )
    ax.scatter(
        landmarks_gt[:, 0],
        landmarks_gt[:, 2],
        c="red",
        s=5,
        alpha=0.4,
        label="Tags (GT)",
    )
    ax.plot(positions_gt[:, 0], positions_gt[:, 2], "b-", alpha=0.5, lw=0.6)
    ax.set(xlabel="x (m)", ylabel="z (m)", ylim=(-0.5, cfg.height + 0.5))

    # 3D view
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.set_title("3D View", fontweight="bold")
    ax.scatter(*landmarks_prior.T, c="orange", s=4, alpha=0.3, label="Prior")
    ax.scatter(*landmarks_gt.T, c="red", s=4, alpha=0.3, label="Tags (GT)")
    ax.plot(*positions_gt.T, "b-", alpha=0.4, lw=0.5)
    ax.set(xlabel="x", ylabel="y", zlabel="z")

    plt.tight_layout()
    out = cfg.output_path.with_stem(cfg.output_path.stem + "_overview").with_suffix(
        ".png"
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    main(tyro.cli(WarehouseDataConfig))
