"""
Minimal loader: COLMAP sparse reconstruction -> arrays for jaxls factors.

    data = load("outputs/my_scene/colmap/sparse/0")

Gives you two types of factors:

  1. Landmark observations (per frame):
       data.frames[i].keypoints      # (K_i, 2) pixel coords
       data.frames[i].landmark_idx   # (K_i,)   index into data.landmarks

  2. Between factors (sequential smoothness):
       data.between_wxyz[i]          # (4,) relative rotation    T_rel = T_wc[i]^{-1} @ T_wc[i+1]
       data.between_t[i]             # (3,) relative translation

Initial guesses:
       data.poses_wxyz[i]            # (4,) world-from-camera, [w,x,y,z] for jaxlie.SO3
       data.poses_t[i]               # (3,) camera origin in world
       data.landmarks[j]             # (3,) landmark position

Camera:
       data.K                        # (3,3) intrinsic matrix
       data.fx, data.fy, data.cx, data.cy

Poses are sorted by filename (= frame order from video).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class FrameObs:
    """Observations for a single frame."""

    keypoints: np.ndarray  # (K, 2) float64 pixel coords
    landmark_idx: np.ndarray  # (K,)   int, indices into SLAMData.landmarks


@dataclass
class SLAMData:
    """Everything you need to build landmark + between factors in jaxls."""

    # Camera intrinsics (shared)
    K: np.ndarray  # (3, 3)
    fx: float
    fy: float
    cx: float
    cy: float
    W: float
    H: float

    # Initial pose guesses, sorted by frame order — world-from-camera
    poses_wxyz: np.ndarray  # (N, 4) quaternion [w, x, y, z]
    poses_t: np.ndarray  # (N, 3) camera position in world

    # Initial landmark positions
    landmarks: np.ndarray  # (M, 3)

    # Per-frame observations, length N, same order as poses
    frames: list[FrameObs]

    # Between-factor measurements: T_rel[i] = T_wc[i]^{-1} @ T_wc[i+1]
    between_wxyz: np.ndarray  # (N-1, 4)
    between_t: np.ndarray  # (N-1, 3)

    @property
    def n_poses(self) -> int:
        return len(self.poses_wxyz)

    @property
    def n_landmarks(self) -> int:
        return len(self.landmarks)

    @property
    def n_observations(self) -> int:
        return sum(len(f.keypoints) for f in self.frames)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def load(sparse_dir: str | Path) -> SLAMData:
    """
    Load COLMAP sparse reconstruction into minimal arrays for jaxls.

    Args:
        sparse_dir: e.g. "outputs/my_scene/colmap/sparse/0"
    """
    sparse_dir = Path(sparse_dir)
    cameras_raw = _read_cameras_bin(sparse_dir / "cameras.bin")
    images_raw = _read_images_bin(sparse_dir / "images.bin")
    points3d_raw = _read_points3d_bin(sparse_dir / "points3D.bin")

    # --- Camera intrinsics (use first camera) ---
    cam = next(iter(cameras_raw.values()))
    K, fx, fy, cx, cy = _extract_intrinsics(cam)

    # --- Sort images by filename (frame order from video) ---
    sorted_images = sorted(images_raw.values(), key=lambda im: im["name"])

    # --- Build contiguous landmark index ---
    all_pt3d_ids = set()
    for im in sorted_images:
        valid = im["point3d_ids"] >= 0
        all_pt3d_ids.update(im["point3d_ids"][valid].tolist())
    all_pt3d_ids = sorted(all_pt3d_ids)
    pt3d_to_idx = {pid: i for i, pid in enumerate(all_pt3d_ids)}

    landmarks = np.stack([points3d_raw[pid]["xyz"] for pid in all_pt3d_ids])

    # --- Extract poses (invert COLMAP's T_cw to T_wc) + per-frame obs ---
    poses_wxyz = []
    poses_t = []
    frames = []

    for im in sorted_images:
        R_cw = _quat_to_rotmat(im["quat_cw"])
        R_wc = R_cw.T
        t_wc = -R_wc @ im["t_cw"]

        poses_wxyz.append(_rotmat_to_quat(R_wc))
        poses_t.append(t_wc)

        valid = im["point3d_ids"] >= 0
        kp = im["keypoints"][valid]
        lm_idx = np.array(
            [pt3d_to_idx[pid] for pid in im["point3d_ids"][valid]],
            dtype=np.int32,
        )
        frames.append(FrameObs(keypoints=kp, landmark_idx=lm_idx))

    poses_wxyz = np.stack(poses_wxyz)
    poses_t = np.stack(poses_t)

    # --- Between factors: T_rel[i] = T_wc[i]^{-1} @ T_wc[i+1] ---
    n = len(sorted_images)
    between_wxyz = np.empty((n - 1, 4))
    between_t = np.empty((n - 1, 3))

    for i in range(n - 1):
        R_i = _quat_to_rotmat(poses_wxyz[i])
        R_j = _quat_to_rotmat(poses_wxyz[i + 1])
        between_wxyz[i] = _rotmat_to_quat(R_i.T @ R_j)
        between_t[i] = R_i.T @ (poses_t[i + 1] - poses_t[i])

    print(
        f"Loaded: {n} frames, {len(landmarks)} landmarks, "
        f"{sum(len(f.keypoints) for f in frames)} obs"
    )

    return SLAMData(
        K=K,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        W=cam["width"],
        H=cam["height"],
        poses_wxyz=poses_wxyz,
        poses_t=poses_t,
        landmarks=landmarks,
        frames=frames,
        between_wxyz=between_wxyz,
        between_t=between_t,
    )


# ---------------------------------------------------------------------------
# COLMAP binary parsers
# ---------------------------------------------------------------------------

_CAM_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _read_cameras_bin(path: Path) -> dict:
    cams = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cid, mid, w, h = struct.unpack("<IiQQ", f.read(24))
            name, np_ = _CAM_MODELS[mid]
            params = np.array(struct.unpack(f"<{np_}d", f.read(8 * np_)))
            cams[cid] = {"model": name, "width": w, "height": h, "params": params}
    return cams


def _read_images_bin(path: Path) -> dict:
    images = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (image_id,) = struct.unpack("<I", f.read(4))
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            (camera_id,) = struct.unpack("<I", f.read(4))

            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            name = name.decode("ascii")

            (n_pts,) = struct.unpack("<Q", f.read(8))
            keypoints = np.empty((n_pts, 2))
            point3d_ids = np.empty(n_pts, dtype=np.int64)
            for j in range(n_pts):
                x, y = struct.unpack("<2d", f.read(16))
                (pid,) = struct.unpack("<q", f.read(8))
                keypoints[j] = (x, y)
                point3d_ids[j] = pid

            images[image_id] = {
                "name": name,
                "camera_id": camera_id,
                "quat_cw": np.array([qw, qx, qy, qz]),
                "t_cw": np.array([tx, ty, tz]),
                "keypoints": keypoints,
                "point3d_ids": point3d_ids,
            }
    return images


def _read_points3d_bin(path: Path) -> dict:
    pts = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (pid,) = struct.unpack("<Q", f.read(8))
            x, y, z = struct.unpack("<3d", f.read(24))
            f.read(3)  # rgb
            (err,) = struct.unpack("<d", f.read(8))
            (track_len,) = struct.unpack("<Q", f.read(8))
            f.read(8 * track_len)  # skip track
            pts[pid] = {"xyz": np.array([x, y, z]), "error": err}
    return pts


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def _extract_intrinsics(cam: dict):
    p, model = cam["params"], cam["model"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        fx = fy = float(p[0])
        cx = float(p[1])
        cy = float(p[2])
    else:
        fx = float(p[0])
        fy = float(p[1])
        cx = float(p[2])
        cy = float(p[3])
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), fx, fy, cx, cy


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = np.array(
            [
                0.25 * s,
                (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
            ]
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array(
            [
                (R[2, 1] - R[1, 2]) / s,
                0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s,
            ]
        )
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array(
            [
                (R[0, 2] - R[2, 0]) / s,
                (R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s,
            ]
        )
    else:
        s = 2.0 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array(
            [
                (R[1, 0] - R[0, 1]) / s,
                (R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s,
                0.25 * s,
            ]
        )
    return q * np.sign(q[0])
