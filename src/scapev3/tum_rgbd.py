"""TUM RGB-D adapter for the generic 3DScape fusion engine."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
from PIL import Image

from scapev3.rgbd import (
    CameraIntrinsics,
    CameraPose,
    RGBDFrame,
    ReliabilityProvider,
    ReconstructionResult,
    ReconstructionSettings,
    reconstruct_rgbd_frames,
)


TUM_DEPTH_SCALE_M = 1.0 / 5000.0
TUM_INTRINSICS = {
    "freiburg1": (517.3, 516.5, 318.6, 255.3),
    "freiburg2": (520.9, 521.0, 325.1, 249.7),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}
T = TypeVar("T")


@dataclass(frozen=True)
class TUMFileEntry:
    """One timestamped file entry from ``rgb.txt`` or ``depth.txt``."""

    timestamp: float
    path: Path


@dataclass(frozen=True)
class TUMPoseEntry:
    """One timestamped TUM camera-to-world pose."""

    timestamp: float
    translation: np.ndarray
    quaternion_xyzw: np.ndarray


def reconstruct_tum_rgbd(
    *,
    scan_dir: str | Path,
    output_dir: str | Path,
    max_frames: int = 90,
    target_fps: float = 3.0,
    pixel_stride: int = 2,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
    min_reliability: float | None = None,
    use_reliability_weights: bool = True,
    reliability_weight_floor: float = 0.05,
    voxel_size_m: float = 0.04,
    max_points: int = 800_000,
    intrinsics_preset: str = "auto",
    max_association_delta_sec: float = 0.04,
    max_pose_delta_sec: float = 0.04,
    reliability_provider: ReliabilityProvider | None = None,
) -> ReconstructionResult:
    """Fuse a TUM RGB-D sequence into a metric point cloud."""

    scan_dir = Path(scan_dir)
    if not scan_dir.exists():
        raise FileNotFoundError(f"TUM RGB-D scan directory does not exist: {scan_dir}")
    frames = list_tum_rgbd_frames(
        scan_dir=scan_dir,
        intrinsics_preset=intrinsics_preset,
        max_association_delta_sec=max_association_delta_sec,
        max_pose_delta_sec=max_pose_delta_sec,
    )
    resolved_preset = resolve_tum_intrinsics_preset(scan_dir, intrinsics_preset)
    return reconstruct_rgbd_frames(
        frames=frames,
        output_dir=output_dir,
        dataset_name="tum_rgbd",
        output_prefix="tum_rgbd",
        settings=ReconstructionSettings(
            max_frames=max_frames,
            target_fps=target_fps,
            pixel_stride=pixel_stride,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            min_confidence=None,
            min_reliability=min_reliability,
            use_reliability_weights=use_reliability_weights,
            reliability_weight_floor=reliability_weight_floor,
            voxel_size_m=voxel_size_m,
            max_points=max_points,
        ),
        reliability_provider=reliability_provider,
        extra_manifest={
            "scan_dir": str(scan_dir),
            "intrinsics_preset": resolved_preset,
            "max_association_delta_sec": max_association_delta_sec,
            "max_pose_delta_sec": max_pose_delta_sec,
            "depth_scale_m": TUM_DEPTH_SCALE_M,
            "camera_convention": (
                "TUM groundtruth trajectory is treated as camera-to-world; "
                "depth is backprojected with positive camera z and image y down."
            ),
        },
    )


def list_tum_rgbd_frames(
    *,
    scan_dir: str | Path,
    intrinsics_preset: str = "auto",
    max_association_delta_sec: float = 0.04,
    max_pose_delta_sec: float = 0.04,
) -> list[RGBDFrame]:
    """Load TUM RGB-D metadata as generic RGB-D frames."""

    if max_association_delta_sec <= 0:
        raise ValueError("max_association_delta_sec must be positive")
    if max_pose_delta_sec <= 0:
        raise ValueError("max_pose_delta_sec must be positive")
    scan_dir = Path(scan_dir)
    rgb_entries = parse_tum_file_list(scan_dir / "rgb.txt", scan_dir=scan_dir)
    depth_entries = parse_tum_file_list(scan_dir / "depth.txt", scan_dir=scan_dir)
    pose_entries = parse_tum_groundtruth(scan_dir / "groundtruth.txt")
    if not rgb_entries:
        raise RuntimeError(f"No TUM RGB entries found in: {scan_dir / 'rgb.txt'}")
    if not depth_entries:
        raise RuntimeError(f"No TUM depth entries found in: {scan_dir / 'depth.txt'}")
    if not pose_entries:
        raise RuntimeError(f"No TUM groundtruth poses found in: {scan_dir / 'groundtruth.txt'}")

    depth_times = [entry.timestamp for entry in depth_entries]
    pose_times = [entry.timestamp for entry in pose_entries]
    resolved_preset = resolve_tum_intrinsics_preset(scan_dir, intrinsics_preset)
    first_depth = np.asarray(Image.open(depth_entries[0].path))
    if first_depth.ndim != 2:
        raise ValueError(f"Expected single-channel TUM depth image: {depth_entries[0].path}")
    height, width = first_depth.shape
    intrinsics = tum_intrinsics_for_image(width=width, height=height, preset=resolved_preset)

    frames: list[RGBDFrame] = []
    for rgb_entry in rgb_entries:
        depth_entry = nearest_by_timestamp(
            depth_entries,
            depth_times,
            rgb_entry.timestamp,
            max_delta=max_association_delta_sec,
        )
        if depth_entry is None:
            continue
        pose_entry = nearest_by_timestamp(
            pose_entries,
            pose_times,
            rgb_entry.timestamp,
            max_delta=max_pose_delta_sec,
        )
        if pose_entry is None:
            continue
        rotation = quaternion_xyzw_to_rotation_matrix(pose_entry.quaternion_xyzw)
        frames.append(
            RGBDFrame(
                timestamp=rgb_entry.timestamp,
                depth_path=str(depth_entry.path),
                rgb_path=str(rgb_entry.path),
                confidence_path=None,
                intrinsics=intrinsics,
                pose=CameraPose(
                    timestamp=pose_entry.timestamp,
                    rotation=rotation,
                    translation=pose_entry.translation.astype(np.float64),
                ),
                depth_scale_m=TUM_DEPTH_SCALE_M,
                name=f"tum_{rgb_entry.timestamp:.6f}",
            )
        )
    if not frames:
        raise RuntimeError("No TUM RGB-D frames could be associated with depth and poses")
    return frames


def parse_tum_file_list(path: str | Path, *, scan_dir: str | Path) -> list[TUMFileEntry]:
    """Parse TUM ``rgb.txt`` or ``depth.txt`` timestamp/file lists."""

    path = Path(path)
    scan_dir = Path(scan_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing TUM association file: {path}")
    entries: list[TUMFileEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            timestamp = float(parts[0])
            file_path = scan_dir / parts[1]
            if file_path.exists():
                entries.append(TUMFileEntry(timestamp=timestamp, path=file_path))
    return sorted(entries, key=lambda entry: entry.timestamp)


def parse_tum_groundtruth(path: str | Path) -> list[TUMPoseEntry]:
    """Parse TUM groundtruth trajectory: timestamp tx ty tz qx qy qz qw."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing TUM groundtruth file: {path}")
    poses: list[TUMPoseEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 8:
                continue
            values = [float(value) for value in parts]
            timestamp = values[0]
            translation = np.asarray(values[1:4], dtype=np.float64)
            quaternion = np.asarray(values[4:8], dtype=np.float64)
            if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quaternion)):
                continue
            poses.append(
                TUMPoseEntry(
                    timestamp=timestamp,
                    translation=translation,
                    quaternion_xyzw=quaternion,
                )
            )
    return sorted(poses, key=lambda pose: pose.timestamp)


def resolve_tum_intrinsics_preset(scan_dir: str | Path, preset: str) -> str:
    """Resolve ``auto`` to a Freiburg preset using the sequence path name."""

    preset = preset.lower()
    if preset != "auto":
        if preset not in TUM_INTRINSICS:
            raise ValueError(f"Unknown TUM intrinsics preset: {preset}")
        return preset
    name = str(scan_dir).lower()
    if "freiburg2" in name or "fr2" in name:
        return "freiburg2"
    if "freiburg3" in name or "fr3" in name:
        return "freiburg3"
    return "freiburg1"


def tum_intrinsics_for_image(*, width: int, height: int, preset: str) -> CameraIntrinsics:
    """Return TUM intrinsics with dataset focal/principal values and image size."""

    if preset not in TUM_INTRINSICS:
        raise ValueError(f"Unknown TUM intrinsics preset: {preset}")
    fx, fy, cx, cy = TUM_INTRINSICS[preset]
    return CameraIntrinsics(
        width=int(width),
        height=int(height),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
    )


def quaternion_xyzw_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion ``[qx, qy, qz, qw]`` to a 3x3 rotation matrix."""

    qx, qy, qz, qw = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def nearest_by_timestamp(
    entries: list[T],
    times: list[float],
    timestamp: float,
    *,
    max_delta: float,
) -> T | None:
    """Return the nearest timestamped entry within ``max_delta`` seconds."""

    if not entries:
        return None
    index = bisect_left(times, timestamp)
    candidates: list[int] = []
    if index < len(entries):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    best_index = min(candidates, key=lambda idx: abs(times[idx] - timestamp))
    if abs(times[best_index] - timestamp) > max_delta:
        return None
    return entries[best_index]
