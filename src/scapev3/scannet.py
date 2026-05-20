"""ScanNet RGB-D adapter for the generic 3DScape fusion engine."""

from __future__ import annotations

from pathlib import Path

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


def reconstruct_scannet_rgbd(
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
    native_fps: float = 30.0,
    reliability_provider: ReliabilityProvider | None = None,
) -> ReconstructionResult:
    """Fuse a ScanNet scan folder into a metric point cloud.

    Expected ScanNet-style layout:

    ``color/<frame>.jpg``, ``depth/<frame>.png``, ``pose/<frame>.txt``, and
    ``intrinsic/intrinsic_depth.txt``.
    """

    scan_dir = Path(scan_dir)
    if not scan_dir.exists():
        raise FileNotFoundError(f"ScanNet scan directory does not exist: {scan_dir}")
    frames = list_scannet_frames(scan_dir=scan_dir, native_fps=native_fps)
    return reconstruct_rgbd_frames(
        frames=frames,
        output_dir=output_dir,
        dataset_name="scannet",
        output_prefix="scannet_rgbd",
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
            "native_fps": native_fps,
            "camera_convention": (
                "ScanNet pose files are treated as camera-to-world transforms; "
                "depth is backprojected with positive camera z and image y down."
            ),
        },
    )


def list_scannet_frames(*, scan_dir: str | Path, native_fps: float = 30.0) -> list[RGBDFrame]:
    """Load ScanNet frame metadata as generic RGB-D frames."""

    if native_fps <= 0:
        raise ValueError("native_fps must be positive")
    scan_dir = Path(scan_dir)
    color_dir = scan_dir / "color"
    depth_dir = scan_dir / "depth"
    pose_dir = scan_dir / "pose"
    intrinsic_dir = scan_dir / "intrinsic"
    for required in (color_dir, depth_dir, pose_dir, intrinsic_dir):
        if not required.exists():
            raise FileNotFoundError(f"Missing ScanNet directory: {required}")

    depth_paths = sorted(depth_dir.glob("*.png"), key=_frame_sort_key)
    if not depth_paths:
        raise RuntimeError(f"No ScanNet depth frames found in: {depth_dir}")
    intrinsics = load_scannet_intrinsics(
        intrinsic_dir=intrinsic_dir,
        first_depth_path=depth_paths[0],
    )

    frames: list[RGBDFrame] = []
    for index, depth_path in enumerate(depth_paths):
        frame_id = depth_path.stem
        pose_path = pose_dir / f"{frame_id}.txt"
        if not pose_path.exists():
            continue
        pose_matrix = load_scannet_pose(pose_path)
        if pose_matrix is None:
            continue
        color_path = _matching_color_path(color_dir, frame_id)
        if color_path is None:
            continue
        timestamp = _timestamp_from_frame_id(frame_id, fallback_index=index, native_fps=native_fps)
        frames.append(
            RGBDFrame(
                timestamp=timestamp,
                depth_path=str(depth_path),
                rgb_path=str(color_path),
                confidence_path=None,
                intrinsics=intrinsics,
                pose=CameraPose(
                    timestamp=timestamp,
                    rotation=pose_matrix[:3, :3].astype(np.float64),
                    translation=pose_matrix[:3, 3].astype(np.float64),
                ),
                depth_scale_m=0.001,
                name=f"scannet_{frame_id}",
            )
        )
    if not frames:
        raise RuntimeError("No valid ScanNet RGB-D frames could be matched")
    return frames


def load_scannet_intrinsics(*, intrinsic_dir: str | Path, first_depth_path: str | Path) -> CameraIntrinsics:
    """Load ScanNet depth intrinsics and infer the depth image size."""

    intrinsic_dir = Path(intrinsic_dir)
    candidates = [
        intrinsic_dir / "intrinsic_depth.txt",
        intrinsic_dir / "intrinsic_color.txt",
    ]
    intrinsic_path = next((path for path in candidates if path.exists()), None)
    if intrinsic_path is None:
        raise FileNotFoundError(f"Missing ScanNet intrinsic file in: {intrinsic_dir}")
    matrix = np.loadtxt(intrinsic_path, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 ScanNet intrinsic matrix: {intrinsic_path}")
    depth = np.asarray(Image.open(first_depth_path))
    if depth.ndim != 2:
        raise ValueError(f"Expected single-channel depth image: {first_depth_path}")
    height, width = depth.shape
    return CameraIntrinsics(
        width=int(width),
        height=int(height),
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
    )


def load_scannet_pose(path: str | Path) -> np.ndarray | None:
    """Load a ScanNet camera-to-world pose matrix, skipping invalid frames."""

    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 ScanNet pose matrix: {path}")
    if not np.all(np.isfinite(matrix)):
        return None
    return matrix


def _matching_color_path(color_dir: Path, frame_id: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png"):
        path = color_dir / f"{frame_id}{suffix}"
        if path.exists():
            return path
    return None


def _timestamp_from_frame_id(frame_id: str, *, fallback_index: int, native_fps: float) -> float:
    try:
        return float(int(frame_id)) / native_fps
    except ValueError:
        return float(fallback_index) / native_fps


def _frame_sort_key(path: Path) -> tuple[int, int | str]:
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)
