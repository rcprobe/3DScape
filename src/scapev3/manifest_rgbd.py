"""Generic posed RGB-D manifest adapter for the 3DScape fusion engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

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


DEFAULT_DEPTH_SCALE_M = 0.001


def reconstruct_manifest_rgbd(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    max_frames: int = 90,
    target_fps: float = 3.0,
    pixel_stride: int = 2,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
    min_confidence: int | None = None,
    min_reliability: float | None = None,
    use_reliability_weights: bool = True,
    reliability_weight_floor: float = 0.05,
    voxel_size_m: float = 0.04,
    max_points: int = 800_000,
    reliability_provider: ReliabilityProvider | None = None,
) -> ReconstructionResult:
    """Fuse any posed RGB-D manifest into a metric point cloud.

    The manifest adapter is the architecture escape hatch: once a dataset can
    provide RGB image paths, depth image paths, camera intrinsics, and
    camera-to-world poses, it can use the same fusion engine as ARKitScenes,
    ScanNet, and TUM RGB-D.
    """

    manifest_path = _resolve_manifest_path(manifest_path)
    manifest = load_rgbd_manifest(manifest_path)
    frames = list_manifest_rgbd_frames(manifest_path=manifest_path, manifest=manifest)
    name = str(manifest.get("name") or manifest_path.stem)
    return reconstruct_rgbd_frames(
        frames=frames,
        output_dir=output_dir,
        dataset_name="manifest_rgbd",
        output_prefix="manifest_rgbd",
        settings=ReconstructionSettings(
            max_frames=max_frames,
            target_fps=target_fps,
            pixel_stride=pixel_stride,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            min_confidence=min_confidence,
            min_reliability=min_reliability,
            use_reliability_weights=use_reliability_weights,
            reliability_weight_floor=reliability_weight_floor,
            voxel_size_m=voxel_size_m,
            max_points=max_points,
        ),
        reliability_provider=reliability_provider,
        extra_manifest={
            "scan_name": name,
            "manifest_path": str(manifest_path),
            "manifest_frame_count": len(frames),
            "camera_convention": (
                "Generic manifest poses are interpreted as camera-to-world transforms; "
                "depth is backprojected with positive camera z and image y down."
            ),
        },
    )


def load_rgbd_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate the top-level manifest JSON object."""

    path = _resolve_manifest_path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("RGB-D manifest must be a JSON object")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("RGB-D manifest must contain a non-empty 'frames' list")
    return manifest


def list_manifest_rgbd_frames(
    *,
    manifest_path: str | Path,
    manifest: dict[str, Any] | None = None,
) -> list[RGBDFrame]:
    """Convert a generic posed RGB-D manifest into normalized RGBDFrame items."""

    manifest_path = _resolve_manifest_path(manifest_path)
    manifest = manifest or load_rgbd_manifest(manifest_path)
    base_dir = _manifest_base_dir(manifest_path, manifest)
    global_intrinsics = manifest.get("intrinsics")
    global_depth_scale = float(manifest.get("depth_scale_m", DEFAULT_DEPTH_SCALE_M))
    frames_json = manifest["frames"]

    frames: list[RGBDFrame] = []
    for index, frame_json in enumerate(frames_json):
        if not isinstance(frame_json, dict):
            raise ValueError(f"Manifest frame {index} must be an object")
        depth_path = _resolve_required_path(
            frame_json.get("depth_path"),
            base_dir=base_dir,
            field=f"frames[{index}].depth_path",
        )
        rgb_path = _resolve_optional_path(frame_json.get("rgb_path"), base_dir=base_dir)
        confidence_path = _resolve_optional_path(frame_json.get("confidence_path"), base_dir=base_dir)
        intrinsics_json = frame_json.get("intrinsics", global_intrinsics)
        if intrinsics_json is None:
            raise ValueError(f"Missing intrinsics for manifest frame {index}")
        intrinsics = parse_intrinsics(intrinsics_json, depth_path=depth_path)
        pose = parse_pose(frame_json.get("pose"), timestamp=_timestamp_for_frame(frame_json, index))
        depth_scale_m = float(frame_json.get("depth_scale_m", global_depth_scale))
        if depth_scale_m <= 0:
            raise ValueError(f"frames[{index}].depth_scale_m must be positive")
        frames.append(
            RGBDFrame(
                timestamp=_timestamp_for_frame(frame_json, index),
                depth_path=str(depth_path),
                rgb_path=str(rgb_path) if rgb_path else None,
                confidence_path=str(confidence_path) if confidence_path else None,
                intrinsics=intrinsics,
                pose=pose,
                depth_scale_m=depth_scale_m,
                name=str(frame_json.get("name", f"manifest_{index:06d}")),
            )
        )
    return sorted(frames, key=lambda frame: frame.timestamp)


def parse_intrinsics(data: Any, *, depth_path: str | Path) -> CameraIntrinsics:
    """Parse object or matrix intrinsics and infer image size when needed."""

    depth_path = Path(depth_path)
    height, width = _depth_image_shape(depth_path)
    if isinstance(data, dict):
        matrix = data.get("matrix")
        if matrix is not None:
            fx, fy, cx, cy = _intrinsics_from_matrix(matrix)
        else:
            required = ("fx", "fy", "cx", "cy")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"Intrinsics object is missing keys: {missing}")
            fx, fy, cx, cy = (float(data["fx"]), float(data["fy"]), float(data["cx"]), float(data["cy"]))
        width = int(data.get("width", width))
        height = int(data.get("height", height))
    else:
        fx, fy, cx, cy = _intrinsics_from_matrix(data)
    if width <= 0 or height <= 0:
        raise ValueError("Intrinsics width and height must be positive")
    if fx <= 0 or fy <= 0:
        raise ValueError("Intrinsics fx and fy must be positive")
    return CameraIntrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)


def parse_pose(data: Any, *, timestamp: float) -> CameraPose:
    """Parse a camera-to-world pose from a 4x4 matrix or rotation/translation."""

    if data is None:
        raise ValueError("Manifest frame is missing pose")
    if isinstance(data, dict):
        matrix = data.get("camera_to_world", data.get("matrix"))
        if matrix is not None:
            pose_matrix = _as_pose_matrix(matrix)
            return _camera_pose_from_matrix(pose_matrix, timestamp=timestamp)
        if "rotation" not in data or "translation" not in data:
            raise ValueError("Pose object must contain 'matrix' or 'rotation' + 'translation'")
        rotation = np.asarray(data["rotation"], dtype=np.float64)
        translation = np.asarray(data["translation"], dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("Pose rotation must have shape 3x3")
        if translation.shape != (3,):
            raise ValueError("Pose translation must have shape [3]")
        _validate_finite(rotation, field="pose.rotation")
        _validate_finite(translation, field="pose.translation")
        return CameraPose(timestamp=timestamp, rotation=rotation, translation=translation)
    pose_matrix = _as_pose_matrix(data)
    return _camera_pose_from_matrix(pose_matrix, timestamp=timestamp)


def _resolve_manifest_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"RGB-D manifest does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"RGB-D manifest path is not a file: {path}")
    return path


def _manifest_base_dir(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    base_dir = manifest.get("base_dir")
    if base_dir is None:
        return manifest_path.parent
    path = Path(str(base_dir))
    return path if path.is_absolute() else manifest_path.parent / path


def _resolve_required_path(value: Any, *, base_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing required manifest path: {field}")
    path = Path(value)
    path = path if path.is_absolute() else base_dir / path
    if not path.exists():
        raise FileNotFoundError(f"Manifest path for {field} does not exist: {path}")
    return path


def _resolve_optional_path(value: Any, *, base_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"Optional manifest path must be a string when provided: {value!r}")
    path = Path(value)
    path = path if path.is_absolute() else base_dir / path
    if not path.exists():
        raise FileNotFoundError(f"Manifest optional path does not exist: {path}")
    return path


def _timestamp_for_frame(frame_json: dict[str, Any], index: int) -> float:
    timestamp = frame_json.get("timestamp", index)
    try:
        value = float(timestamp)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid timestamp for manifest frame {index}: {timestamp!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Timestamp for manifest frame {index} must be finite")
    return value


def _depth_image_shape(path: Path) -> tuple[int, int]:
    depth = np.asarray(Image.open(path))
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be single-channel: {path}")
    height, width = depth.shape
    return int(height), int(width)


def _intrinsics_from_matrix(data: Any) -> tuple[float, float, float, float]:
    matrix = np.asarray(data, dtype=np.float64)
    if matrix.shape not in ((3, 3), (4, 4)):
        raise ValueError("Intrinsics matrix must have shape 3x3 or 4x4")
    _validate_finite(matrix, field="intrinsics.matrix")
    return float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2])


def _as_pose_matrix(data: Any) -> np.ndarray:
    matrix = np.asarray(data, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("Pose matrix must have shape 4x4")
    _validate_finite(matrix, field="pose.matrix")
    return matrix


def _camera_pose_from_matrix(matrix: np.ndarray, *, timestamp: float) -> CameraPose:
    return CameraPose(
        timestamp=timestamp,
        rotation=matrix[:3, :3].astype(np.float64),
        translation=matrix[:3, 3].astype(np.float64),
    )


def _validate_finite(array: np.ndarray, *, field: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must contain only finite values")
