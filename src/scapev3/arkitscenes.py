"""ARKitScenes raw RGB-D reconstruction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

from scapev3.rgbd import (
    CameraIntrinsics,
    CameraPose,
    RGBDFrame,
    ReliabilityProvider,
    ReconstructionResult,
    ReconstructionSettings,
    backproject_depth,
    camera_to_world,
    load_confidence as load_generic_confidence,
    load_depth_m as load_generic_depth_m,
    reconstruct_rgbd_frames,
    resize_or_crop_rgb,
)


TIMESTAMP_RE = re.compile(r"_(\d+(?:\.\d+)?)\.")


ARKitIntrinsics = CameraIntrinsics
ARKitPose = CameraPose


@dataclass(frozen=True)
class ARKitFrame:
    """Depth/confidence/intrinsics files for one timestamp."""

    timestamp: float
    depth_path: str
    confidence_path: str | None
    intrinsics_path: str


ARKitReconstructionResult = ReconstructionResult


def reconstruct_arkitscenes_rgbd(
    *,
    scan_dir: str | Path,
    output_dir: str | Path,
    video_path: str | Path | None = None,
    rgb_dir: str | Path | None = None,
    max_frames: int = 90,
    target_fps: float = 3.0,
    pixel_stride: int = 2,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
    min_confidence: int = 2,
    min_reliability: float | None = None,
    use_reliability_weights: bool = True,
    reliability_weight_floor: float = 0.05,
    voxel_size_m: float = 0.04,
    max_points: int = 800_000,
    pose_max_delta_sec: float = 0.08,
    reliability_provider: ReliabilityProvider | None = None,
) -> ARKitReconstructionResult:
    """Fuse ARKitScenes metric depth frames into a world-space point cloud.

    This path uses real ARKit trajectory and depth when available. It is meant
    as a practical evaluation baseline before adding learned reliability or
    upstream pose/depth estimation.
    """

    scan_dir = Path(scan_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not scan_dir.exists():
        raise FileNotFoundError(f"ARKitScenes scan directory does not exist: {scan_dir}")
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be >= 1")
    if min_confidence < 0 or min_confidence > 2:
        raise ValueError("min_confidence should be 0, 1, or 2 for ARKit confidence maps")

    depth_dir = scan_dir / "lowres_depth"
    confidence_dir = scan_dir / "confidence"
    intrinsics_dir = scan_dir / "lowres_wide_intrinsics"
    traj_path = scan_dir / "lowres_wide.traj"
    if video_path is None:
        mov_candidates = sorted(scan_dir.glob("*.mov"))
        video_path = mov_candidates[0] if mov_candidates else None
    if rgb_dir is None:
        candidate_rgb_dir = scan_dir / "lowres_wide"
        rgb_dir = candidate_rgb_dir if candidate_rgb_dir.exists() else None

    raw_frames = list_arkitscenes_frames(
        depth_dir=depth_dir,
        intrinsics_dir=intrinsics_dir,
        confidence_dir=confidence_dir if confidence_dir.exists() else None,
    )
    trajectory = load_trajectory(traj_path)
    if not trajectory:
        raise RuntimeError(f"No trajectory poses were loaded from: {traj_path}")

    rgb_by_time: dict[float, Path] = {}
    if rgb_dir and Path(rgb_dir).exists():
        rgb_by_time = {_timestamp_from_path(path): path for path in sorted(Path(rgb_dir).glob("*.png"))}
    frames: list[RGBDFrame] = []
    skipped_by_pose = 0
    for raw_frame in raw_frames:
        pose, pose_delta = nearest_pose(trajectory, raw_frame.timestamp)
        if pose_delta > pose_max_delta_sec:
            skipped_by_pose += 1
            continue
        intrinsics = load_intrinsics(raw_frame.intrinsics_path)
        rgb_path = _nearest_timestamp_path(rgb_by_time, raw_frame.timestamp, max_delta=0.04)
        frames.append(
            RGBDFrame(
                timestamp=raw_frame.timestamp,
                depth_path=raw_frame.depth_path,
                confidence_path=raw_frame.confidence_path,
                intrinsics=intrinsics,
                pose=pose,
                rgb_path=str(rgb_path) if rgb_path else None,
                depth_scale_m=0.001,
                name=f"arkit_{raw_frame.timestamp:.3f}",
            )
        )
    if not frames:
        raise RuntimeError("No ARKitScenes frames had usable nearby poses")

    color_source = _ColorSource(
        video_path=Path(video_path) if video_path else None,
        rgb_dir=Path(rgb_dir) if rgb_dir else None,
        first_timestamp=raw_frames[0].timestamp,
    )
    try:
        return reconstruct_rgbd_frames(
            frames=frames,
            output_dir=output_dir,
            dataset_name="arkitscenes",
            output_prefix="arkitscenes_rgbd",
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
            color_provider=lambda frame: color_source.colors_for_frame(
                frame.timestamp,
                frame.intrinsics.width,
                frame.intrinsics.height,
            ),
            reliability_provider=reliability_provider,
            extra_manifest={
                "scan_dir": str(scan_dir),
                "video_path": str(video_path) if video_path else None,
                "rgb_dir": str(rgb_dir) if rgb_dir else None,
                "frames_matched_before_pose_filter": len(raw_frames),
                "frames_skipped_by_pose_delta": skipped_by_pose,
                "pose_max_delta_sec": pose_max_delta_sec,
                "camera_convention": (
                    "ARKitScenes benchmark convention: image x/y with positive camera z; "
                    "trajectory is inverted to camera-to-world before fusion"
                ),
            },
        )
    finally:
        color_source.close()


def list_arkitscenes_frames(
    *,
    depth_dir: str | Path,
    intrinsics_dir: str | Path,
    confidence_dir: str | Path | None = None,
) -> list[ARKitFrame]:
    """Match ARKitScenes depth, confidence, and intrinsics files by timestamp."""

    depth_dir = Path(depth_dir)
    intrinsics_dir = Path(intrinsics_dir)
    if not depth_dir.exists():
        raise FileNotFoundError(f"Missing depth directory: {depth_dir}")
    if not intrinsics_dir.exists():
        raise FileNotFoundError(f"Missing intrinsics directory: {intrinsics_dir}")
    confidence_dir = Path(confidence_dir) if confidence_dir else None

    intrinsics_by_time = {_timestamp_from_path(path): path for path in intrinsics_dir.glob("*.pincam")}
    confidence_by_time = (
        {_timestamp_from_path(path): path for path in confidence_dir.glob("*.png")}
        if confidence_dir and confidence_dir.exists()
        else {}
    )
    frames: list[ARKitFrame] = []
    for depth_path in sorted(depth_dir.glob("*.png")):
        timestamp = _timestamp_from_path(depth_path)
        intrinsics_path = _nearest_timestamp_path(intrinsics_by_time, timestamp, max_delta=0.03)
        if intrinsics_path is None:
            continue
        confidence_path = _nearest_timestamp_path(confidence_by_time, timestamp, max_delta=0.03)
        frames.append(
            ARKitFrame(
                timestamp=timestamp,
                depth_path=str(depth_path),
                confidence_path=str(confidence_path) if confidence_path else None,
                intrinsics_path=str(intrinsics_path),
            )
        )
    if not frames:
        raise RuntimeError("No ARKitScenes frames could be matched")
    return frames


def select_frames_by_rate(
    frames: list[ARKitFrame],
    *,
    target_fps: float,
    max_frames: int,
) -> list[ARKitFrame]:
    """Select frames with roughly even temporal coverage."""

    if not frames:
        return []
    start = frames[0].timestamp
    end = frames[-1].timestamp
    if end <= start:
        return frames[:max_frames]
    desired = np.arange(start, end + 1e-6, 1.0 / target_fps)
    if desired.shape[0] > max_frames:
        desired = np.linspace(start, end, num=max_frames)
    timestamps = np.asarray([frame.timestamp for frame in frames], dtype=np.float64)
    selected: list[ARKitFrame] = []
    used: set[int] = set()
    for timestamp in desired:
        index = int(np.argmin(np.abs(timestamps - timestamp)))
        if index not in used:
            selected.append(frames[index])
            used.add(index)
    return selected[:max_frames]


def load_intrinsics(path: str | Path) -> ARKitIntrinsics:
    """Load a single-line ARKitScenes ``.pincam`` file."""

    parts = Path(path).read_text(encoding="utf-8").strip().split()
    if len(parts) != 6:
        raise ValueError(f"Expected 6 fields in .pincam file: {path}")
    return ARKitIntrinsics(
        width=int(float(parts[0])),
        height=int(float(parts[1])),
        fx=float(parts[2]),
        fy=float(parts[3]),
        cx=float(parts[4]),
        cy=float(parts[5]),
    )


def load_depth_m(path: str | Path) -> np.ndarray:
    """Load ARKitScenes uint16 millimeter depth as meters."""

    return load_generic_depth_m(path, scale_m=0.001)


def load_confidence(path: str | Path) -> np.ndarray:
    """Load ARKitScenes uint8 confidence map."""

    return load_generic_confidence(path)


def load_trajectory(path: str | Path) -> list[ARKitPose]:
    """Load ARKitScenes axis-angle trajectory poses.

    Apple's benchmark loader parses the trajectory as a world-to-camera
    transform, then inverts it before projecting frame points into world space.
    We follow that convention here.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {path}")
    poses: list[ARKitPose] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 7:
                raise ValueError(f"Expected 7 fields in trajectory line: {line}")
            timestamp = float(parts[0])
            axis_angle = np.asarray([float(value) for value in parts[1:4]], dtype=np.float64)
            translation = np.asarray([float(value) for value in parts[4:7]], dtype=np.float64)
            world_to_camera = np.eye(4, dtype=np.float64)
            world_to_camera[:3, :3] = axis_angle_to_rotation_matrix(axis_angle)
            world_to_camera[:3, 3] = translation
            camera_to_world = np.linalg.inv(world_to_camera)
            poses.append(
                ARKitPose(
                    timestamp=timestamp,
                    rotation=camera_to_world[:3, :3],
                    translation=camera_to_world[:3, 3],
                )
            )
    return sorted(poses, key=lambda pose: pose.timestamp)


def nearest_pose(poses: list[ARKitPose], timestamp: float) -> tuple[ARKitPose, float]:
    """Find the nearest trajectory pose by timestamp."""

    pose_times = np.asarray([pose.timestamp for pose in poses], dtype=np.float64)
    index = int(np.argmin(np.abs(pose_times - timestamp)))
    delta = abs(float(pose_times[index]) - timestamp)
    return poses[index], delta


def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle radians to a 3x3 rotation matrix."""

    axis_angle = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis_angle / theta
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def backproject_arkit_depth(
    depth_m: np.ndarray,
    intrinsics: ARKitIntrinsics,
    *,
    confidence: np.ndarray | None = None,
    min_confidence: int = 2,
    pixel_stride: int = 2,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject ARKit depth to camera coordinates.

    Output follows the ARKitScenes benchmark loader convention: image x/y and
    positive camera z before applying the camera-to-world pose.
    """

    return backproject_depth(
        depth_m,
        intrinsics,
        confidence=confidence,
        min_confidence=min_confidence,
        pixel_stride=pixel_stride,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )


def arkit_camera_to_world(points_camera: np.ndarray, pose: ARKitPose) -> np.ndarray:
    """Transform ARKit camera-space points into world coordinates."""

    return camera_to_world(points_camera, pose)


class _ColorSource:
    def __init__(
        self,
        *,
        video_path: Path | None,
        rgb_dir: Path | None,
        first_timestamp: float,
    ) -> None:
        self.video_path = video_path
        self.rgb_dir = rgb_dir
        self.first_timestamp = first_timestamp
        self._capture = cv2.VideoCapture(str(video_path)) if video_path and video_path.exists() else None
        self._rgb_files = sorted(rgb_dir.glob("*.png")) if rgb_dir and rgb_dir.exists() else []
        self._rgb_by_time = {_timestamp_from_path(path): path for path in self._rgb_files}

    def colors_for_frame(self, timestamp: float, width: int, height: int) -> np.ndarray:
        if self._rgb_by_time:
            rgb_path = _nearest_timestamp_path(self._rgb_by_time, timestamp, max_delta=0.04)
            if rgb_path is not None:
                image = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
                return _resize_or_crop_rgb(image, width=width, height=height)
        if self._capture and self._capture.isOpened():
            relative_sec = max(0.0, timestamp - self.first_timestamp)
            self._capture.set(cv2.CAP_PROP_POS_MSEC, relative_sec * 1000.0)
            ok, bgr = self._capture.read()
            if ok:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return _resize_or_crop_rgb(rgb, width=width, height=height)
        return np.full((height, width, 3), 210, dtype=np.uint8)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()


def _resize_or_crop_rgb(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    return resize_or_crop_rgb(image, width=width, height=height)


def _timestamp_from_path(path: Path) -> float:
    match = TIMESTAMP_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse timestamp from filename: {path}")
    return float(match.group(1))


def _nearest_timestamp_path(paths_by_time: dict[float, Path], timestamp: float, *, max_delta: float) -> Path | None:
    if not paths_by_time:
        return None
    times = np.asarray(list(paths_by_time.keys()), dtype=np.float64)
    index = int(np.argmin(np.abs(times - timestamp)))
    nearest_time = float(times[index])
    if abs(nearest_time - timestamp) > max_delta:
        return None
    return paths_by_time[nearest_time]
