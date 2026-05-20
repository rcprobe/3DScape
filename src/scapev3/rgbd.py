"""Dataset-agnostic RGB-D reconstruction primitives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scapev3.config import save_json
from scapev3.ply import write_ply
from scapev3.voxel import save_occupancy_grid, trim_outliers_mask, voxel_downsample, weighted_voxel_downsample


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics for one depth image size."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CameraPose:
    """Camera-to-world pose for one RGB-D frame."""

    timestamp: float
    rotation: np.ndarray
    translation: np.ndarray


@dataclass(frozen=True)
class RGBDFrame:
    """One dataset-normalized RGB-D frame.

    The generic fusion engine only needs these fields. Dataset-specific loaders
    are responsible for converting raw dataset files into this representation.
    """

    timestamp: float
    depth_path: str
    intrinsics: CameraIntrinsics
    pose: CameraPose
    rgb_path: str | None = None
    confidence_path: str | None = None
    depth_scale_m: float = 0.001
    name: str | None = None


@dataclass(frozen=True)
class ReconstructionSettings:
    """Shared settings for RGB-D point-cloud fusion."""

    max_frames: int = 90
    target_fps: float = 3.0
    pixel_stride: int = 2
    min_depth_m: float = 0.2
    max_depth_m: float = 8.0
    min_confidence: int | None = None
    min_reliability: float | None = None
    use_reliability_weights: bool = True
    reliability_weight_floor: float = 0.05
    voxel_size_m: float = 0.04
    max_points: int = 800_000
    trim_percentile: float = 99.7


@dataclass(frozen=True)
class ReconstructionResult:
    """Artifacts written by the generic RGB-D fusion path."""

    output_dir: str
    raw_ply: str
    downsampled_ply: str
    occupancy_npz: str
    top_down_png: str
    manifest_json: str
    frames_used: int
    raw_points: int
    downsampled_points: int
    reliability_ply: str | None = None


ColorProvider = Callable[[RGBDFrame], np.ndarray]
ReliabilityProvider = Callable[[RGBDFrame, np.ndarray, np.ndarray], np.ndarray]


def reconstruct_rgbd_frames(
    *,
    frames: Sequence[RGBDFrame],
    output_dir: str | Path,
    dataset_name: str,
    output_prefix: str,
    settings: ReconstructionSettings,
    color_provider: ColorProvider | None = None,
    reliability_provider: ReliabilityProvider | None = None,
    extra_manifest: dict | None = None,
) -> ReconstructionResult:
    """Fuse normalized RGB-D frames into a world-space colored point cloud."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_settings(settings)
    if not frames:
        raise RuntimeError(f"No RGB-D frames were provided for dataset: {dataset_name}")

    sorted_frames = sorted(frames, key=lambda frame: frame.timestamp)
    selected = select_frames_by_rate(
        sorted_frames,
        target_fps=settings.target_fps,
        max_frames=settings.max_frames,
    )

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    all_reliability: list[np.ndarray] = []
    frame_reports: list[dict] = []
    for frame in selected:
        depth_m = load_depth_m(frame.depth_path, scale_m=frame.depth_scale_m)
        confidence = load_confidence(frame.confidence_path) if frame.confidence_path else None
        colors = _colors_for_frame(frame, color_provider=color_provider)
        reliability = (
            reliability_provider(frame, depth_m, colors) if reliability_provider is not None else None
        )
        points_camera, pixels = backproject_depth(
            depth_m,
            frame.intrinsics,
            confidence=confidence,
            min_confidence=settings.min_confidence,
            reliability=reliability,
            min_reliability=settings.min_reliability,
            pixel_stride=settings.pixel_stride,
            min_depth_m=settings.min_depth_m,
            max_depth_m=settings.max_depth_m,
        )
        if points_camera.shape[0] == 0:
            frame_reports.append(_frame_report(frame, used=False, reason="no valid points"))
            continue

        points_world = camera_to_world(points_camera, frame.pose).astype(np.float32)
        point_colors = colors[pixels[:, 1], pixels[:, 0]].astype(np.uint8)
        point_weights, point_reliability = _point_reliability_weights(
            reliability=reliability,
            pixels=pixels,
            use_reliability_weights=settings.use_reliability_weights,
            weight_floor=settings.reliability_weight_floor,
        )
        all_points.append(points_world)
        all_colors.append(point_colors)
        all_weights.append(point_weights)
        if point_reliability is not None:
            all_reliability.append(point_reliability)
        frame_reports.append(
            _frame_report(
                frame,
                used=True,
                points=int(points_world.shape[0]),
                reliability=point_reliability,
            )
        )

    if not all_points:
        raise RuntimeError(f"No {dataset_name} RGB-D points were fused. Check depth, poses, and filters.")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    weights = np.concatenate(all_weights, axis=0)
    reliability_values = np.concatenate(all_reliability, axis=0) if all_reliability else None
    keep = trim_outliers_mask(points, percentile=settings.trim_percentile)
    points = points[keep]
    colors = colors[keep]
    weights = weights[keep]
    if reliability_values is not None:
        reliability_values = reliability_values[keep]
    if points.shape[0] > settings.max_points:
        keep = np.linspace(0, points.shape[0] - 1, num=settings.max_points, dtype=np.int64)
        points = points[keep]
        colors = colors[keep]
        weights = weights[keep]
        if reliability_values is not None:
            reliability_values = reliability_values[keep]

    raw_ply = write_ply(output_dir / f"{output_prefix}_raw_metric.ply", points, colors)
    reliability_ply = None
    down_reliability = None
    if reliability_values is not None:
        down_points, down_colors, _down_weights, down_reliability = weighted_voxel_downsample(
            points,
            colors,
            weights,
            voxel_size=settings.voxel_size_m,
            scalar_values=reliability_values,
        )
    else:
        down_points, down_colors = voxel_downsample(points, colors, voxel_size=settings.voxel_size_m)
    down_ply = write_ply(output_dir / f"{output_prefix}_downsampled_metric.ply", down_points, down_colors)
    if down_reliability is not None:
        reliability_colors = reliability_to_colors(down_reliability)
        reliability_ply = write_ply(
            output_dir / f"{output_prefix}_reliability_downsampled_metric.ply",
            down_points,
            reliability_colors,
        )
    occupancy_npz = save_occupancy_grid(
        output_dir / f"{output_prefix}_occupied_only.npz",
        down_points,
        voxel_size_m=settings.voxel_size_m,
    )
    used_pose_centers = np.asarray(
        [frame.pose.translation for frame, report in zip(selected, frame_reports, strict=False) if report.get("used")],
        dtype=np.float32,
    )
    frames_used = sum(1 for report in frame_reports if report.get("used"))
    top_down_png = save_top_down_density(
        output_dir / f"{output_prefix}_top_down.png",
        down_points,
        trajectory_points=used_pose_centers,
    )
    manifest = {
        "dataset": dataset_name,
        "output_prefix": output_prefix,
        "raw_ply": str(raw_ply),
        "downsampled_ply": str(down_ply),
        "reliability_ply": str(reliability_ply) if reliability_ply else None,
        "occupancy_npz": str(occupancy_npz),
        "top_down_png": str(top_down_png),
        "frames_available": len(sorted_frames),
        "frames_selected": len(selected),
        "frames_used": frames_used,
        "raw_points": int(points.shape[0]),
        "downsampled_points": int(down_points.shape[0]),
        "learned_reliability_enabled": reliability_provider is not None,
        "reliability_summary": _summarize_values(reliability_values),
        "settings": {
            "max_frames": settings.max_frames,
            "target_fps": settings.target_fps,
            "pixel_stride": settings.pixel_stride,
            "min_depth_m": settings.min_depth_m,
            "max_depth_m": settings.max_depth_m,
            "min_confidence": settings.min_confidence,
            "min_reliability": settings.min_reliability,
            "use_reliability_weights": settings.use_reliability_weights,
            "reliability_weight_floor": settings.reliability_weight_floor,
            "voxel_size_m": settings.voxel_size_m,
            "max_points": settings.max_points,
            "trim_percentile": settings.trim_percentile,
        },
        "frame_reports": frame_reports,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest_json = save_json(manifest, output_dir / f"{output_prefix}_manifest.json")
    return ReconstructionResult(
        output_dir=str(output_dir),
        raw_ply=str(raw_ply),
        downsampled_ply=str(down_ply),
        occupancy_npz=str(occupancy_npz),
        top_down_png=str(top_down_png),
        manifest_json=str(manifest_json),
        frames_used=frames_used,
        raw_points=int(points.shape[0]),
        downsampled_points=int(down_points.shape[0]),
        reliability_ply=str(reliability_ply) if reliability_ply else None,
    )


def select_frames_by_rate(
    frames: Sequence[RGBDFrame],
    *,
    target_fps: float,
    max_frames: int,
) -> list[RGBDFrame]:
    """Select frames with roughly even temporal coverage."""

    if not frames:
        return []
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    sorted_frames = sorted(frames, key=lambda frame: frame.timestamp)
    start = sorted_frames[0].timestamp
    end = sorted_frames[-1].timestamp
    if end <= start:
        return sorted_frames[:max_frames]
    desired = np.arange(start, end + 1e-6, 1.0 / target_fps)
    if desired.shape[0] > max_frames:
        desired = np.linspace(start, end, num=max_frames)
    timestamps = np.asarray([frame.timestamp for frame in sorted_frames], dtype=np.float64)
    selected: list[RGBDFrame] = []
    used: set[int] = set()
    for timestamp in desired:
        index = int(np.argmin(np.abs(timestamps - timestamp)))
        if index not in used:
            selected.append(sorted_frames[index])
            used.add(index)
    return selected[:max_frames]


def load_depth_m(path: str | Path, *, scale_m: float = 0.001) -> np.ndarray:
    """Load a depth image and convert it to meters."""

    depth = np.asarray(Image.open(path))
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be single-channel: {path}")
    return depth.astype(np.float32) * float(scale_m)


def load_confidence(path: str | Path) -> np.ndarray:
    """Load an integer confidence map."""

    confidence = np.asarray(Image.open(path), dtype=np.uint8)
    if confidence.ndim != 2:
        raise ValueError(f"Confidence image must be single-channel: {path}")
    return confidence


def backproject_depth(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    confidence: np.ndarray | None = None,
    min_confidence: int | None = None,
    reliability: np.ndarray | None = None,
    min_reliability: float | None = None,
    pixel_stride: int = 2,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject metric depth to camera coordinates using pinhole intrinsics."""

    if depth_m.ndim != 2:
        raise ValueError("depth_m must have shape [H, W]")
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be >= 1")
    height, width = depth_m.shape
    if (width, height) != (intrinsics.width, intrinsics.height):
        raise ValueError(
            f"Depth shape {width}x{height} does not match intrinsics "
            f"{intrinsics.width}x{intrinsics.height}"
        )
    yy, xx = np.mgrid[0:height:pixel_stride, 0:width:pixel_stride]
    z_metric = depth_m[0:height:pixel_stride, 0:width:pixel_stride].astype(np.float64)
    valid = np.isfinite(z_metric) & (z_metric >= min_depth_m) & (z_metric <= max_depth_m)
    if confidence is not None and min_confidence is not None:
        if confidence.shape != depth_m.shape:
            raise ValueError("confidence must have the same shape as depth_m")
        conf = confidence[0:height:pixel_stride, 0:width:pixel_stride]
        valid &= conf >= min_confidence
    if reliability is not None and min_reliability is not None:
        if reliability.shape != depth_m.shape:
            raise ValueError("reliability must have the same shape as depth_m")
        rel = reliability[0:height:pixel_stride, 0:width:pixel_stride]
        valid &= np.isfinite(rel) & (rel >= min_reliability)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int64)

    x = (xx.astype(np.float64) - intrinsics.cx) * z_metric / intrinsics.fx
    y = (yy.astype(np.float64) - intrinsics.cy) * z_metric / intrinsics.fy
    points_camera = np.stack([x, y, z_metric], axis=-1)[valid].astype(np.float32)
    pixels = np.stack([xx, yy], axis=-1)[valid].astype(np.int64)
    return points_camera, pixels


def camera_to_world(points_camera: np.ndarray, pose: CameraPose) -> np.ndarray:
    """Transform camera-space points into world coordinates."""

    return (pose.rotation @ points_camera.astype(np.float64).T).T + pose.translation[None, :]


def reliability_to_colors(values: np.ndarray) -> np.ndarray:
    """Map reliability scores to high-contrast diagnostic RGB colors.

    Reliability probabilities often occupy a narrow range, so this visualization
    contrast-stretches scores between their 5th and 95th percentiles before
    assigning colors. Low reliability is blue/purple; high reliability is
    yellow, matching the viewer legend's highest-to-lowest ramp.
    """

    scores = np.clip(np.asarray(values, dtype=np.float32).reshape(-1), 0.0, 1.0)
    stretched = _contrast_stretch(scores)
    palette = np.asarray(
        [
            [91, 86, 162],
            [179, 90, 154],
            [232, 119, 131],
            [249, 178, 118],
            [245, 234, 158],
        ],
        dtype=np.uint8,
    )
    bins = np.minimum((stretched * len(palette)).astype(np.int64), len(palette) - 1)
    return palette[bins]


def _contrast_stretch(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(finite, [5, 95])
    if high - low < 1e-6:
        return np.clip(values, 0.0, 1.0).astype(np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def save_top_down_density(
    path: str | Path,
    points: np.ndarray,
    *,
    image_size: int = 900,
    trajectory_points: np.ndarray | None = None,
) -> Path:
    """Write a simple XZ top-down density preview."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if points.shape[0] == 0:
        raise ValueError("Cannot render top-down density for an empty point cloud")
    xz = points[:, [0, 2]].astype(np.float64)
    low = np.percentile(xz, 1, axis=0)
    high = np.percentile(xz, 99, axis=0)
    span = np.maximum(high - low, 1e-6)
    normalized = (xz - low[None, :]) / span[None, :]
    pixels = np.floor(normalized * (image_size - 1)).astype(np.int32)
    valid = np.all((pixels >= 0) & (pixels < image_size), axis=1)
    canvas = np.zeros((image_size, image_size), dtype=np.float32)
    np.add.at(canvas, (image_size - 1 - pixels[valid, 1], pixels[valid, 0]), 1.0)
    if float(canvas.max()) > 0:
        canvas = np.log1p(canvas)
        canvas = canvas / float(canvas.max())
    image = cv2.applyColorMap((canvas * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    if trajectory_points is not None and trajectory_points.size:
        trajectory_xz = trajectory_points[:, [0, 2]].astype(np.float64)
        trajectory_pixels = np.floor(((trajectory_xz - low[None, :]) / span[None, :]) * (image_size - 1))
        trajectory_pixels = trajectory_pixels.astype(np.int32)
        valid = np.all((trajectory_pixels >= 0) & (trajectory_pixels < image_size), axis=1)
        trajectory_pixels = trajectory_pixels[valid]
        for idx in range(1, trajectory_pixels.shape[0]):
            start = (int(trajectory_pixels[idx - 1, 0]), int(image_size - 1 - trajectory_pixels[idx - 1, 1]))
            end = (int(trajectory_pixels[idx, 0]), int(image_size - 1 - trajectory_pixels[idx, 1]))
            cv2.line(image, start, end, (255, 255, 255), 2, cv2.LINE_AA)
        for pixel in trajectory_pixels[:: max(1, trajectory_pixels.shape[0] // 24)]:
            center = (int(pixel[0]), int(image_size - 1 - pixel[1]))
            cv2.circle(image, center, 4, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.imwrite(str(path), image)
    return path


def resize_or_crop_rgb(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Resize RGB image to the depth/intrinsics size with light aspect correction."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB image must have shape [H, W, 3]")
    source_height, source_width = image.shape[:2]
    target_aspect = width / height
    source_aspect = source_width / source_height
    if abs(source_aspect - target_aspect) > 0.02:
        if source_aspect > target_aspect:
            new_width = int(round(source_height * target_aspect))
            left = max(0, (source_width - new_width) // 2)
            image = image[:, left : left + new_width]
        else:
            new_height = int(round(source_width / target_aspect))
            top = max(0, (source_height - new_height) // 2)
            image = image[top : top + new_height, :]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _colors_for_frame(frame: RGBDFrame, *, color_provider: ColorProvider | None) -> np.ndarray:
    if color_provider is not None:
        colors = color_provider(frame)
        return resize_or_crop_rgb(
            colors.astype(np.uint8),
            width=frame.intrinsics.width,
            height=frame.intrinsics.height,
        )
    if frame.rgb_path:
        image = np.asarray(Image.open(frame.rgb_path).convert("RGB"), dtype=np.uint8)
        return resize_or_crop_rgb(image, width=frame.intrinsics.width, height=frame.intrinsics.height)
    return np.full((frame.intrinsics.height, frame.intrinsics.width, 3), 210, dtype=np.uint8)


def _frame_report(
    frame: RGBDFrame,
    *,
    used: bool,
    points: int | None = None,
    reason: str | None = None,
    reliability: np.ndarray | None = None,
) -> dict:
    report = {
        "name": frame.name,
        "timestamp": frame.timestamp,
        "used": used,
        "depth_path": frame.depth_path,
        "rgb_path": frame.rgb_path,
        "confidence_path": frame.confidence_path,
    }
    if points is not None:
        report["points"] = points
    if reason:
        report["reason"] = reason
    summary = _summarize_values(reliability)
    if summary is not None:
        report["reliability"] = summary
    return report


def _point_reliability_weights(
    *,
    reliability: np.ndarray | None,
    pixels: np.ndarray,
    use_reliability_weights: bool,
    weight_floor: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    if reliability is None:
        return np.ones((pixels.shape[0],), dtype=np.float32), None
    scores = reliability[pixels[:, 1], pixels[:, 0]].astype(np.float32)
    scores = np.clip(scores, 0.0, 1.0)
    if not use_reliability_weights:
        return np.ones_like(scores, dtype=np.float32), scores
    weights = np.clip(scores, weight_floor, 1.0).astype(np.float32)
    return weights, scores


def _summarize_values(values: np.ndarray | None) -> dict[str, float] | None:
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _validate_settings(settings: ReconstructionSettings) -> None:
    if settings.max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if settings.target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if settings.pixel_stride < 1:
        raise ValueError("pixel_stride must be >= 1")
    if settings.min_depth_m <= 0:
        raise ValueError("min_depth_m must be positive")
    if settings.max_depth_m <= settings.min_depth_m:
        raise ValueError("max_depth_m must be greater than min_depth_m")
    if settings.min_confidence is not None and settings.min_confidence < 0:
        raise ValueError("min_confidence must be non-negative when provided")
    if settings.min_reliability is not None and not (0.0 <= settings.min_reliability <= 1.0):
        raise ValueError("min_reliability must be in [0, 1] when provided")
    if not 0.0 <= settings.reliability_weight_floor <= 1.0:
        raise ValueError("reliability_weight_floor must be in [0, 1]")
    if settings.voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    if settings.max_points < 1:
        raise ValueError("max_points must be >= 1")
