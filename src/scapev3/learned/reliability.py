"""PyTorch depth reliability model and geometry-derived labels."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from scapev3.rgbd import (
    CameraIntrinsics,
    CameraPose,
    RGBDFrame,
    backproject_depth,
    camera_to_world,
    resize_or_crop_rgb,
)


class ConvBlock(nn.Module):
    """Small convolution block used by TinyReliabilityNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TinyReliabilityNet(nn.Module):
    """Tiny U-Net that predicts per-pixel depth reliability logits.

    Input channels are intended to be RGB, normalized depth, valid-depth mask,
    and depth gradients. The output is one logit per pixel; apply sigmoid to get
    a probability that the pixel should be trusted during fusion.
    """

    def __init__(self, in_channels: int = 7, base_channels: int = 16) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.enc1 = ConvBlock(self.in_channels, self.base_channels)
        self.enc2 = ConvBlock(self.base_channels, self.base_channels * 2)
        self.bottleneck = ConvBlock(self.base_channels * 2, self.base_channels * 4)
        self.dec2 = ConvBlock(self.base_channels * 4 + self.base_channels * 2, self.base_channels * 2)
        self.dec1 = ConvBlock(self.base_channels * 2 + self.base_channels, self.base_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head = nn.Conv2d(self.base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("TinyReliabilityNet input must have shape [B, C, H, W]")
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))
        bottleneck = self.bottleneck(self.pool(skip2))
        up2 = _match_spatial(self.up2(bottleneck), skip2)
        dec2 = self.dec2(torch.cat([up2, skip2], dim=1))
        up1 = _match_spatial(self.up1(dec2), skip1)
        dec1 = self.dec1(torch.cat([up1, skip1], dim=1))
        return self.head(dec1)


def make_reliability_features(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    max_depth_m: float = 8.0,
) -> np.ndarray:
    """Create a 7-channel feature tensor from RGB and metric depth.

    Returns a float32 array with shape [7, H, W]:
    RGB/255, normalized depth, valid mask, depth-gradient-x, depth-gradient-y.
    """

    if depth_m.ndim != 2:
        raise ValueError("depth_m must have shape [H, W]")
    if max_depth_m <= 0:
        raise ValueError("max_depth_m must be positive")
    height, width = depth_m.shape
    rgb = resize_or_crop_rgb(rgb, width=width, height=height).astype(np.float32) / 255.0
    valid = np.isfinite(depth_m) & (depth_m > 0)
    depth_norm = np.zeros_like(depth_m, dtype=np.float32)
    depth_norm[valid] = np.clip(depth_m[valid], 0.0, max_depth_m) / max_depth_m
    grad_y, grad_x = np.gradient(depth_norm)
    grad_x = np.clip(grad_x, -1.0, 1.0).astype(np.float32)
    grad_y = np.clip(grad_y, -1.0, 1.0).astype(np.float32)
    return np.concatenate(
        [
            rgb.transpose(2, 0, 1).astype(np.float32),
            depth_norm[None, ...],
            valid.astype(np.float32)[None, ...],
            grad_x[None, ...],
            grad_y[None, ...],
        ],
        axis=0,
    )


def predict_reliability(
    *,
    model: nn.Module,
    features: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Run a reliability network and return a [H, W] probability map."""

    if features.ndim != 3:
        raise ValueError("features must have shape [C, H, W]")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(features[None, ...]).to(device=device, dtype=torch.float32)
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return probabilities.astype(np.float32)


def make_reliability_provider(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    max_depth_m: float = 8.0,
) -> Callable[[RGBDFrame, np.ndarray, np.ndarray], np.ndarray]:
    """Load a checkpoint and return a provider usable by reconstruct_rgbd_frames."""

    model = load_reliability_checkpoint(checkpoint_path, device=device)

    def provider(_frame: RGBDFrame, depth_m: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        features = make_reliability_features(rgb=rgb, depth_m=depth_m, max_depth_m=max_depth_m)
        return predict_reliability(model=model, features=features, device=device)

    return provider


def save_reliability_checkpoint(
    path: str | Path,
    *,
    model: TinyReliabilityNet,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a reliability model checkpoint with its architecture config."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_name": "TinyReliabilityNet",
        "model_config": {
            "in_channels": model.in_channels,
            "base_channels": model.base_channels,
        },
        "model_state": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_reliability_checkpoint(path: str | Path, *, device: str = "cpu") -> TinyReliabilityNet:
    """Load a TinyReliabilityNet checkpoint."""

    checkpoint = torch.load(Path(path), map_location=device)
    if "model_state" in checkpoint:
        config = checkpoint.get("model_config", {})
        model = TinyReliabilityNet(**config)
        model.load_state_dict(checkpoint["model_state"])
    else:
        model = TinyReliabilityNet()
        model.load_state_dict(checkpoint)
    return model.to(device)


def build_multiview_reliability_label(
    *,
    source_depth_m: np.ndarray,
    source_intrinsics: CameraIntrinsics,
    source_pose: CameraPose,
    target_depth_m: np.ndarray,
    target_intrinsics: CameraIntrinsics,
    target_pose: CameraPose,
    pixel_stride: int = 2,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
    depth_tolerance_m: float = 0.08,
    relative_tolerance: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Create labels by checking whether source depth agrees with a target view.

    Returns ``(labels, supervision_mask)`` with shape [H, W]. Pixels are labeled
    reliable when their backprojected 3D point lands in the target frame and its
    projected target depth agrees with the observed target depth. Pixels outside
    the target view or without valid target depth are masked out.
    """

    if source_depth_m.shape != (source_intrinsics.height, source_intrinsics.width):
        raise ValueError("source_depth_m shape must match source_intrinsics")
    if target_depth_m.shape != (target_intrinsics.height, target_intrinsics.width):
        raise ValueError("target_depth_m shape must match target_intrinsics")
    if depth_tolerance_m <= 0:
        raise ValueError("depth_tolerance_m must be positive")
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative")

    labels = np.zeros_like(source_depth_m, dtype=np.float32)
    supervision = np.zeros_like(source_depth_m, dtype=bool)
    points_camera, source_pixels = backproject_depth(
        source_depth_m,
        source_intrinsics,
        pixel_stride=pixel_stride,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    if points_camera.shape[0] == 0:
        return labels, supervision
    points_world = camera_to_world(points_camera, source_pose)
    points_target = world_to_camera(points_world, target_pose)
    target_pixels, projected_z, in_front = project_camera_points(points_target, target_intrinsics)
    rounded = np.rint(target_pixels).astype(np.int64)
    in_bounds = (
        in_front
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < target_intrinsics.width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < target_intrinsics.height)
    )
    if not np.any(in_bounds):
        return labels, supervision
    source_pixels = source_pixels[in_bounds]
    rounded = rounded[in_bounds]
    projected_z = projected_z[in_bounds]
    observed = target_depth_m[rounded[:, 1], rounded[:, 0]]
    comparable = np.isfinite(observed) & (observed >= min_depth_m) & (observed <= max_depth_m)
    if not np.any(comparable):
        return labels, supervision
    source_pixels = source_pixels[comparable]
    observed = observed[comparable]
    projected_z = projected_z[comparable]
    tolerance = np.maximum(depth_tolerance_m, relative_tolerance * projected_z)
    reliable = np.abs(observed - projected_z) <= tolerance
    supervision[source_pixels[:, 1], source_pixels[:, 0]] = True
    labels[source_pixels[:, 1], source_pixels[:, 0]] = reliable.astype(np.float32)
    return labels, supervision


def build_multiview_reliability_target(
    *,
    source_depth_m: np.ndarray,
    source_intrinsics: CameraIntrinsics,
    source_pose: CameraPose,
    target_depths_m: Sequence[np.ndarray],
    target_intrinsics: Sequence[CameraIntrinsics],
    target_poses: Sequence[CameraPose],
    pixel_stride: int = 2,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
    depth_tolerance_m: float = 0.08,
    relative_tolerance: float = 0.03,
    min_target_observations: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a soft reliability target from agreement across nearby views.

    A one-neighbor target is brittle: a true surface can be hidden by occlusion,
    motion, or a bad neighboring depth pixel. This helper compares the source
    depth against several target views and labels each supervised source pixel
    by the fraction of target views that agree with it.

    The returned label is continuous in ``[0, 1]``:

    - ``1.0`` means every comparable target view agreed with the source depth.
    - ``0.0`` means comparable target views consistently disagreed.
    - intermediate values mean mixed multi-view evidence.
    """

    if min_target_observations < 1:
        raise ValueError("min_target_observations must be >= 1")
    if not (len(target_depths_m) == len(target_intrinsics) == len(target_poses)):
        raise ValueError("target depth, intrinsics, and pose lists must have equal length")
    if not target_depths_m:
        raise ValueError("at least one target view is required")

    reliable_votes = np.zeros_like(source_depth_m, dtype=np.float32)
    comparable_votes = np.zeros_like(source_depth_m, dtype=np.float32)
    for target_depth_m, intrinsics, pose in zip(
        target_depths_m,
        target_intrinsics,
        target_poses,
        strict=True,
    ):
        labels, mask = build_multiview_reliability_label(
            source_depth_m=source_depth_m,
            source_intrinsics=source_intrinsics,
            source_pose=source_pose,
            target_depth_m=target_depth_m,
            target_intrinsics=intrinsics,
            target_pose=pose,
            pixel_stride=pixel_stride,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            depth_tolerance_m=depth_tolerance_m,
            relative_tolerance=relative_tolerance,
        )
        reliable_votes[mask] += labels[mask]
        comparable_votes[mask] += 1.0

    supervision = comparable_votes >= float(min_target_observations)
    soft_labels = np.zeros_like(source_depth_m, dtype=np.float32)
    soft_labels[supervision] = reliable_votes[supervision] / comparable_votes[supervision]
    return soft_labels, supervision


def world_to_camera(points_world: np.ndarray, pose: CameraPose) -> np.ndarray:
    """Transform world-space points into a camera coordinate frame."""

    return (points_world.astype(np.float64) - pose.translation[None, :]) @ pose.rotation


def project_camera_points(
    points_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-space points into pixel coordinates."""

    z = points_camera[:, 2]
    in_front = np.isfinite(z) & (z > 1e-6)
    safe_z = np.where(in_front, z, 1.0)
    u = intrinsics.fx * points_camera[:, 0] / safe_z + intrinsics.cx
    v = intrinsics.fy * points_camera[:, 1] / safe_z + intrinsics.cy
    pixels = np.stack([u, v], axis=-1)
    return pixels.astype(np.float64), z.astype(np.float64), in_front


def _match_spatial(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape[-2:] == target.shape[-2:]:
        return source
    return nn.functional.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)
