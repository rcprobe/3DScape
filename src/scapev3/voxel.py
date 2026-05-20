"""Point cloud cleanup and voxel utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    voxel_size: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-average points and colors without requiring Open3D."""

    down_points, down_colors, _down_weights, _down_scalars = weighted_voxel_downsample(
        points,
        colors,
        np.ones((points.shape[0],), dtype=np.float32),
        voxel_size=voxel_size,
    )
    return down_points, down_colors


def weighted_voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    weights: np.ndarray,
    *,
    voxel_size: float = 0.05,
    scalar_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Voxel-average points/colors with optional per-point confidence weights.

    ``weights`` lets learned reliability influence the representative point and
    color for each voxel without deleting all low-confidence geometry. When
    ``scalar_values`` is provided, it is averaged per voxel with the same
    weights; this is useful for reliability-colored exports.
    """

    if points.shape[0] == 0:
        scalar_out = scalar_values.astype(np.float32) if scalar_values is not None else None
        return points, colors, weights.astype(np.float32), scalar_out
    if voxel_size <= 0:
        scalar_out = scalar_values.astype(np.float32) if scalar_values is not None else None
        return points, colors, weights.astype(np.float32), scalar_out
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if colors.shape != (points.shape[0], 3):
        raise ValueError("colors must have shape [N, 3]")
    if weights.shape[0] != points.shape[0]:
        raise ValueError("weights must have shape [N]")
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1e-6)
    if scalar_values is not None:
        scalar_values = np.asarray(scalar_values, dtype=np.float64).reshape(-1)
        if scalar_values.shape[0] != points.shape[0]:
            raise ValueError("scalar_values must have shape [N]")

    voxel = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(voxel, axis=0, return_inverse=True)
    weight_sum = np.bincount(inverse, weights=weights).astype(np.float64)
    point_sum = np.zeros((weight_sum.shape[0], 3), dtype=np.float64)
    color_sum = np.zeros((weight_sum.shape[0], 3), dtype=np.float64)
    np.add.at(point_sum, inverse, points * weights[:, None])
    np.add.at(color_sum, inverse, colors.astype(np.float64) * weights[:, None])
    down_points = (point_sum / weight_sum[:, None]).astype(np.float32)
    down_colors = np.clip(color_sum / weight_sum[:, None], 0, 255).astype(np.uint8)
    down_scalars = None
    if scalar_values is not None:
        scalar_sum = np.zeros((weight_sum.shape[0],), dtype=np.float64)
        np.add.at(scalar_sum, inverse, scalar_values * weights)
        down_scalars = (scalar_sum / weight_sum).astype(np.float32)
    return down_points, down_colors, weight_sum.astype(np.float32), down_scalars


def trim_outliers(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    percentile: float = 99.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove extreme points by distance from the median."""

    keep = trim_outliers_mask(points, percentile=percentile)
    return points[keep], colors[keep]


def trim_outliers_mask(points: np.ndarray, *, percentile: float = 99.7) -> np.ndarray:
    """Return a boolean mask that removes extreme points by median distance."""

    center = np.median(points, axis=0)
    distance = np.linalg.norm(points - center[None, :], axis=1)
    return distance <= np.percentile(distance, percentile)


def save_occupancy_grid(path: str | Path, points: np.ndarray, *, voxel_size_m: float = 0.05) -> Path:
    """Save an occupied-only voxel grid as compressed NPZ.

    This marks observed surface voxels as occupied. It does not yet distinguish
    free from unknown space.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    min_bound = np.floor(np.min(points, axis=0) / voxel_size_m) * voxel_size_m
    voxel_indices = np.floor((points - min_bound[None, :]) / voxel_size_m).astype(np.int32)
    unique = np.unique(voxel_indices, axis=0)
    np.savez_compressed(
        path,
        occupied_indices=unique,
        min_bound_m=min_bound.astype(np.float32),
        voxel_size_m=np.asarray(voxel_size_m, dtype=np.float32),
    )
    return path
