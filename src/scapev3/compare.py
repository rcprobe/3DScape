"""Point-cloud comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from scapev3.config import save_json
from scapev3.ply import read_ply_vertices


@dataclass(frozen=True)
class ComparisonResult:
    """Artifacts and metrics from point cloud comparison."""

    metrics_json: str
    overlay_png: str
    source_points: int
    reference_points: int
    source_to_reference: dict[str, float]
    reference_to_source: dict[str, float]


def compare_ply_point_sets(
    *,
    source_ply: str | Path,
    reference_ply: str | Path,
    output_dir: str | Path,
    max_source_points: int = 80_000,
    max_reference_points: int = 160_000,
    grid_size_m: float = 0.08,
    max_search_radius_m: float = 0.48,
) -> ComparisonResult:
    """Compare a reconstructed PLY against a reference mesh/point cloud.

    Distances are approximate nearest-neighbor distances over sampled vertices.
    This avoids heavy dependencies while still producing useful MVP diagnostics.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_points, _source_colors = read_ply_vertices(source_ply, max_points=max_source_points)
    reference_points, _reference_colors = read_ply_vertices(reference_ply, max_points=max_reference_points)
    if source_points.shape[0] == 0 or reference_points.shape[0] == 0:
        raise ValueError("Both source and reference PLY files must contain vertices")

    s2r = approximate_nearest_distances(
        source_points,
        reference_points,
        grid_size_m=grid_size_m,
        max_search_radius_m=max_search_radius_m,
    )
    r2s = approximate_nearest_distances(
        reference_points,
        source_points,
        grid_size_m=grid_size_m,
        max_search_radius_m=max_search_radius_m,
    )
    s2r_stats = summarize_distances(s2r)
    r2s_stats = summarize_distances(r2s)
    overlay_png = save_top_down_overlay(
        output_dir / "comparison_top_down_overlay.png",
        source_points=source_points,
        reference_points=reference_points,
    )
    metrics_json = save_json(
        {
            "source_ply": str(source_ply),
            "reference_ply": str(reference_ply),
            "source_points": int(source_points.shape[0]),
            "reference_points": int(reference_points.shape[0]),
            "grid_size_m": grid_size_m,
            "max_search_radius_m": max_search_radius_m,
            "source_to_reference_m": s2r_stats,
            "reference_to_source_m": r2s_stats,
            "interpretation": {
                "source_to_reference": "How far reconstructed points are from the reference mesh vertices.",
                "reference_to_source": "How much of the reference mesh is covered by reconstructed points.",
                "unmatched_fraction": "Queries that found no neighbor within max_search_radius_m.",
            },
        },
        output_dir / "comparison_metrics.json",
    )
    return ComparisonResult(
        metrics_json=str(metrics_json),
        overlay_png=str(overlay_png),
        source_points=int(source_points.shape[0]),
        reference_points=int(reference_points.shape[0]),
        source_to_reference=s2r_stats,
        reference_to_source=r2s_stats,
    )


def approximate_nearest_distances(
    queries: np.ndarray,
    references: np.ndarray,
    *,
    grid_size_m: float,
    max_search_radius_m: float,
) -> np.ndarray:
    """Approximate nearest-neighbor distances with a voxel hash grid."""

    if grid_size_m <= 0:
        raise ValueError("grid_size_m must be positive")
    if max_search_radius_m <= 0:
        raise ValueError("max_search_radius_m must be positive")
    queries = np.asarray(queries, dtype=np.float32)
    references = np.asarray(references, dtype=np.float32)
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("queries must have shape [N, 3]")
    if references.ndim != 2 or references.shape[1] != 3:
        raise ValueError("references must have shape [N, 3]")

    origin = np.minimum(np.min(queries, axis=0), np.min(references, axis=0))
    reference_voxels = np.floor((references - origin[None, :]) / grid_size_m).astype(np.int32)
    grid: dict[tuple[int, int, int], list[int]] = {}
    for idx, voxel in enumerate(reference_voxels):
        grid.setdefault((int(voxel[0]), int(voxel[1]), int(voxel[2])), []).append(idx)

    max_ring = max(1, int(np.ceil(max_search_radius_m / grid_size_m)))
    distances = np.full((queries.shape[0],), np.nan, dtype=np.float32)
    query_voxels = np.floor((queries - origin[None, :]) / grid_size_m).astype(np.int32)
    for idx, query in enumerate(queries):
        voxel = query_voxels[idx]
        best_sq = np.inf
        for ring in range(max_ring + 1):
            candidate_indices = _candidate_indices_for_ring(grid, voxel, ring)
            if not candidate_indices:
                continue
            candidate_points = references[np.asarray(candidate_indices, dtype=np.int64)]
            diff = candidate_points - query[None, :]
            local_best_sq = float(np.min(np.einsum("ij,ij->i", diff, diff)))
            if local_best_sq < best_sq:
                best_sq = local_best_sq
            if best_sq <= (ring * grid_size_m) ** 2:
                break
        if np.isfinite(best_sq) and best_sq <= max_search_radius_m * max_search_radius_m:
            distances[idx] = float(np.sqrt(best_sq))
    return distances


def summarize_distances(distances: np.ndarray) -> dict[str, float]:
    """Summarize nearest-neighbor distances in meters."""

    distances = np.asarray(distances, dtype=np.float32)
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return {
            "matched_fraction": 0.0,
            "unmatched_fraction": 1.0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "rmse": float("nan"),
        }
    return {
        "matched_fraction": float(finite.size / max(1, distances.size)),
        "unmatched_fraction": float(1.0 - finite.size / max(1, distances.size)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "rmse": float(np.sqrt(np.mean(finite * finite))),
    }


def save_top_down_overlay(
    path: str | Path,
    *,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    image_size: int = 1000,
) -> Path:
    """Save a top-down XZ overlay: reference in blue, source in orange."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source_xz = source_points[:, [0, 2]].astype(np.float64)
    reference_xz = reference_points[:, [0, 2]].astype(np.float64)
    both = np.concatenate([source_xz, reference_xz], axis=0)
    low = np.percentile(both, 1, axis=0)
    high = np.percentile(both, 99, axis=0)
    span = np.maximum(high - low, 1e-6)
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    _draw_density(image, reference_xz, low=low, span=span, color=np.array([70, 130, 255], dtype=np.float32))
    _draw_density(image, source_xz, low=low, span=span, color=np.array([255, 170, 40], dtype=np.float32))
    cv2.imwrite(str(path), image)
    return path


def _candidate_indices_for_ring(
    grid: dict[tuple[int, int, int], list[int]],
    voxel: np.ndarray,
    ring: int,
) -> list[int]:
    indices: list[int] = []
    vx, vy, vz = int(voxel[0]), int(voxel[1]), int(voxel[2])
    for dx in range(-ring, ring + 1):
        for dy in range(-ring, ring + 1):
            for dz in range(-ring, ring + 1):
                if ring > 0 and max(abs(dx), abs(dy), abs(dz)) != ring:
                    continue
                indices.extend(grid.get((vx + dx, vy + dy, vz + dz), []))
    return indices


def _draw_density(
    image: np.ndarray,
    xz: np.ndarray,
    *,
    low: np.ndarray,
    span: np.ndarray,
    color: np.ndarray,
) -> None:
    image_size = image.shape[0]
    pixels = np.floor(((xz - low[None, :]) / span[None, :]) * (image_size - 1)).astype(np.int32)
    valid = np.all((pixels >= 0) & (pixels < image_size), axis=1)
    pixels = pixels[valid]
    density = np.zeros((image_size, image_size), dtype=np.float32)
    np.add.at(density, (image_size - 1 - pixels[:, 1], pixels[:, 0]), 1.0)
    if float(density.max()) <= 0:
        return
    density = np.log1p(density)
    density = density / float(density.max())
    alpha = np.clip(density[..., None] * 0.85, 0.0, 0.85)
    image[:] = np.clip(image.astype(np.float32) * (1.0 - alpha) + color[None, None, :] * alpha, 0, 255)
