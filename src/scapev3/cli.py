"""Shared command-line helpers for 3DScape scripts."""

from __future__ import annotations

from scapev3.rgbd import ReconstructionResult


def print_reconstruction_result(dataset_label: str, result: ReconstructionResult) -> None:
    """Print a consistent artifact summary after reconstruction."""

    print(f"{dataset_label} RGB-D reconstruction complete")
    print(f"frames_used: {result.frames_used}")
    print(f"raw_points: {result.raw_points}")
    print(f"downsampled_points: {result.downsampled_points}")
    print(f"raw_ply: {result.raw_ply}")
    print(f"downsampled_ply: {result.downsampled_ply}")
    if result.reliability_ply:
        print(f"reliability_ply: {result.reliability_ply}")
    print(f"occupancy_npz: {result.occupancy_npz}")
    print(f"top_down_png: {result.top_down_png}")
    print(f"manifest_json: {result.manifest_json}")
