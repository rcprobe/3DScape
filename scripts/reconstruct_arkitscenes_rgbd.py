#!/usr/bin/env python
"""Fuse an ARKitScenes raw scan using metric depth and ARKit trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scapev3.arkitscenes import reconstruct_arkitscenes_rgbd  # noqa: E402
from scapev3.cli import print_reconstruction_result  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct a metric ARKitScenes RGB-D point cloud.")
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--rgb-dir", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument("--target-fps", type=float, default=3.0)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--min-depth-m", type=float, default=0.2)
    parser.add_argument("--max-depth-m", type=float, default=8.0)
    parser.add_argument("--min-confidence", type=int, default=2)
    parser.add_argument("--voxel-size-m", type=float, default=0.04)
    parser.add_argument("--max-points", type=int, default=800_000)
    parser.add_argument("--pose-max-delta-sec", type=float, default=0.08)
    args = parser.parse_args()

    result = reconstruct_arkitscenes_rgbd(
        scan_dir=args.scan_dir,
        output_dir=args.out_dir,
        video_path=args.video,
        rgb_dir=args.rgb_dir,
        max_frames=args.max_frames,
        target_fps=args.target_fps,
        pixel_stride=args.pixel_stride,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        min_confidence=args.min_confidence,
        voxel_size_m=args.voxel_size_m,
        max_points=args.max_points,
        pose_max_delta_sec=args.pose_max_delta_sec,
    )
    print_reconstruction_result("ARKitScenes", result)


if __name__ == "__main__":
    main()
