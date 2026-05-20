#!/usr/bin/env python
"""Export adapter-loaded RGB-D frames as a generic posed RGB-D manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scapev3.arkitscenes import (  # noqa: E402
    _nearest_timestamp_path,
    _timestamp_from_path,
    list_arkitscenes_frames,
    load_intrinsics,
    load_trajectory,
    nearest_pose,
)
from scapev3.config import save_json  # noqa: E402
from scapev3.rgbd import RGBDFrame  # noqa: E402
from scapev3.scannet import list_scannet_frames  # noqa: E402
from scapev3.tum_rgbd import list_tum_rgbd_frames  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dataset frames to generic RGB-D manifest JSON.")
    parser.add_argument(
        "--dataset",
        choices=["arkitscenes", "scannet", "tum_rgbd"],
        required=True,
    )
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--rgb-dir", type=Path, default=None, help="ARKitScenes lowres_wide directory.")
    parser.add_argument("--pose-max-delta-sec", type=float, default=0.08)
    parser.add_argument("--scannet-native-fps", type=float, default=30.0)
    parser.add_argument(
        "--tum-intrinsics",
        choices=["auto", "freiburg1", "freiburg2", "freiburg3"],
        default="auto",
    )
    parser.add_argument("--tum-association-delta-sec", type=float, default=0.04)
    parser.add_argument("--tum-pose-delta-sec", type=float, default=0.04)
    args = parser.parse_args()

    if args.dataset == "arkitscenes":
        frames = _list_arkitscenes_manifest_frames(
            scan_dir=args.scan_dir,
            rgb_dir=args.rgb_dir,
            pose_max_delta_sec=args.pose_max_delta_sec,
        )
    elif args.dataset == "scannet":
        frames = list_scannet_frames(scan_dir=args.scan_dir, native_fps=args.scannet_native_fps)
    else:
        frames = list_tum_rgbd_frames(
            scan_dir=args.scan_dir,
            intrinsics_preset=args.tum_intrinsics,
            max_association_delta_sec=args.tum_association_delta_sec,
            max_pose_delta_sec=args.tum_pose_delta_sec,
        )
    if not frames:
        raise RuntimeError(f"No frames were exported for dataset: {args.dataset}")

    manifest = {
        "name": args.name or f"{args.dataset}_{args.scan_dir.name}",
        "dataset": args.dataset,
        "base_dir": "/",
        "frames": [_frame_to_json(frame) for frame in frames],
    }
    out_path = save_json(manifest, args.out_manifest)
    print(f"Exported {len(frames)} frames")
    print(f"manifest_json: {out_path}")


def _list_arkitscenes_manifest_frames(
    *,
    scan_dir: Path,
    rgb_dir: Path | None,
    pose_max_delta_sec: float,
) -> list[RGBDFrame]:
    scan_dir = Path(scan_dir)
    depth_dir = scan_dir / "lowres_depth"
    confidence_dir = scan_dir / "confidence"
    intrinsics_dir = scan_dir / "lowres_wide_intrinsics"
    traj_path = scan_dir / "lowres_wide.traj"
    rgb_dir = rgb_dir or scan_dir / "lowres_wide"

    raw_frames = list_arkitscenes_frames(
        depth_dir=depth_dir,
        intrinsics_dir=intrinsics_dir,
        confidence_dir=confidence_dir if confidence_dir.exists() else None,
    )
    trajectory = load_trajectory(traj_path)
    rgb_by_time: dict[float, Path] = {}
    if rgb_dir.exists():
        rgb_by_time = {_timestamp_from_path(path): path for path in sorted(rgb_dir.glob("*.png"))}

    frames: list[RGBDFrame] = []
    for raw_frame in raw_frames:
        pose, pose_delta = nearest_pose(trajectory, raw_frame.timestamp)
        if pose_delta > pose_max_delta_sec:
            continue
        intrinsics = load_intrinsics(raw_frame.intrinsics_path)
        rgb_path = _nearest_timestamp_path(rgb_by_time, raw_frame.timestamp, max_delta=0.04)
        frames.append(
            RGBDFrame(
                timestamp=raw_frame.timestamp,
                depth_path=raw_frame.depth_path,
                rgb_path=str(rgb_path) if rgb_path else None,
                confidence_path=raw_frame.confidence_path,
                intrinsics=intrinsics,
                pose=pose,
                depth_scale_m=0.001,
                name=f"arkit_{raw_frame.timestamp:.3f}",
            )
        )
    return frames


def _frame_to_json(frame: RGBDFrame) -> dict:
    return {
        "name": frame.name,
        "timestamp": frame.timestamp,
        "rgb_path": frame.rgb_path,
        "depth_path": frame.depth_path,
        "confidence_path": frame.confidence_path,
        "depth_scale_m": frame.depth_scale_m,
        "intrinsics": {
            "width": frame.intrinsics.width,
            "height": frame.intrinsics.height,
            "fx": frame.intrinsics.fx,
            "fy": frame.intrinsics.fy,
            "cx": frame.intrinsics.cx,
            "cy": frame.intrinsics.cy,
        },
        "pose": {
            "rotation": np.asarray(frame.pose.rotation, dtype=float).tolist(),
            "translation": np.asarray(frame.pose.translation, dtype=float).tolist(),
        },
    }


if __name__ == "__main__":
    main()
