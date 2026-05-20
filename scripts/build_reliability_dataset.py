#!/usr/bin/env python
"""Build TinyReliabilityNet training samples from a posed RGB-D manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scapev3.config import save_json  # noqa: E402
from scapev3.learned.reliability import (  # noqa: E402
    build_multiview_reliability_target,
    make_reliability_features,
)
from scapev3.manifest_rgbd import list_manifest_rgbd_frames  # noqa: E402
from scapev3.rgbd import load_depth_m, resize_or_crop_rgb, select_frames_by_rate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create reliability-network .npz samples from adjacent posed RGB-D frames."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--target-fps", type=float, default=4.0)
    parser.add_argument("--pair-gap", type=int, default=1)
    parser.add_argument(
        "--target-window",
        type=int,
        default=2,
        help="Use this many pair-gap steps before and after each source frame as target views.",
    )
    parser.add_argument("--label-pixel-stride", type=int, default=2)
    parser.add_argument("--min-depth-m", type=float, default=0.2)
    parser.add_argument("--max-depth-m", type=float, default=8.0)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.08)
    parser.add_argument("--relative-tolerance", type=float, default=0.03)
    parser.add_argument(
        "--min-target-observations",
        type=int,
        default=1,
        help="Require each supervised pixel to be comparable in at least this many target views.",
    )
    parser.add_argument("--min-supervised-pixels", type=int, default=64)
    parser.add_argument(
        "--sample-max-side",
        type=int,
        default=None,
        help="Optionally resize saved feature/label samples so the largest side is at most this value.",
    )
    args = parser.parse_args()

    frames = select_frames_by_rate(
        list_manifest_rgbd_frames(manifest_path=args.manifest),
        target_fps=args.target_fps,
        max_frames=args.max_frames,
    )
    if len(frames) <= args.pair_gap:
        raise RuntimeError("Not enough manifest frames to build reliability pairs")
    if args.target_window < 1:
        raise ValueError("--target-window must be >= 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    sample_index = 0
    for source_idx in range(len(frames)):
        source = frames[source_idx]
        target_indices = _target_indices(
            source_idx,
            frame_count=len(frames),
            pair_gap=args.pair_gap,
            target_window=args.target_window,
        )
        if not target_indices:
            continue
        targets = [frames[index] for index in target_indices]
        source_depth = load_depth_m(source.depth_path, scale_m=source.depth_scale_m)
        target_depths = [load_depth_m(target.depth_path, scale_m=target.depth_scale_m) for target in targets]
        rgb = _load_rgb_for_frame(source, source_depth.shape)
        labels, mask = build_multiview_reliability_target(
            source_depth_m=source_depth,
            source_intrinsics=source.intrinsics,
            source_pose=source.pose,
            target_depths_m=target_depths,
            target_intrinsics=[target.intrinsics for target in targets],
            target_poses=[target.pose for target in targets],
            pixel_stride=args.label_pixel_stride,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            depth_tolerance_m=args.depth_tolerance_m,
            relative_tolerance=args.relative_tolerance,
            min_target_observations=args.min_target_observations,
        )
        supervised_pixels = int(mask.sum())
        if supervised_pixels < args.min_supervised_pixels:
            continue
        features = make_reliability_features(
            rgb=rgb,
            depth_m=source_depth,
            max_depth_m=args.max_depth_m,
        )
        features, labels, mask = _resize_sample(
            features=features,
            labels=labels,
            mask=mask,
            max_side=args.sample_max_side,
        )
        sample_path = args.out_dir / f"sample_{sample_index:06d}.npz"
        np.savez_compressed(
            sample_path,
            features=features.astype(np.float32),
            labels=labels.astype(np.float32),
            mask=mask.astype(np.uint8),
        )
        records.append(
            {
                "sample_path": str(sample_path),
                "source_name": source.name,
                "target_names": [target.name for target in targets],
                "source_timestamp": source.timestamp,
                "target_timestamps": [target.timestamp for target in targets],
                "supervised_pixels": supervised_pixels,
                "positive_label_mass": float(labels[mask].sum()),
                "positive_pixels_at_05": int((labels[mask] >= 0.5).sum()),
            }
        )
        sample_index += 1

    if not records:
        raise RuntimeError("No reliability samples were written; try looser pairing or thresholds")
    manifest_path = save_json(
        {
            "source_manifest": str(args.manifest),
            "sample_count": len(records),
            "settings": {
                "max_frames": args.max_frames,
                "target_fps": args.target_fps,
                "pair_gap": args.pair_gap,
                "target_window": args.target_window,
                "label_pixel_stride": args.label_pixel_stride,
                "min_depth_m": args.min_depth_m,
                "max_depth_m": args.max_depth_m,
                "depth_tolerance_m": args.depth_tolerance_m,
                "relative_tolerance": args.relative_tolerance,
                "min_target_observations": args.min_target_observations,
                "min_supervised_pixels": args.min_supervised_pixels,
                "sample_max_side": args.sample_max_side,
            },
            "samples": records,
        },
        args.out_dir / "reliability_dataset_manifest.json",
    )
    print(f"Wrote {len(records)} reliability samples")
    print(f"manifest_json: {manifest_path}")


def _target_indices(
    source_idx: int,
    *,
    frame_count: int,
    pair_gap: int,
    target_window: int,
) -> list[int]:
    if pair_gap < 1:
        raise ValueError("--pair-gap must be >= 1")
    indices: list[int] = []
    for step in range(1, target_window + 1):
        offset = step * pair_gap
        for candidate in (source_idx - offset, source_idx + offset):
            if 0 <= candidate < frame_count:
                indices.append(candidate)
    return indices


def _load_rgb_for_frame(frame, depth_shape: tuple[int, int]) -> np.ndarray:
    height, width = depth_shape
    if frame.rgb_path is None:
        return np.full((height, width, 3), 128, dtype=np.uint8)
    image = np.asarray(Image.open(frame.rgb_path).convert("RGB"), dtype=np.uint8)
    return resize_or_crop_rgb(image, width=width, height=height)


def _resize_sample(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    max_side: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_side is None:
        return features, labels, mask
    if max_side < 16:
        raise ValueError("--sample-max-side must be >= 16")
    height, width = labels.shape
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale >= 1.0:
        return features, labels, mask
    out_width = max(1, int(round(width * scale)))
    out_height = max(1, int(round(height * scale)))
    resized_features = np.stack(
        [
            cv2.resize(channel, (out_width, out_height), interpolation=cv2.INTER_AREA)
            for channel in features
        ],
        axis=0,
    )
    resized_labels = cv2.resize(labels, (out_width, out_height), interpolation=cv2.INTER_NEAREST)
    resized_mask = cv2.resize(mask.astype(np.uint8), (out_width, out_height), interpolation=cv2.INTER_NEAREST)
    return resized_features.astype(np.float32), resized_labels.astype(np.float32), resized_mask.astype(bool)


if __name__ == "__main__":
    main()
