#!/usr/bin/env python
"""Dataset-agnostic RGB-D reconstruction entrypoint."""

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
from scapev3.manifest_rgbd import reconstruct_manifest_rgbd  # noqa: E402
from scapev3.rgbd import ReliabilityProvider  # noqa: E402
from scapev3.scannet import reconstruct_scannet_rgbd  # noqa: E402
from scapev3.tum_rgbd import reconstruct_tum_rgbd  # noqa: E402


def _load_reliability_provider(
    args: argparse.Namespace,
) -> tuple[ReliabilityProvider | None, float | None]:
    """Load the optional learned reliability provider from a checkpoint."""

    if args.reliability_checkpoint is None:
        return None, None
    if not 0.0 <= args.reliability_threshold <= 1.0:
        raise ValueError("--reliability-threshold must be in [0, 1]")
    try:
        from scapev3.learned.reliability import make_reliability_provider
    except ImportError as exc:
        raise RuntimeError(
            "Torch is required for --reliability-checkpoint. "
            'Install the ML extra with: pip install -e ".[ml]"'
        ) from exc
    provider = make_reliability_provider(
        args.reliability_checkpoint,
        device=args.reliability_device,
        max_depth_m=args.max_depth_m,
    )
    return provider, args.reliability_threshold


def _shared_reconstruction_kwargs(
    args: argparse.Namespace,
    min_reliability: float | None,
) -> dict[str, object]:
    """Build keyword arguments shared by every dataset reconstruction path."""

    return {
        "output_dir": args.out_dir,
        "max_frames": args.max_frames,
        "target_fps": args.target_fps,
        "pixel_stride": args.pixel_stride,
        "min_depth_m": args.min_depth_m,
        "max_depth_m": args.max_depth_m,
        "min_reliability": min_reliability,
        "use_reliability_weights": not args.no_reliability_soft_weights,
        "reliability_weight_floor": args.reliability_weight_floor,
        "voxel_size_m": args.voxel_size_m,
        "max_points": args.max_points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct an RGB-D scan into 3DScape outputs.")
    parser.add_argument(
        "--dataset",
        choices=["arkitscenes", "scannet", "tum_rgbd", "manifest"],
        required=True,
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        required=True,
        help="Dataset directory, or a manifest JSON path/directory when --dataset manifest.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument("--target-fps", type=float, default=3.0)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--min-depth-m", type=float, default=0.2)
    parser.add_argument("--max-depth-m", type=float, default=8.0)
    parser.add_argument("--voxel-size-m", type=float, default=0.04)
    parser.add_argument("--max-points", type=int, default=800_000)
    parser.add_argument(
        "--reliability-checkpoint",
        type=Path,
        default=None,
        help="Optional TinyReliabilityNet checkpoint for learned depth-pixel filtering.",
    )
    parser.add_argument(
        "--reliability-threshold",
        type=float,
        default=0.05,
        help=(
            "Low hard cutoff for obvious junk. Soft reliability weighting is applied after this "
            "when --reliability-checkpoint is provided."
        ),
    )
    parser.add_argument(
        "--no-reliability-soft-weights",
        action="store_true",
        help="Disable reliability-weighted voxel fusion and use the checkpoint only as a hard filter.",
    )
    parser.add_argument(
        "--reliability-weight-floor",
        type=float,
        default=0.05,
        help="Minimum point weight used during soft reliability-weighted voxel fusion.",
    )
    parser.add_argument(
        "--reliability-device",
        type=str,
        default="cpu",
        help="Torch device for reliability inference, for example cpu, cuda, or mps.",
    )
    parser.add_argument("--rgb-dir", type=Path, default=None, help="ARKitScenes lowres_wide directory.")
    parser.add_argument("--video", type=Path, default=None, help="Optional ARKitScenes .mov color source.")
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=2,
        help="Confidence threshold for ARKitScenes or manifest confidence maps.",
    )
    parser.add_argument("--pose-max-delta-sec", type=float, default=0.08, help="ARKitScenes pose match limit.")
    parser.add_argument("--scannet-native-fps", type=float, default=30.0)
    parser.add_argument(
        "--tum-intrinsics",
        choices=["auto", "freiburg1", "freiburg2", "freiburg3"],
        default="auto",
        help="TUM RGB-D Freiburg intrinsics preset.",
    )
    parser.add_argument(
        "--tum-association-delta-sec",
        type=float,
        default=0.04,
        help="Maximum RGB/depth timestamp gap for TUM RGB-D.",
    )
    parser.add_argument(
        "--tum-pose-delta-sec",
        type=float,
        default=0.04,
        help="Maximum RGB/pose timestamp gap for TUM RGB-D.",
    )
    args = parser.parse_args()
    reliability_provider, min_reliability = _load_reliability_provider(args)
    shared_kwargs = _shared_reconstruction_kwargs(args, min_reliability)

    if args.dataset == "arkitscenes":
        result = reconstruct_arkitscenes_rgbd(
            scan_dir=args.scan_dir,
            video_path=args.video,
            rgb_dir=args.rgb_dir,
            min_confidence=args.min_confidence,
            pose_max_delta_sec=args.pose_max_delta_sec,
            reliability_provider=reliability_provider,
            **shared_kwargs,
        )
    elif args.dataset == "scannet":
        result = reconstruct_scannet_rgbd(
            scan_dir=args.scan_dir,
            native_fps=args.scannet_native_fps,
            reliability_provider=reliability_provider,
            **shared_kwargs,
        )
    elif args.dataset == "tum_rgbd":
        result = reconstruct_tum_rgbd(
            scan_dir=args.scan_dir,
            intrinsics_preset=args.tum_intrinsics,
            max_association_delta_sec=args.tum_association_delta_sec,
            max_pose_delta_sec=args.tum_pose_delta_sec,
            reliability_provider=reliability_provider,
            **shared_kwargs,
        )
    else:
        result = reconstruct_manifest_rgbd(
            manifest_path=args.scan_dir,
            min_confidence=args.min_confidence,
            reliability_provider=reliability_provider,
            **shared_kwargs,
        )

    print_reconstruction_result(args.dataset, result)


if __name__ == "__main__":
    main()
