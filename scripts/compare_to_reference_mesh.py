#!/usr/bin/env python
"""Compare a reconstructed point cloud against a reference mesh/point cloud."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scapev3.compare import compare_ply_point_sets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two PLY vertex sets.")
    parser.add_argument("--source-ply", type=Path, required=True)
    parser.add_argument("--reference-ply", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-source-points", type=int, default=80_000)
    parser.add_argument("--max-reference-points", type=int, default=160_000)
    parser.add_argument("--grid-size-m", type=float, default=0.08)
    parser.add_argument("--max-search-radius-m", type=float, default=0.48)
    args = parser.parse_args()

    result = compare_ply_point_sets(
        source_ply=args.source_ply,
        reference_ply=args.reference_ply,
        output_dir=args.out_dir,
        max_source_points=args.max_source_points,
        max_reference_points=args.max_reference_points,
        grid_size_m=args.grid_size_m,
        max_search_radius_m=args.max_search_radius_m,
    )
    print("PLY comparison complete")
    print(f"source_points: {result.source_points}")
    print(f"reference_points: {result.reference_points}")
    print(f"source_to_reference median_m: {result.source_to_reference['median']:.4f}")
    print(f"source_to_reference p90_m: {result.source_to_reference['p90']:.4f}")
    print(f"reference_to_source median_m: {result.reference_to_source['median']:.4f}")
    print(f"reference_to_source p90_m: {result.reference_to_source['p90']:.4f}")
    print(f"metrics_json: {result.metrics_json}")
    print(f"overlay_png: {result.overlay_png}")


if __name__ == "__main__":
    main()
