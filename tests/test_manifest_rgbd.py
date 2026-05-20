from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scapev3.manifest_rgbd import (
    DEFAULT_DEPTH_SCALE_M,
    list_manifest_rgbd_frames,
    parse_intrinsics,
    parse_pose,
    reconstruct_manifest_rgbd,
)


def test_list_manifest_rgbd_frames_uses_common_schema(tmp_path: Path) -> None:
    manifest_path = _write_minimal_manifest_scan(tmp_path)

    frames = list_manifest_rgbd_frames(manifest_path=manifest_path)

    assert len(frames) == 1
    frame = frames[0]
    assert frame.name == "frame_000"
    assert frame.depth_scale_m == DEFAULT_DEPTH_SCALE_M
    assert frame.intrinsics.width == 4
    assert frame.intrinsics.fx == 2.0
    assert np.allclose(frame.pose.rotation, np.eye(3))
    assert np.allclose(frame.pose.translation, [0.0, 0.0, 0.0])


def test_parse_intrinsics_accepts_matrix_and_infers_depth_size(tmp_path: Path) -> None:
    depth_path = tmp_path / "depth.png"
    Image.fromarray(np.full((5, 6), 1000, dtype=np.uint16)).save(depth_path)

    intrinsics = parse_intrinsics(
        [
            [3.0, 0.0, 2.5],
            [0.0, 4.0, 2.0],
            [0.0, 0.0, 1.0],
        ],
        depth_path=depth_path,
    )

    assert intrinsics.width == 6
    assert intrinsics.height == 5
    assert intrinsics.fx == 3.0
    assert intrinsics.fy == 4.0


def test_parse_pose_accepts_rotation_translation_object() -> None:
    pose = parse_pose(
        {
            "rotation": np.eye(3).tolist(),
            "translation": [1.0, 2.0, 3.0],
        },
        timestamp=7.0,
    )

    assert pose.timestamp == 7.0
    assert np.allclose(pose.rotation, np.eye(3))
    assert np.allclose(pose.translation, [1.0, 2.0, 3.0])


def test_reconstruct_manifest_rgbd_writes_manifest_outputs(tmp_path: Path) -> None:
    manifest_path = _write_minimal_manifest_scan(tmp_path)

    result = reconstruct_manifest_rgbd(
        manifest_path=manifest_path,
        output_dir=tmp_path / "out",
        max_frames=1,
        target_fps=1.0,
        pixel_stride=1,
    )

    assert Path(result.downsampled_ply).name == "manifest_rgbd_downsampled_metric.ply"
    manifest = json.loads(Path(result.manifest_json).read_text(encoding="utf-8"))
    assert manifest["dataset"] == "manifest_rgbd"
    assert manifest["frames_used"] == 1
    assert manifest["manifest_frame_count"] == 1


def _write_minimal_manifest_scan(tmp_path: Path) -> Path:
    scan_dir = tmp_path / "manifest_scan"
    rgb_dir = scan_dir / "rgb"
    depth_dir = scan_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(np.full((4, 4), 1000, dtype=np.uint16)).save(depth_dir / "000.png")
    Image.fromarray(np.full((4, 4, 3), [80, 120, 160], dtype=np.uint8)).save(rgb_dir / "000.png")
    manifest = {
        "name": "synthetic_manifest_scan",
        "depth_scale_m": DEFAULT_DEPTH_SCALE_M,
        "intrinsics": {
            "width": 4,
            "height": 4,
            "fx": 2.0,
            "fy": 2.0,
            "cx": 1.5,
            "cy": 1.5,
        },
        "frames": [
            {
                "name": "frame_000",
                "timestamp": 0.0,
                "rgb_path": "rgb/000.png",
                "depth_path": "depth/000.png",
                "pose": np.eye(4).tolist(),
            }
        ],
    }
    manifest_path = scan_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path
