from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scapev3.tum_rgbd import (
    TUM_DEPTH_SCALE_M,
    list_tum_rgbd_frames,
    quaternion_xyzw_to_rotation_matrix,
    reconstruct_tum_rgbd,
    resolve_tum_intrinsics_preset,
)


def test_quaternion_xyzw_to_rotation_matrix_identity() -> None:
    rotation = quaternion_xyzw_to_rotation_matrix(np.array([0.0, 0.0, 0.0, 1.0]))

    assert np.allclose(rotation, np.eye(3))


def test_resolve_tum_intrinsics_preset_from_sequence_name(tmp_path: Path) -> None:
    assert resolve_tum_intrinsics_preset(tmp_path / "rgbd_dataset_freiburg2_xyz", "auto") == "freiburg2"
    assert resolve_tum_intrinsics_preset(tmp_path / "rgbd_dataset_fr3_office", "auto") == "freiburg3"
    assert resolve_tum_intrinsics_preset(tmp_path / "unknown_sequence", "auto") == "freiburg1"


def test_list_tum_rgbd_frames_matches_standard_layout(tmp_path: Path) -> None:
    scan_dir = _write_minimal_tum_scan(tmp_path)

    frames = list_tum_rgbd_frames(scan_dir=scan_dir, intrinsics_preset="freiburg1")

    assert len(frames) == 1
    assert frames[0].name == "tum_1.000000"
    assert frames[0].depth_scale_m == TUM_DEPTH_SCALE_M
    assert frames[0].intrinsics.width == 4
    assert frames[0].intrinsics.fx == 517.3
    assert np.allclose(frames[0].pose.rotation, np.eye(3))
    assert np.allclose(frames[0].pose.translation, [0.0, 0.0, 0.0])


def test_reconstruct_tum_rgbd_writes_tum_outputs(tmp_path: Path) -> None:
    scan_dir = _write_minimal_tum_scan(tmp_path)

    result = reconstruct_tum_rgbd(
        scan_dir=scan_dir,
        output_dir=tmp_path / "out",
        max_frames=1,
        target_fps=1.0,
        pixel_stride=1,
        intrinsics_preset="freiburg1",
    )

    assert Path(result.downsampled_ply).name == "tum_rgbd_downsampled_metric.ply"
    manifest = json.loads(Path(result.manifest_json).read_text(encoding="utf-8"))
    assert manifest["dataset"] == "tum_rgbd"
    assert manifest["frames_used"] == 1
    assert manifest["depth_scale_m"] == TUM_DEPTH_SCALE_M


def _write_minimal_tum_scan(tmp_path: Path) -> Path:
    scan_dir = tmp_path / "rgbd_dataset_freiburg1_synthetic"
    rgb_dir = scan_dir / "rgb"
    depth_dir = scan_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = rgb_dir / "1.000000.png"
    depth_path = depth_dir / "1.000010.png"
    Image.fromarray(np.full((4, 4, 3), [30, 120, 210], dtype=np.uint8)).save(rgb_path)
    Image.fromarray(np.full((4, 4), 5000, dtype=np.uint16)).save(depth_path)

    (scan_dir / "rgb.txt").write_text(
        "# timestamp rgb_file\n"
        "1.000000 rgb/1.000000.png\n",
        encoding="utf-8",
    )
    (scan_dir / "depth.txt").write_text(
        "# timestamp depth_file\n"
        "1.000010 depth/1.000010.png\n",
        encoding="utf-8",
    )
    (scan_dir / "groundtruth.txt").write_text(
        "# timestamp tx ty tz qx qy qz qw\n"
        "1.000000 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n",
        encoding="utf-8",
    )
    return scan_dir
