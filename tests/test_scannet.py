from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scapev3.scannet import list_scannet_frames, reconstruct_scannet_rgbd


def test_list_scannet_frames_matches_standard_layout(tmp_path: Path) -> None:
    scan_dir = _write_minimal_scannet_scan(tmp_path)

    frames = list_scannet_frames(scan_dir=scan_dir, native_fps=30.0)

    assert len(frames) == 1
    assert frames[0].name == "scannet_0"
    assert frames[0].intrinsics.width == 4
    assert frames[0].intrinsics.fx == 2.0
    assert np.allclose(frames[0].pose.translation, [0.0, 0.0, 0.0])


def test_reconstruct_scannet_rgbd_writes_scannet_outputs(tmp_path: Path) -> None:
    scan_dir = _write_minimal_scannet_scan(tmp_path)

    result = reconstruct_scannet_rgbd(
        scan_dir=scan_dir,
        output_dir=tmp_path / "out",
        max_frames=1,
        target_fps=1.0,
        pixel_stride=1,
    )

    assert Path(result.downsampled_ply).name == "scannet_rgbd_downsampled_metric.ply"
    manifest = json.loads(Path(result.manifest_json).read_text(encoding="utf-8"))
    assert manifest["dataset"] == "scannet"
    assert manifest["frames_used"] == 1


def _write_minimal_scannet_scan(tmp_path: Path) -> Path:
    scan_dir = tmp_path / "scene0000_00"
    for subdir in ("color", "depth", "pose", "intrinsic"):
        (scan_dir / subdir).mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4), 1000, dtype=np.uint16)).save(scan_dir / "depth" / "0.png")
    Image.fromarray(np.full((4, 4, 3), [20, 90, 160], dtype=np.uint8)).save(scan_dir / "color" / "0.jpg")
    np.savetxt(
        scan_dir / "intrinsic" / "intrinsic_depth.txt",
        np.array(
            [
                [2.0, 0.0, 1.5, 0.0],
                [0.0, 2.0, 1.5, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )
    np.savetxt(scan_dir / "pose" / "0.txt", np.eye(4, dtype=np.float64))
    return scan_dir
