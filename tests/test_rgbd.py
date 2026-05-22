from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scapev3.rgbd import (
    CameraIntrinsics,
    CameraPose,
    RGBDFrame,
    ReconstructionSettings,
    backproject_depth,
    camera_to_world,
    reconstruct_rgbd_frames,
    reliability_to_colors,
)
from scapev3.voxel import weighted_voxel_downsample


def test_generic_backprojection_and_transform() -> None:
    intrinsics = CameraIntrinsics(width=3, height=3, fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    depth = np.ones((3, 3), dtype=np.float32)

    points, pixels = backproject_depth(depth, intrinsics, pixel_stride=1)
    pose = CameraPose(timestamp=0.0, rotation=np.eye(3), translation=np.array([1.0, 2.0, 3.0]))
    world = camera_to_world(points, pose)

    center_index = int(np.where((pixels[:, 0] == 1) & (pixels[:, 1] == 1))[0][0])
    assert np.allclose(points[center_index], [0.0, 0.0, 1.0])
    assert np.allclose(world[center_index], [1.0, 2.0, 4.0])


def test_backprojection_can_filter_by_reliability() -> None:
    intrinsics = CameraIntrinsics(width=3, height=3, fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    depth = np.ones((3, 3), dtype=np.float32)
    reliability = np.zeros((3, 3), dtype=np.float32)
    reliability[1, 1] = 0.9

    points, pixels = backproject_depth(
        depth,
        intrinsics,
        reliability=reliability,
        min_reliability=0.5,
        pixel_stride=1,
    )

    assert points.shape == (1, 3)
    assert np.allclose(points[0], [0.0, 0.0, 1.0])
    assert np.allclose(pixels[0], [1, 1])


def test_reconstruct_rgbd_frames_writes_generic_outputs(tmp_path: Path) -> None:
    depth_path = tmp_path / "depth.png"
    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(np.full((4, 4), 1000, dtype=np.uint16)).save(depth_path)
    Image.fromarray(np.full((4, 4, 3), [120, 80, 40], dtype=np.uint8)).save(rgb_path)
    frame = RGBDFrame(
        timestamp=0.0,
        depth_path=str(depth_path),
        rgb_path=str(rgb_path),
        intrinsics=CameraIntrinsics(width=4, height=4, fx=2.0, fy=2.0, cx=1.5, cy=1.5),
        pose=CameraPose(timestamp=0.0, rotation=np.eye(3), translation=np.zeros(3)),
        name="synthetic",
    )

    result = reconstruct_rgbd_frames(
        frames=[frame],
        output_dir=tmp_path / "out",
        dataset_name="synthetic",
        output_prefix="synthetic_rgbd",
        settings=ReconstructionSettings(max_frames=1, target_fps=1.0, pixel_stride=1),
    )

    assert Path(result.downsampled_ply).exists()
    assert Path(result.occupancy_npz).exists()
    manifest = json.loads(Path(result.manifest_json).read_text(encoding="utf-8"))
    assert manifest["dataset"] == "synthetic"
    assert manifest["frames_used"] == 1
    assert not any(key.endswith("_png") for key in manifest)


def test_weighted_voxel_downsample_biases_toward_reliable_points() -> None:
    points = np.array([[0.0, 0.0, 1.0], [0.09, 0.0, 1.0]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    weights = np.array([0.1, 1.0], dtype=np.float32)

    down_points, down_colors, down_weights, _scalars = weighted_voxel_downsample(
        points,
        colors,
        weights,
        voxel_size=0.2,
    )

    assert down_points.shape == (1, 3)
    assert down_points[0, 0] > 0.07
    assert down_colors[0, 1] > down_colors[0, 0]
    assert np.allclose(down_weights, [1.1], atol=1e-5)


def test_reconstruct_rgbd_frames_writes_reliability_colored_output(tmp_path: Path) -> None:
    depth_path = tmp_path / "depth.png"
    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(np.full((4, 4), 1000, dtype=np.uint16)).save(depth_path)
    Image.fromarray(np.full((4, 4, 3), [120, 80, 40], dtype=np.uint8)).save(rgb_path)
    frame = RGBDFrame(
        timestamp=0.0,
        depth_path=str(depth_path),
        rgb_path=str(rgb_path),
        intrinsics=CameraIntrinsics(width=4, height=4, fx=2.0, fy=2.0, cx=1.5, cy=1.5),
        pose=CameraPose(timestamp=0.0, rotation=np.eye(3), translation=np.zeros(3)),
        name="synthetic",
    )

    def reliability_provider(_frame: RGBDFrame, depth_m: np.ndarray, _rgb: np.ndarray) -> np.ndarray:
        reliability = np.full(depth_m.shape, 0.75, dtype=np.float32)
        reliability[0, 0] = 0.02
        return reliability

    result = reconstruct_rgbd_frames(
        frames=[frame],
        output_dir=tmp_path / "reliable",
        dataset_name="synthetic",
        output_prefix="synthetic_rgbd",
        settings=ReconstructionSettings(
            max_frames=1,
            target_fps=1.0,
            pixel_stride=1,
            min_reliability=0.05,
        ),
        reliability_provider=reliability_provider,
    )

    assert result.reliability_ply is not None
    assert Path(result.reliability_ply).exists()
    manifest = json.loads(Path(result.manifest_json).read_text(encoding="utf-8"))
    assert manifest["learned_reliability_enabled"] is True
    assert manifest["reliability_ply"] == result.reliability_ply
    assert manifest["settings"]["use_reliability_weights"] is True


def test_reliability_colors_are_high_contrast_for_clustered_scores() -> None:
    scores = np.array([0.39, 0.43, 0.50, 0.57, 0.63], dtype=np.float32)

    colors = reliability_to_colors(scores)
    old_palette = {
        (255, 0, 150),
        (255, 110, 0),
        (255, 235, 40),
        (0, 220, 255),
    }

    assert colors.shape == (5, 3)
    assert np.unique(colors, axis=0).shape[0] >= 4
    assert colors[0].tolist() != colors[-1].tolist()
    assert colors[0].tolist() == [91, 86, 162]
    assert colors[-1].tolist() == [245, 234, 158]
    assert not any(tuple(color.tolist()) in old_palette for color in colors)
