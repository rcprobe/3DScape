from __future__ import annotations

from pathlib import Path

import numpy as np

from scapev3.arkitscenes import (
    ARKitIntrinsics,
    ARKitPose,
    arkit_camera_to_world,
    axis_angle_to_rotation_matrix,
    backproject_arkit_depth,
    load_intrinsics,
    load_trajectory,
    select_frames_by_rate,
)
from scapev3.arkitscenes import ARKitFrame


def test_load_intrinsics(tmp_path: Path) -> None:
    path = tmp_path / "frame.pincam"
    path.write_text("256 192 212.8 212.8 126.6 93.1\n", encoding="utf-8")

    intrinsics = load_intrinsics(path)

    assert intrinsics.width == 256
    assert intrinsics.height == 192
    assert intrinsics.fx == 212.8


def test_backproject_arkit_depth_center_pixel() -> None:
    intrinsics = ARKitIntrinsics(width=3, height=3, fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    depth = np.ones((3, 3), dtype=np.float32)
    confidence = np.full((3, 3), 2, dtype=np.uint8)

    points, pixels = backproject_arkit_depth(
        depth,
        intrinsics,
        confidence=confidence,
        min_confidence=2,
        pixel_stride=1,
    )

    center_index = int(np.where((pixels[:, 0] == 1) & (pixels[:, 1] == 1))[0][0])
    assert np.allclose(points[center_index], [0.0, 0.0, 1.0])


def test_axis_angle_identity_and_camera_to_world() -> None:
    rotation = axis_angle_to_rotation_matrix(np.zeros(3))
    pose = ARKitPose(timestamp=0.0, rotation=rotation, translation=np.array([1.0, 2.0, 3.0]))

    points = arkit_camera_to_world(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), pose)

    assert np.allclose(points, [[1.0, 2.0, 2.0]])


def test_load_trajectory_inverts_world_to_camera_transform(tmp_path: Path) -> None:
    path = tmp_path / "lowres_wide.traj"
    path.write_text("1.0 0.0 0.0 0.0 1.0 2.0 3.0\n", encoding="utf-8")

    pose = load_trajectory(path)[0]

    assert np.allclose(pose.rotation, np.eye(3))
    assert np.allclose(pose.translation, [-1.0, -2.0, -3.0])


def test_select_frames_by_rate_limits_count() -> None:
    frames = [
        ARKitFrame(timestamp=float(index) * 0.1, depth_path="d", confidence_path=None, intrinsics_path="i")
        for index in range(20)
    ]

    selected = select_frames_by_rate(frames, target_fps=5.0, max_frames=4)

    assert len(selected) <= 4
    assert selected[0].timestamp == 0.0
