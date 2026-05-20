from __future__ import annotations

import numpy as np
import pytest

from scapev3.rgbd import CameraIntrinsics, CameraPose

torch = pytest.importorskip("torch")

from scapev3.learned.reliability import (  # noqa: E402
    TinyReliabilityNet,
    build_multiview_reliability_label,
    build_multiview_reliability_target,
    make_reliability_features,
)


def test_reliability_features_and_model_forward() -> None:
    rgb = np.full((8, 8, 3), [30, 120, 210], dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)
    features = make_reliability_features(rgb=rgb, depth_m=depth)
    model = TinyReliabilityNet(base_channels=4)

    logits = model(torch.from_numpy(features[None, ...]))

    assert features.shape == (7, 8, 8)
    assert logits.shape == (1, 1, 8, 8)


def test_multiview_reliability_label_marks_identity_views_reliable() -> None:
    intrinsics = CameraIntrinsics(width=6, height=6, fx=4.0, fy=4.0, cx=2.5, cy=2.5)
    pose = CameraPose(timestamp=0.0, rotation=np.eye(3), translation=np.zeros(3))
    depth = np.ones((6, 6), dtype=np.float32)

    labels, mask = build_multiview_reliability_label(
        source_depth_m=depth,
        source_intrinsics=intrinsics,
        source_pose=pose,
        target_depth_m=depth,
        target_intrinsics=intrinsics,
        target_pose=pose,
        pixel_stride=1,
    )

    assert mask.sum() > 0
    assert labels[mask].mean() == 1.0


def test_multiview_reliability_target_can_return_soft_labels() -> None:
    intrinsics = CameraIntrinsics(width=6, height=6, fx=4.0, fy=4.0, cx=2.5, cy=2.5)
    pose = CameraPose(timestamp=0.0, rotation=np.eye(3), translation=np.zeros(3))
    source_depth = np.ones((6, 6), dtype=np.float32)
    agreeing_depth = np.ones((6, 6), dtype=np.float32)
    disagreeing_depth = np.full((6, 6), 2.0, dtype=np.float32)

    labels, mask = build_multiview_reliability_target(
        source_depth_m=source_depth,
        source_intrinsics=intrinsics,
        source_pose=pose,
        target_depths_m=[agreeing_depth, disagreeing_depth],
        target_intrinsics=[intrinsics, intrinsics],
        target_poses=[pose, pose],
        pixel_stride=1,
        min_target_observations=2,
    )

    assert mask.sum() > 0
    assert np.allclose(labels[mask], 0.5)
