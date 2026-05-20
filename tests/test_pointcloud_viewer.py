from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_viewer_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_pointcloud_viewer.py"
    spec = importlib.util.spec_from_file_location("make_pointcloud_viewer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load make_pointcloud_viewer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_axis_map_can_convert_y_up_to_z_up() -> None:
    viewer = _load_viewer_module()
    tokens, matrix = viewer._validate_axis_map("x,-z,y")
    points = np.asarray([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32)

    remapped = viewer._apply_axis_map(points, matrix)

    assert tokens == ("x", "-z", "y")
    assert np.allclose(remapped, [[1.0, -3.0, 2.0], [-4.0, 6.0, 5.0]])


def test_axis_map_rejects_repeated_axes() -> None:
    viewer = _load_viewer_module()

    with pytest.raises(ValueError, match="repeats source axis"):
        viewer._validate_axis_map("x,x,z")
