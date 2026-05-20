from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

from scapev3.ply import read_ply_vertices, write_ply


def test_read_ascii_ply_vertices(tmp_path: Path) -> None:
    path = write_ply(
        tmp_path / "points.ply",
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        np.array([[10, 20, 30]], dtype=np.uint8),
    )

    points, colors = read_ply_vertices(path)

    assert np.allclose(points, [[1.0, 2.0, 3.0]])
    assert colors is not None
    assert colors.tolist() == [[10, 20, 30]]


def test_read_binary_little_ply_vertices(tmp_path: Path) -> None:
    path = tmp_path / "mesh.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar alpha\n"
        "element face 0\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(struct.pack("<fffBBBB", 1.0, 2.0, 3.0, 10, 20, 30, 255))
        handle.write(struct.pack("<fffBBBB", 4.0, 5.0, 6.0, 40, 50, 60, 255))

    points, colors = read_ply_vertices(path)

    assert np.allclose(points, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert colors is not None
    assert colors.tolist() == [[10, 20, 30], [40, 50, 60]]
