"""PLY IO utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


_PLY_DTYPE_MAP = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def write_ply(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> Path:
    """Write an ASCII PLY point cloud."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if colors is None:
        colors = np.full((points.shape[0], 3), 210, dtype=np.uint8)
    colors = np.asarray(colors, dtype=np.uint8)
    if colors.shape != (points.shape[0], 3):
        raise ValueError("colors must have shape [N, 3]")
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32)
        if normals.shape != (points.shape[0], 3):
            raise ValueError("normals must have shape [N, 3]")

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        if normals is not None:
            handle.write("property float nx\n")
            handle.write("property float ny\n")
            handle.write("property float nz\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for idx, point in enumerate(points):
            fields = [f"{point[0]:.6f}", f"{point[1]:.6f}", f"{point[2]:.6f}"]
            if normals is not None:
                normal = normals[idx]
                fields.extend([f"{normal[0]:.6f}", f"{normal[1]:.6f}", f"{normal[2]:.6f}"])
            color = colors[idx]
            fields.extend([str(int(color[0])), str(int(color[1])), str(int(color[2]))])
            handle.write(" ".join(fields) + "\n")
    return output_path


def read_ply_vertices(
    path: str | Path,
    *,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read vertex XYZ and optional RGB from an ASCII or binary little-endian PLY.

    The reader intentionally focuses on vertex properties. Face data and other
    elements are ignored, which is enough for point-cloud/mesh-vertex metrics.
    """

    path = Path(path)
    if max_points is not None and max_points < 1:
        raise ValueError("max_points must be >= 1")
    with path.open("rb") as handle:
        format_name, vertex_count, properties = _read_ply_header(handle)
        if vertex_count < 1:
            raise ValueError(f"PLY has no vertices: {path}")
        if format_name == "ascii":
            points, colors = _read_ascii_vertices(handle, vertex_count, properties)
        elif format_name == "binary_little_endian":
            points, colors = _read_binary_little_vertices(handle, vertex_count, properties)
        else:
            raise ValueError(f"Unsupported PLY format `{format_name}` in {path}")

    if max_points is not None and points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
        points = points[indices]
        colors = colors[indices] if colors is not None else None
    return points.astype(np.float32), colors


def _read_ply_header(handle) -> tuple[str, int, list[tuple[str, str]]]:
    first = handle.readline().decode("utf-8", errors="replace").strip()
    if first != "ply":
        raise ValueError("File is not a PLY")

    format_name: str | None = None
    vertex_count: int | None = None
    properties: list[tuple[str, str]] = []
    in_vertex = False
    while True:
        raw = handle.readline()
        if not raw:
            raise RuntimeError("PLY ended before end_header")
        line = raw.decode("utf-8", errors="replace").strip()
        if line == "end_header":
            break
        if not line or line.startswith("comment"):
            continue
        parts = line.split()
        if parts[:1] == ["format"]:
            format_name = parts[1]
        elif parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
        elif parts[:1] == ["element"]:
            in_vertex = False
        elif in_vertex and parts[:1] == ["property"]:
            if len(parts) >= 3 and parts[1] != "list":
                properties.append((parts[2], parts[1]))

    if format_name is None:
        raise RuntimeError("PLY header has no format")
    if vertex_count is None:
        raise RuntimeError("PLY header has no element vertex")
    for required in ["x", "y", "z"]:
        if required not in {name for name, _dtype in properties}:
            raise RuntimeError(f"PLY vertex is missing `{required}` property")
    return format_name, vertex_count, properties


def _read_ascii_vertices(
    handle,
    vertex_count: int,
    properties: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray | None]:
    names = [name for name, _dtype in properties]
    x_idx, y_idx, z_idx = names.index("x"), names.index("y"), names.index("z")
    color_indices = _color_indices(names)
    points = np.zeros((vertex_count, 3), dtype=np.float32)
    colors = np.zeros((vertex_count, 3), dtype=np.uint8) if color_indices else None
    for idx in range(vertex_count):
        line = handle.readline().decode("utf-8", errors="replace").strip()
        if not line:
            raise RuntimeError("PLY ended before all ASCII vertices were read")
        values = line.split()
        points[idx] = [float(values[x_idx]), float(values[y_idx]), float(values[z_idx])]
        if colors is not None and color_indices is not None:
            colors[idx] = [int(float(values[index])) for index in color_indices]
    return points, colors


def _read_binary_little_vertices(
    handle,
    vertex_count: int,
    properties: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray | None]:
    dtype_fields = []
    for name, ply_type in properties:
        if ply_type not in _PLY_DTYPE_MAP:
            raise ValueError(f"Unsupported PLY property type: {ply_type}")
        dtype_fields.append((name, _PLY_DTYPE_MAP[ply_type]))
    data = np.fromfile(handle, dtype=np.dtype(dtype_fields), count=vertex_count)
    points = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    color_indices = _color_indices([name for name, _dtype in properties])
    colors = None
    if color_indices:
        color_names = [properties[index][0] for index in color_indices]
        colors = np.stack([data[name] for name in color_names], axis=1).astype(np.uint8)
    return points, colors


def _color_indices(names: list[str]) -> tuple[int, int, int] | None:
    if {"red", "green", "blue"}.issubset(set(names)):
        return names.index("red"), names.index("green"), names.index("blue")
    if {"r", "g", "b"}.issubset(set(names)):
        return names.index("r"), names.index("g"), names.index("b")
    return None
