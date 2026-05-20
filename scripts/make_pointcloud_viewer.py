#!/usr/bin/env python
"""Create a standalone browser orbit viewer for a colored PLY point cloud."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scapev3.ply import read_ply_vertices  # noqa: E402

DEFAULT_ROTATION_DEG = (90.0, 1.0, -180.0)
DEFAULT_AXIS_MAP = "x,y,z"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a standalone HTML orbit viewer for a PLY.")
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--out-html", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=160_000)
    parser.add_argument("--title", type=str, default="3DScape Point Cloud")
    parser.add_argument(
        "--default-rotation-deg",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_ROTATION_DEG,
        help=(
            "Default object-space viewer rotation in degrees. "
            "Use the X/Y/Z dial values from a good manual orientation."
        ),
    )
    parser.add_argument(
        "--axis-map",
        type=str,
        default=DEFAULT_AXIS_MAP,
        metavar="MAP",
        help=(
            "Display-axis remap applied before embedding the cloud. "
            "Use comma-separated source axes with optional signs, e.g. "
            "'x,y,z' or 'x,-z,y' to show a Y-up cloud in a Z-up viewer convention."
        ),
    )
    parser.add_argument(
        "--reliability-legend",
        action="store_true",
        help="Show a color key for reliability-colored point clouds.",
    )
    args = parser.parse_args()

    output = make_pointcloud_viewer(
        ply_path=args.ply,
        output_html=args.out_html,
        max_points=args.max_points,
        title=args.title,
        default_rotation_deg=args.default_rotation_deg,
        axis_map=args.axis_map,
        show_reliability_legend=args.reliability_legend,
    )
    print(f"Wrote point cloud viewer: {output}")


def make_pointcloud_viewer(
    *,
    ply_path: str | Path,
    output_html: str | Path,
    max_points: int,
    title: str,
    default_rotation_deg: tuple[float, float, float] | list[float] = DEFAULT_ROTATION_DEG,
    axis_map: str | tuple[str, str, str] | list[str] = DEFAULT_AXIS_MAP,
    show_reliability_legend: bool = False,
) -> Path:
    """Build an offline WebGL viewer with embedded point data."""

    default_rotation_deg = _validate_rotation_degrees(default_rotation_deg)
    axis_map_tokens, axis_matrix = _validate_axis_map(axis_map)
    points, colors = read_ply_vertices(ply_path, max_points=max_points)
    if colors is None:
        colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)
    points = _apply_axis_map(points, axis_matrix)
    center = np.median(points, axis=0).astype(np.float32)
    centered = points - center[None, :]
    radius = float(np.percentile(np.linalg.norm(centered, axis=1), 98))
    if not np.isfinite(radius) or radius <= 1e-6:
        radius = 1.0
    bounds_min = np.min(centered, axis=0).astype(np.float32)
    bounds_max = np.max(centered, axis=0).astype(np.float32)

    positions = centered.astype("<f4")
    colors = colors.astype(np.uint8)
    payload = {
        "title": title,
        "source": Path(ply_path).name,
        "pointCount": int(points.shape[0]),
        "center": [float(value) for value in center],
        "radius": radius,
        "boundsMin": [float(value) for value in bounds_min],
        "boundsMax": [float(value) for value in bounds_max],
        "defaultRotationDeg": [float(value) for value in default_rotation_deg],
        "axisMap": list(axis_map_tokens),
        "showReliabilityLegend": bool(show_reliability_legend),
        "positionsBase64": base64.b64encode(positions.tobytes()).decode("ascii"),
        "colorsBase64": base64.b64encode(colors.tobytes()).decode("ascii"),
    }
    html = _html_template(json.dumps(payload))
    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _validate_rotation_degrees(rotation: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    if len(rotation) != 3:
        raise ValueError("default_rotation_deg must contain exactly three values: X Y Z degrees")
    values = tuple(float(value) for value in rotation)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("default_rotation_deg must contain finite numeric values")
    return values


def _validate_axis_map(axis_map: str | tuple[str, str, str] | list[str]) -> tuple[tuple[str, str, str], np.ndarray]:
    """Validate and convert a display-axis remap into a 3x3 transform matrix."""

    if isinstance(axis_map, str):
        tokens = tuple(token.strip() for token in axis_map.replace(",", " ").split())
    else:
        tokens = tuple(str(token).strip() for token in axis_map)
    if len(tokens) != 3:
        raise ValueError("axis_map must contain exactly three axes, for example 'x,y,z'")

    axis_to_index = {"x": 0, "y": 1, "z": 2}
    used_axes: set[str] = set()
    matrix = np.zeros((3, 3), dtype=np.float32)
    normalized_tokens: list[str] = []
    for output_index, token in enumerate(tokens):
        if not token:
            raise ValueError("axis_map contains an empty axis token")
        sign = -1.0 if token.startswith("-") else 1.0
        axis = token[1:] if token[0] in "+-" else token
        axis = axis.lower()
        if axis not in axis_to_index:
            raise ValueError(f"Unsupported axis token '{token}'. Use x, y, z with optional +/- signs.")
        if axis in used_axes:
            raise ValueError(f"axis_map repeats source axis '{axis}'")
        used_axes.add(axis)
        matrix[output_index, axis_to_index[axis]] = sign
        normalized_tokens.append(f"-{axis}" if sign < 0 else axis)
    return (normalized_tokens[0], normalized_tokens[1], normalized_tokens[2]), matrix


def _apply_axis_map(points: np.ndarray, axis_matrix: np.ndarray) -> np.ndarray:
    """Return points in display-axis coordinates."""

    return (points.astype(np.float32) @ axis_matrix.T).astype(np.float32)


def _html_template(payload_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>3DScape Point Cloud Viewer</title>
  <style>
    html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: #111; color: #eee; font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; }}
    canvas {{ display: block; width: 100vw; height: 100vh; cursor: grab; }}
    canvas:active {{ cursor: grabbing; }}
    .hud {{ position: fixed; left: 14px; top: 14px; padding: 10px 12px; background: rgba(20, 20, 22, 0.78); border: 1px solid rgba(255,255,255,0.16); border-radius: 8px; backdrop-filter: blur(8px); max-width: min(420px, calc(100vw - 28px)); }}
    .title {{ font-size: 14px; font-weight: 650; margin-bottom: 5px; }}
    .meta {{ font-size: 12px; opacity: 0.78; line-height: 1.35; }}
    .controls {{ display: flex; gap: 8px; align-items: center; margin-top: 10px; flex-wrap: wrap; }}
    .view-title {{ font-size: 12px; font-weight: 650; margin-top: 12px; margin-bottom: 7px; }}
    .preset-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; }}
    .rotation-title {{ font-size: 12px; font-weight: 650; margin-top: 12px; margin-bottom: 7px; }}
    .dial-row {{ display: grid; grid-template-columns: repeat(3, 70px); gap: 10px; align-items: start; margin-top: 4px; }}
    .dial-control {{ display: flex; flex-direction: column; align-items: center; gap: 5px; user-select: none; }}
    .dial-face {{ position: relative; width: 58px; height: 58px; border-radius: 50%; border: 1px solid rgba(255,255,255,0.24); background: radial-gradient(circle at 50% 45%, rgba(255,255,255,0.16), rgba(255,255,255,0.04)); box-shadow: inset 0 0 0 1px rgba(0,0,0,0.22); cursor: grab; touch-action: none; }}
    .dial-face:active {{ cursor: grabbing; }}
    .dial-face:focus-visible {{ outline: 2px solid #8fd3ff; outline-offset: 3px; }}
    .dial-face::before {{ content: ""; position: absolute; left: 50%; top: 6px; width: 3px; height: 7px; border-radius: 999px; background: rgba(255,255,255,0.72); transform: translateX(-50%); }}
    .dial-needle-wrap {{ position: absolute; left: 50%; top: 50%; width: 0; height: 0; transform: rotate(0rad); }}
    .dial-needle {{ position: absolute; left: -2px; bottom: 0; width: 4px; height: 25px; border-radius: 999px; background: #9bd2ff; transform: translateY(-4px); box-shadow: 0 0 8px rgba(155,210,255,0.58); }}
    .dial-center {{ position: absolute; left: 50%; top: 50%; width: 9px; height: 9px; border-radius: 50%; background: #f3f7fb; transform: translate(-50%, -50%); }}
    .dial-label {{ font-size: 11px; line-height: 1.1; opacity: 0.88; text-align: center; min-height: 24px; }}
    .reliability-legend {{ margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.14); font-size: 12px; line-height: 1.35; }}
    .legend-title {{ font-weight: 700; margin-bottom: 6px; }}
    .legend-row {{ display: grid; grid-template-columns: 14px 56px 1fr; gap: 7px; align-items: center; margin: 4px 0; }}
    .swatch {{ width: 13px; height: 13px; border-radius: 3px; border: 1px solid rgba(255,255,255,0.24); }}
    .legend-label {{ font-weight: 650; opacity: 0.9; }}
    .legend-note {{ margin-top: 7px; opacity: 0.74; }}
    button {{ border: 1px solid rgba(255,255,255,0.2); background: rgba(255,255,255,0.08); color: #eee; border-radius: 6px; padding: 5px 8px; font: inherit; font-size: 12px; }}
    button.active {{ background: rgba(155,210,255,0.18); border-color: rgba(155,210,255,0.45); }}
    label {{ font-size: 12px; display: inline-flex; gap: 6px; align-items: center; }}
    input[type="range"] {{ width: 110px; }}
  </style>
</head>
<body>
  <canvas id="view"></canvas>
  <div class="hud">
    <div class="title" id="title"></div>
    <div class="meta" id="meta"></div>
    <div class="controls">
      <button id="reset">Reset View</button>
      <button id="resetObject">Reset Object</button>
      <button id="guideToggle">Hide Axes</button>
      <label>Point Size <input id="pointSize" type="range" min="1" max="8" step="0.5" value="2.5"></label>
    </div>
    <div class="view-title">View Presets</div>
    <div class="preset-grid" aria-label="Camera view presets">
      <button class="view-preset" data-view="iso">Iso</button>
      <button class="view-preset" data-view="top">Top</button>
      <button class="view-preset" data-view="front">Front</button>
      <button class="view-preset" data-view="right">Right</button>
      <button class="view-preset" data-view="left">Left</button>
      <button class="view-preset" data-view="back">Back</button>
    </div>
    <div class="rotation-title">Object Rotation</div>
    <div class="dial-row" aria-label="Object rotation dials">
      <div class="dial-control">
        <div class="dial-face" id="dialX" role="slider" tabindex="0" aria-label="Rotate object around X axis" aria-valuemin="-180" aria-valuemax="180" aria-valuenow="0">
          <div class="dial-needle-wrap"><div class="dial-needle"></div></div>
          <div class="dial-center"></div>
        </div>
        <div class="dial-label">X<br><span id="rotXValue">0°</span></div>
      </div>
      <div class="dial-control">
        <div class="dial-face" id="dialY" role="slider" tabindex="0" aria-label="Rotate object around Y axis" aria-valuemin="-180" aria-valuemax="180" aria-valuenow="0">
          <div class="dial-needle-wrap"><div class="dial-needle"></div></div>
          <div class="dial-center"></div>
        </div>
        <div class="dial-label">Y<br><span id="rotYValue">0°</span></div>
      </div>
      <div class="dial-control">
        <div class="dial-face" id="dialZ" role="slider" tabindex="0" aria-label="Rotate object around Z axis" aria-valuemin="-180" aria-valuemax="180" aria-valuenow="0">
          <div class="dial-needle-wrap"><div class="dial-needle"></div></div>
          <div class="dial-center"></div>
        </div>
        <div class="dial-label">Z<br><span id="rotZValue">0°</span></div>
      </div>
    </div>
    <div id="reliabilityLegend" class="reliability-legend" hidden>
      <div class="legend-title">Reliability Key</div>
      <div class="legend-row"><span class="swatch" style="background:#f5ea9e"></span><span class="legend-label">Highest</span><span>most trusted local surface evidence</span></div>
      <div class="legend-row"><span class="swatch" style="background:#f9b276"></span><span class="legend-label">High</span><span>more stable multi-view depth</span></div>
      <div class="legend-row"><span class="swatch" style="background:#e87783"></span><span class="legend-label">Mid</span><span>usable but less certain geometry</span></div>
      <div class="legend-row"><span class="swatch" style="background:#b35a9a"></span><span class="legend-label">Low</span><span>possible edge/noise/occlusion artifacts</span></div>
      <div class="legend-row"><span class="swatch" style="background:#5b56a2"></span><span class="legend-label">Lowest</span><span>least trusted depth observations</span></div>
      <div class="legend-note">Reliability matters because low-confidence depth can create floating points, fuzzy walls, and bad training data. These colors show what the model distrusts without hiding the scan.</div>
    </div>
  </div>
<script>
const DATA = {payload_json};
const canvas = document.getElementById("view");
const gl = canvas.getContext("webgl", {{ antialias: true, alpha: false }});
if (!gl) {{
  document.body.innerHTML = "<p style='padding:20px'>WebGL is not available in this browser.</p>";
  throw new Error("WebGL unavailable");
}}

document.getElementById("title").textContent = DATA.title;
document.getElementById("meta").textContent = `${{DATA.pointCount.toLocaleString()}} points. Drag to rotate freely, Option-drag to roll, right/Shift-drag to pan, scroll to zoom. Keys 1-6 switch views.`;
const reliabilityLegend = document.getElementById("reliabilityLegend");
if (DATA.showReliabilityLegend && reliabilityLegend) reliabilityLegend.hidden = false;

function decodeFloat32(base64) {{
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}}
function decodeUint8(base64) {{
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}}

const positions = decodeFloat32(DATA.positionsBase64);
const colors = decodeUint8(DATA.colorsBase64);

const vertexShaderSource = `
attribute vec3 aPosition;
attribute vec3 aColor;
uniform mat4 uViewProj;
uniform mat4 uModel;
uniform float uPointSize;
uniform int uDrawPoints;
varying vec3 vColor;
void main() {{
  gl_Position = uViewProj * uModel * vec4(aPosition, 1.0);
  gl_PointSize = uPointSize;
  vColor = aColor / 255.0;
}}`;
const fragmentShaderSource = `
precision mediump float;
varying vec3 vColor;
uniform int uDrawPoints;
void main() {{
  if (uDrawPoints == 1) {{
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    if (dot(p, p) > 1.0) discard;
  }}
  gl_FragColor = vec4(vColor, 1.0);
}}`;

function compileShader(type, source) {{
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}}
const program = gl.createProgram();
gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vertexShaderSource));
gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fragmentShaderSource));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
gl.useProgram(program);

const posBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
const aPosition = gl.getAttribLocation(program, "aPosition");
gl.enableVertexAttribArray(aPosition);
gl.vertexAttribPointer(aPosition, 3, gl.FLOAT, false, 0, 0);

const colorBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
const aColor = gl.getAttribLocation(program, "aColor");
gl.enableVertexAttribArray(aColor);
gl.vertexAttribPointer(aColor, 3, gl.UNSIGNED_BYTE, false, 0, 0);

const guideLines = buildGuideLines();
const guidePosBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, guidePosBuffer);
gl.bufferData(gl.ARRAY_BUFFER, guideLines.positions, gl.STATIC_DRAW);
const guideColorBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, guideColorBuffer);
gl.bufferData(gl.ARRAY_BUFFER, guideLines.colors, gl.STATIC_DRAW);

const uViewProj = gl.getUniformLocation(program, "uViewProj");
const uModel = gl.getUniformLocation(program, "uModel");
const uPointSize = gl.getUniformLocation(program, "uPointSize");
const uDrawPoints = gl.getUniformLocation(program, "uDrawPoints");

let distance = DATA.radius * 2.6;
let yaw = -0.75;
let pitch = -0.35;
let roll = 0.0;
let pan = [0, 0, 0];
const DEFAULT_OBJECT_ROTATION_DEG = DATA.defaultRotationDeg || [90, 1, -180];
const DEFAULT_OBJECT_ROTATION = Object.freeze({{
  x: degreesToRadians(DEFAULT_OBJECT_ROTATION_DEG[0]),
  y: degreesToRadians(DEFAULT_OBJECT_ROTATION_DEG[1]),
  z: degreesToRadians(DEFAULT_OBJECT_ROTATION_DEG[2])
}});
let objectRotation = copyObjectRotation(DEFAULT_OBJECT_ROTATION);
let showGuides = true;
let activeView = "iso";
let dragging = false;
let last = [0, 0];
let mode = "rotate";
let pointSize = 2.5;

const dialControls = {{
  x: {{
    face: document.getElementById("dialX"),
    value: document.getElementById("rotXValue"),
    needle: document.querySelector("#dialX .dial-needle-wrap")
  }},
  y: {{
    face: document.getElementById("dialY"),
    value: document.getElementById("rotYValue"),
    needle: document.querySelector("#dialY .dial-needle-wrap")
  }},
  z: {{
    face: document.getElementById("dialZ"),
    value: document.getElementById("rotZValue"),
    needle: document.querySelector("#dialZ .dial-needle-wrap")
  }}
}};

document.getElementById("pointSize").addEventListener("input", event => {{
  pointSize = Number(event.target.value);
}});
document.getElementById("reset").addEventListener("click", () => {{
  setViewPreset("iso");
}});
document.getElementById("resetObject").addEventListener("click", () => {{
  objectRotation = copyObjectRotation(DEFAULT_OBJECT_ROTATION);
  updateAllDials();
}});
document.getElementById("guideToggle").addEventListener("click", toggleGuides);
document.querySelectorAll(".view-preset").forEach(button => {{
  button.addEventListener("click", () => setViewPreset(button.dataset.view));
}});

setupDial("x");
setupDial("y");
setupDial("z");
updateAllDials();
updatePresetButtons();

canvas.addEventListener("contextmenu", event => event.preventDefault());
canvas.addEventListener("pointerdown", event => {{
  dragging = true;
  last = [event.clientX, event.clientY];
  mode = event.button === 2 || event.shiftKey ? "pan" : event.altKey ? "roll" : "rotate";
  canvas.setPointerCapture(event.pointerId);
}});
canvas.addEventListener("pointerup", () => dragging = false);
canvas.addEventListener("pointermove", event => {{
  if (!dragging) return;
  const dx = event.clientX - last[0];
  const dy = event.clientY - last[1];
  last = [event.clientX, event.clientY];
  if (mode === "pan") {{
    const scale = distance * 0.0017;
    const right = applyOrbitRotation([1, 0, 0]);
    const up = applyOrbitRotation([0, 1, 0]);
    pan[0] += (-right[0] * dx + up[0] * dy) * scale;
    pan[1] += (-right[1] * dx + up[1] * dy) * scale;
    pan[2] += (-right[2] * dx + up[2] * dy) * scale;
  }} else if (mode === "roll") {{
    roll += dx * 0.008;
  }} else {{
    yaw += dx * 0.006;
    pitch = clamp(pitch + dy * 0.006, -Math.PI / 2 + 0.02, Math.PI / 2 - 0.02);
    activeView = "custom";
    updatePresetButtons();
  }}
}});
canvas.addEventListener("wheel", event => {{
  event.preventDefault();
  distance *= Math.exp(event.deltaY * 0.001);
  distance = Math.max(DATA.radius * 0.08, Math.min(DATA.radius * 20.0, distance));
}}, {{ passive: false }});
window.addEventListener("keydown", event => {{
  if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
  const activeTag = document.activeElement?.tagName;
  if (activeTag === "INPUT" || activeTag === "BUTTON") return;
  const keyViews = {{
    "1": "iso",
    "2": "top",
    "3": "front",
    "4": "right",
    "5": "left",
    "6": "back"
  }};
  if (keyViews[event.key]) {{
    event.preventDefault();
    setViewPreset(keyViews[event.key]);
  }} else if (event.key.toLowerCase() === "r") {{
    event.preventDefault();
    setViewPreset("iso");
  }} else if (event.key.toLowerCase() === "o") {{
    event.preventDefault();
    objectRotation = copyObjectRotation(DEFAULT_OBJECT_ROTATION);
    updateAllDials();
  }} else if (event.key.toLowerCase() === "a") {{
    event.preventDefault();
    toggleGuides();
  }}
}});

function setupDial(axis) {{
  const control = dialControls[axis];
  const updateFromPointer = event => {{
    const rect = control.face.getBoundingClientRect();
    const x = event.clientX - (rect.left + rect.width / 2);
    const y = event.clientY - (rect.top + rect.height / 2);
    setObjectRotation(axis, normalizeAngle(Math.atan2(y, x) + Math.PI / 2));
  }};
  control.face.addEventListener("pointerdown", event => {{
    event.preventDefault();
    control.face.setPointerCapture(event.pointerId);
    updateFromPointer(event);
  }});
  control.face.addEventListener("pointermove", event => {{
    if (event.buttons) updateFromPointer(event);
  }});
  control.face.addEventListener("wheel", event => {{
    event.preventDefault();
    setObjectRotation(axis, objectRotation[axis] + (event.deltaY > 0 ? -0.05 : 0.05));
  }}, {{ passive: false }});
  control.face.addEventListener("keydown", event => {{
    const smallStep = 5 * Math.PI / 180;
    const largeStep = 15 * Math.PI / 180;
    if (event.key === "ArrowUp" || event.key === "ArrowRight") {{
      event.preventDefault();
      setObjectRotation(axis, objectRotation[axis] + (event.shiftKey ? largeStep : smallStep));
    }} else if (event.key === "ArrowDown" || event.key === "ArrowLeft") {{
      event.preventDefault();
      setObjectRotation(axis, objectRotation[axis] - (event.shiftKey ? largeStep : smallStep));
    }} else if (event.key === "Home" || event.key === "0") {{
      event.preventDefault();
      setObjectRotation(axis, DEFAULT_OBJECT_ROTATION[axis]);
    }}
  }});
}}

function toggleGuides() {{
  showGuides = !showGuides;
  document.getElementById("guideToggle").textContent = showGuides ? "Hide Axes" : "Show Axes";
}}

function setViewPreset(name) {{
  const presets = {{
    iso: {{ yaw: -0.75, pitch: -0.35, roll: 0.0 }},
    top: {{ yaw: 0.0, pitch: -Math.PI / 2 + 0.02, roll: 0.0 }},
    front: {{ yaw: 0.0, pitch: 0.0, roll: 0.0 }},
    right: {{ yaw: Math.PI / 2, pitch: 0.0, roll: 0.0 }},
    left: {{ yaw: -Math.PI / 2, pitch: 0.0, roll: 0.0 }},
    back: {{ yaw: Math.PI, pitch: 0.0, roll: 0.0 }}
  }};
  const preset = presets[name] || presets.iso;
  yaw = preset.yaw;
  pitch = preset.pitch;
  roll = preset.roll;
  distance = DATA.radius * 2.6;
  pan = [0, 0, 0];
  objectRotation = copyObjectRotation(DEFAULT_OBJECT_ROTATION);
  activeView = name;
  updateAllDials();
  updatePresetButtons();
}}

function updatePresetButtons() {{
  document.querySelectorAll(".view-preset").forEach(button => {{
    button.classList.toggle("active", button.dataset.view === activeView);
  }});
}}

function copyObjectRotation(rotation) {{
  return {{ x: rotation.x, y: rotation.y, z: rotation.z }};
}}

function degreesToRadians(degrees) {{
  return degrees * Math.PI / 180;
}}

function setObjectRotation(axis, angle) {{
  objectRotation[axis] = normalizeAngle(angle);
  updateDial(axis);
}}

function updateAllDials() {{
  updateDial("x");
  updateDial("y");
  updateDial("z");
}}

function updateDial(axis) {{
  const control = dialControls[axis];
  const angle = objectRotation[axis];
  const degrees = Math.round(angle * 180 / Math.PI);
  control.needle.style.transform = `rotate(${{angle}}rad)`;
  control.value.textContent = `${{degrees}}°`;
  control.face.setAttribute("aria-valuenow", String(degrees));
}}

function normalizeAngle(angle) {{
  while (angle <= -Math.PI) angle += Math.PI * 2;
  while (angle > Math.PI) angle -= Math.PI * 2;
  return angle;
}}

function clamp(value, low, high) {{
  return Math.max(low, Math.min(high, value));
}}

function resize() {{
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {{
    canvas.width = width; canvas.height = height;
  }}
  gl.viewport(0, 0, canvas.width, canvas.height);
}}

function perspective(fovy, aspect, near, far) {{
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, (2 * far * near) * nf, 0
  ]);
}}
function lookAt(eye, target, up) {{
  const z = normalize(sub(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
  ]);
}}
function multiply(a, b) {{
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {{
    for (let r = 0; r < 4; r++) {{
      out[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }}
  }}
  return out;
}}
function bindAttributeBuffers(positionBuffer, colorBuffer, colorType) {{
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.vertexAttribPointer(aPosition, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.vertexAttribPointer(aColor, 3, colorType, false, 0, 0);
}}
function buildGuideLines() {{
  const mn = DATA.boundsMin;
  const mx = DATA.boundsMax;
  const corners = [
    [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]], [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
    [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]], [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]]
  ];
  const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  const positions = [];
  const colors = [];
  const addLine = (a, b, color) => {{
    positions.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    colors.push(color[0], color[1], color[2], color[0], color[1], color[2]);
  }};
  for (const [a, b] of edges) addLine(corners[a], corners[b], [140, 154, 168]);
  const axis = DATA.radius * 1.2;
  addLine([-axis, 0, 0], [axis, 0, 0], [255, 88, 88]);
  addLine([0, -axis, 0], [0, axis, 0], [95, 220, 130]);
  addLine([0, 0, -axis], [0, 0, axis], [105, 165, 255]);
  return {{
    positions: new Float32Array(positions),
    colors: new Uint8Array(colors),
    count: positions.length / 3
  }};
}}
function matrixRotationX(angle) {{
  const c = Math.cos(angle), s = Math.sin(angle);
  return new Float32Array([
    1, 0, 0, 0,
    0, c, s, 0,
    0, -s, c, 0,
    0, 0, 0, 1
  ]);
}}
function matrixRotationY(angle) {{
  const c = Math.cos(angle), s = Math.sin(angle);
  return new Float32Array([
    c, 0, -s, 0,
    0, 1, 0, 0,
    s, 0, c, 0,
    0, 0, 0, 1
  ]);
}}
function matrixRotationZ(angle) {{
  const c = Math.cos(angle), s = Math.sin(angle);
  return new Float32Array([
    c, s, 0, 0,
    -s, c, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  ]);
}}
function objectRotationMatrix() {{
  const rx = matrixRotationX(objectRotation.x);
  const ry = matrixRotationY(objectRotation.y);
  const rz = matrixRotationZ(objectRotation.z);
  return multiply(multiply(rz, ry), rx);
}}
function sub(a, b) {{ return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }}
function dot(a, b) {{ return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }}
function cross(a, b) {{ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }}
function normalize(v) {{
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0]/len, v[1]/len, v[2]/len];
}}
function rotateX(v, angle) {{
  const c = Math.cos(angle), s = Math.sin(angle);
  return [v[0], v[1] * c - v[2] * s, v[1] * s + v[2] * c];
}}
function rotateY(v, angle) {{
  const c = Math.cos(angle), s = Math.sin(angle);
  return [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c];
}}
function rotateZ(v, angle) {{
  const c = Math.cos(angle), s = Math.sin(angle);
  return [v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]];
}}
function applyOrbitRotation(v) {{
  return rotateY(rotateX(rotateZ(v, roll), pitch), yaw);
}}

function draw() {{
  resize();
  gl.clearColor(0.06, 0.06, 0.065, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.useProgram(program);

  const offset = applyOrbitRotation([0, 0, distance]);
  const up = applyOrbitRotation([0, 1, 0]);
  const eye = [pan[0] + offset[0], pan[1] + offset[1], pan[2] + offset[2]];
  const target = [pan[0], pan[1], pan[2]];
  const view = lookAt(eye, target, up);
  const proj = perspective(50 * Math.PI / 180, canvas.width / canvas.height, Math.max(0.01, DATA.radius * 0.005), DATA.radius * 50);
  gl.uniformMatrix4fv(uViewProj, false, multiply(proj, view));
  gl.uniformMatrix4fv(uModel, false, objectRotationMatrix());
  gl.uniform1f(uPointSize, pointSize * Math.min(window.devicePixelRatio || 1, 2));
  gl.uniform1i(uDrawPoints, 1);
  bindAttributeBuffers(posBuffer, colorBuffer, gl.UNSIGNED_BYTE);
  gl.drawArrays(gl.POINTS, 0, DATA.pointCount);
  if (showGuides) {{
    gl.uniform1i(uDrawPoints, 0);
    gl.uniform1f(uPointSize, 1.0);
    bindAttributeBuffers(guidePosBuffer, guideColorBuffer, gl.UNSIGNED_BYTE);
    gl.drawArrays(gl.LINES, 0, guideLines.count);
  }}
  requestAnimationFrame(draw);
}}
draw();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
