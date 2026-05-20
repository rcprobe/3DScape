# Dataset And Output Notes

3DScape V4 is code-first. It does not include downloaded RGB-D datasets, raw
videos, full reconstructed point clouds, trained checkpoints, or large generated
workspaces.

## What Is Not Committed

The following are intentionally ignored by Git:

- downloaded ARKitScenes, ScanNet, TUM RGB-D, or other raw dataset files
- generated `workspaces/`
- generated `demo_outputs/`
- `.ply` point clouds and meshes
- `.npz` occupancy grids
- `.pt`, `.pth`, and `.ckpt` model checkpoints
- `.mov`, `.MOV`, and `.mp4` videos
- local virtual environments

This keeps the repo small and avoids accidentally redistributing files whose
licenses may not permit redistribution.

The exception is `docs/`, which includes lightweight HTML viewers and a short
compressed ARKitScenes reference clip for public demonstration. These are small
portfolio artifacts, not full dataset redistribution. See `docs/ATTRIBUTION.md`.

## Dataset Responsibilities

If you run the demos, download each dataset from its official source and follow
that dataset's terms.

| Dataset | Used for | Notes |
|---|---|---|
| ARKitScenes | RGB-D fusion, reference-mesh comparison, and small demo viewers | Requires Apple's dataset terms. Raw assets and full scans are excluded. |
| ScanNet | Adapter support for ScanNet-style posed RGB-D folders | Access requires approval from the ScanNet maintainers. Do not include ScanNet data in the repo. |
| TUM RGB-D | Lightweight RGB-D geometry debugging and public demo viewers | Public research dataset; cite it and follow its CC BY 4.0 license/usage notes. |
| Generic manifest | Dataset-agnostic integration | Use this path when another source can provide RGB, depth, intrinsics, and camera poses. |

## Generated Outputs

Generated outputs are meant to be reproduced locally:

```text
workspaces/<scan_name>/
  *_raw_metric.ply
  *_downsampled_metric.ply
  *_occupied_only.npz
  *_top_down.png
  *_manifest.json
  pointcloud_viewer.html
```

The public repo should include commands and docs that explain how to regenerate
these files, not the heavy artifacts themselves.

## Portfolio Presentation

Safe phrasing:

- "I built the dataset adapters, geometry fusion pipeline, viewer, tests,
  documentation, and an experimental PyTorch reliability module."
- "The system requires RGB-D frames, camera intrinsics, and camera poses."
- "It does not reconstruct arbitrary phone video by itself."
- "The reliability module is experimental and currently serves as a soft
  confidence signal; it is not yet a proven quality improvement."

Avoid phrasing:

- "This is SOTA."
- "This turns any phone video into a room scan."
- "Reliability Net improves reconstruction quality."
- "The repo contains raw ARKitScenes/ScanNet/TUM dataset files."
