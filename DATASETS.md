# Dataset And Output Notes

3DScape is code-first. It does not include downloaded RGB-D datasets, raw
dataset videos, full reconstructed point clouds, trained checkpoints, or large
generated workspaces.

## What Is Not Committed

The following are intentionally ignored by Git:

- downloaded ARKitScenes, ScanNet, TUM RGB-D, or other raw dataset files
- generated `workspaces/`
- generated `demo_outputs/`
- `.ply` point clouds and meshes
- `.npz` occupancy grids
- `.pt`, `.pth`, and `.ckpt` model checkpoints
- `.mov`, `.MOV`, and large `.mp4` videos
- local virtual environments

This keeps the repo small and avoids redistributing files whose licenses may
not permit redistribution.

The exception is `docs/`, which contains lightweight generated demo viewers,
static preview images, and one web-compressed ARKitScenes reference clip for
public inspection. These are demonstration artifacts, not full dataset
redistribution. See `docs/ATTRIBUTION.md`.

## Dataset Responsibilities

If you run the demos, download each dataset from its official source and follow
that dataset's terms.

| Dataset | Used for | Notes |
| --- | --- | --- |
| ARKitScenes | RGB-D fusion, reference-mesh comparison, and small demo viewers | Requires Apple's dataset terms. Raw assets and full scans are excluded. |
| ScanNet | Adapter support for ScanNet-style posed RGB-D folders | Access requires approval from the ScanNet maintainers. Do not include ScanNet data in the repo. |
| TUM RGB-D | Lightweight RGB-D geometry debugging and public demo viewers | Public research dataset; cite it and follow its license and usage notes. |
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

The public repository should include the code, docs, tests, lightweight demo
media, and reproduction commands needed to regenerate these files, not the
heavy artifacts themselves.
