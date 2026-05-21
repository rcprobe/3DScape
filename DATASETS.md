# Datasets and Generated Outputs

This repository contains the 3DScape source code, tests, documentation, reproduction commands, and lightweight demo artifacts. It does **not** include downloaded RGB-D datasets, raw dataset videos, full reconstructed point clouds, trained checkpoints, or generated workspaces.

The goal is to keep the repository small, reproducible, and respectful of dataset-specific redistribution terms.

## What Is Not Committed

The following files and directories are intentionally excluded from Git:

- downloaded ARKitScenes, ScanNet, TUM RGB-D, or other raw dataset files
- generated `workspaces/`
- generated `demo_outputs/`
- generated `.ply` point clouds and meshes
- generated `.npz` occupancy grids
- trained `.pt`, `.pth`, and `.ckpt` model checkpoints
- raw `.mov`, `.MOV`, and large `.mp4` videos
- local virtual environments and machine-specific files

Generated artifacts should be reproduced locally from the code and commands in the README.

## Lightweight Demo Artifacts

The `docs/` directory may include lightweight artifacts for public inspection, such as:

- static preview images
- small browser-based demo viewers
- limited compressed demo media
- attribution notes

These files are included to make the project inspectable without requiring a full local reconstruction run. They are not intended to redistribute full datasets, raw scans, full generated workspaces, or large reconstruction outputs.

For dataset and media attribution, see [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md).

## Dataset Responsibilities

To run 3DScape on real RGB-D data, download each dataset from its official source and follow that dataset's license, citation requirements, and terms of use.

| Dataset / input | Used for | Repository policy |
| --- | --- | --- |
| ARKitScenes | RGB-D fusion, reference-mesh comparison, and demo reconstruction examples | Download from Apple's official ARKitScenes release and follow its dataset terms. Raw assets, full scans, meshes, and full-resolution videos are not committed. |
| ScanNet | Adapter support for ScanNet-style posed RGB-D folders | Access requires approval and agreement to the ScanNet Terms of Use. Do not commit ScanNet data or derived large artifacts. |
| TUM RGB-D | Lightweight RGB-D geometry debugging and public demo examples | Download from the official TUM RGB-D benchmark page, cite the benchmark publication, and follow its license and usage notes. |
| Generic manifest | Dataset-agnostic integration | Use this path when another source can provide RGB images, depth maps, intrinsics, and camera poses. Dataset ownership and redistribution rules remain the user's responsibility. |

## Generated Outputs

3DScape writes reconstruction artifacts to local output directories such as:

```text
workspaces/<scan_name>/
  *_raw_metric.ply
  *_downsampled_metric.ply
  *_occupied_only.npz
  *_top_down.png
  *_manifest.json
  pointcloud_viewer.html
```
