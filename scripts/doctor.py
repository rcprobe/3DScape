#!/usr/bin/env python
"""Check local dependencies for 3DScape V4."""

from __future__ import annotations

import importlib.util
import sys


def main() -> None:
    print("3DScape V4 dependency check")
    print(f"python: {sys.executable}")
    for module in ["cv2", "numpy", "PIL"]:
        status = "ok" if importlib.util.find_spec(module) is not None else "missing"
        print(f"{module}: {status}")


if __name__ == "__main__":
    main()
