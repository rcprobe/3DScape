from __future__ import annotations

import numpy as np
import pytest

from scapev3.compare import approximate_nearest_distances, summarize_distances


def test_approximate_nearest_distances_matches_simple_points() -> None:
    queries = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    refs = np.array([[0.0, 0.0, 0.1], [3.0, 0.0, 0.0]], dtype=np.float32)

    distances = approximate_nearest_distances(
        queries,
        refs,
        grid_size_m=0.2,
        max_search_radius_m=2.5,
    )

    assert distances[0] == pytest.approx(0.1)
    assert distances[1] == pytest.approx(np.sqrt(1.01))


def test_summarize_distances_counts_unmatched() -> None:
    stats = summarize_distances(np.array([0.1, np.nan, 0.3], dtype=np.float32))

    assert stats["matched_fraction"] == pytest.approx(2 / 3)
    assert stats["unmatched_fraction"] == pytest.approx(1 / 3)
    assert stats["median"] == pytest.approx(0.2)
