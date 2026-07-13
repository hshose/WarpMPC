from __future__ import annotations

import itertools
import unittest

import numpy as np

from warpmpc.jax_qdldl.core import (
    _make_adaptive_segments,
    _make_fixed_segments,
    _make_optimal_segments,
    _make_segments,
)


def _mask_from_counts(counts: list[int]) -> np.ndarray:
    width = max(counts, default=0)
    mask = np.zeros((len(counts), width), dtype=bool)
    for row, count in enumerate(counts):
        mask[row, :count] = True
    return mask


def _cost(counts: np.ndarray, segments) -> int:
    out = 0
    for _, columns in segments:
        start = int(columns[0])
        stop = int(columns[-1]) + 1
        out += (stop - start) * int(np.max(counts[start:stop], initial=0))
    return out


def _intervals(segments) -> list[tuple[int, int]]:
    return [
        (int(columns[0]), int(columns[-1]) + 1)
        for _, columns in segments
    ]


def _brute_force_cost(counts: np.ndarray, k: int) -> int:
    n = len(counts)
    best = None
    for splits in itertools.combinations(range(1, n), k - 1):
        starts = (0, *splits)
        stops = (*splits, n)
        cost = sum(
            (stop - start) * int(np.max(counts[start:stop], initial=0))
            for start, stop in zip(starts, stops, strict=True)
        )
        best = cost if best is None else min(best, cost)
    assert best is not None
    return int(best)


class QDLDLSegmentTest(unittest.TestCase):
    def test_strategy_selector_uses_segment_budget(self):
        counts = np.array([5, 1, 4, 1, 3], dtype=np.int64)
        mask = _mask_from_counts(counts.tolist())
        fixed = _make_segments(mask, segment_budget=3, segment_strategy="fixed")
        greedy = _make_segments(mask, segment_budget=3, segment_strategy="greedy")
        optimal = _make_segments(mask, segment_budget=3, segment_strategy="optimal")

        self.assertEqual(_intervals(fixed), _intervals(_make_fixed_segments(mask, 3)))
        self.assertEqual(_intervals(greedy), _intervals(_make_adaptive_segments(mask, 3)))
        self.assertEqual(_intervals(optimal), _intervals(_make_optimal_segments(mask, 3)))
        self.assertEqual(len(fixed), 3)
        self.assertLessEqual(len(greedy), 3)
        self.assertEqual(len(optimal), 3)

    def test_optimal_segments_can_cross_zero_saving_splits(self):
        counts = np.array([100, 1, 1, 100], dtype=np.int64)
        mask = _mask_from_counts(counts.tolist())
        greedy = _make_adaptive_segments(mask, 3)
        optimal = _make_optimal_segments(mask, 3)
        self.assertEqual(len(optimal), 3)
        self.assertEqual(_cost(counts, greedy), 400)
        self.assertEqual(_cost(counts, optimal), 202)

    def test_optimal_segments_match_bruteforce_for_small_inputs(self):
        rng = np.random.default_rng(0)
        for n in range(1, 9):
            for _ in range(20):
                counts = rng.integers(0, 8, size=n, dtype=np.int64)
                mask = _mask_from_counts(counts.tolist())
                for k in range(1, n + 1):
                    optimal = _make_optimal_segments(mask, k)
                    self.assertEqual(len(optimal), k)
                    self.assertEqual(_cost(counts, optimal), _brute_force_cost(counts, k))


if __name__ == "__main__":
    unittest.main()
