"""Chunk selection and independent pose histories without optional GMR assets."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from examples.gmr_retargeting.chunks import chunk_indices, rollout_chunks
from examples.gmr_retargeting.prepare_pool import select_chunks


class ChunkTests(unittest.TestCase):
    def test_pool_selection_is_balanced_and_never_repeats_or_overlaps(self):
        clips = [
            {"subject": subject, "member": f"{subject}/{i}", "frames": length}
            for subject, i, length in (("a", 0, 23), ("a", 1, 16), ("b", 0, 8), ("c", 0, 13))
        ]
        selected = select_chunks([{"clips": clips}], 4, 8, 42)
        self.assertEqual(selected, select_chunks([{"clips": clips}], 4, 8, 42))
        self.assertEqual(len({c["subject"] for c in selected[:3]}), 3)
        windows = {(c["member"], c["first_frame"]) for c in selected}
        self.assertEqual(len(windows), 8)
        for c in selected:
            source = next(x for x in clips if x["member"] == c["member"])
            self.assertEqual(c["first_frame"] % 4, 0)
            self.assertLessEqual(c["first_frame"] + 4, source["frames"])

    def test_chunks_never_cross_clip_boundaries(self):
        np.testing.assert_array_equal(
            chunk_indices(15, [0, 7], 4), [[0, 1, 2, 3], [7, 8, 9, 10], [11, 12, 13, 14]]
        )

    def test_schedule_and_own_previous_pose(self):
        bank = jnp.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
        cold = lambda q, target: q + 10 * target
        warm = lambda q, target: q + target
        identity = lambda value: value
        result, finite, _ = jax.jit(
            lambda: rollout_chunks(
                cold, warm, identity, jnp.array([0.0]), bank, jnp.array([1, 0, 1]), collect=True
            )
        )()
        np.testing.assert_array_equal(
            result[..., 0], [[40.0, 45.0, 51.0], [10.0, 12.0, 15.0], [40.0, 45.0, 51.0]]
        )
        self.assertTrue(finite)
        final, _, _ = rollout_chunks(
            cold, warm, identity, jnp.array([0.0]), bank, jnp.array([1, 0, 1]), collect=False
        )
        np.testing.assert_array_equal(final[:, 0], [51.0, 15.0, 51.0])
