"""Independent serial oracle and explicit absence of frame barriers."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from examples.gmr_retargeting.async_chunks import rollout_async


class AsyncChunkTests(unittest.TestCase):
    def test_uneven_work_advances_frames_independently(self):
        bank = np.array(
            [
                [[[0.0], [0.0]]] * 4,
                [[[5.0], [-2.0]], [[3.0], [1.0]], [[0.0], [2.0]], [[-1.0], [0.0]]],
            ],
            dtype=np.float32,
        )
        lanes = np.array([1, 0, 1], dtype=np.int32)
        expected, counts = [], []
        for lane in lanes:
            q, work, poses = np.float32(0), 0, []
            for frame, targets in enumerate(bank[lane]):
                for stage in range(2):
                    target, factor = targets[stage, 0], np.float32(0.5 if stage == 0 else 0.8)
                    for _ in range(3 if frame == 0 else 11):
                        before = abs(q - target)
                        q = target + factor * (q - target)
                        work += 1
                        if before - abs(q - target) <= 0.001:
                            break
                poses.append([q])
            expected.append(poses)
            counts.append(work)

        def advance(q, target, stage):
            factor = jnp.where(stage == 0, 0.5, 0.8)[:, None]
            candidate = target + factor * (q - target)
            return candidate, jnp.abs(q - target)[:, 0], jnp.abs(candidate - target)[:, 0]

        def run(collect):
            return jax.jit(
                lambda b, l: rollout_async(
                    advance,
                    lambda x: x,
                    jnp.zeros(1),
                    b,
                    l,
                    initial_cap=3,
                    warm_cap=11,
                    collect=collect,
                )
            )(jnp.asarray(bank), jnp.asarray(lanes))

        poses, finite, violation, stats = run(True)
        np.testing.assert_allclose(poses, expected, atol=5e-7, rtol=1e-6)
        np.testing.assert_array_equal(stats["lane_qps"], counts)
        # The easy lane finishes frame 1 before the hard lane finishes frame 0.
        self.assertLess(
            int(stats["frame_completion_rounds"][1, 1]), int(stats["frame_completion_rounds"][0, 0])
        )
        np.testing.assert_array_equal(stats["frame_completion_rounds"][1], [2, 4, 6, 8])
        self.assertEqual(int(stats["rounds"]), max(counts))
        self.assertEqual(int(stats["completed_frames"]), len(lanes) * bank.shape[1])
        self.assertAlmostEqual(
            float(stats["lane_utilization"]), sum(counts) / (len(lanes) * max(counts)), places=6
        )
        self.assertTrue(finite)
        self.assertEqual(float(violation), 0.0)
        final, finite, _, final_stats = run(False)
        np.testing.assert_allclose(final, np.array(expected)[:, -1], atol=5e-7, rtol=1e-6)
        self.assertTrue(finite)
        self.assertNotIn("frame_completion_rounds", final_stats)


if __name__ == "__main__":
    unittest.main()
