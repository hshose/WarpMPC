"""Body-model format checks using a tiny structural fixture, not AMASS data."""

import pathlib
import pickle
import tempfile
import unittest

import numpy as np
from scipy import sparse

from examples.gmr_retargeting.prepare import find_body_model, joints_from_motion


class BodyModelFormatTests(unittest.TestCase):
    def test_shape_and_joint_positions_with_sparse_regressor_in_both_formats(self):
        rest = np.zeros((22, 3))
        rest[:, 0] = np.arange(22)
        shapes = np.zeros((22, 3, 1))
        shapes[:, 0, 0] = np.arange(22) * 0.1
        body = dict(
            v_template=rest,
            shapedirs=shapes,
            J_regressor=sparse.eye(22, format="csr"),
            kintree_table=np.array([[-1] + list(range(21)), list(range(22))]),
        )
        motion = dict(
            betas=np.array([2.0]),
            poses=np.zeros((2, 165)),
            trans=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        )
        expected = np.broadcast_to(rest * 1.2, (2, 22, 3)) + motion["trans"][:, None]
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for suffix in ("npz", "pkl"):
                with self.subTest(format=suffix):
                    folder = root / suffix / "smplx"
                    folder.mkdir(parents=True)
                    path = folder / f"SMPLX_MALE.{suffix}"
                    if suffix == "npz":
                        np.savez(path, **body)
                    else:
                        with path.open("wb") as stream:
                            pickle.dump(body, stream)
                    found = find_body_model(root / suffix, "smplx", "male")
                    self.assertEqual(found, path)
                    positions, rotations, parents = joints_from_motion(motion, found)
                    np.testing.assert_allclose(positions, expected)
                    np.testing.assert_allclose(
                        rotations, np.broadcast_to([1.0, 0.0, 0.0, 0.0], (2, 22, 4))
                    )
                    self.assertEqual(parents, [-1] + list(range(21)))


if __name__ == "__main__":
    unittest.main()
