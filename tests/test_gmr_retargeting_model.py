"""Independent numerical checks against MuJoCo and Mink (no GPU required)."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from examples.gmr_retargeting.prepare import resample
from examples.gmr_retargeting.prepare_pool import fold_height, target_vector

try:
    import mink
    from examples.gmr_retargeting.model import ROOT, RetargetModel, load_gmr
    from examples.gmr_retargeting.validate import (
        check_math,
        synthetic_frames,
        reference,
        matched_chunk_reference,
    )
except ImportError as exc:
    raise unittest.SkipTest("GMR tests require the optional retargeting dependencies") from exc


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "resources/GMR/general_motion_retargeting/motion_retarget.py").is_file():
            raise unittest.SkipTest("Clone GMR into resources/GMR to run model parity checks")
        cls.model = RetargetModel(load_gmr(ROOT / "resources/GMR"))

    def test_mink_kinematics_linearization_and_integration(self):
        check_math(self.model, synthetic_frames(self.model, 8))

    def test_zero_and_near_pi_pose_errors(self):
        model = self.model
        q = model.mj.qpos0.copy()
        model.gmr.configuration.update(q)
        fk = np.asarray(model.fk_function(q)).reshape(-1, 7)
        for angle in (0.0, 1e-7, 0.001, np.pi - 1e-5):
            target = fk.copy()
            for i in range(len(target)):
                rotation = Rotation.from_quat(fk[i, 3:], scalar_first=True) * Rotation.from_rotvec(
                    [angle, 0, 0]
                )
                target[i, 3:] = rotation.as_quat(scalar_first=True)
            error = np.asarray(model.error_function(q, target.ravel())).reshape(-1, 6)
            J = np.asarray(model.jacobian_function(q, target.ravel())).reshape(-1, 6, model.nv)
            for task in model.gmr.tasks1:
                k = model.names.index(task.frame_name)
                task.set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3(target[k, 3:]), target[k, :3])
                )
                np.testing.assert_allclose(
                    error[k], task.compute_error(model.gmr.configuration), atol=2e-7
                )
                np.testing.assert_allclose(
                    J[k], task.compute_jacobian(model.gmr.configuration), atol=2e-7
                )

    def test_gpu_target_formula_matches_gmr(self):
        model = self.model
        frames = synthetic_frames(model, 8)
        expected = []
        for frame in frames:
            model.gmr.update_targets(frame)
            expected.append(model.targets())
        actual = jax.jit(model.preprocess)(jnp.asarray(model.pack_human(frames), dtype=jnp.float32))
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_subject_height_folding_preserves_original_gmr_targets(self):
        common = load_gmr(ROOT / "resources/GMR", 1.66)
        for height in (1.45, 1.78, 1.96):
            original = load_gmr(ROOT / "resources/GMR", height)
            for frame in synthetic_frames(self.model, 4, stress=True):
                folded = {
                    name: (fold_height(pos, height, 1.66), quat.copy())
                    for name, (pos, quat) in frame.items()
                }
                original.update_targets(frame)
                common.update_targets(folded)
                np.testing.assert_allclose(
                    target_vector(original), target_vector(common), atol=1e-14, rtol=0
                )

    def test_adaptive_chunk_cpu_policy_matches_original_gmr(self):
        frames = synthetic_frames(self.model, 8, stress=True)
        starts = {0, 4}
        _, _, original, _ = reference(self.model, frames, starts)
        matched = matched_chunk_reference(self.model, frames, starts, 11, 11, adaptive=True)
        np.testing.assert_allclose(matched, original, atol=1e-12, rtol=0)

    def test_resample_composes_local_orientations(self):
        positions = np.zeros((8, 22, 3))
        angles = np.linspace(0, 0.3, 8)
        local = Rotation.from_rotvec(np.c_[angles, angles * 0, angles * 0]).as_quat(
            scalar_first=True
        )
        quats = np.repeat(local[:, None, :], 22, axis=1)
        parents = [-1] + list(range(21))
        _, global_quats = resample(positions, quats, parents, 120.0, 30.0)
        expected = Rotation.from_rotvec([0.3 * 22, 0, 0])
        actual = Rotation.from_quat(global_quats[-1, -1], scalar_first=True)
        self.assertLess((actual.inv() * expected).magnitude(), 1e-12)


if __name__ == "__main__":
    unittest.main()
