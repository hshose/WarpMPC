"""The committed fixes must load without changing a user's MPAX checkout."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from warpmpc.jax_osqp.mpax_support import PATCH, load_mpax, patched_package


class MPAXLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            load_mpax()
        except ImportError as exc:
            if "requires resources/MPAX or an installed MPAX package" in str(exc):
                raise unittest.SkipTest(str(exc)) from exc
            raise

    def test_pristine_source_is_patched_atomically_without_mutation(self):
        _, solver = load_mpax()
        import inspect
        source = Path(inspect.getfile(solver)).parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original/mpax"
            shutil.copytree(source, original, ignore=shutil.ignore_patterns("__pycache__"))
            subprocess.run(["git", "apply", "--reverse", str(PATCH)],
                cwd=original.parent, check=True, capture_output=True)
            before = {p.name: p.read_bytes() for p in original.glob("*.py")}
            def prepare(_):
                return patched_package(original, cache_root=root / "cache")
            with ThreadPoolExecutor(max_workers=2) as pool:
                paths = list(pool.map(prepare, range(2)))
            self.assertEqual(paths[0], paths[1])
            self.assertEqual(before, {p.name: p.read_bytes() for p in original.glob("*.py")})
            for name in ("rapdhg.py", "restart.py"):
                self.assertEqual((paths[0] / name).read_bytes(), (source / name).read_bytes())

    def test_solver_factories_share_selected_numerical_defaults(self):
        from warpmpc.jax_sqp import MPAXSettings
        from benchmarks.mpax_linear_mpc_utils import make_solver
        settings, linear = MPAXSettings(), make_solver()
        for name in ("iteration_limit", "eps_abs", "eps_rel", "primal_importance",
                     "scale_invariant_initial_primal_weight", "termination_evaluation_frequency",
                     "l_inf_ruiz_iterations", "pock_chambolle_alpha"):
            self.assertEqual(getattr(settings, name), getattr(linear, name), name)
        self.assertEqual(settings.iteration_limit, 500)
        self.assertTrue(settings.warm_start)
        # Independent linear QPs have no previous solution state.
        self.assertFalse(linear.warm_start)


if __name__ == "__main__":
    unittest.main()
