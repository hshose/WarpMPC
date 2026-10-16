"""Regression checks for MPAX's QP restarts, stopping, and SQP warm starts.

Requires a compatible MPAX package; the loader applies the bundled correctness patch.
Run: JAX_PLATFORMS=cpu python -m unittest discover -s tests -p test_mpax_regressions.py
"""
from pathlib import Path
import os
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.environ.get("MPAX_SOURCE", str(Path(__file__).resolve().parents[1] / "resources/MPAX")))
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.experimental.sparse import BCOO
try:
    if os.environ.get("MPAX_SOURCE"):
        from mpax import create_qp, raPDHG
    else:
        from warpmpc.jax_osqp.mpax_support import load_mpax
        create_qp, raPDHG = load_mpax()
except ImportError as exc:
    raise unittest.SkipTest("MPAX is an optional benchmark dependency") from exc
from mpax.restart import compute_weight_kkt_residual
from mpax.utils import TerminationStatus
from warpmpc.jax_sqp.sparse_mpc import MPAXSettings, _compile_mpax_qp


def scalar_qp(*, equality=True):
    matrix = BCOO.fromdense(jnp.ones((1, 1)))
    empty = BCOO.fromdense(jnp.zeros((0, 1)))
    return create_qp(
        matrix, jnp.array([-1.0]), matrix if equality else empty,
        jnp.array([2.0]) if equality else jnp.zeros(0),
        empty if equality else matrix,
        jnp.zeros(0) if equality else jnp.array([-1.0]),
        jnp.array([-jnp.inf]), jnp.array([jnp.inf]),
    )


class MPAXRegressions(unittest.TestCase):
    def test_restart_residual_includes_qp_stationarity(self):
        qp = scalar_qp(equality=False)
        def residual(x):
            x = jnp.array([x])
            y = jnp.zeros(1)
            return compute_weight_kkt_residual(qp, x, y, qp.constraint_matrix @ x,
                qp.constraint_matrix_t @ y, qp.objective_matrix @ x, 1.0, jnp.inf)
        # Both points are feasible with zero primal-dual objective gap, but
        # only x=1 is stationary. The original restart metric calls both zero.
        self.assertGreater(float(residual(0.0)), 0.1)
        self.assertAlmostEqual(float(residual(1.0)), 0.0, places=6)

    def test_iteration_budget_and_returned_residuals(self):
        qp = scalar_qp()
        for limit in (0, 1, 9, 25, 101):
            with self.subTest(limit=limit):
                solver = raPDHG(iteration_limit=limit, termination_evaluation_frequency=100,
                    eps_abs=1e-9, eps_rel=1e-9)
                result = jax.jit(solver.optimize)(qp)
                self.assertLessEqual(int(result.iteration_count), limit)
                if int(result.termination_status) == TerminationStatus.ITERATION_LIMIT:
                    self.assertEqual(int(result.iteration_count), limit)
                primal = abs(float(result.primal_solution[0]) - 2.0)
                dual = abs(float(result.primal_solution[0] - 1.0 - result.dual_solution[0]))
                self.assertAlmostEqual(float(result.primal_residual_norm), primal, delta=3e-6)
                self.assertAlmostEqual(float(result.dual_residual_norm), dual, delta=3e-6)

    def test_vmap_preserves_each_stopped_solution(self):
        import dataclasses
        base = scalar_qp()
        solver = raPDHG(iteration_limit=500, termination_evaluation_frequency=25,
            eps_abs=1e-5, eps_rel=1e-5)
        def solve(rhs):
            return solver.optimize(dataclasses.replace(base, right_hand_side=rhs))
        results = jax.jit(jax.vmap(solve))(jnp.array([[0.0], [2.0], [10.0]]))
        x, y = np.asarray(results.primal_solution), np.asarray(results.dual_solution)
        np.testing.assert_allclose(results.primal_residual_norm, np.abs(x[:, 0] - [0, 2, 10]), atol=3e-6)
        np.testing.assert_allclose(results.dual_residual_norm, np.abs(x[:, 0] - 1 - y[:, 0]), atol=3e-6)
        self.assertTrue(np.all(np.asarray(results.iteration_count) <= 500))

    def test_unrolled_and_while_solve_agree_at_partial_block(self):
        qp = scalar_qp()
        outputs = []
        for unroll in (False, True):
            solver = raPDHG(iteration_limit=25, termination_evaluation_frequency=10,
                eps_abs=1e-9, eps_rel=1e-9, unroll=unroll)
            outputs.append(jax.jit(solver.optimize)(qp))
        self.assertEqual(int(outputs[0].iteration_count), 25)
        self.assertEqual(int(outputs[1].iteration_count), 25)
        np.testing.assert_allclose(outputs[0].primal_solution, outputs[1].primal_solution, atol=1e-6)
        np.testing.assert_allclose(outputs[0].dual_solution, outputs[1].dual_solution, atol=1e-6)

    def test_static_blocks_match_bounded_steps(self):
        class BoundedSolver(raPDHG):
            def take_main_steps(self, state, problem):
                return self.take_multiple_steps(state, problem)

        qp = scalar_qp()
        for limit, frequency in ((0, 100), (9, 100), (25, 10), (110, 100), (250, 100)):
            with self.subTest(limit=limit, frequency=frequency):
                kwargs = dict(iteration_limit=limit, termination_evaluation_frequency=frequency,
                    eps_abs=1e-9, eps_rel=1e-9)
                expected = jax.jit(BoundedSolver(**kwargs).optimize)(qp)
                actual = jax.jit(raPDHG(**kwargs).optimize)(qp)
                self.assertEqual(int(actual.iteration_count), int(expected.iteration_count))
                np.testing.assert_allclose(actual.primal_solution, expected.primal_solution, atol=3e-6)
                np.testing.assert_allclose(actual.dual_solution, expected.dual_solution, atol=3e-6)

    def test_adapter_preserves_warm_start_and_dual_signs(self):
        plan = SimpleNamespace(n_variables=2, n_constraints=3,
            representative_l=np.array([0., -2., -1e30]),
            representative_u=np.array([0., 2., 3.]),
            p_pattern=sp.csc_matrix([[2., .5], [0., 1.]]),
            a_pattern=sp.csc_matrix([[1., 1.], [1., 0.], [0., 1.]]))
        p = jnp.asarray(plan.p_pattern.data, dtype=jnp.float32)[None, :]
        a = jnp.asarray(plan.a_pattern.data, dtype=jnp.float32)[None, :]
        q = jnp.array([[.3, -.4]], dtype=jnp.float32)
        lo = jnp.array([[0., -2., -1e30]], dtype=jnp.float32)
        hi = jnp.array([[0., 2., 3.]], dtype=jnp.float32)
        x = jnp.array([[.1, -.2]], dtype=jnp.float32)
        y = jnp.array([[.7, -.6, .8]], dtype=jnp.float32)
        warm = _compile_mpax_qp(plan, dtype=np.dtype("float32"),
            settings=MPAXSettings(iteration_limit=0, warm_start=True))
        result = warm.solve_warm_start(p, a, q, lo, hi, x, jnp.zeros_like(y), y)
        np.testing.assert_allclose(result[0], x, atol=1e-6)
        np.testing.assert_allclose(result[2], y, atol=1e-6)

        self.assertAlmostEqual(float(result[3][0]), 0.1, delta=1e-6)
        # Reconstruct the full symmetric Hessian, including off-diagonals.
        expected_dual = np.max(np.abs(np.array([[2., .5], [.5, 1.]]) @ np.asarray(x[0])
            + np.asarray(q[0]) + plan.a_pattern.T @ np.asarray(y[0])))
        self.assertAlmostEqual(float(result[4][0]), expected_dual, delta=2e-6)
        damped = _compile_mpax_qp(plan, dtype=np.dtype("float32"),
            settings=MPAXSettings(iteration_limit=0, warm_start=True, warm_start_primal_scale=.5))
        result = damped.solve_warm_start(p, a, q, lo, hi, x, jnp.zeros_like(y), y)
        np.testing.assert_allclose(result[0], .5 * x, atol=1e-6)
        np.testing.assert_allclose(result[2], y, atol=1e-6)
        cold = _compile_mpax_qp(plan, dtype=np.dtype("float32"),
            settings=MPAXSettings(iteration_limit=0, warm_start=False))
        result = cold.solve_warm_start(p, a, q, lo, hi, x, jnp.zeros_like(y), y)
        np.testing.assert_array_equal(result[0], np.zeros((1, 2)))
        dual_only = _compile_mpax_qp(plan, dtype=np.dtype("float32"),
            settings=MPAXSettings(iteration_limit=0, warm_start=True, warm_start_primal=False))
        result = dual_only.solve_warm_start(p, a, q, lo, hi, x, jnp.zeros_like(y), y)
        np.testing.assert_array_equal(result[0], np.zeros((1, 2)))
        np.testing.assert_allclose(result[2], y, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
