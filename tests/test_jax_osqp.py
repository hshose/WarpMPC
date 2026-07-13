from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from warpmpc.jax_osqp import OSQPSettings, build_osqp_plan, compile_osqp


class JaxOSQPScalingTest(unittest.TestCase):
    def test_vector_rho_matches_osqp_fixed_iterations(self):
        try:
            import osqp
        except ModuleNotFoundError:
            self.skipTest("python osqp bindings are not installed")

        p_matrix = sp.csc_matrix([[4.0, 1.0], [1.0, 2.0]])
        q = np.array([-1.0, -0.5])
        a_matrix = sp.csc_matrix(
            [
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
        lower = np.array([0.2, -0.1, -1e30, -1e30])
        upper = np.array([0.2, 1e30, 0.4, 1e30])
        settings = OSQPSettings(
            rho=0.1,
            sigma=1e-6,
            alpha=1.6,
            max_iter=25,
            scaling=0,
            adaptive_rho=False,
            rho_is_vec=True,
            check_termination=0,
            warm_starting=False,
            polishing=False,
        )

        plan = build_osqp_plan(p_matrix, a_matrix, lower, upper, settings, scaling_q=q)
        np.testing.assert_allclose(plan.rho_vec, np.array([100.0, 0.1, 0.1, 1e-6]))
        np.testing.assert_array_equal(plan.constr_type, np.array([1, 0, 0, -1]))

        jax_solver = compile_osqp(plan, dtype=np.float64)
        jax_result = jax_solver.solve(
            jnp.asarray(plan.p_upper.data[None, :]),
            jnp.asarray(plan.a_matrix.data[None, :]),
            jnp.asarray(q[None, :]),
            jnp.asarray(lower[None, :]),
            jnp.asarray(upper[None, :]),
        )
        x_jax, _z_jax, y_jax, prim_jax, dual_jax, obj_jax = [
            np.asarray(value)[0] for value in jax.device_get(jax_result)
        ]

        osqp_solver = (
            osqp.OSQP(algebra="builtin")
            if hasattr(osqp, "algebras_available")
            else osqp.OSQP()
        )
        osqp_solver.setup(
            P=p_matrix,
            q=q,
            A=a_matrix,
            l=lower,
            u=upper,
            rho=settings.rho,
            sigma=settings.sigma,
            alpha=settings.alpha,
            max_iter=settings.max_iter,
            scaling=settings.scaling,
            adaptive_rho=settings.adaptive_rho,
            rho_is_vec=settings.rho_is_vec,
            check_termination=settings.check_termination,
            warm_starting=settings.warm_starting,
            polishing=settings.polishing,
            eps_abs=settings.eps_abs,
            eps_rel=settings.eps_rel,
            verbose=False,
        )
        osqp_result = osqp_solver.solve()

        np.testing.assert_allclose(x_jax, osqp_result.x, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(y_jax, osqp_result.y, rtol=1e-11, atol=1e-11)
        self.assertAlmostEqual(float(prim_jax), float(osqp_result.info.prim_res), places=12)
        self.assertAlmostEqual(float(dual_jax), float(osqp_result.info.dual_res), places=12)
        self.assertAlmostEqual(float(obj_jax), float(osqp_result.info.obj_val), places=12)

    def test_scaling_solves_ill_conditioned_problem_in_original_units(self):
        p_matrix = sp.csc_matrix([[1e-4, 0.0], [0.0, 1e4]])
        a_matrix = sp.eye(2, format="csc")
        l = np.array([0.0, 0.0])
        u = np.array([10.0, 10.0])
        q = np.array([-1.0, -1.0])
        settings = OSQPSettings(max_iter=200, scaling=10, check_termination=0)

        plan = build_osqp_plan(p_matrix, a_matrix, l, u, settings, scaling_q=q)
        self.assertFalse(np.allclose(plan.p_scale, 1.0))
        self.assertFalse(np.allclose(plan.a_scale, 1.0))

        solver = compile_osqp(plan, dtype=np.float64)
        result = solver.solve(
            jnp.asarray(plan.p_upper.data[None, :]),
            jnp.asarray(plan.a_matrix.data[None, :]),
            jnp.asarray(q[None, :]),
            jnp.asarray(l[None, :]),
            jnp.asarray(u[None, :]),
        )
        x, z, y, prim_res, dual_res, obj_val = jax.device_get(result)

        expected_x = np.array([[10.0, 1e-4]])
        np.testing.assert_allclose(x, expected_x, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(z, expected_x, rtol=1e-6, atol=1e-6)
        self.assertLess(float(prim_res[0]), 1e-8)
        self.assertLess(float(dual_res[0]), 1e-8)
        self.assertGreater(float(y[0, 0]), 0.0)
        np.testing.assert_allclose(obj_val, np.array([-9.99505]), rtol=1e-8, atol=1e-8)

    def test_scaled_factor_can_be_reused_with_batched_vectors(self):
        p_matrix = sp.csc_matrix([[4.0, 1.0], [1.0, 2.0]])
        a_matrix = sp.eye(2, format="csc")
        l = np.array([0.0, 0.0])
        u = np.array([1.0, 1.0])
        q = np.array([[-1.0, -1.0], [-2.0, -0.5]])
        settings = OSQPSettings(max_iter=80, scaling=5, check_termination=0)

        plan = build_osqp_plan(p_matrix, a_matrix, l, u, settings, scaling_q=q[0])
        solver = compile_osqp(plan, dtype=np.float64)
        p_values = jnp.asarray(plan.p_upper.data[None, :])
        a_values = jnp.asarray(plan.a_matrix.data[None, :])
        q_values = jnp.asarray(q)
        l_values = jnp.asarray(np.broadcast_to(l, q.shape))
        u_values = jnp.asarray(np.broadcast_to(u, q.shape))

        lx, dinv = solver.factor(p_values, a_values)
        from_factor = solver.solve_with_factor(
            lx, dinv, p_values, a_values, q_values, l_values, u_values
        )
        direct = solver.solve(p_values, a_values, q_values, l_values, u_values)

        for actual, expected in zip(
            jax.device_get(from_factor), jax.device_get(direct), strict=True
        ):
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
