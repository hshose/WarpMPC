"""Equivalent QP coordinates, padding, and warm starts for the Turbo baseline."""
from __future__ import annotations

import unittest
from dataclasses import replace

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

try:
    from benchmarks.nonlinear_mpc.turbompc_adapter import (
        DenseStageTurbompcProblem, DenseStageQPScaling, ScaledTurbompcADMM,
        make_problem_params, make_solver, make_solver_params, qp_residuals,
    )
    from turbompc.solvers.admm.admm import _apply_C, _apply_Ct, _apply_G, _apply_Gt, _apply_P
except ModuleNotFoundError as exc:
    if not (exc.name or '').startswith('turbompc'):
        raise
    raise unittest.SkipTest('TurboMPC is optional; install it or clone resources/turbompc')

from warpmpc.jax_osqp import OSQPSettings
from warpmpc.jax_sqp import CasadiStageFunction, SparseMPCProblem, build_sparse_mpc_plan


def fixture():
    stages = []
    for i, width in enumerate((3, 2, 1)):
        z = ca.SX.sym(f'z{i}', width)
        p = ca.SX.sym(f'p{i}', 1)
        cost = 0.5 * (100 * (z[0] - p[0]) ** 2 + ca.sumsqr(z[1:]))
        g, lower, upper = [], [], []
        if i == 0:
            g.append(z[0] - p[0]); lower.append(0); upper.append(0)
        if i < 2:
            zn = ca.SX.sym(f'zn{i}', width - 1)
            cost += 0.3 * (zn[0] - z[0]) ** 2
            g.append(10 * (zn[0] - z[0]) - z[1]); lower.append(0); upper.append(0)
            g.append(z[-1]); lower.append(-2); upper.append(2)
            inputs = [z, zn, p]
        else:
            g.append(z[0]); lower.append(-3); upper.append(3)
            inputs = [z, p]
        fn = ca.Function(f'scaling_stage{i}', inputs,
                         [cost, ca.vertcat(*g), ca.vertcat(*lower), ca.vertcat(*upper)])
        stages.append(CasadiStageFunction.from_function(fn, has_next=i < 2))
    sparse = SparseMPCProblem(tuple(stages))
    problem = DenseStageTurbompcProblem(sparse, state_dim=1, control_dim=2)
    base = np.array([.2, -.1, .3, .4, .1, .5])
    params = np.array([.2, -.3, .5])
    plan = build_sparse_mpc_plan(sparse, osqp_settings=OSQPSettings(scaling=2),
                                representative_z=base, representative_params=params)
    settings = make_solver_params(sqp_iterations=1, admm_max_iter=100,
                                  rho=.1, sigma=1e-6, alpha=1.4,
                                  eps_abs=1e-8, eps_rel=1e-8, line_search_step_min=.1)
    solver = make_solver(problem, settings, forward_backend='admm_jax_loop_jax_dense',
                         backward_backend='direct_jax_dense')
    states, controls = problem.split_packed_z(base)
    qp = solver._build_qp_data(states, controls, make_problem_params(params, base))
    scaling = DenseStageQPScaling.from_sparse_plan(problem, plan)
    return problem, solver, qp, scaling, plan


class TurboScalingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem, cls.solver, cls.qp, cls.scaling, cls.plan = fixture()

    def test_padding_and_original_sparse_scaling_map(self):
        p, s = self.problem, self.scaling
        np.testing.assert_allclose(p.pack_states_controls(s.variables[:, :1], s.variables[:, 1:]),
                                   self.plan.osqp_plan.scaling_d)
        self.assertEqual(s.variables[1, 2], 1)
        np.testing.assert_array_equal(s.variables[2, 1:], 1)
        for i, rows in enumerate(p._local_rows):
            np.testing.assert_allclose(s.local[i, :len(rows)],
                self.plan.osqp_plan.scaling_e[self.plan.constraint_offsets[i] + rows])
            np.testing.assert_array_equal(s.local[i, len(rows):], 1)

    def test_qp_objective_constraints_and_duals_are_equivalent(self):
        qp, scaling = self.qp, self.scaling
        scaled = scaling.scale_qp(qp)
        rng = np.random.default_rng(42)
        x = jnp.asarray(rng.normal(size=qp.cost.q.shape))
        init = self.solver._admm_solver_fwd.initial_state(qp, states0=x[:, :1], controls0=x[:, 1:])
        state = init._replace(y_f_0=jnp.asarray(rng.normal(size=init.y_f_0.shape)),
                              y_f_dyn=jnp.asarray(rng.normal(size=init.y_f_dyn.shape)),
                              y_g=jnp.asarray(rng.normal(size=init.y_g.shape)))
        transformed = scaling.transform_state(state)
        restored = scaling.transform_state(transformed, inverse=True)
        for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(state)):
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        xhat = transformed.x_blocks
        objective = lambda q, x: jnp.sum(.5 * x * _apply_P(q, x) + q.cost.q * x)
        np.testing.assert_allclose(objective(scaled, xhat), scaling.cost * objective(qp, x), rtol=1e-13)
        erows = np.r_[scaling.initial, scaling.dynamics.ravel()]
        np.testing.assert_allclose(_apply_C(scaled, xhat), erows * _apply_C(qp, x), atol=1e-13)
        np.testing.assert_allclose(_apply_G(scaled, xhat), scaling.local * _apply_G(qp, x), atol=1e-13)
        gradient = lambda q, s: _apply_P(q, s.x_blocks) + q.cost.q + _apply_Ct(q, s.y_f_0, s.y_f_dyn) + _apply_Gt(q, s.y_g)
        np.testing.assert_allclose(gradient(scaled, transformed),
                                  scaling.cost * scaling.variables * gradient(qp, state), atol=1e-13)

    def test_wrapper_preserves_native_admm_across_warm_started_updates(self):
        native = self.solver._admm_solver_fwd
        wrapped = ScaledTurbompcADMM(native, self.scaling)
        physical = native.initial_state(self.qp, rho_bar=.1)
        scaled = self.scaling.transform_state(physical)
        for offset in (0., .1, -.05):
            qp = replace(self.qp, cost=replace(self.qp.cost, q=self.qp.cost.q + offset))
            (states, controls), _, physical = jax.jit(wrapped.solve)(qp, physical, rho_bar=.1, alpha=1.4)
            _, _, scaled = jax.jit(native.solve)(self.scaling.scale_qp(qp), scaled, rho_bar=.1, alpha=1.4)
            expected = self.scaling.transform_state(scaled, inverse=True)
            for actual, reference in zip(jax.tree.leaves(physical), jax.tree.leaves(expected)):
                np.testing.assert_allclose(actual, reference, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(jnp.concatenate([states, controls], axis=-1), physical.x_blocks)
            primal, dual = qp_residuals(qp, physical)
            self.assertTrue(np.isfinite(primal) and np.isfinite(dual))

    def test_split_diagnostic_matches_upstream_sqp_step(self):
        from benchmarks.diagnostics.diagnose_turbompc_humanoid import single_sqp_iteration, split_assembly
        problem, solver, qp, scaling, _ = fixture()
        solver._admm_solver_fwd = ScaledTurbompcADMM(solver._admm_solver_fwd, scaling)
        base = jnp.array([.2, -.1, .3, .4, .1, .5])
        states, controls = problem.split_packed_z(base)
        params = make_problem_params(jnp.array([.2, -.3, .5]), base)
        initial = solver._admm_solver_fwd.initial_state(qp, rho_bar=.1)
        expected = jax.jit(solver._solve_impl)(states, controls, params, initial)
        step = single_sqp_iteration()
        split_assembly(solver)
        actual = step(solver, states, controls, params, initial)
        for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_allclose(value, reference, rtol=1e-9, atol=1e-9)

    def test_gpu_float32_stationarity_with_dense_equalities(self):
        if not any(device.platform == 'gpu' for device in jax.devices()):
            self.skipTest('Requires the cuDSS GPU backend')
        try:
            from turbompc.solvers.admm import admm_cudss_ffi_backend
        except (ImportError, FileNotFoundError, OSError):
            self.skipTest('Requires the optional compiled cuDSS extension')
        from turbompc.solvers.admm.admm import ADMMSolver
        from turbompc.solvers.linear_systems_solvers.backends import AdmmBackend
        from turbompc.solvers.qp_data import QPData, QPCostBlocks, QPEqualityBlocks, QPInequalityBlocks
        from turbompc.solvers.qp_utils import ZShape
        rng = np.random.default_rng(13)
        nodes, nx, nu = 3, 48, 30
        n = nx + nu
        array = lambda value: jnp.asarray(value, dtype=jnp.float32)
        identity = np.broadcast_to(np.eye(n), (nodes, n, n))
        qp = QPData(
            QPCostBlocks(array(identity), array(np.zeros((nodes - 1, n, n))),
                         array(rng.normal(size=(nodes, n)))),
            QPEqualityBlocks(array(.25 * rng.normal(size=(nx, n))),
                             array(.25 * rng.normal(size=(nodes - 1, nx, n))),
                             array(.25 * rng.normal(size=(nodes - 1, nx, n))),
                             array(rng.normal(size=nx)), array(rng.normal(size=(nodes - 1, nx)))),
            QPInequalityBlocks(array(identity), array(np.full((nodes, n), -1e10)),
                               array(np.full((nodes, n), 1e10))),
        )
        native = ADMMSolver(ZShape(nodes - 1, nx, nu), {}, max_iter=200,
                            check_termination_every=201, adapt_rho_every=201,
                            admm_backend=AdmmBackend.FUSED_CUDSS)
        scaling = DenseStageQPScaling(1., np.ones((nodes, n)), np.ones(nx),
                                      np.ones((nodes - 1, nx)), np.ones((nodes, n)))
        solver = ScaledTurbompcADMM(native, scaling)
        # A caller's default precision must not degrade the normal equations.
        with jax.default_matmul_precision('default'):
            state = jax.jit(lambda q: solver.solve(q, rho_bar=.02, alpha=1.4)[2])(qp)
        q, state = jax.device_get((qp, state))
        # Independent float64 stationarity calculation on the original QP.
        dual = state.x_blocks.astype(float) + q.cost.q
        dual[0] += q.eq.A0.astype(float).T @ state.y_f_0
        dual[:-1] += np.einsum('tji,tj->ti', q.eq.A_minus.astype(float), state.y_f_dyn)
        dual[1:] += np.einsum('tji,tj->ti', q.eq.A_plus.astype(float), state.y_f_dyn)
        dual += state.y_g
        self.assertLess(np.max(np.abs(dual)), 1e-3)


if __name__ == '__main__':
    unittest.main()
