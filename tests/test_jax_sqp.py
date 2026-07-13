from __future__ import annotations

import unittest

import casadi as ca
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import scipy.sparse as sp

from benchmarks.problems.crazyflie_sqp import (
    CRAZYFLIE_NU,
    CRAZYFLIE_NX,
    CRAZYFLIE_NZ,
    crazyflie_dt_schedule,
    crazyflie_initial_guess_and_params,
    crazyflie_jax_dynamics,
    make_crazyflie_sqp_problem,
)
from benchmarks.problems.mixed_crazyflie_sqp import (
    MIXED_SIMPLE_NU,
    MIXED_SIMPLE_NX,
    MIXED_SIMPLE_NZ,
    make_mixed_crazyflie_sqp_problem,
    mixed_crazyflie_initial_guess_and_params,
    mixed_crazyflie_stage_dims,
    mixed_crazyflie_z_offsets,
)
from benchmarks.problems.humanoid_mpc import (
    HUMANOID_LOW_LEVEL_TIMESTEP,
    HUMANOID_MPC_PERIOD_S,
    HUMANOID_MPC_PERIOD_STEPS,
    HUMANOID_NQ,
    HUMANOID_REFERENCE_VELOCITY_Y_LIMIT,
    HUMANOID_SIM_TIMESTEP,
    gait_schedule_walking,
    humanoid_dt_schedule,
    humanoid_initial_guess_and_params,
    humanoid_qhome,
    humanoid_z_offsets,
    update_humanoid_params_gait,
    update_humanoid_params_initial_state,
)
from warpmpc.jax_osqp import OSQPSettings
from warpmpc.jax_sqp import (
    CasadiStageFunction,
    FilterLineSearchSettings,
    LINE_SEARCH_COST,
    LINE_SEARCH_MAX_ITER,
    constraint_violation,
    filter_line_search_from_evaluations,
    make_step_lengths,
    SparseMPCProblem,
    build_sparse_mpc_plan,
    compile_sparse_mpc_sqp,
)


def _make_tiny_problem():
    z = ca.SX.sym("z", 2)
    zn = ca.SX.sym("zn", 2)
    p = ca.SX.sym("p", 1)
    cost = 0.5 * ((z[0] - p[0]) ** 2 + 2.0 * z[1] ** 2) + 0.25 * zn[0] ** 2
    g = ca.vertcat(z[0] + zn[0] ** 2, z[1] + zn[1])
    lower = ca.vertcat(-1.0, 0.0)
    upper = ca.vertcat(1.0, 0.0)
    first_fn = ca.Function("tiny_first", [z, zn, p], [cost, g, lower, upper])

    zt = ca.SX.sym("zt", 2)
    pt = ca.SX.sym("pt", 1)
    terminal_cost = 0.5 * ((zt[0] - pt[0]) ** 2 + zt[1] ** 2)
    terminal_g = ca.vertcat(zt[0])
    terminal_l = ca.vertcat(-2.0)
    terminal_u = ca.vertcat(2.0)
    terminal_fn = ca.Function(
        "tiny_terminal", [zt, pt], [terminal_cost, terminal_g, terminal_l, terminal_u]
    )

    first = CasadiStageFunction.from_function(first_fn, has_next=True)
    terminal = CasadiStageFunction.from_function(terminal_fn, has_next=False)
    problem = SparseMPCProblem.from_stage_functions(
        horizon=1,
        first=first,
        intermediate=first,
        terminal=terminal,
    )
    return first, terminal, problem


def _dense_from_csc(pattern: sp.csc_matrix, values: np.ndarray) -> np.ndarray:
    return sp.csc_matrix((values, pattern.indices, pattern.indptr), shape=pattern.shape).toarray()


class JaxSQPTest(unittest.TestCase):
    def test_filter_line_search_selects_first_accepted_candidate(self):
        settings = FilterLineSearchSettings(
            line_search_step_min=0.1,
            line_search_step_decay=0.5,
        )
        steps = make_step_lengths(settings)
        np.testing.assert_allclose(steps, np.array([1.0, 0.5, 0.25, 0.125]))

        result = filter_line_search_from_evaluations(
            settings=settings,
            step_lengths=steps,
            baseline_cost=np.array([10.0]),
            baseline_constraint_violation=np.array([1e-4]),
            armijo_descent_metric=np.array([-2.0]),
            candidate_costs=np.array([[10.5], [9.0], [8.0]]),
            candidate_constraint_violations=np.array([[1e-4], [1e-4], [1e-4]]),
        )

        self.assertAlmostEqual(float(result.step_length[0]), 0.5)
        self.assertEqual(int(result.reason[0]), LINE_SEARCH_COST)
        self.assertTrue(bool(result.accepted[0]))

    def test_filter_line_search_can_match_reference_cost_branch(self):
        settings = FilterLineSearchSettings(
            line_search_step_min=0.5,
            line_search_step_decay=0.5,
            line_search_cost_accept_uses_trial_cost=False,
        )
        steps = make_step_lengths(settings)
        result = filter_line_search_from_evaluations(
            settings=settings,
            step_lengths=steps,
            baseline_cost=np.array([10.0]),
            baseline_constraint_violation=np.array([1e-4]),
            armijo_descent_metric=np.array([-2.0]),
            candidate_costs=np.array([[9.0], [8.0]]),
            candidate_constraint_violations=np.array([[1e-4], [1e-4]]),
        )

        self.assertAlmostEqual(float(result.step_length[0]), 0.5)
        self.assertEqual(int(result.reason[0]), LINE_SEARCH_MAX_ITER)
        self.assertFalse(bool(result.accepted[0]))

    def test_constraint_violation_matches_bound_residual_definition(self):
        lower = np.array([[0.0, -1.0, 2.0]])
        g = np.array([[-1.0, 0.0, 3.0]])
        upper = np.array([[2.0, 1.0, 2.5]])
        violation = constraint_violation(lower, g, upper, scale=0.1)
        self.assertAlmostEqual(float(violation[0]), 0.1 * np.sqrt(1.0**2 + 0.5**2))

    def test_mx_stage_is_expanded_to_sx_and_sparsified(self):
        z = ca.MX.sym("z", 2)
        zn = ca.MX.sym("zn", 2)
        p = ca.MX.sym("p", 1)
        cost = (z[0] - p[0]) ** 2 + 0.0 * zn[1] ** 2
        g = ca.vertcat(z[0], 0.0 * zn[1], zn[0])
        lower = ca.vertcat(-1.0, 0.0, -2.0)
        upper = ca.vertcat(1.0, 0.0, 2.0)
        mx_function = ca.Function("mx_stage", [z, zn, p], [cost, g, lower, upper])

        stage = CasadiStageFunction.from_function(mx_function, has_next=True)

        self.assertIn("SXFunction", str(stage.function))
        self.assertIsInstance(stage.function.sx_in(0), ca.SX)
        self.assertEqual(stage.constraint_dim, 3)
        np.testing.assert_array_equal(stage.jac_rows, np.array([0, 2], dtype=np.int32))
        np.testing.assert_array_equal(stage.jac_cols, np.array([0, 2], dtype=np.int32))

    def test_stage_sparse_values_match_casadi(self):
        first, _, _ = _make_tiny_problem()
        z = np.array([0.2, 0.3])
        zn = np.array([0.4, 0.5])
        p = np.array([1.0])
        casadi_out = [np.asarray(v).reshape(-1) for v in first.sparse_function(z, zn, p)]
        jax_out = [
            np.asarray(v).reshape(-1)
            for v in jax.device_get(first.jax_function(z, zn, p))
        ]
        for actual, expected in zip(jax_out, casadi_out, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_global_qp_assembly_matches_dense_reference(self):
        _, _, problem = _make_tiny_problem()
        settings = OSQPSettings(max_iter=10, scaling=0, check_termination=0)
        plan = build_sparse_mpc_plan(problem, osqp_settings=settings)
        compiled = compile_sparse_mpc_sqp(
            problem,
            plan,
            dtype=np.float64,
            osqp_settings=settings,
            transpose_work=False,
            segmented=False,
        )
        z_value = np.array([[0.2, 0.3, 0.4, 0.5]])
        params = np.array([[1.0, 0.5]])
        linearization = jax.device_get(compiled.build_qp(z_value, params))

        z = ca.SX.sym("z", 4)
        p = ca.SX.sym("p", 2)
        z0 = z[:2]
        z1 = z[2:]
        p0 = p[:1]
        p1 = p[1:]
        first_cost = 0.5 * ((z0[0] - p0[0]) ** 2 + 2.0 * z0[1] ** 2) + 0.25 * z1[0] ** 2
        terminal_cost = 0.5 * ((z1[0] - p1[0]) ** 2 + z1[1] ** 2)
        cost = first_cost + terminal_cost
        g = ca.vertcat(z0[0] + z1[0] ** 2, z0[1] + z1[1], z1[0])
        lower = ca.vertcat(-1.0, 0.0, -2.0)
        upper = ca.vertcat(1.0, 0.0, 2.0)
        ref = ca.Function(
            "ref",
            [z, p],
            [
                ca.hessian(cost, z)[0],
                ca.gradient(cost, z),
                ca.jacobian(g, z),
                lower - g,
                upper - g,
            ],
        )
        hess, grad, jac, lower_qp, upper_qp = [
            np.asarray(v, dtype=float) for v in ref(z_value[0], params[0])
        ]
        p_dense = _dense_from_csc(plan.p_pattern, np.asarray(linearization.p_values[0]))
        a_dense = _dense_from_csc(plan.a_pattern, np.asarray(linearization.a_values[0]))
        np.testing.assert_allclose(p_dense, np.triu(hess), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(a_dense, jac, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(linearization.q[0], grad.reshape(-1), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(linearization.l[0], lower_qp.reshape(-1), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(linearization.u[0], upper_qp.reshape(-1), rtol=1e-10, atol=1e-10)

    def test_sparse_plan_scaling_can_use_representative_linearization(self):
        _, _, problem = _make_tiny_problem()
        settings = OSQPSettings(max_iter=10, scaling=1, check_termination=0)
        z_value = np.array([0.2, 0.3, 0.4, 0.5])
        params = np.array([1.0, 0.5])

        pattern_plan = build_sparse_mpc_plan(problem, osqp_settings=settings)
        representative_plan = build_sparse_mpc_plan(
            problem,
            osqp_settings=settings,
            representative_z=z_value,
            representative_params=params,
        )

        np.testing.assert_allclose(pattern_plan.osqp_plan.scaling_d, 1.0)
        self.assertFalse(np.allclose(representative_plan.osqp_plan.scaling_d, 1.0))

    def test_sqp_step_shapes_are_fixed_and_finite(self):
        _, _, problem = _make_tiny_problem()
        settings = OSQPSettings(max_iter=20, scaling=0, check_termination=0)
        plan = build_sparse_mpc_plan(problem, osqp_settings=settings)
        compiled = compile_sparse_mpc_sqp(
            problem,
            plan,
            dtype=np.float64,
            osqp_settings=settings,
            transpose_work=True,
            segmented=True,
            segment_budget=2,
            segment_strategy="optimal",
        )
        z_value = np.array([[0.2, 0.3, 0.4, 0.5], [0.0, -0.2, 0.1, 0.0]])
        params = np.array([[1.0, 0.5], [0.2, -0.1]])
        state = compiled.init_state(z_value.shape[0])
        result, next_state = compiled.fixed_step(z_value, params, beta=0.25, state=state)
        result = jax.device_get(result)
        next_state = jax.device_get(next_state)
        self.assertEqual(result.z_next.shape, z_value.shape)
        self.assertEqual(result.solve.direction.shape, z_value.shape)
        self.assertEqual(next_state.x.shape, z_value.shape)
        self.assertEqual(next_state.z.shape, (z_value.shape[0], plan.n_constraints))
        self.assertEqual(next_state.y.shape, (z_value.shape[0], plan.n_constraints))
        self.assertTrue(np.all(np.isfinite(result.z_next)))

    def test_sqp_step_derivatives_match_finite_differences(self):
        _, _, problem = _make_tiny_problem()
        settings = OSQPSettings(
            max_iter=80,
            scaling=0,
            check_termination=0,
            warm_starting=True,
        )
        plan = build_sparse_mpc_plan(
            problem,
            osqp_settings=settings,
            derivatives=True,
        )
        compiled = compile_sparse_mpc_sqp(
            problem,
            plan,
            dtype=np.float64,
            osqp_settings=settings,
            derivatives=True,
            derivative_refinement_iters=2,
            transpose_work=True,
            segmented=True,
            segment_budget=2,
            segment_strategy="optimal",
        )
        self.assertIsNotNone(compiled.osqp.adjoint_derivative_compute)

        z_value = jax.numpy.array([[0.12, -0.08, 0.05, 0.03]], dtype=jax.numpy.float64)
        params = jax.numpy.array([[0.7, -0.4]], dtype=jax.numpy.float64)
        weights = jax.numpy.array([[1.0, -0.5, 0.25, 0.75]], dtype=jax.numpy.float64)
        state = compiled.init_state(1)

        def loss(params_in):
            result, _ = compiled.fixed_step(z_value, params_in, beta=0.35, state=state)
            return (
                jax.numpy.sum(result.z_next * weights)
                + 0.1 * jax.numpy.sum(result.solve.direction**2)
            )

        grad = jax.device_get(jax.grad(loss)(params))
        finite_diff = np.zeros_like(np.asarray(params))
        params_np = np.asarray(params)
        eps = 1e-5
        for index in np.ndindex(params_np.shape):
            delta = np.zeros_like(params_np)
            delta[index] = eps
            plus = float(loss(jax.numpy.asarray(params_np + delta)))
            minus = float(loss(jax.numpy.asarray(params_np - delta)))
            finite_diff[index] = (plus - minus) / (2.0 * eps)

        self.assertEqual(grad.shape, params.shape)
        np.testing.assert_allclose(grad, finite_diff, rtol=1e-5, atol=1e-7)

    def test_sqp_line_search_step_shapes_are_fixed_and_finite(self):
        _, _, problem = _make_tiny_problem()
        settings = OSQPSettings(max_iter=20, scaling=0, check_termination=0)
        plan = build_sparse_mpc_plan(problem, osqp_settings=settings)
        compiled = compile_sparse_mpc_sqp(
            problem,
            plan,
            dtype=np.float64,
            osqp_settings=settings,
            line_search_settings=FilterLineSearchSettings(line_search_step_min=0.25),
        )
        z_value = np.array([[0.2, 0.3, 0.4, 0.5], [0.0, -0.2, 0.1, 0.0]])
        params = np.array([[1.0, 0.5], [0.2, -0.1]])
        state = compiled.init_state(z_value.shape[0])
        result, next_state = compiled.step(z_value, params, state=state)
        result = jax.device_get(result)
        next_state = jax.device_get(next_state)
        self.assertEqual(result.z_next.shape, z_value.shape)
        self.assertEqual(result.solve.direction.shape, z_value.shape)
        self.assertEqual(result.line_search.step_length.shape, (2,))
        self.assertEqual(next_state.x.shape, z_value.shape)
        self.assertEqual(next_state.z.shape, (z_value.shape[0], plan.n_constraints))
        self.assertEqual(next_state.y.shape, (z_value.shape[0], plan.n_constraints))
        self.assertTrue(np.all(np.isfinite(result.z_next)))
        self.assertTrue(np.all(result.is_finite))

    def test_grouped_repeated_stage_evaluation_matches_ungrouped(self):
        first, terminal, _ = _make_tiny_problem()
        problem = SparseMPCProblem.from_stage_functions(
            horizon=3,
            first=first,
            intermediate=[first, first],
            terminal=terminal,
        )
        settings = OSQPSettings(max_iter=10, scaling=0, check_termination=0)
        plan = build_sparse_mpc_plan(problem, osqp_settings=settings)
        grouped = compile_sparse_mpc_sqp(
            problem,
            plan,
            dtype=np.float64,
            osqp_settings=settings,
            group_repeated_stages=True,
        )
        ungrouped = compile_sparse_mpc_sqp(
            problem,
            plan,
            dtype=np.float64,
            osqp_settings=settings,
            group_repeated_stages=False,
        )
        z_value = np.array([[0.2, 0.3, 0.4, 0.5, -0.1, 0.2, 0.0, 0.1]])
        params = np.array([[1.0, 0.5, -0.2, 0.1]])
        grouped_lin = jax.device_get(grouped.build_qp(z_value, params))
        ungrouped_lin = jax.device_get(ungrouped.build_qp(z_value, params))

        for left, right in zip(grouped_lin, ungrouped_lin, strict=True):
            np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-10)

    def test_humanoid_gait_schedule_uses_cumulative_dt(self):
        dt = np.array([0.1, 0.2, 0.4, 0.8])
        phi_vel = 1.25
        gait, _ = gait_schedule_walking(0.1, phi_vel, dt, 4)
        offsets = np.array([0.0, 0.5, 0.0, 0.5])
        phase_2 = (0.1 + (dt[0] + dt[1]) * phi_vel + offsets) % 1.0
        np.testing.assert_array_equal(gait[:, 2], (phase_2 <= 0.7).astype(float))

    def test_humanoid_debug_timing_matches_reference_simulation(self):
        self.assertAlmostEqual(HUMANOID_SIM_TIMESTEP, 0.0005)
        self.assertAlmostEqual(HUMANOID_LOW_LEVEL_TIMESTEP, 0.001)
        self.assertEqual(HUMANOID_MPC_PERIOD_STEPS, 20)
        self.assertAlmostEqual(
            HUMANOID_MPC_PERIOD_STEPS * HUMANOID_SIM_TIMESTEP,
            HUMANOID_MPC_PERIOD_S,
        )
        self.assertAlmostEqual(HUMANOID_REFERENCE_VELOCITY_Y_LIMIT, 0.2)

    def test_humanoid_parameter_packing_updates_initial_state_and_gait(self):
        q0 = humanoid_qhome()[None, :]
        z, params = humanoid_initial_guess_and_params(q0, n_nodes=4, dtype=np.float64)
        self.assertEqual(z.shape, (1, int(humanoid_z_offsets(4)[-1])))
        self.assertEqual(params.shape[0], 1)
        q_new = q0.copy()
        q_new[:, 0] = 0.12
        updated = update_humanoid_params_initial_state(params, q_new, np.zeros((1, HUMANOID_NQ)))
        np.testing.assert_allclose(updated[:, :HUMANOID_NQ], q_new)

        gait_updated = update_humanoid_params_gait(
            params,
            phi=0.3,
            n_nodes=4,
            phi_vel=1.0 / 0.6,
        )
        self.assertFalse(np.allclose(gait_updated, params))

    def test_crazyflie_sqp_problem_shapes_and_hover_dynamics(self):
        dt = crazyflie_dt_schedule(25)
        self.assertEqual(dt.shape, (25,))
        self.assertAlmostEqual(float(dt[0]), 0.01)
        self.assertAlmostEqual(float(dt.sum()), 0.47727098817987723)

        x0 = np.zeros((3, CRAZYFLIE_NX), dtype=np.float32)
        x0[:, :3] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.2, 0.3, -0.4],
            ],
            dtype=np.float32,
        )
        z, params = crazyflie_initial_guess_and_params(x0, n_steps=2, dtype=np.float32)
        self.assertEqual(z.shape, (3, 3 * CRAZYFLIE_NZ))
        self.assertEqual(params.shape, (3, 2 * CRAZYFLIE_NX + 1 + CRAZYFLIE_NX + 1 + CRAZYFLIE_NX))

        problem = make_crazyflie_sqp_problem(2)
        plan = build_sparse_mpc_plan(
            problem,
            osqp_settings=OSQPSettings(max_iter=5, scaling=0, check_termination=0),
        )
        self.assertEqual(plan.n_variables, 3 * CRAZYFLIE_NZ)
        self.assertEqual(
            plan.n_constraints,
            (2 * CRAZYFLIE_NX + CRAZYFLIE_NU)
            + (CRAZYFLIE_NX + CRAZYFLIE_NU)
            + CRAZYFLIE_NU,
        )

        hover_dx = np.asarray(
            jax.device_get(
                crazyflie_jax_dynamics(
                    np.zeros((1, CRAZYFLIE_NX), dtype=np.float32),
                    np.zeros((1, CRAZYFLIE_NU), dtype=np.float32),
                )
            )
        )
        np.testing.assert_allclose(hover_dx, 0.0, atol=1e-6)

    def test_mixed_crazyflie_stage_dimensions_and_plan_shapes(self):
        full_nodes = 3
        simple_nodes = 2
        x0 = np.zeros((2, CRAZYFLIE_NX), dtype=np.float32)
        x0[:, :3] = np.array([[0.4, -0.2, 0.3], [-0.1, 0.2, -0.3]], dtype=np.float32)
        z, params = mixed_crazyflie_initial_guess_and_params(
            x0,
            full_model_nodes=full_nodes,
            simple_model_nodes=simple_nodes,
            dtype=np.float32,
        )
        dims = mixed_crazyflie_stage_dims(
            full_model_nodes=full_nodes,
            simple_model_nodes=simple_nodes,
        )
        offsets = mixed_crazyflie_z_offsets(
            full_model_nodes=full_nodes,
            simple_model_nodes=simple_nodes,
        )
        self.assertEqual(dims, (CRAZYFLIE_NZ, CRAZYFLIE_NZ, CRAZYFLIE_NZ, 9, 9))
        self.assertEqual(z.shape, (2, int(offsets[-1])))
        self.assertEqual(z.shape[1], full_nodes * CRAZYFLIE_NZ + simple_nodes * MIXED_SIMPLE_NZ)
        self.assertEqual(params.shape, (2, 64))

        problem = make_mixed_crazyflie_sqp_problem(
            full_model_nodes=full_nodes,
            simple_model_nodes=simple_nodes,
        )
        plan = build_sparse_mpc_plan(
            problem,
            osqp_settings=OSQPSettings(max_iter=5, scaling=0, check_termination=0),
        )
        self.assertEqual(plan.n_variables, z.shape[1])
        self.assertEqual(plan.n_constraints, z.shape[1])
        self.assertEqual(plan.stage_var_dims, dims)
        self.assertEqual(problem.stages[2].z_dim, CRAZYFLIE_NZ)
        self.assertEqual(problem.stages[2].next_z_dim, MIXED_SIMPLE_NZ)
        self.assertEqual(problem.stages[-1].z_dim, MIXED_SIMPLE_NZ)
        self.assertEqual(problem.stages[-1].constraint_dim, MIXED_SIMPLE_NU)
        self.assertEqual(problem.stages[-1].param_dim, MIXED_SIMPLE_NX)


if __name__ == "__main__":
    unittest.main()
