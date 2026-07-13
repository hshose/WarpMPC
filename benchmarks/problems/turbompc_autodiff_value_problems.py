"""Value-only nonlinear MPC problem factories for TurboMPC autodiff baselines."""

from __future__ import annotations

import casadi as ca
import numpy as np

from benchmarks.nonlinear_mpc.turbompc_autodiff_adapter import (
    ValueCasadiStageFunction,
    ValueSparseMPCProblem,
)
from benchmarks.problems import cartpole_physical_quadratic as cartpole
from benchmarks.problems import crazyflie_sqp as crazyflie
from benchmarks.problems import humanoid_mpc as humanoid


def make_cartpole_turbompc_autodiff_problem(
    n_steps: int = cartpole.CARTPOLE_N_STEPS,
    *,
    rail_constraint: bool = False,
    compile_jax: bool = True,
) -> ValueSparseMPCProblem:
    """Create the cartpole problem with value-only stage conversion."""

    if n_steps < 1:
        raise ValueError("cartpole MPC needs at least one shooting interval")

    def make_first() -> ca.Function:
        z = ca.SX.sym("cartpole_phys_quad_ad_z0", cartpole.CARTPOLE_NZ)
        zn = ca.SX.sym("cartpole_phys_quad_ad_z1", cartpole.CARTPOLE_NZ)
        p = ca.SX.sym(
            "cartpole_phys_quad_ad_p0",
            cartpole.CARTPOLE_NX + 1 + cartpole.CARTPOLE_N_COST_PARAMS + 1,
        )
        state, action = cartpole._split_stage(z)
        next_state, _ = cartpole._split_stage(zn)
        x0 = p[: cartpole.CARTPOLE_NX]
        dt = p[cartpole.CARTPOLE_NX]
        cost_params = p[
            cartpole.CARTPOLE_NX + 1 : cartpole.CARTPOLE_NX + 1 + cartpole.CARTPOLE_N_COST_PARAMS
        ]
        tube_s = p[cartpole.CARTPOLE_NX + 1 + cartpole.CARTPOLE_N_COST_PARAMS]
        cost = cartpole._stage_cost_ca(state, action, dt, cost_params)
        action_lower, action_upper = cartpole._tightened_action_bounds_ca(cost_params, tube_s)
        g = ca.vertcat(
            state - x0,
            next_state - cartpole._cartpole_euler_step_ca(state, action, dt),
            action[0],
        )
        lower = ca.vertcat(
            np.zeros(cartpole.CARTPOLE_NX),
            np.zeros(cartpole.CARTPOLE_NX),
            action_lower,
        )
        upper = ca.vertcat(
            np.zeros(cartpole.CARTPOLE_NX),
            np.zeros(cartpole.CARTPOLE_NX),
            action_upper,
        )
        if rail_constraint:
            rail_lower, rail_upper = cartpole._tightened_rail_bounds_ca(cost_params, tube_s)
            g = ca.vertcat(g, state[0])
            lower = ca.vertcat(lower, rail_lower)
            upper = ca.vertcat(upper, rail_upper)
        return ca.Function("cartpole_phys_quad_ad_first", [z, zn, p], [cost, g, lower, upper])

    def make_middle() -> ca.Function:
        z = ca.SX.sym("cartpole_phys_quad_ad_z", cartpole.CARTPOLE_NZ)
        zn = ca.SX.sym("cartpole_phys_quad_ad_zn", cartpole.CARTPOLE_NZ)
        p = ca.SX.sym(
            "cartpole_phys_quad_ad_p",
            1 + cartpole.CARTPOLE_N_COST_PARAMS + 1,
        )
        state, action = cartpole._split_stage(z)
        next_state, _ = cartpole._split_stage(zn)
        dt = p[0]
        cost_params = p[1 : 1 + cartpole.CARTPOLE_N_COST_PARAMS]
        tube_s = p[1 + cartpole.CARTPOLE_N_COST_PARAMS]
        cost = cartpole._stage_cost_ca(state, action, dt, cost_params)
        action_lower, action_upper = cartpole._tightened_action_bounds_ca(cost_params, tube_s)
        g = ca.vertcat(
            next_state - cartpole._cartpole_euler_step_ca(state, action, dt),
            action[0],
        )
        lower = ca.vertcat(np.zeros(cartpole.CARTPOLE_NX), action_lower)
        upper = ca.vertcat(np.zeros(cartpole.CARTPOLE_NX), action_upper)
        if rail_constraint:
            rail_lower, rail_upper = cartpole._tightened_rail_bounds_ca(cost_params, tube_s)
            g = ca.vertcat(g, state[0])
            lower = ca.vertcat(lower, rail_lower)
            upper = ca.vertcat(upper, rail_upper)
        return ca.Function("cartpole_phys_quad_ad_middle", [z, zn, p], [cost, g, lower, upper])

    def make_terminal() -> ca.Function:
        z = ca.SX.sym("cartpole_phys_quad_ad_zN", cartpole.CARTPOLE_NZ)
        p = ca.SX.sym(
            "cartpole_phys_quad_ad_pN",
            1 + cartpole.CARTPOLE_N_COST_PARAMS + 1,
        )
        state, action = cartpole._split_stage(z)
        dt = p[0]
        cost_params = p[1 : 1 + cartpole.CARTPOLE_N_COST_PARAMS]
        tube_s = p[1 + cartpole.CARTPOLE_N_COST_PARAMS]
        cost = cartpole._terminal_cost_ca(state, dt, cost_params)
        g = ca.vertcat(action[0])
        lower = ca.vertcat(0.0)
        upper = ca.vertcat(0.0)
        if rail_constraint:
            rail_lower, rail_upper = cartpole._tightened_rail_bounds_ca(cost_params, tube_s)
            g = ca.vertcat(g, state[0])
            lower = ca.vertcat(lower, rail_lower)
            upper = ca.vertcat(upper, rail_upper)
        return ca.Function("cartpole_phys_quad_ad_terminal", [z, p], [cost, g, lower, upper])

    first = ValueCasadiStageFunction.from_function(make_first(), has_next=True, compile_jax=compile_jax)
    middle = ValueCasadiStageFunction.from_function(make_middle(), has_next=True, compile_jax=compile_jax)
    terminal = ValueCasadiStageFunction.from_function(make_terminal(), has_next=False, compile_jax=compile_jax)
    return ValueSparseMPCProblem.from_stage_functions(
        horizon=n_steps,
        first=first,
        intermediate=middle,
        terminal=terminal,
    )


def make_crazyflie_turbompc_autodiff_problem(
    n_steps: int = crazyflie.CRAZYFLIE_N_STEPS,
    *,
    compile_jax: bool = True,
) -> ValueSparseMPCProblem:
    """Create the Crazyflie problem with value-only stage conversion."""

    def make_first() -> ca.Function:
        z = ca.SX.sym("crazyflie_ad_z0", crazyflie.CRAZYFLIE_NZ)
        zn = ca.SX.sym("crazyflie_ad_z1", crazyflie.CRAZYFLIE_NZ)
        p = ca.SX.sym("crazyflie_ad_p0", 2 * crazyflie.CRAZYFLIE_NX + 1)
        x, u = crazyflie._split_stage(z)
        x_next, _ = crazyflie._split_stage(zn)
        x0 = p[: crazyflie.CRAZYFLIE_NX]
        x_ref = p[crazyflie.CRAZYFLIE_NX : 2 * crazyflie.CRAZYFLIE_NX]
        dt = p[2 * crazyflie.CRAZYFLIE_NX]
        cost = crazyflie._stage_cost(x, u, x_ref, dt)
        g = ca.vertcat(
            x - x0,
            x_next - crazyflie._crazyflie_euler_step(x, u, dt),
            crazyflie._motor_mixing(u),
        )
        lower = ca.vertcat(
            np.zeros(crazyflie.CRAZYFLIE_NX),
            np.zeros(crazyflie.CRAZYFLIE_NX),
            np.zeros(crazyflie.CRAZYFLIE_NU),
        )
        upper = ca.vertcat(
            np.zeros(crazyflie.CRAZYFLIE_NX),
            np.zeros(crazyflie.CRAZYFLIE_NX),
            crazyflie.CRAZYFLIE_THRUST_MAX * np.ones(crazyflie.CRAZYFLIE_NU),
        )
        return ca.Function("crazyflie_ad_first", [z, zn, p], [cost, g, lower, upper])

    def make_middle() -> ca.Function:
        z = ca.SX.sym("crazyflie_ad_z", crazyflie.CRAZYFLIE_NZ)
        zn = ca.SX.sym("crazyflie_ad_zn", crazyflie.CRAZYFLIE_NZ)
        p = ca.SX.sym("crazyflie_ad_p", crazyflie.CRAZYFLIE_NX + 1)
        x, u = crazyflie._split_stage(z)
        x_next, _ = crazyflie._split_stage(zn)
        x_ref = p[: crazyflie.CRAZYFLIE_NX]
        dt = p[crazyflie.CRAZYFLIE_NX]
        cost = crazyflie._stage_cost(x, u, x_ref, dt)
        g = ca.vertcat(
            x_next - crazyflie._crazyflie_euler_step(x, u, dt),
            crazyflie._motor_mixing(u),
        )
        lower = ca.vertcat(np.zeros(crazyflie.CRAZYFLIE_NX), np.zeros(crazyflie.CRAZYFLIE_NU))
        upper = ca.vertcat(
            np.zeros(crazyflie.CRAZYFLIE_NX),
            crazyflie.CRAZYFLIE_THRUST_MAX * np.ones(crazyflie.CRAZYFLIE_NU),
        )
        return ca.Function("crazyflie_ad_middle", [z, zn, p], [cost, g, lower, upper])

    def make_terminal() -> ca.Function:
        z = ca.SX.sym("crazyflie_ad_zN", crazyflie.CRAZYFLIE_NZ)
        p = ca.SX.sym("crazyflie_ad_pN", crazyflie.CRAZYFLIE_NX)
        x, u = crazyflie._split_stage(z)
        cost = crazyflie._diag_quadratic(np.diag(crazyflie.CRAZYFLIE_Q), x - p)
        g = u
        lower = np.zeros(crazyflie.CRAZYFLIE_NU)
        upper = np.zeros(crazyflie.CRAZYFLIE_NU)
        return ca.Function("crazyflie_ad_terminal", [z, p], [cost, g, lower, upper])

    first = ValueCasadiStageFunction.from_function(make_first(), has_next=True, compile_jax=compile_jax)
    middle = ValueCasadiStageFunction.from_function(make_middle(), has_next=True, compile_jax=compile_jax)
    terminal = ValueCasadiStageFunction.from_function(make_terminal(), has_next=False, compile_jax=compile_jax)
    return ValueSparseMPCProblem.from_stage_functions(
        horizon=n_steps,
        first=first,
        intermediate=middle,
        terminal=terminal,
    )


def make_humanoid_turbompc_autodiff_problem(
    n_nodes: int = humanoid.HUMANOID_N_NODES,
    *,
    model_name: str = humanoid.HUMANOID_MODEL_NAME,
    compile_jax: bool = True,
) -> ValueSparseMPCProblem:
    """Create the humanoid problem with value-only stage conversion."""

    if n_nodes < humanoid.HUMANOID_N_TORQUE_STAGES + 1:
        raise ValueError("humanoid MPC requires at least three shooting nodes")

    inverse_dynamics = humanoid._load_humanoid_casadi_function(f"{model_name}_inverse_dynamics")
    contact_pos_jac = humanoid._load_humanoid_casadi_function(f"{model_name}_contact_points_pos_jac")
    signed_distance = humanoid._load_humanoid_casadi_function(f"{model_name}_signed_distance")
    friction_model = humanoid._load_humanoid_casadi_function("friction_model")

    first_param_dim = 231
    running_param_dim = 147
    terminal_param_dim = 105

    def unpack_first(p):
        offset = 0
        q0 = p[offset : offset + humanoid.HUMANOID_NQ]
        offset += humanoid.HUMANOID_NQ
        dq0 = p[offset : offset + humanoid.HUMANOID_NDQ]
        offset += humanoid.HUMANOID_NDQ
        tau0 = p[offset : offset + humanoid.HUMANOID_NTAU]
        offset += humanoid.HUMANOID_NTAU
        gamma = p[offset : offset + 4]
        offset += 4
        foot = p[offset : offset + 4]
        offset += 4
        qref = p[offset : offset + humanoid.HUMANOID_NQ]
        offset += humanoid.HUMANOID_NQ
        dqref = p[offset : offset + humanoid.HUMANOID_NDQ]
        offset += humanoid.HUMANOID_NDQ
        fref = p[offset : offset + humanoid.HUMANOID_NF]
        offset += humanoid.HUMANOID_NF
        dt = p[offset]
        offset += 1
        weights = {
            "Q_q": p[offset : offset + humanoid.HUMANOID_NQ],
            "Q_dq": p[offset + humanoid.HUMANOID_NQ : offset + humanoid.HUMANOID_NQ + humanoid.HUMANOID_NDQ],
            "Q_ddq": p[offset + 48 : offset + 66],
            "Q_F": p[offset + 66 : offset + 78],
            "Q_tau_first": p[offset + 78 : offset + 96],
        }
        return q0, dq0, tau0, gamma, foot, qref, dqref, fref, dt, weights

    def unpack_running(p):
        offset = 0
        gamma = p[offset : offset + 4]
        offset += 4
        foot = p[offset : offset + 4]
        offset += 4
        qref = p[offset : offset + humanoid.HUMANOID_NQ]
        offset += humanoid.HUMANOID_NQ
        dqref = p[offset : offset + humanoid.HUMANOID_NDQ]
        offset += humanoid.HUMANOID_NDQ
        fref = p[offset : offset + humanoid.HUMANOID_NF]
        offset += humanoid.HUMANOID_NF
        dt = p[offset]
        offset += 1
        weights = {
            "Q_q": p[offset : offset + humanoid.HUMANOID_NQ],
            "Q_dq": p[offset + humanoid.HUMANOID_NQ : offset + humanoid.HUMANOID_NQ + humanoid.HUMANOID_NDQ],
            "Q_ddq": p[offset + 48 : offset + 66],
            "Q_F": p[offset + 66 : offset + 78],
        }
        return gamma, foot, qref, dqref, fref, dt, weights

    def unpack_terminal(p):
        offset = 0
        gamma = p[offset : offset + 4]
        offset += 4
        foot = p[offset : offset + 4]
        offset += 4
        qref = p[offset : offset + humanoid.HUMANOID_NQ]
        offset += humanoid.HUMANOID_NQ
        dqref = p[offset : offset + humanoid.HUMANOID_NDQ]
        offset += humanoid.HUMANOID_NDQ
        dt = p[offset]
        offset += 1
        weights = {
            "Q_q": p[offset : offset + humanoid.HUMANOID_NQ],
            "Q_dq": p[offset + humanoid.HUMANOID_NQ : offset + humanoid.HUMANOID_NQ + humanoid.HUMANOID_NDQ],
        }
        return gamma, foot, qref, dqref, dt, weights

    def make_first() -> ca.Function:
        z = ca.SX.sym("humanoid_ad_z0", humanoid.HUMANOID_NZ_TORQUE)
        zn = ca.SX.sym("humanoid_ad_z1", humanoid.HUMANOID_NZ_TORQUE)
        p = ca.SX.sym("humanoid_ad_p0", first_param_dim)
        q, dq, tau, force = humanoid._split_torque_stage(z)
        q_next, dq_next = humanoid._split_q_dq(zn)
        q0, dq0, tau0, gamma, _foot, qref, dqref, fref, dt, weights = unpack_first(p)
        ddq = (dq_next - dq) / dt
        p_contact, j_contact = humanoid._robot_contact_terms(contact_pos_jac, q)
        parts = [
            q - q0,
            dq - dq0,
            humanoid._full_inverse_dynamics_residual(
                inverse_dynamics, friction_model, q, dq, tau, force, ddq, j_contact
            ),
            q_next - humanoid._configuration_integrator(q, dq, ddq, dt),
        ]
        lower = [
            ca.SX.zeros(humanoid.HUMANOID_NQ),
            ca.SX.zeros(humanoid.HUMANOID_NDQ),
            ca.SX.zeros(humanoid.HUMANOID_NQ),
            ca.SX.zeros(humanoid.HUMANOID_NQ),
        ]
        upper = [
            ca.SX.zeros(humanoid.HUMANOID_NQ),
            ca.SX.zeros(humanoid.HUMANOID_NDQ),
            ca.SX.zeros(humanoid.HUMANOID_NQ),
            ca.SX.zeros(humanoid.HUMANOID_NQ),
        ]
        humanoid._append_force_constraints(parts, lower, upper, force=force, gamma=gamma)
        humanoid._append_torque_constraints(parts, lower, upper, tau=tau, dq_joint=dq[6:])
        cost = humanoid._stage_cost(
            q,
            dq,
            force,
            qref,
            dqref,
            fref,
            dt,
            weights,
            tau=tau,
            tau0=tau0,
            ddq=ddq,
        )
        return ca.Function("humanoid_ad_first", [z, zn, p], list(humanoid._pack_stage_output(cost, parts, lower, upper)))

    def make_second(next_z_dim: int) -> ca.Function:
        z = ca.SX.sym("humanoid_ad_z_second", humanoid.HUMANOID_NZ_TORQUE)
        zn = ca.SX.sym("humanoid_ad_zn_second", next_z_dim)
        p = ca.SX.sym("humanoid_ad_p_second", running_param_dim)
        q, dq, tau, force = humanoid._split_torque_stage(z)
        q_next, dq_next = humanoid._split_q_dq(zn)
        gamma, foot, qref, dqref, fref, dt, weights = unpack_running(p)
        ddq = (dq_next - dq) / dt
        p_contact, j_contact = humanoid._robot_contact_terms(contact_pos_jac, q)
        parts = [
            humanoid._full_inverse_dynamics_residual(
                inverse_dynamics, friction_model, q, dq, tau, force, ddq, j_contact
            ),
            q_next - humanoid._configuration_integrator(q, dq, ddq, dt),
        ]
        lower = [ca.SX.zeros(humanoid.HUMANOID_NQ), ca.SX.zeros(humanoid.HUMANOID_NQ)]
        upper = [ca.SX.zeros(humanoid.HUMANOID_NQ), ca.SX.zeros(humanoid.HUMANOID_NQ)]
        humanoid._append_stage_constraints(
            parts,
            lower,
            upper,
            signed_distance=signed_distance,
            q=q,
            dq=dq,
            gamma=gamma,
            foot_height_des=foot,
            p_contact=p_contact,
            j_contact=j_contact,
        )
        humanoid._append_force_constraints(parts, lower, upper, force=force, gamma=gamma)
        humanoid._append_torque_constraints(parts, lower, upper, tau=tau, dq_joint=dq[6:])
        cost = humanoid._stage_cost(q, dq, force, qref, dqref, fref, dt, weights, ddq=ddq)
        return ca.Function(
            f"humanoid_ad_second_next{next_z_dim}",
            [z, zn, p],
            list(humanoid._pack_stage_output(cost, parts, lower, upper)),
        )

    def make_middle(next_z_dim: int) -> ca.Function:
        z = ca.SX.sym("humanoid_ad_z_middle", humanoid.HUMANOID_NZ_FORCE)
        zn = ca.SX.sym("humanoid_ad_zn_middle", next_z_dim)
        p = ca.SX.sym("humanoid_ad_p_middle", running_param_dim)
        q, dq, force = humanoid._split_force_stage(z)
        q_next, dq_next = humanoid._split_q_dq(zn)
        gamma, foot, qref, dqref, fref, dt, weights = unpack_running(p)
        ddq = (dq_next - dq) / dt
        p_contact, j_contact = humanoid._robot_contact_terms(contact_pos_jac, q)
        parts = [
            humanoid._floating_base_dynamics_residual(inverse_dynamics, q, dq, force, ddq, j_contact),
            q_next - humanoid._configuration_integrator(q, dq, ddq, dt),
        ]
        lower = [ca.SX.zeros(humanoid.HUMANOID_N_FB), ca.SX.zeros(humanoid.HUMANOID_NQ)]
        upper = [ca.SX.zeros(humanoid.HUMANOID_N_FB), ca.SX.zeros(humanoid.HUMANOID_NQ)]
        humanoid._append_stage_constraints(
            parts,
            lower,
            upper,
            signed_distance=signed_distance,
            q=q,
            dq=dq,
            gamma=gamma,
            foot_height_des=foot,
            p_contact=p_contact,
            j_contact=j_contact,
        )
        humanoid._append_force_constraints(parts, lower, upper, force=force, gamma=gamma)
        cost = humanoid._stage_cost(q, dq, force, qref, dqref, fref, dt, weights, ddq=ddq)
        return ca.Function(
            f"humanoid_ad_middle_next{next_z_dim}",
            [z, zn, p],
            list(humanoid._pack_stage_output(cost, parts, lower, upper)),
        )

    def make_terminal() -> ca.Function:
        z = ca.SX.sym("humanoid_ad_zN", humanoid.HUMANOID_NZ_TERMINAL)
        p = ca.SX.sym("humanoid_ad_pN", terminal_param_dim)
        q, dq = humanoid._split_q_dq(z)
        gamma, foot, qref, dqref, dt, weights = unpack_terminal(p)
        p_contact, j_contact = humanoid._robot_contact_terms(contact_pos_jac, q)
        parts: list[ca.SX] = []
        lower: list[ca.SX] = []
        upper: list[ca.SX] = []
        humanoid._append_stage_constraints(
            parts,
            lower,
            upper,
            signed_distance=signed_distance,
            q=q,
            dq=dq,
            gamma=gamma,
            foot_height_des=foot,
            p_contact=p_contact,
            j_contact=j_contact,
        )
        cost = dt * (
            humanoid._diag_quadratic(weights["Q_q"], q - qref)
            + humanoid._diag_quadratic(weights["Q_dq"], dq - dqref)
        )
        return ca.Function("humanoid_ad_terminal", [z, p], list(humanoid._pack_stage_output(cost, parts, lower, upper)))

    first = ValueCasadiStageFunction.from_function(make_first(), has_next=True, compile_jax=compile_jax)
    terminal = ValueCasadiStageFunction.from_function(make_terminal(), has_next=False, compile_jax=compile_jax)
    if n_nodes == 3:
        intermediate = [
            ValueCasadiStageFunction.from_function(
                make_second(humanoid.HUMANOID_NZ_TERMINAL),
                has_next=True,
                compile_jax=compile_jax,
            )
        ]
    else:
        second = ValueCasadiStageFunction.from_function(
            make_second(humanoid.HUMANOID_NZ_FORCE),
            has_next=True,
            compile_jax=compile_jax,
        )
        penultimate = ValueCasadiStageFunction.from_function(
            make_middle(humanoid.HUMANOID_NZ_TERMINAL),
            has_next=True,
            compile_jax=compile_jax,
        )
        if n_nodes == 4:
            intermediate = [second, penultimate]
        else:
            middle = ValueCasadiStageFunction.from_function(
                make_middle(humanoid.HUMANOID_NZ_FORCE),
                has_next=True,
                compile_jax=compile_jax,
            )
            intermediate = [second, *([middle] * (n_nodes - 4)), penultimate]
    return ValueSparseMPCProblem.from_stage_functions(
        horizon=n_nodes - 1,
        first=first,
        intermediate=intermediate,
        terminal=terminal,
    )
