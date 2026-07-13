"""MPX relaxed-barrier adapter for existing sparse MPC stage functions."""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
MPX_ROOT = ROOT / "resources" / "mpx"
if MPX_ROOT.exists() and str(MPX_ROOT) not in sys.path:
    sys.path.insert(0, str(MPX_ROOT))

from mpx.jax_ocp_solvers.jax_ocp_solvers import optimizers as mpx_optimizers
from mpx.jax_ocp_solvers.jax_ocp_solvers.primal_tvlqr import direct_optimal_control_rollout


MPX_INFTY = 1e30
FINITE_BOUND_THRESHOLD = 1e26


@dataclass(frozen=True)
class MPXBarrierSettings:
    equality_weight: float = 1.0e4
    barrier_alpha: float = 0.1
    barrier_sigma: float = 1.0
    num_alpha: int = 11
    limited_memory: bool = True
    solver_mode: str = "primal_dual"


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 10, 11, 12))
def ilqr_mpc(
    cost,
    dynamics,
    hessian_approx,
    limited_memory,
    reference,
    parameter,
    W,
    x0,
    X_in,
    U_in,
    num_alpha=11,
    goldstein_b1=0.1,
    goldstein_b2=2.0,
):
    """Single MPX direct iLQR step without FDDP defect-closing rollout."""

    _cost = partial(cost, W, reference)
    _hessian_approx = (
        partial(hessian_approx, W, reference) if hessian_approx is not None else None
    )
    _dynamics = partial(dynamics, parameter=parameter)

    defects0 = mpx_optimizers.direct_dynamics_defect_helper(_dynamics, x0, X_in, U_in)
    cost0 = mpx_optimizers.direct_cost_evaluator_helper(_cost, X_in, U_in)
    K, k, _, _, _, _, delta1, delta2 = mpx_optimizers.compute_fddp_search_direction(
        _cost,
        _dynamics,
        _hessian_approx,
        limited_memory,
        x0,
        X_in,
        U_in,
        defects0,
    )

    alpha_values = jnp.exp2(-jnp.arange(num_alpha, dtype=X_in.dtype))

    def evaluate_alpha(alpha):
        X_new, U_new = direct_optimal_control_rollout(
            _dynamics,
            x0,
            X_in,
            U_in,
            K,
            k,
            alpha,
        )
        new_cost = mpx_optimizers.direct_cost_evaluator_helper(_cost, X_new, U_new)
        new_defects = mpx_optimizers.direct_dynamics_defect_helper(_dynamics, x0, X_new, U_new)
        theta_new = jnp.sum(new_defects * new_defects)

        expected_delta = delta1 * alpha + 0.5 * delta2 * alpha * alpha
        actual_delta = new_cost - cost0
        goldstein_bound = jnp.where(
            expected_delta <= 0.0,
            goldstein_b1 * expected_delta,
            goldstein_b2 * expected_delta,
        )
        finite = jnp.all(jnp.isfinite(X_new))
        finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(U_new)))
        finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(new_defects)))
        accepted = jnp.logical_and(finite, actual_delta <= goldstein_bound)
        safe_cost = jnp.where(finite, new_cost, jnp.inf)
        safe_theta = jnp.where(finite, theta_new, jnp.inf)
        return X_new, U_new, new_defects, safe_cost, safe_theta, accepted

    X_candidates, U_candidates, defect_candidates, _costs, _thetas, accepted = jax.vmap(
        evaluate_alpha
    )(alpha_values)
    any_accepted = jnp.any(accepted)
    best_index = jnp.where(any_accepted, jnp.argmax(accepted), 0)
    X_new = X_candidates[best_index]
    U_new = U_candidates[best_index]
    defects_new = defect_candidates[best_index]
    X = jnp.where(any_accepted, X_new, X_in)
    U = jnp.where(any_accepted, U_new, U_in)
    return X, U, defects_new


@dataclass(frozen=True)
class MPXLiftedProblem:
    stage_z_dims: tuple[int, ...]
    stage_next_z_dims: tuple[int, ...]
    stage_param_dims: tuple[int, ...]
    z_offsets: np.ndarray
    param_offsets: np.ndarray
    max_z_dim: int
    max_param_dim: int
    settings: MPXBarrierSettings
    solve: object
    cost: object
    dynamics: object
    violation: object

    @property
    def n_nodes(self) -> int:
        return len(self.stage_z_dims)

    @property
    def horizon(self) -> int:
        return self.n_nodes - 1

    @property
    def n_variables(self) -> int:
        # MPX optimizes X[0:N+1] and U[0:N].  U[k] is a lifted copy of X[k+1].
        return (self.n_nodes + self.horizon) * self.max_z_dim

    @property
    def n_constraints(self) -> int:
        return self.horizon * self.max_z_dim


def relaxed_barrier(margin, alpha, sigma):
    """Relaxed log barrier used by MPX examples for positive margins."""

    margin = jnp.asarray(margin)
    alpha = jnp.asarray(alpha, dtype=margin.dtype)
    sigma = jnp.asarray(sigma, dtype=margin.dtype)
    safe_margin = jnp.clip(margin, 1.0e-10, 1.0e6)
    safe_sigma = jnp.clip(sigma, 1.0e-10, 1.0e6)
    log_barrier = -alpha * jnp.log(safe_margin)
    quadratic = alpha * 0.5 * ((margin - 2.0 * safe_sigma) / safe_sigma) ** 2
    quadratic = quadratic - alpha * 0.5
    relaxed = jnp.where(margin >= safe_sigma, log_barrier, quadratic - alpha * jnp.log(safe_sigma))
    return jnp.clip(relaxed, 0.0, 1.0e8)


def relaxed_barrier_curvature(margin, alpha, sigma):
    """Second derivative of the relaxed log barrier with respect to its margin."""

    margin = jnp.asarray(margin)
    alpha = jnp.asarray(alpha, dtype=margin.dtype)
    sigma = jnp.asarray(sigma, dtype=margin.dtype)
    safe_margin = jnp.clip(margin, 1.0e-6, 1.0e6)
    safe_sigma = jnp.clip(sigma, 1.0e-6, 1.0e6)
    return jnp.where(margin >= safe_sigma, alpha / (safe_margin * safe_margin), alpha / (safe_sigma * safe_sigma))


def _stage_offsets(widths: tuple[int, ...]) -> np.ndarray:
    offsets = np.zeros(len(widths) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(np.asarray(widths, dtype=np.int32))
    return offsets


def pack_lifted_trajectory(
    z: np.ndarray,
    params: np.ndarray,
    lifted: MPXLiftedProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert flat sparse-MPC arrays into MPX X/U/reference arrays."""

    z = np.asarray(z)
    params = np.asarray(params)
    batch = z.shape[0]
    dtype = z.dtype
    x = np.zeros((batch, lifted.n_nodes, lifted.max_z_dim), dtype=dtype)
    u = np.zeros((batch, lifted.horizon, lifted.max_z_dim), dtype=dtype)
    reference = np.zeros((batch, lifted.n_nodes, lifted.max_param_dim), dtype=dtype)

    for stage, width in enumerate(lifted.stage_z_dims):
        start = int(lifted.z_offsets[stage])
        x[:, stage, :width] = z[:, start : start + width]
    for stage in range(lifted.horizon):
        next_width = lifted.stage_z_dims[stage + 1]
        u[:, stage, :next_width] = x[:, stage + 1, :next_width]
    for stage, width in enumerate(lifted.stage_param_dims):
        start = int(lifted.param_offsets[stage])
        reference[:, stage, :width] = params[:, start : start + width]
    return x, u, reference


def update_lifted_reference(
    params: np.ndarray,
    reference: np.ndarray,
    lifted: MPXLiftedProblem,
) -> np.ndarray:
    """Update packed per-stage parameters in a preallocated MPX reference array."""

    out = np.array(reference, copy=True)
    params = np.asarray(params)
    for stage, width in enumerate(lifted.stage_param_dims):
        start = int(lifted.param_offsets[stage])
        out[:, stage, :width] = params[:, start : start + width]
    return out


def lifted_to_flat_z(x, lifted: MPXLiftedProblem) -> np.ndarray:
    x = np.asarray(x)
    batch = x.shape[0]
    out = np.zeros((batch, int(lifted.z_offsets[-1])), dtype=x.dtype)
    for stage, width in enumerate(lifted.stage_z_dims):
        start = int(lifted.z_offsets[stage])
        out[:, start : start + width] = x[:, stage, :width]
    return out


def _stage_objective(cost, g, lower, upper, settings_vec):
    dtype = jnp.asarray(g).dtype
    eq_weight = jnp.asarray(settings_vec[0], dtype=dtype)
    barrier_alpha = jnp.asarray(settings_vec[1], dtype=dtype)
    barrier_sigma = jnp.asarray(settings_vec[2], dtype=dtype)

    cost = jnp.ravel(cost)[0]
    g = jnp.ravel(g)
    lower = jnp.ravel(lower)
    upper = jnp.ravel(upper)

    finite_lower = lower > -FINITE_BOUND_THRESHOLD
    finite_upper = upper < FINITE_BOUND_THRESHOLD
    equality = finite_lower & finite_upper & (jnp.abs(upper - lower) <= 1.0e-8)
    lower_only = finite_lower & ~equality
    upper_only = finite_upper & ~equality

    eq_res = jnp.where(equality, g - lower, 0.0)
    lower_margin = jnp.where(lower_only, g - lower, 1.0)
    upper_margin = jnp.where(upper_only, upper - g, 1.0)

    eq_penalty = 0.5 * eq_weight * jnp.sum(eq_res * eq_res)
    ineq_penalty = jnp.sum(jnp.where(lower_only, relaxed_barrier(lower_margin, barrier_alpha, barrier_sigma), 0.0))
    ineq_penalty += jnp.sum(jnp.where(upper_only, relaxed_barrier(upper_margin, barrier_alpha, barrier_sigma), 0.0))
    return cost + eq_penalty + ineq_penalty


def _constraint_gn_weights(g, lower, upper, settings_vec):
    dtype = jnp.asarray(g).dtype
    eq_weight = jnp.asarray(settings_vec[0], dtype=dtype)
    barrier_alpha = jnp.asarray(settings_vec[1], dtype=dtype)
    barrier_sigma = jnp.asarray(settings_vec[2], dtype=dtype)

    g = jnp.ravel(g)
    lower = jnp.ravel(lower)
    upper = jnp.ravel(upper)

    finite_lower = lower > -FINITE_BOUND_THRESHOLD
    finite_upper = upper < FINITE_BOUND_THRESHOLD
    equality = finite_lower & finite_upper & (jnp.abs(upper - lower) <= 1.0e-8)
    lower_only = finite_lower & ~equality
    upper_only = finite_upper & ~equality

    lower_margin = jnp.where(lower_only, g - lower, 1.0)
    upper_margin = jnp.where(upper_only, upper - g, 1.0)
    weights = jnp.where(equality, eq_weight, 0.0)
    weights += jnp.where(
        lower_only,
        relaxed_barrier_curvature(lower_margin, barrier_alpha, barrier_sigma),
        0.0,
    )
    weights += jnp.where(
        upper_only,
        relaxed_barrier_curvature(upper_margin, barrier_alpha, barrier_sigma),
        0.0,
    )
    return weights


def _split_hessian_blocks(hessian, x_dim: int, u_dim: int, max_x_dim: int):
    dtype = hessian.dtype
    q = jnp.zeros((max_x_dim, max_x_dim), dtype=dtype)
    r = jnp.zeros((max_x_dim, max_x_dim), dtype=dtype)
    m = jnp.zeros((max_x_dim, max_x_dim), dtype=dtype)
    q = q.at[:x_dim, :x_dim].set(hessian[:x_dim, :x_dim])
    if u_dim > 0:
        r = r.at[:u_dim, :u_dim].set(hessian[x_dim : x_dim + u_dim, x_dim : x_dim + u_dim])
        m = m.at[:x_dim, :u_dim].set(hessian[:x_dim, x_dim : x_dim + u_dim])
    return q, r, m


def _stage_gn_hessian(base_cost_fn, constraint_fn, variable, settings_vec, x_dim: int, u_dim: int, max_x_dim: int):
    cost_hessian = jax.hessian(base_cost_fn)(variable)
    g, lower, upper = constraint_fn(variable)
    jacobian = jax.jacobian(lambda value: jnp.ravel(constraint_fn(value)[0]))(variable)
    weights = _constraint_gn_weights(g, lower, upper, settings_vec)
    constraint_hessian = (jacobian * weights[:, None]).T @ jacobian
    hessian = cost_hessian + constraint_hessian
    hessian = 0.5 * (hessian + hessian.T)
    return _split_hessian_blocks(hessian, x_dim, u_dim, max_x_dim)


def _stage_violation(g, lower, upper):
    g = jnp.ravel(g)
    lower = jnp.ravel(lower)
    upper = jnp.ravel(upper)
    finite_lower = lower > -FINITE_BOUND_THRESHOLD
    finite_upper = upper < FINITE_BOUND_THRESHOLD
    lower_violation = jnp.where(finite_lower, jnp.maximum(lower - g, 0.0), 0.0)
    upper_violation = jnp.where(finite_upper, jnp.maximum(g - upper, 0.0), 0.0)
    return jnp.max(jnp.maximum(lower_violation, upper_violation))


def compile_lifted_mpx_problem(
    problem,
    *,
    settings: MPXBarrierSettings | None = None,
):
    """Compile an MPX primal-dual solver over a lifted stage-function problem.

    The lifted state at node ``k`` is the original sparse-MPC stage variable
    ``z_k`` padded to a uniform width.  The MPX control at interval ``k`` is a
    padded copy of ``z_{k+1}``, and the MPX dynamics is ``x_next = u``.  Original
    stage costs and finite-bound constraints are evaluated inside the scalar
    cost callback; inequalities use the relaxed barrier from MPX examples.
    """

    settings = settings or MPXBarrierSettings()
    stages = tuple(problem.stages)
    stage_z_dims = tuple(int(stage.z_dim) for stage in stages)
    stage_next_z_dims = tuple(int(stage.next_z_dim) for stage in stages)
    stage_param_dims = tuple(int(stage.param_dim) for stage in stages)
    max_z_dim = max(stage_z_dims)
    max_param_dim = max(stage_param_dims)
    z_offsets = _stage_offsets(stage_z_dims)
    param_offsets = _stage_offsets(stage_param_dims)

    branches = []
    violation_branches = []
    hessian_branches = []
    for stage in stages:
        z_dim = int(stage.z_dim)
        next_z_dim = int(stage.next_z_dim)
        param_dim = int(stage.param_dim)
        stage_fn = stage.jax_value_function

        if stage.has_next:

            def branch(operand, *, stage_fn=stage_fn, z_dim=z_dim, next_z_dim=next_z_dim, param_dim=param_dim):
                x, u, p, settings_vec = operand
                cost, g, lower, upper = stage_fn(x[:z_dim], u[:next_z_dim], p[:param_dim])
                return _stage_objective(cost, g, lower, upper, settings_vec)

            def violation_branch(operand, *, stage_fn=stage_fn, z_dim=z_dim, next_z_dim=next_z_dim, param_dim=param_dim):
                x, u, p = operand
                _, g, lower, upper = stage_fn(x[:z_dim], u[:next_z_dim], p[:param_dim])
                return _stage_violation(g, lower, upper)

            def hessian_branch(
                operand,
                *,
                stage_fn=stage_fn,
                z_dim=z_dim,
                next_z_dim=next_z_dim,
                param_dim=param_dim,
                max_z_dim=max_z_dim,
            ):
                x, u, p, settings_vec = operand
                variable = jnp.concatenate([x[:z_dim], u[:next_z_dim]])

                def base_cost_fn(value):
                    cost, _, _, _ = stage_fn(value[:z_dim], value[z_dim : z_dim + next_z_dim], p[:param_dim])
                    return jnp.ravel(cost)[0]

                def constraint_fn(value):
                    _, g, lower, upper = stage_fn(
                        value[:z_dim],
                        value[z_dim : z_dim + next_z_dim],
                        p[:param_dim],
                    )
                    return g, lower, upper

                return _stage_gn_hessian(
                    base_cost_fn,
                    constraint_fn,
                    variable,
                    settings_vec,
                    z_dim,
                    next_z_dim,
                    max_z_dim,
                )

        else:

            def branch(operand, *, stage_fn=stage_fn, z_dim=z_dim, param_dim=param_dim):
                x, u, p, settings_vec = operand
                del u
                cost, g, lower, upper = stage_fn(x[:z_dim], p[:param_dim])
                return _stage_objective(cost, g, lower, upper, settings_vec)

            def violation_branch(operand, *, stage_fn=stage_fn, z_dim=z_dim, param_dim=param_dim):
                x, u, p = operand
                del u
                _, g, lower, upper = stage_fn(x[:z_dim], p[:param_dim])
                return _stage_violation(g, lower, upper)

            def hessian_branch(
                operand,
                *,
                stage_fn=stage_fn,
                z_dim=z_dim,
                param_dim=param_dim,
                max_z_dim=max_z_dim,
            ):
                x, u, p, settings_vec = operand
                del u
                variable = x[:z_dim]

                def base_cost_fn(value):
                    cost, _, _, _ = stage_fn(value[:z_dim], p[:param_dim])
                    return jnp.ravel(cost)[0]

                def constraint_fn(value):
                    _, g, lower, upper = stage_fn(value[:z_dim], p[:param_dim])
                    return g, lower, upper

                return _stage_gn_hessian(
                    base_cost_fn,
                    constraint_fn,
                    variable,
                    settings_vec,
                    z_dim,
                    0,
                    max_z_dim,
                )

        branches.append(branch)
        violation_branches.append(violation_branch)
        hessian_branches.append(hessian_branch)

    def cost(settings_vec, reference, x, u, t):
        idx = jnp.minimum(t, len(branches) - 1)
        return jax.lax.switch(idx, branches, (x, u, reference[idx], settings_vec))

    def dynamics(x, u, t, parameter):
        del x, t, parameter
        return u

    def hessian_approx(settings_vec, reference, x, u, t):
        idx = jnp.minimum(t, len(hessian_branches) - 1)
        return jax.lax.switch(idx, hessian_branches, (x, u, reference[idx], settings_vec))

    def max_stage_violation(x, u, reference):
        def one_stage(t):
            u_t = jnp.where(t < len(stages) - 1, u[jnp.minimum(t, u.shape[0] - 1)], jnp.zeros_like(u[0]))
            return jax.lax.switch(t, violation_branches, (x[t], u_t, reference[t]))

        return jnp.max(jax.vmap(one_stage)(jnp.arange(len(stages))))

    if settings.solver_mode == "primal_dual":
        work = partial(
            mpx_optimizers.mpc,
            cost,
            dynamics,
            hessian_approx,
            settings.limited_memory,
            num_alpha=settings.num_alpha,
        )
    elif settings.solver_mode == "fddp":
        def work(reference, parameter, settings_vec, x0, x, u, v):
            del v
            return mpx_optimizers.fddp_mpc(
                cost,
                dynamics,
                hessian_approx,
                settings.limited_memory,
                reference,
                parameter,
                settings_vec,
                x0,
                x,
                u,
                settings.num_alpha,
            )
    elif settings.solver_mode == "ilqr":
        def work(reference, parameter, settings_vec, x0, x, u, v):
            del v
            return ilqr_mpc(
                cost,
                dynamics,
                hessian_approx,
                settings.limited_memory,
                reference,
                parameter,
                settings_vec,
                x0,
                x,
                u,
                settings.num_alpha,
            )
    else:
        raise ValueError(f"unsupported MPX solver mode: {settings.solver_mode!r}")
    solve = jax.jit(jax.vmap(work))
    violation = jax.jit(jax.vmap(max_stage_violation))
    return MPXLiftedProblem(
        stage_z_dims=stage_z_dims,
        stage_next_z_dims=stage_next_z_dims,
        stage_param_dims=stage_param_dims,
        z_offsets=z_offsets,
        param_offsets=param_offsets,
        max_z_dim=max_z_dim,
        max_param_dim=max_param_dim,
        settings=settings,
        solve=solve,
        cost=cost,
        dynamics=dynamics,
        violation=violation,
    )


def settings_array(settings: MPXBarrierSettings, batch_size: int, dtype: np.dtype | str):
    dtype = np.dtype(dtype)
    values = np.asarray(
        [settings.equality_weight, settings.barrier_alpha, settings.barrier_sigma],
        dtype=dtype,
    )
    return np.broadcast_to(values[None, :], (batch_size, values.size))


def zeros_parameter(batch_size: int, dtype: np.dtype | str):
    return np.zeros((batch_size, 1), dtype=np.dtype(dtype))
