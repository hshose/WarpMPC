"""Run the humanoid evaluator in small GPU batches for numerical diagnosis.

This is not a throughput benchmark. Keep the upstream single SQP iteration,
but compile stage evaluation, assembly, ADMM, and line search separately to
avoid a very large monolithic humanoid compilation. Each trajectory retains
its own feedback and ADMM warm state. Remaining CLI arguments go to the
ordinary evaluator (including --scaling 0 to reproduce the old baseline).
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys
import textwrap
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

from benchmarks.nonlinear_mpc import evaluate_turbompc_humanoid_mpc as evaluator
from benchmarks.nonlinear_mpc.turbompc_adapter import DenseStageTurbompcProblem
from turbompc.solvers.turbompc_solver import TurboMPCSolver


class _EvaluatedStages(DenseStageTurbompcProblem):
    def __init__(self, original, raw, initial):
        self.original, self.raw, self.initial = original, raw, initial

    def __getattr__(self, name):
        return getattr(self.original, name)

    def _eval_sparse(self, states, controls, params, index):
        return self.raw[index]

    def _eval_values(self, states, controls, params, index):
        return self.raw[index]

    def get_initial_equality_linearized_matrices(self, params, dtype):
        return self.initial


def split_assembly(solver):
    problem = solver.program
    assert not problem.rescale_optimization_variables
    evaluate = problem._eval_sparse

    @jax.jit
    def initial(params):
        return problem.get_initial_equality_linearized_matrices(
            params, params['packed_params'].dtype
        )

    @jax.jit
    def assemble(states, controls, params, raw, initial_data):
        proxy = types.SimpleNamespace(program=_EvaluatedStages(problem, raw, initial_data))
        return TurboMPCSolver._build_qp_data(proxy, states, controls, params)

    def build(states, controls, params, apply_variable_scaling=True):
        raw = tuple(evaluate(states, controls, params, i) for i in range(problem.horizon + 1))
        return assemble(states, controls, params, raw, initial(params))

    solver._build_qp_data = build
    problem.cost = jax.jit(problem.cost)

    def split_constraint_values(method):
        @jax.jit
        def assemble_values(states, controls, params, raw):
            return method(_EvaluatedStages(problem, raw, None), states, controls, params)

        def constraints(states, controls, params):
            raw = tuple(problem._eval_values(states, controls, params, i)
                        for i in range(problem.horizon + 1))
            return assemble_values(states, controls, params, raw)

        return constraints

    problem.equality_constraints = split_constraint_values(DenseStageTurbompcProblem.equality_constraints)
    problem.inequality_constraints = split_constraint_values(DenseStageTurbompcProblem.inequality_constraints)


def single_sqp_iteration():
    # The upstream while loop executes its first iteration unconditionally.
    # Fail explicitly if that implementation changes, rather than silently
    # changing the numerical algorithm during diagnosis.
    source = textwrap.dedent(inspect.getsource(TurboMPCSolver._solve_impl))
    marker = '    out = jax.lax.while_loop(cond_fun, body_fun, init_state)'
    assert source.count(marker) == 1
    source = source.replace(marker, '    assert max_iter == 1\n    out = body_fun(init_state)')
    namespace = {}
    exec(compile(source, '<single_turbompc_iteration>', 'exec'),
         TurboMPCSolver._solve_impl.__globals__, namespace)
    return namespace['_solve_impl']


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--device-batch-size', type=int, default=20)
    args, remaining = parser.parse_known_args()
    if args.device_batch_size < 1:
        parser.error('--device-batch-size must be positive')
    if not any(d.platform == 'gpu' for d in jax.devices()):
        raise RuntimeError('This diagnostic requires a GPU')
    TurboMPCSolver._solve_impl = single_sqp_iteration()

    original_make_solver = evaluator.make_solver

    def make_solver(*positional, **keywords):
        solver = original_make_solver(*positional, **keywords)
        split_assembly(solver)
        return solver

    evaluator.make_solver = make_solver

    def chunked(function):
        def call(*values):
            batch = jax.tree.leaves(values)[0].shape[0]
            outputs = []
            for start in range(0, batch, args.device_batch_size):
                chunk = jax.tree.map(lambda v: v[start:start + args.device_batch_size], values)
                outputs.append(jax.block_until_ready(function(*chunk)))
            return jax.tree.map(lambda *v: jnp.concatenate(v, axis=0), *outputs)
        return call

    original_init = evaluator.initial_admm_state_batch

    def initialize(solver, states, controls, params, *, rho):
        # The CUDA FFI encodes relaxation as a static attribute.
        solver._admm_solver_fwd.solve = jax.jit(
            solver._admm_solver_fwd.solve, static_argnames=('alpha',)
        )
        print(f'Initializing in GPU batches of {args.device_batch_size}', flush=True)
        return chunked(lambda x, u, p: original_init(solver, x, u, p, rho=rho))(
            states, controls, params
        )

    evaluator.initial_admm_state_batch = initialize
    source = textwrap.dedent(inspect.getsource(evaluator.run))
    source = source.replace('    dtype = np.dtype(args.dtype)',
                            '    requested_sim_steps = args.sim_steps\n    dtype = np.dtype(args.dtype)')
    marker = 'solve_batch = jax.jit(jax.vmap(solve_one))'
    assert source.count(marker) == 1
    source = source.replace(marker, 'solve_batch = _chunked(jax.vmap(solve_one))')
    marker = '        step_total_s.append(time.perf_counter() - step_start)'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '''
        finite = np.all(np.isfinite(np.concatenate([q_measured, dq_measured, tau0_measured], axis=1)), axis=1)
        print(f'Step {step + 1}: finite feedback {finite.sum()}/{args.batch_size}, '
              f'QP primal median={np.median(prim_res[-1]):.6g}, '
              f'dual median={np.median(dual_res[-1]):.6g}', flush=True)
        if not np.any(finite):
            args.sim_steps = step + 1
            break
''')
    marker = '    return summary'
    assert source.count(marker) == 1
    source = source.replace(marker, '''    summary.update(
        diagnostic_only=True,
        device_batch_size=_diagnostic_device_batch_size,
        requested_sim_steps=requested_sim_steps,
    )
''' + marker)
    namespace = evaluator.run.__globals__ | {
        '_chunked': chunked, '_diagnostic_device_batch_size': args.device_batch_size,
    }
    exec(compile(source, '<chunked_humanoid_diagnostic>', 'exec'), namespace)
    evaluator.run = namespace['run']
    sys.argv = [str(evaluator.__file__), *remaining]
    evaluator.main()


if __name__ == '__main__':
    main()
