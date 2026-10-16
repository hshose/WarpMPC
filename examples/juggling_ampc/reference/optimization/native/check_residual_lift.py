"""Frozen-point residual-lift algebra check; never advances a plant.

--inputs NPZ contains normalized z (physical n), parameter p (30), and optional
reference_offset (3), reference_posture (7), contact_bias (3). --initialize
activates the existing causal event cache from p before checking the given z.
--solve optionally compares two high-accuracy frozen QPs; it is not policy work.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cpp.python_controller import NativeController, DOUBLE, INTEGER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('library', 'factors', 'inputs', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--solve', action='store_true')
    parser.add_argument('--compare-qp-data', type=Path,
                        help='Prior checker JSON: require bit-identical lifted QP/warmstart data (e.g. link rho 1 vs 100)')
    args = parser.parse_args()
    saved = np.load(args.inputs)
    z, p = (np.ascontiguousarray(saved[key], dtype=float) for key in ('z', 'p'))
    c = NativeController(args.library, args.factors)
    try:
        if z.shape != (c.n,) or p.shape != (30,):
            raise ValueError('Expected z[n] and p[30]')
        if 'reference_offset' in saved:
            c.set_reference(saved['reference_offset'], saved['reference_posture'])
        if 'contact_bias' in saved:
            c.set_contact_bias(saved['contact_bias'])
        if args.initialize:
            c.initialize(p[:23], p[23:])
        error = [ct.c_char_p, ct.c_size_t]
        counts = np.empty(12, np.int64)
        c.lib.juggling_qp_diagnostics.argtypes = [ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), *error]
        c.lib.juggling_lift_pattern.argtypes = [ct.c_void_p, INTEGER, INTEGER, INTEGER, INTEGER, *error]
        c.lib.juggling_lift_prepare.argtypes = [ct.c_void_p, *([DOUBLE]*11), *error]
        c.lib.juggling_value.argtypes = [ct.c_void_p, DOUBLE, DOUBLE, DOUBLE, DOUBLE, *error]
        c._call('juggling_qp_diagnostics', counts)
        physical_n, physical_m, nr, n, m, np_, na = map(int, counts[:7])
        assert physical_n == c.n and physical_m == c.m+c.n and n == c.n+nr and m == physical_m+nr
        pp, pi = np.empty(n+1, np.int32), np.empty(np_, np.int32)
        ap, ai = np.empty(n+1, np.int32), np.empty(na, np.int32)
        c._call('juggling_lift_pattern', pp, pi, ap, ai)
        H, gradient, original_A, original_c, original_cost = c.linearize(z, p)
        pv, q, av = np.empty(np_), np.empty(n), np.empty(na)
        l, u, residual = np.empty(m), np.empty(m), np.empty(nr)
        warm_x, warm_y, cost = np.empty(n), np.empty(m), np.empty(1)
        c._call('juggling_lift_prepare', z, p, pv, q, av, l, u, residual, warm_x, warm_y, cost)
        P, A = sp.csc_matrix((pv, pi, pp), (n, n)), sp.csc_matrix((av, ai, ap), (m, n))
        J = -A[physical_m:, :physical_n]
        full_H = H+H.T-sp.diags(H.diagonal())
        reduced_H = J.T@J+P[:physical_n, :physical_n]
        reduced_q = J.T@l[physical_m:]+q[:physical_n]
        cv, value_cost = np.empty(c.m), np.empty(1)
        c._call('juggling_value', z, p, cv, value_cost)
        maxima = {}

        def equal(name, actual, expected, atol=3e-10, rtol=3e-10):
            if sp.issparse(actual):
                difference = actual-expected
                maximum = float(np.max(np.abs(difference.data), initial=0.))
                reference = float(np.max(np.abs(expected.data), initial=0.))
                assert maximum <= atol+rtol*reference, (name, maximum, reference)
            else:
                maximum = float(np.max(np.abs(np.asarray(actual)-expected), initial=0.))
                np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=name)
            maxima[name] = maximum

        equal('H_schur', reduced_H, full_H)
        equal('gradient', J.T@residual, gradient)
        equal('absolute_q', reduced_q, gradient-full_H@z)
        equal('nonlinear_cost', .5*residual@residual, original_cost)
        equal('prepare_cost', cost[0], original_cost)
        equal('value_cost', value_cost[0], original_cost)
        equal('value_constraints', cv, original_c)
        equal('physical_A', A[:physical_m, :physical_n], original_A, atol=0., rtol=0.)
        equal('no_auxiliary_original_rows', A[:physical_m, physical_n:], sp.csc_matrix((physical_m, nr)), atol=0., rtol=0.)
        equal('auxiliary_identity', A[physical_m:, physical_n:], sp.eye(nr, format='csc'), atol=0., rtol=0.)
        equal('equality_rhs', l[physical_m:], residual-J@z)
        equal('equality_bounds', l[physical_m:], u[physical_m:], atol=0., rtol=0.)
        equal('warm_equality', A[physical_m:]@warm_x, l[physical_m:])
        equal('warm_aux_stationarity', warm_x[physical_n:]+warm_y[physical_m:], np.zeros(nr), atol=0., rtol=0.)
        assert np.array_equal(P.diagonal()[physical_n:], np.ones(nr)) and P.nnz == n
        arrays = dict(P_values=pv, P_indptr=pp, P_indices=pi, A_values=av,
                      A_indptr=ap, A_indices=ai, q=q, l=l, u=u,
                      residual=residual, warm_primal=warm_x, warm_dual=warm_y,
                      nonlinear_constraints=cv, nonlinear_cost=value_cost)
        digests = {name: hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                   for name, array in arrays.items()}
        if args.compare_qp_data:
            previous = json.loads(args.compare_qp_data.read_text())
            if previous.get('qp_data_sha256') != digests:
                changed = sorted(name for name, digest in digests.items()
                                 if previous.get('qp_data_sha256', {}).get(name) != digest)
                raise AssertionError(f'Lifted QP/warmstart data changed: {changed}')
        result = dict(passed=True, diagnostic_counts=counts.tolist(), max_errors=maxima,
                      qp_data_sha256=digests,
                      compared_qp_data=str(args.compare_qp_data) if args.compare_qp_data else None,
                      initialized_event_cache=args.initialize, qp_solved=False, physical_plant_advanced=False)
        if args.solve:
            import osqp
            def solve(P_, q_, A_, l_, u_, x_, y_):
                solver = osqp.OSQP()
                solver.setup(P=P_, q=q_, A=A_, l=l_, u=u_, eps_abs=1e-9, eps_rel=1e-9,
                             max_iter=20000, polishing=True, verbose=False, adaptive_rho=True)
                solver.warm_start(x=x_, y=y_)
                answer = solver.solve()
                assert answer.info.status_val in (1, 2), answer.info.status
                return answer
            old = solve(H, gradient-full_H@z, original_A, l[:physical_m], u[:physical_m], z, warm_y[:physical_m])
            new = solve(P, q, A, l, u, warm_x, warm_y)
            # Different ADMM systems can have different iterate histories. The
            # optional high-accuracy comparison is physical solution/objective.
            equal('high_accuracy_physical_solution', new.x[:physical_n], old.x, atol=2e-4, rtol=2e-4)
            def physical_objective(x):
                return .5*x@(full_H@x)+(gradient-full_H@z)@x
            equal('high_accuracy_physical_objective', physical_objective(new.x[:physical_n]), physical_objective(old.x), atol=1e-6, rtol=1e-6)
            result.update(qp_solved=True, validation_iterations=[int(old.info.iter), int(new.info.iter)])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(result))
    finally:
        c.close()
        saved.close()


if __name__ == '__main__':
    main()
