#!/usr/bin/env python3
"""Audit scaling and rho on QPs captured from original GMR on real motions."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import mink
import numpy as np
import scipy.sparse as sp
from qpsolvers import solve_qp

from examples.gmr_retargeting.model import RetargetModel, load_gmr
from examples.gmr_retargeting.reporting import provenance, write_csv, write_json
from warpmpc.jax_osqp import OSQPSettings, build_osqp_plan, compile_osqp


def capture_qps(model, data):
    """First and last QPs of both stages at cold, early and later frames."""
    selected = set()
    starts = list(map(int, data["starts"]))
    for begin, end in zip(starts, starts[1:] + [len(data["human"])]):
        selected.update((begin, min(begin + 1, end - 1), (begin + end) // 2, end - 1))
    captured = []
    original = mink.solve_ik
    for frame in sorted(selected):
        calls = [[], []]

        def record(configuration, tasks, dt, *args, **kwargs):
            stage = 0 if tasks is model.gmr.tasks1 else 1
            q = configuration.q.copy()
            p, c = model.qp_function(q, model.targets()[stage], model.weights[stage])
            lower = model.limit_gain * (model.lower - q[model.qindices])
            upper = model.limit_gain * (model.upper - q[model.qindices])
            velocity = original(configuration, tasks, dt, *args, **kwargs)
            calls[stage].append(
                (
                    np.asarray(p).ravel(),
                    np.asarray(c).ravel(),
                    lower,
                    upper,
                    np.asarray(velocity) * dt,
                )
            )
            return velocity

        model.gmr.configuration.update(data["seeds"][frame])
        human = {
            name: (data["human"][frame, j, :3], data["human"][frame, j, 3:])
            for j, name in enumerate(model.human_names)
        }
        try:
            mink.solve_ik = record
            model.gmr.retarget(human)
        finally:
            mink.solve_ik = original
        for stage, steps in enumerate(calls):
            for iteration in sorted({0, len(steps) - 1}):
                captured.append((frame, stage, iteration, *steps[iteration]))
    return captured


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-dir", type=pathlib.Path, required=True)
    p.add_argument("--output-dir", type=pathlib.Path, required=True)
    p.add_argument("--gmr-root", type=pathlib.Path, default=ROOT / "resources/GMR")
    p.add_argument("--rhos", default="0.01,0.03,0.1,0.3,1")
    p.add_argument("--admm-iterations", default="25,50,100")
    p.add_argument(
        "--scaled-only", action="store_true", help="Refine rho after the raw/scaled comparison"
    )
    args = p.parse_args()
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("Scaling audit requires the GPU solver")
    metadata = json.loads((args.reference_dir / "metadata.json").read_text())
    model = RetargetModel(load_gmr(args.gmr_root, metadata["height"]))
    with np.load(args.reference_dir / "reference.npz", allow_pickle=False) as data:
        captured = capture_qps(model, data)
    identities = np.array([row[:3] for row in captured])
    pvals, c, lower, upper, original_step = [
        np.stack([row[i] for row in captured]) for i in range(3, 8)
    ]
    hessian = np.zeros((len(c), model.nv, model.nv))
    hessian[:, model.rows, model.cols] = pvals
    hessian[:, model.cols, model.rows] = pvals
    diagonal = np.diagonal(hessian, axis1=1, axis2=2)
    d = 1 / np.sqrt(diagonal)
    scaled_hessian = hessian * d[:, :, None] * d[:, None, :]
    raw_condition, scaled_condition = np.linalg.cond(hessian), np.linalg.cond(scaled_hessian)
    reference, scaled_reference = [], []
    for i in range(len(c)):
        lb, ub = np.full(model.nv, -np.inf), np.full(model.nv, np.inf)
        lb[model.dofs], ub[model.dofs] = lower[i], upper[i]
        exact = solve_qp(
            hessian[i], c[i], lb=lb, ub=ub, solver="daqp", primal_tol=1e-10, dual_tol=1e-10
        )
        scaled = solve_qp(
            scaled_hessian[i],
            c[i] * d[i],
            lb=lb / d[i],
            ub=ub / d[i],
            solver="daqp",
            primal_tol=1e-10,
            dual_tol=1e-10,
        )
        if exact is None or scaled is None:
            raise RuntimeError("DAQP failed on captured GMR QP")
        reference.append(exact)
        scaled_reference.append(scaled * d[i])
    reference, scaled_reference = np.array(reference), np.array(scaled_reference)
    equivalence = float(np.max(np.abs(reference - scaled_reference)))
    if equivalence > 1e-7:
        raise AssertionError(f"Scaling changed the double-precision QP solution: {equivalence}")
    stats = dict(
        provenance=provenance(),
        source=metadata["source"],
        qp_count=len(c),
        diagonal_min=float(diagonal.min()),
        diagonal_max=float(diagonal.max()),
        raw_condition_min=float(raw_condition.min()),
        raw_condition_max=float(raw_condition.max()),
        scaled_condition_min=float(scaled_condition.min()),
        scaled_condition_max=float(scaled_condition.max()),
        scaled_diagonal_max_error=float(
            np.max(np.abs(np.diagonal(scaled_hessian, axis1=1, axis2=2) - 1))
        ),
        daqp_scaling_equivalence_max=equivalence,
        original_gmr_step_vs_tight_daqp_max=float(np.max(np.abs(original_step - reference))),
    )
    write_json(args.output_dir / "scaling.json", stats)
    np.savez_compressed(
        args.output_dir / "captured_qps.npz",
        frame_stage_iteration=identities,
        p_values=pvals,
        c=c,
        lower=lower,
        upper=upper,
        reference=reference,
        original_gmr_step=original_step,
        diagonal=diagonal,
        raw_condition=raw_condition,
        scaled_condition=scaled_condition,
    )
    print(json.dumps(stats), flush=True)
    constraints = sp.csc_matrix(
        (np.ones(len(model.dofs)), (np.arange(len(model.dofs)), model.dofs)),
        shape=(len(model.dofs), model.nv),
    )
    ref_objective = 0.5 * np.einsum("bi,bij,bj->b", reference, hessian, reference) + np.sum(
        c * reference, axis=1
    )
    rows = []
    for scaled in ((True,) if args.scaled_only else (False, True)):
        scale = d if scaled else np.ones_like(c)
        P = pvals * scale[:, model.rows] * scale[:, model.cols]
        C, L, U = c * scale, lower / scale[:, model.dofs], upper / scale[:, model.dofs]
        inputs = [
            jnp.asarray(value, dtype=jnp.float32)
            for value in (P, np.ones((1, len(model.dofs))), C, L, U)
        ]
        for rho in map(float, args.rhos.split(",")):
            for iterations in map(int, args.admm_iterations.split(",")):
                plan = build_osqp_plan(
                    model.pattern,
                    constraints,
                    -np.ones(len(model.dofs)),
                    np.ones(len(model.dofs)),
                    OSQPSettings(max_iter=iterations, rho=rho, sigma=1e-6, alpha=1.6, scaling=0),
                )
                solver = compile_osqp(
                    plan,
                    dtype=np.float32,
                    qdldl_backend="warp",
                    transpose_work=True,
                    segmented=True,
                    segment_budget=128,
                )
                solution = jax.block_until_ready(solver.solve(*inputs))
                x = np.asarray(solution[0], dtype=np.float64) * scale
                y = np.asarray(solution[2], dtype=np.float64) / scale[:, model.dofs]
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise FloatingPointError("Nonfinite QP audit output")
                hx = np.einsum("bij,bj->bi", hessian, x)
                aty = np.zeros_like(x)
                aty[:, model.dofs] = y
                normalized_dual = np.max(np.abs(hx + c + aty), axis=1) / np.maximum(
                    1.0,
                    np.maximum(
                        np.max(np.abs(hx), axis=1),
                        np.maximum(np.max(np.abs(c), axis=1), np.max(np.abs(aty), axis=1)),
                    ),
                )
                violations = np.maximum(
                    0.0, np.maximum(lower - x[:, model.dofs], x[:, model.dofs] - upper)
                )
                objective = 0.5 * np.sum(x * hx, axis=1) + np.sum(c * x, axis=1)
                errors = np.max(np.abs(x - reference), axis=1)
                row = dict(
                    scaled=scaled,
                    rho=rho,
                    admm_iterations=iterations,
                    delta_max_error=float(errors.max()),
                    delta_p95_error=float(np.quantile(errors, 0.95)),
                    fraction_qps_within_1e3=float(np.mean(errors <= 1e-3)),
                    max_bound_violation_rad=float(violations.max()),
                    max_relative_stationarity=float(normalized_dual.max()),
                    max_relative_objective_error=float(
                        np.max(
                            np.abs(objective - ref_objective)
                            / np.maximum(1.0, np.abs(ref_objective))
                        )
                    ),
                )
                rows.append(row)
                write_csv(args.output_dir / "qp_accuracy.csv", rows)
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
