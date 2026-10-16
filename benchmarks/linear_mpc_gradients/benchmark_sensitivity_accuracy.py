#!/usr/bin/env python3
"""Accuracy of implicit linear-MPC sensitivities against converged central FD.

By default, differentiate only the four components of the first MPC action
u_0 with respect to all 12 initial-state coordinates (a 4x12 Jacobian).
Legacy random primal/dual projections are available via --objective. Equality bounds
move together. Reference solves are independent, cold-started CPU OSQP float64
solves, not differences of the truncated ADMM algorithm. See README_accuracy.md.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from benchmarks.problems import batched_problem_data, make_linear_mpc

PAPER_FORWARD = 25
PAPER_REFINEMENT = 100


def integers(value):
    values = sorted(set(int(v) for v in value.split(",")))
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def statistics(values, prefix):
    # Never discard nonfinite values: they must remain visible in the artifacts.
    return {f"{prefix}_{name}": float(fn(values)) for name, fn in
            (("median", np.median), ("mean", np.mean), ("max", np.max))}


def reference_one(task):
    import osqp
    index, horizon, dtype, q, lower, upper, dx, dy, step, tol, max_iter = task
    problem = make_linear_mpc(horizon)
    problem.p_matrix.data = problem.p_matrix.data.astype(dtype).astype(float)
    problem.a_matrix.data = problem.a_matrix.data.astype(dtype).astype(float)
    solver = osqp.OSQP()
    solver.setup(problem.p_matrix, q, problem.a_matrix, lower, upper,
                 verbose=False, scaling=0, adaptive_rho=False, rho_is_vec=True,
                 check_termination=25, max_iter=max_iter, polishing=False,
                 warm_starting=False, rho=0.1, sigma=1e-6, alpha=1.6,
                 eps_abs=tol, eps_rel=tol)
    residuals = []
    iterations = []

    def solve(l, u):
        solver.update(l=l, u=u)
        result = solver.solve(raise_error=False)
        if result.info.status_val != 1:
            raise RuntimeError(f"FD sample {index}: {result.info.status}")
        residuals.append((result.info.prim_res, result.info.dual_res))
        iterations.append(result.info.iter)
        return result.x.copy(), result.y.copy()

    x, y = solve(lower, upper)
    gradients = []
    for h in (step, step / 2):
        gradient = np.empty((*dx.shape[:-1], problem.nx))
        for component in range(problem.nx):
            lp, up, lm, um = (v.copy() for v in (lower, upper, lower, upper))
            lp[component] -= h
            up[component] -= h
            lm[component] += h
            um[component] += h
            xp, yp = solve(lp, up)
            xm, ym = solve(lm, um)
            gradient[..., component] = (dx @ (xp - xm) + dy @ (yp - ym)) / (2 * h)
        gradients.append(gradient)
    return index, x, y, gradients[0], gradients[1], np.max(residuals, axis=0), max(iterations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--refinements", type=integers, default=integers("1,5,10,20,25,50,100,200,250,400,500"))
    parser.add_argument("--forward-iterations", type=integers, default=integers("25,100,200,800,2000"))
    parser.add_argument("--objective", choices=("u0-jacobian", "primal-dual", "primal"), default="u0-jacobian")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--fd-step", type=float, default=1e-3)
    parser.add_argument("--reference-tolerance", type=float, default=1e-10)
    parser.add_argument("--reference-max-iter", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--reference-only", action="store_true", help="Save matched inputs and CPU finite differences without the GPU grid")
    args = parser.parse_args()
    if min(args.batch_size, args.horizon, args.workers, args.reference_max_iter) < 1:
        parser.error("batch size, horizon, workers, and reference max iter must be positive")
    if not (args.fd_step > 0 and args.reference_tolerance > 0):
        parser.error("FD step and tolerance must be positive")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.json")) or any(out.glob("*.csv")):
        raise FileExistsError(f"Use a fresh output directory: {out}")
    start = time.time()
    problem = make_linear_mpc(args.horizon)
    # Generate in the target dtype first so FD and GPU evaluate identical inputs.
    data = batched_problem_data(problem, args.batch_size, args.dtype,
                                seed=args.seed + args.batch_size, x0_variation=0.15)
    p, a, q, lower, upper = data
    action_start = (args.horizon + 1) * problem.nx
    action_end = action_start + problem.nu
    is_action = args.objective == "u0-jacobian"
    if is_action:
        # Each basis vector selects one component of u_0 only; all dual seeds are zero.
        dx = np.zeros((problem.nu, q.shape[1]), dtype=args.dtype)
        dx[:, action_start:action_end] = np.eye(problem.nu, dtype=args.dtype)
        dy = np.zeros((problem.nu, lower.shape[1]), dtype=args.dtype)
    else:
        rng = np.random.default_rng(args.seed + 3000 + args.batch_size)
        dx = rng.standard_normal(q.shape).astype(args.dtype)
        dy = rng.standard_normal(lower.shape).astype(args.dtype)
        if args.objective == "primal":
            dy.fill(0)
    np.savez_compressed(out / "inputs.npz", p=p, a=a, q=q, l=lower, u=upper, dx=dx, dy=dy)
    metadata = dict(vars(args), output_dir=str(out), status="reference", started=time.ctime(),
                    hostname=platform.node(), paper_forward_iterations=PAPER_FORWARD,
                    paper_refinement_iterations=PAPER_REFINEMENT,
                    parameter="initial state (all 12 components)",
                    objective_description=("Full du_0/dx_0 Jacobian using four action basis cotangents; zero dual seeds"
                                           if is_action else "dx.T @ primal + dy.T @ dual; dy=0 for primal objective"),
                    study="full_iteration_grid",
                    reference="cold-started converged float64 CPU OSQP central FD at h and h/2",
                    primary_error=("per-QP relative Frobenius norm of 4x12 Jacobian error; denominator max(norm(FD), 1e-12)"
                                   if is_action else "per-QP L2 relative error; denominator max(norm(FD), 1e-12)"),
                    sensitivity_shape=([args.batch_size, problem.nu, problem.nx] if is_action
                                       else [args.batch_size, problem.nx]),
                    action_slice=[action_start, action_end],
                    variant="transpose+segmented+levelsolve", segment_budget=16)
    metadata["source_sha256"] = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                                 for folder in ("warpmpc/jax_osqp", "warpmpc/jax_qdldl", "benchmarks/linear_mpc_gradients", "benchmarks/problems")
                                 for f in sorted((ROOT / folder).glob("*.py"))}

    def save_metadata():
        temporary = out / "metadata.tmp"
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(out / "metadata.json")

    save_metadata()
    tasks = [(i, args.horizon, args.dtype, q[i].astype(float), lower[i].astype(float), upper[i].astype(float),
              (dx if is_action else dx[i]).astype(float), (dy if is_action else dy[i]).astype(float), args.fd_step, args.reference_tolerance,
              args.reference_max_iter) for i in range(args.batch_size)]
    xref, yref = np.empty(q.shape), np.empty(lower.shape)
    sensitivity_shape = tuple(metadata["sensitivity_shape"])
    fd, fd_half = np.empty(sensitivity_shape), np.empty(sensitivity_shape)
    norm_axes = (1, 2) if is_action else 1
    residuals, iters = np.empty((args.batch_size, 2)), np.empty(args.batch_size, dtype=int)
    print(f"Reference: {args.batch_size} QPs, 49 cold solves each, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for count, result in enumerate(pool.map(reference_one, tasks), 1):
            i, xref[i], yref[i], fd[i], fd_half[i], residuals[i], iters[i] = result
            if count % max(1, args.batch_size // 20) == 0:
                print(f"Reference {count}/{args.batch_size}; elapsed {time.time()-start:.1f}s", flush=True)
    denominator = np.maximum(np.linalg.norm(fd_half, axis=norm_axes), 1e-12)
    fd_disagreement = np.linalg.norm(fd - fd_half, axis=norm_axes) / denominator
    np.savez_compressed(out / "reference.npz", x=xref, y=yref, fd=fd, fd_half=fd_half,
                        relative_step_disagreement=fd_disagreement, residuals=residuals, iterations=iters)
    metadata.update(statistics(fd_disagreement, "fd_step_relative_disagreement"))
    metadata["reference_seconds"] = time.time() - start
    metadata["status"] = "gpu_sweeps"
    save_metadata()
    print("FD step check:", statistics(fd_disagreement, "relative"), flush=True)
    if args.reference_only:
        metadata.update(status="complete", study="finite_difference_reference", completed_points=0, total_points=0)
        save_metadata()
        return

    import jax
    import jax.numpy as jnp
    import osqp
    from benchmarks.jax_cache import configure_jax_compilation_cache
    from warpmpc.jax_osqp import OSQPSettings, build_osqp_plan, compile_osqp
    configure_jax_compilation_cache()
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    if args.require_gpu and not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("GPU requested but unavailable")
    metadata.update(devices=[str(d) for d in jax.devices()], jax_version=jax.__version__,
                    numpy_version=np.__version__, osqp_version=osqp.__version__)
    save_metadata()
    print("Devices:", jax.devices(), flush=True)
    arrays = tuple(jnp.asarray(v) for v in (*data, dx, dy))
    rows = []
    action_rows = []
    configurations = [("grid", f, r) for f in args.forward_iterations for r in args.refinements]
    for number, (sweep, forward, refinement) in enumerate(configurations, 1):
        point_start = time.time()
        print(f"Point {number}/{len(configurations)}: {sweep}, forward={forward}, refinement={refinement}", flush=True)
        settings = OSQPSettings(max_iter=forward, scaling=0, adaptive_rho=False, rho_is_vec=True,
                                check_termination=0, warm_starting=False, polishing=False)
        plan = build_osqp_plan(problem.p_matrix, problem.a_matrix, problem.l, problem.u, settings, derivatives=True)
        solver = compile_osqp(plan, dtype=args.dtype, derivatives=True,
                              derivative_refinement_iters=refinement, transpose_work=True,
                              segmented=True, segment_budget=16, segment_strategy="optimal",
                              level_scheduled_solve=True)
        def evaluate(pv, av, qv, lv, uv, dxx, dyy):
            def objective(x0):
                ll = lv.at[:, :problem.nx].set(-x0)
                uu = uv.at[:, :problem.nx].set(-x0)
                xx, yy = solver.solve_xy(pv, av, qv, ll, uu)
                if is_action:
                    # QPs are independent, so differentiating the sum over the
                    # batch returns each QP's own action Jacobian, without a
                    # wasteful (batch, 4, batch, 12) cross-QP Jacobian.
                    value = jnp.sum(xx[:, action_start:action_end], axis=0)
                else:
                    value = jnp.sum(xx * dxx) + jnp.sum(yy * dyy)
                return value, (xx, yy)
            x0 = -lv[:, :problem.nx]
            if is_action:
                _, pullback, (xx, yy) = jax.vjp(objective, x0, has_aux=True)
                gradients = jax.lax.map(lambda seed: pullback(seed)[0], jnp.eye(problem.nu, dtype=x0.dtype))
                return jnp.moveaxis(gradients, 0, 1), xx, yy
            (_, (xx, yy)), gradient = jax.value_and_grad(objective, has_aux=True)(x0)
            return gradient, xx, yy
        values = arrays
        gradient, x, y = (np.asarray(v, dtype=float) for v in jax.jit(evaluate)(*values))
        absolute = np.linalg.norm(gradient - fd_half, axis=norm_axes)
        relative = absolute / denominator
        row = dict(sweep=sweep, forward_iterations=forward, refinement_iterations=refinement,
                   batch_size=args.batch_size, horizon=args.horizon, dtype=args.dtype,
                   nonfinite_count=int(np.count_nonzero(~np.isfinite(relative))),
                   seconds=time.time()-point_start,
                   **statistics(relative, "relative_l2"), **statistics(absolute, "absolute_l2"),
                   **statistics(np.abs(gradient-fd_half), "component_absolute"),
                   **statistics(np.linalg.norm(x-xref, axis=1), "primal_l2"),
                   **statistics(np.linalg.norm(y-yref, axis=1), "dual_l2"))
        extra_arrays = {}
        if is_action:
            action_absolute = np.linalg.norm(gradient - fd_half, axis=2)
            action_relative = action_absolute / np.maximum(np.linalg.norm(fd_half, axis=2), 1e-12)
            extra_arrays = dict(jacobian=gradient, action_absolute_l2=action_absolute,
                                action_relative_l2=action_relative)
            row.update(statistics(action_relative, "action_relative_l2"))
            row.update(statistics(action_absolute, "action_absolute_l2"))
            for action in range(problem.nu):
                action_rows.append(dict(forward_iterations=forward, refinement_iterations=refinement,
                                        action=action, batch_size=args.batch_size,
                                        **statistics(action_relative[:, action], "relative_l2"),
                                        **statistics(action_absolute[:, action], "absolute_l2")))
            with (out / "per_action_summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(action_rows[0]))
                writer.writeheader()
                writer.writerows(action_rows)
        np.savez_compressed(out / f"{sweep}_f{forward}_r{refinement}.npz", gradient=gradient,
                            absolute_l2=absolute, relative_l2=relative, x=x, y=y, **extra_arrays)
        rows.append(row)
        with (out / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        metadata.update(completed_points=number, total_points=len(configurations), solver_settings=asdict(settings))
        save_metadata()
        print(f"  relative norm error median={row['relative_l2_median']:.6g}, mean={row['relative_l2_mean']:.6g}, max={row['relative_l2_max']:.6g}; {row['seconds']:.1f}s", flush=True)
        del solver, evaluate
        jax.clear_caches()
    metadata.update(status="complete", elapsed_seconds=time.time()-start)
    save_metadata()
    print(f"COMPLETE: {out}", flush=True)


if __name__ == "__main__":
    main()
