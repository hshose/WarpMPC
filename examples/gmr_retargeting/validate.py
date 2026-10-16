#!/usr/bin/env python3
"""Local parity sweep against the unmodified GMR/Mink/DAQP implementation."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-gmr-cache")
import jax
import jax.numpy as jnp
import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from examples.gmr_retargeting.model import RetargetModel, SolverSettings, integrate, load_gmr
from examples.gmr_retargeting.reporting import provenance, source_revision, write_csv, write_json


def integers(text):
    return [int(x) for x in text.split(",")]


def source_hashes(gmr_root):
    paths = [
        gmr_root / "general_motion_retargeting/motion_retarget.py",
        gmr_root / "general_motion_retargeting/ik_configs/smplx_to_g1.json",
        gmr_root / "assets/unitree_g1/g1_mocap_29dof.xml",
        pathlib.Path(__file__),
        pathlib.Path(__file__).with_name("model.py"),
        pathlib.Path(__file__).with_name("chunks.py"),
        pathlib.Path(__file__).with_name("async_chunks.py"),
    ]
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def synthetic_frames(model, count=64, stress=False):
    """Explicit model-based smoke fixture; never presented as AMASS data."""
    gmr, m = model.gmr, model.mj
    frames = []
    for t in np.linspace(0, 2 * np.pi, count):
        q = m.qpos0.copy()
        q[:3] += [0.12 * np.sin(t), 0.04 * np.cos(t), 0.025 * np.sin(2 * t)]
        q[3:7] = Rotation.from_rotvec([0.04 * np.sin(t), 0.03, 0.1 * np.sin(t)]).as_quat(
            scalar_first=True
        )
        q[7:] = np.clip(
            0.15 * np.sin(t + np.arange(m.nv - 6) * 0.6), model.lower + 0.02, model.upper - 0.02
        )
        fk = np.asarray(model.fk_function(q)).reshape(-1, 7)
        source = {}
        scaled = {}
        for name, (human, *_rest) in gmr.ik_match_table1.items():
            k = model.names.index(name)
            rot = Rotation.from_quat(fk[k, 3:], scalar_first=True)
            scaled[human] = (
                fk[k, :3] - rot.apply(gmr.pos_offsets1[human]),
                (rot * gmr.rot_offsets1[human].inv()).as_quat(scalar_first=True),
            )
        root = gmr.human_root_name
        root_pos = scaled[root][0] / gmr.human_scale_table[root]
        for name, (pos, quat) in scaled.items():
            p = (
                root_pos
                if name == root
                else root_pos + (pos - scaled[root][0]) / gmr.human_scale_table[name]
            )
            if stress and name != root:
                # Deliberately inconsistent, partly unreachable task targets.
                k = model.human_names.index(name)
                p = p + 0.18 * np.array([np.sin(t + k), np.cos(2 * t + k), np.sin(t / 2 + k)])
                quat = (
                    Rotation.from_quat(quat, scalar_first=True)
                    * Rotation.from_rotvec(0.35 * np.array([np.sin(t + k), np.cos(t + k), 0.0]))
                ).as_quat(scalar_first=True)
            source[name] = (p, quat)
        frames.append(source)
    return frames


def check_math(model, frames):
    """Compare FK, analytic manifold Jacobians and raw QPs to Mink in float64."""
    gmr = model.gmr
    worst = dict(
        fk_position=0.0,
        fk_rotation=0.0,
        error=0.0,
        jacobian=0.0,
        hessian_relative=0.0,
        gradient_relative=0.0,
        integration=0.0,
    )
    rng = np.random.default_rng(42)
    for i in sorted(set([0, len(frames) // 2, len(frames) - 1])):
        gmr.update_targets(frames[i])
        q = model.mj.qpos0.copy()
        q[:3] += rng.normal(0, 0.1, 3)
        q[3:7] = Rotation.from_rotvec(rng.normal(0, 0.25, 3)).as_quat(scalar_first=True)
        q[7:] = rng.uniform(model.lower, model.upper) * 0.3
        gmr.configuration.update(q)
        fk = np.asarray(model.fk_function(q)).reshape(-1, 7)
        for k, b in enumerate(model.body_ids):
            worst["fk_position"] = max(
                worst["fk_position"],
                float(np.max(np.abs(fk[k, :3] - gmr.configuration.data.xpos[b]))),
            )
            dot = abs(float(fk[k, 3:] @ gmr.configuration.data.xquat[b]))
            worst["fk_rotation"] = max(worst["fk_rotation"], 1 - min(dot, 1.0))
        for stage, tasks in enumerate((gmr.tasks1, gmr.tasks2)):
            target = model.targets()[stage]
            e = np.asarray(model.error_function(q, target)).reshape(-1, 6)
            J = np.asarray(model.jacobian_function(q, target)).reshape(-1, 6, model.nv)
            for task in tasks:
                k = model.names.index(task.frame_name)
                worst["error"] = max(
                    worst["error"],
                    float(np.max(np.abs(e[k] - task.compute_error(gmr.configuration)))),
                )
                worst["jacobian"] = max(
                    worst["jacobian"],
                    float(np.max(np.abs(J[k] - task.compute_jacobian(gmr.configuration)))),
                )
            raw = mink.build_ik(
                gmr.configuration, tasks, model.mj.opt.timestep, gmr.damping, gmr.ik_limits
            )
            p, c = model.qp_function(q, target, model.weights[stage])
            expected = raw.P[model.rows, model.cols]
            worst["hessian_relative"] = max(
                worst["hessian_relative"],
                float(
                    np.max(np.abs(np.asarray(p).ravel() - expected))
                    / max(1, np.max(np.abs(expected)))
                ),
            )
            worst["gradient_relative"] = max(
                worst["gradient_relative"],
                float(
                    np.max(np.abs(np.asarray(c).ravel() - raw.q)) / max(1, np.max(np.abs(raw.q)))
                ),
            )
        dx = rng.normal(0, 0.1, model.nv)
        expected = q.copy()
        mujoco.mj_integratePos(model.mj, expected, dx, 1.0)
        actual = np.asarray(
            integrate(
                jnp.asarray(q[None], dtype=jnp.float32), jnp.asarray(dx[None], dtype=jnp.float32)
            )
        )[0]
        worst["integration"] = max(worst["integration"], float(np.max(np.abs(actual - expected))))
    if max(worst.values()) > 2e-5:
        raise AssertionError(f"GMR mathematical parity failed: {worst}")
    return worst


def reference(model, frames, starts):
    gmr = model.gmr
    targets, seeds, result, durations = [], [], [], []
    for i, frame in enumerate(frames):
        if i in starts:
            gmr.configuration.update(model.mj.qpos0)
        seeds.append(gmr.configuration.q.copy())
        begin = time.perf_counter()
        result.append(gmr.retarget(frame))
        durations.append(time.perf_counter() - begin)
        targets.append(model.targets())
        if (i + 1) % 10000 == 0:
            print(f"Original GMR/DAQP reference: {i+1}/{len(frames)} frames", flush=True)
    return np.array(targets), np.array(seeds), np.array(result), durations


def matched_budget_reference(model, frames, seeds, sqp, adaptive):
    """DAQP with the same seed and exact same number/stopping rule of SQP steps."""
    gmr = model.gmr
    result = []
    for frame, seed in zip(frames, seeds):
        gmr.configuration.update(seed)
        gmr.update_targets(frame)
        for tasks in (gmr.tasks1, gmr.tasks2):
            error = lambda: np.linalg.norm(
                np.concatenate([t.compute_error(gmr.configuration) for t in tasks])
            )
            previous = error() if adaptive else 0
            for _ in range(sqp):
                velocity = mink.solve_ik(
                    gmr.configuration,
                    tasks,
                    model.mj.opt.timestep,
                    "daqp",
                    gmr.damping,
                    gmr.ik_limits,
                )
                gmr.configuration.integrate_inplace(velocity, model.mj.opt.timestep)
                if adaptive:
                    current = error()
                    if previous - current <= 0.001:
                        break
                    previous = current
        result.append(gmr.configuration.q.copy())
    return np.array(result)


def matched_chunk_reference(model, frames, starts, initial_sqp, warm_sqp, adaptive=False):
    """Serial DAQP with independent pose warm starts and the chunk schedule."""
    gmr = model.gmr
    result = []
    for i, frame in enumerate(frames):
        if i in starts:
            gmr.configuration.update(model.mj.qpos0)
        gmr.update_targets(frame)
        for tasks in (gmr.tasks1, gmr.tasks2):
            error = lambda: np.linalg.norm(
                np.concatenate([t.compute_error(gmr.configuration) for t in tasks])
            )
            previous = error() if adaptive else 0.0
            for _ in range(initial_sqp if i in starts else warm_sqp):
                velocity = mink.solve_ik(
                    gmr.configuration,
                    tasks,
                    model.mj.opt.timestep,
                    solver="daqp",
                    damping=gmr.damping,
                    limits=gmr.ik_limits,
                )
                gmr.configuration.integrate_inplace(velocity, model.mj.opt.timestep)
                if adaptive:
                    current = error()
                    if previous - current <= 0.001:
                        break
                    previous = current
        result.append(gmr.configuration.q.copy())
    return np.array(result)


def metrics(model, actual, expected):
    if not np.isfinite(actual).all():
        raise FloatingPointError("Nonfinite GPU retargeting output")
    diffs = []
    pos = []
    for q, ref in zip(actual, expected):
        d = np.empty(model.nv)
        mujoco.mj_differentiatePos(model.mj, d, 1.0, ref, q)
        diffs.append(d)
        a = np.asarray(model.fk_function(q)).reshape(-1, 7)[:, :3]
        b = np.asarray(model.fk_function(ref)).reshape(-1, 7)[:, :3]
        pos.append(np.linalg.norm(a - b, axis=1))
    diffs, pos = np.array(diffs), np.array(pos)
    bound = np.maximum(
        model.lower - actual[:, model.qindices], actual[:, model.qindices] - model.upper
    )
    per_frame = np.max(np.abs(diffs), axis=1)
    return {
        "q_tangent_rms": float(np.sqrt(np.mean(diffs**2))),
        "q_tangent_max": float(np.max(per_frame)),
        "q_tangent_p95": float(np.quantile(per_frame, 0.95)),
        "fraction_q_within_1e2": float(np.mean(per_frame <= 1e-2)),
        "fraction_q_within_1e3": float(np.mean(per_frame <= 1e-3)),
        "joint_rms_rad": float(np.sqrt(np.mean(diffs[:, 6:] ** 2))),
        "body_position_rms_m": float(np.sqrt(np.mean(pos**2))),
        "body_position_max_m": float(np.max(pos)),
        "joint_limit_violation_rad": float(max(0, np.max(bound))),
    }


def task_quality(model, actual, expected, targets):
    """Final task costs and geometric tracking errors, separate from pose parity."""
    result = {}
    for label, configurations in (("", actual), ("reference_", expected)):
        fk = np.array([np.asarray(model.fk_function(q)).reshape(-1, 7) for q in configurations])
        for stage in range(2):
            error = np.array(
                [
                    np.asarray(model.error_function(q, t[stage])).ravel()
                    for q, t in zip(configurations, targets)
                ]
            )
            cost = np.sum((error * model.weights[stage]) ** 2, axis=1)
            prefix = f"{label}stage{stage+1}_"
            result[prefix + "weighted_cost_mean"] = float(np.mean(cost))
            task_weights = model.weights[stage].reshape(-1, 6)
            positions = np.any(task_weights[:, :3] != 0, axis=1)
            rotations = np.any(task_weights[:, 3:] != 0, axis=1)
            goal = targets[:, stage].reshape(len(targets), -1, 7)
            result[prefix + "tracking_position_rms_m"] = float(
                np.sqrt(
                    np.mean(np.sum((fk[:, positions, :3] - goal[:, positions, :3]) ** 2, axis=-1))
                )
            )
            result[prefix + "tracking_rotation_rms_rad"] = float(
                np.sqrt(
                    np.mean(
                        np.sum(error.reshape(len(error), -1, 6)[:, rotations, 3:] ** 2, axis=-1)
                    )
                )
            )
    for stage in (1, 2):
        name = f"stage{stage}_weighted_cost_mean"
        result[f"stage{stage}_cost_relative_difference"] = (
            result[name] - result["reference_" + name]
        ) / max(result["reference_" + name], 1e-12)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gmr-root", type=pathlib.Path, default=ROOT / "resources/GMR")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data", type=pathlib.Path, help="Prepared human motion NPZ from prepare.py"
    )
    source.add_argument("--synthetic", action="store_true", help="Explicit non-AMASS smoke fixture")
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--synthetic-stress", action="store_true")
    p.add_argument("--output-dir", type=pathlib.Path, required=True)
    p.add_argument("--sqp-iterations", type=integers, default=[1, 2, 3, 5])
    p.add_argument("--admm-iterations", type=integers, default=[25, 50])
    p.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Sequential chunks; 0 retains the single-frame diagnostic sweep",
    )
    p.add_argument("--initial-sqp-iterations", type=int, default=10)
    p.add_argument("--rho", type=float, default=0.1)
    p.add_argument("--backend", choices=["warp", "jax"], default="warp")
    p.add_argument("--derivative-backend", choices=["warp", "jax"], default="warp")
    p.add_argument("--no-scaling", action="store_true")
    p.add_argument("--adaptive-stopping", action="store_true")
    p.add_argument("--scheduler", choices=["frame", "independent"], default="frame")
    p.add_argument("--skip-rollout", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()
    if args.synthetic_stress and not args.synthetic:
        p.error("--synthetic-stress requires --synthetic")
    if args.scheduler == "independent" and (not args.chunk_size or not args.adaptive_stopping):
        p.error("--scheduler independent requires --chunk-size and --adaptive-stopping")
    if not args.allow_cpu and jax.devices()[0].platform != "gpu":
        raise RuntimeError(
            "GPU required; CPU results must be explicitly requested with --allow-cpu"
        )
    height = None
    if args.data:
        from examples.gmr_retargeting.prepare import load_frames

        frames, starts, height, source_info = load_frames(args.data, args.frames)
    model = RetargetModel(load_gmr(args.gmr_root, height))
    if args.synthetic:
        frames, starts = synthetic_frames(model, args.frames, args.synthetic_stress), {0}
        source_info = {
            "dataset": "synthetic_stress" if args.synthetic_stress else "synthetic_robot_fixture",
            "amass": False,
        }
    if args.chunk_size:
        from examples.gmr_retargeting.chunks import chunk_indices, compile_chunks

        if args.scheduler == "independent":
            from examples.gmr_retargeting.async_chunks import compile_async_chunks as compile_chunks
        selected = chunk_indices(len(frames), starts, args.chunk_size)
        dropped = len(frames) - selected.size
        frames = [frames[i] for i in selected.ravel()]
        starts = set(range(0, len(frames), args.chunk_size))
        source_info = source_info | {"dropped_tail_frames": dropped}
    math_check = check_math(model, frames)
    print("Math parity:", math_check, flush=True)
    targets, seeds, cpu, durations = reference(model, frames, starts)
    raw_human = model.pack_human(frames)
    gpu_targets = np.asarray(jax.jit(model.preprocess)(jnp.asarray(raw_human, dtype=jnp.float32)))
    np.testing.assert_allclose(gpu_targets, targets, atol=1e-6, rtol=1e-6)
    math_check["target_preprocessing"] = float(np.max(np.abs(gpu_targets - targets)))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "reference.npz",
        targets=targets,
        seeds=seeds,
        qpos=cpu,
        starts=np.array(sorted(starts)),
        weights=model.weights,
        human=raw_human,
    )
    warm_durations = [duration for i, duration in enumerate(durations) if i not in starts]
    metadata = {
        "source": source_info,
        "frames": len(frames),
        "height": height,
        "math_parity": math_check,
        "provenance": provenance(),
        "source_sha256": source_hashes(args.gmr_root),
        "git_commit": source_revision(ROOT),
        "cpu_frames_per_s": len(frames) / sum(durations),
        "cpu_seconds": sum(durations),
        "cpu_warm_frames_per_s": (
            len(warm_durations) / sum(warm_durations) if warm_durations else None
        ),
        "cpu_timing_scope": "Original GMR.retarget including target preprocessing, both adaptive IK stages",
        "gmr_max_iterations_per_stage": 11,
        "dtype": "float32",
    }
    if args.chunk_size:
        metadata.update(
            chunk_size=args.chunk_size,
            initial_sqp_iterations=args.initial_sqp_iterations,
            chunks=len(starts),
            scheduler=args.scheduler,
        )
    write_json(out / "metadata.json", metadata)
    rows = []
    q0, t = jnp.asarray(seeds, dtype=jnp.float32), jnp.asarray(targets, dtype=jnp.float32)
    if args.chunk_size:
        bank = jnp.asarray(
            raw_human.reshape(len(starts), args.chunk_size, *raw_human.shape[1:]), dtype=jnp.float32
        )
        lanes = jnp.arange(len(starts), dtype=jnp.int32)
        cold_mask = np.isin(np.arange(len(frames)), list(starts))
        for sqp in args.sqp_iterations:
            # This exact policy is the original GMR reference already evaluated above.
            matched = (
                cpu
                if args.adaptive_stopping and args.initial_sqp_iterations == sqp == 11
                else matched_chunk_reference(
                    model, frames, starts, args.initial_sqp_iterations, sqp, args.adaptive_stopping
                )
            )
            for admm in args.admm_iterations:
                settings = SolverSettings(
                    sqp,
                    admm,
                    args.rho,
                    backend=args.backend,
                    derivative_backend=args.derivative_backend,
                    scaled=not args.no_scaling,
                    adaptive_stopping=args.adaptive_stopping,
                )
                begin = time.perf_counter()
                run = compile_chunks(model, settings, args.initial_sqp_iterations, collect=True)
                result = jax.block_until_ready(run(bank, lanes))
                trajectory, finite, violation = result[:3]
                gpu = np.asarray(trajectory).reshape(len(frames), model.nq)
                if not bool(finite):
                    raise FloatingPointError("Nonfinite chunk trajectory")
                prefix = dataclasses.asdict(settings) | {
                    "compile_and_first_run_s": time.perf_counter() - begin,
                    "frames": len(frames),
                    "chunk_size": args.chunk_size,
                    "initial_sqp_iterations": args.initial_sqp_iterations,
                    "scheduler": args.scheduler,
                }
                if args.scheduler == "independent":
                    stats = {key: np.asarray(value).tolist() for key, value in result[3].items()}
                    write_json(out / f"schedule_sqp{sqp}_admm{admm}.json", stats)
                    prefix.update(
                        {key: value for key, value in stats.items() if not isinstance(value, list)}
                    )
                for mode, reference_q in (
                    ("chunk_vs_original_gmr", cpu),
                    ("chunk_vs_matched_daqp", matched),
                ):
                    for suffix, mask in (
                        ("", np.ones(len(frames), dtype=bool)),
                        ("_cold", cold_mask),
                        ("_warm", ~cold_mask),
                    ):
                        if not mask.any():
                            continue
                        row = prefix | {"mode": mode + suffix, "evaluated_frames": int(mask.sum())}
                        row.update(metrics(model, gpu[mask], reference_q[mask]))
                        if not suffix:
                            row.update(task_quality(model, gpu, reference_q, targets))
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                policy = "adaptive_" if args.adaptive_stopping else ""
                if args.scheduler == "independent":
                    policy = "independent_" + policy
                np.savez_compressed(
                    out
                    / f"chunks_{policy}initial{args.initial_sqp_iterations}_sqp{sqp}_admm{admm}.npz",
                    gpu=gpu,
                    matched_daqp=matched,
                    original_gmr=cpu,
                )
                write_csv(out / "accuracy.csv", rows)
        print(
            f"Saved chunk accuracy in {out}; original GMR {metadata['cpu_frames_per_s']:.1f} frames/s",
            flush=True,
        )
        return
    for sqp in args.sqp_iterations:
        matched = matched_budget_reference(model, frames, seeds, sqp, args.adaptive_stopping)
        for admm in args.admm_iterations:
            settings = SolverSettings(
                sqp,
                admm,
                args.rho,
                backend=args.backend,
                derivative_backend=args.derivative_backend,
                scaled=not args.no_scaling,
                adaptive_stopping=args.adaptive_stopping,
            )
            begin = time.perf_counter()
            solve, _ = model.compile(settings)
            gpu = np.asarray(solve(q0, t).block_until_ready())
            compile_s = time.perf_counter() - begin
            prefix = dataclasses.asdict(settings) | {"compile_s": compile_s, "frames": len(frames)}
            matched_row = prefix | {"mode": "matched_cpu_budget"} | metrics(model, gpu, matched)
            rows.append(matched_row)
            print(json.dumps(matched_row), flush=True)
            for label, mask in [
                ("shared_previous_cpu", np.ones(len(frames), dtype=bool)),
                ("shared_previous_cpu_warm", ~np.isin(np.arange(len(frames)), list(starts))),
            ]:
                if mask.any():
                    row = prefix | {"mode": label} | metrics(model, gpu[mask], cpu[mask])
                    rows.append(row)
                    print(json.dumps(row), flush=True)
            if not args.skip_rollout:
                reset = jnp.asarray(np.isin(np.arange(len(frames)), list(starts)))
                initial = jnp.asarray(model.mj.qpos0[None], dtype=jnp.float32)

                def rollout():
                    def step(state, inputs):
                        target, restart = inputs
                        state = solve(jnp.where(restart, initial, state), target[None])
                        return state, state[0]

                    return jax.lax.scan(step, initial, (t, reset))[1]

                rolled = np.asarray(jax.jit(rollout)().block_until_ready())
                row = prefix | {"mode": "independent_sequence"} | metrics(model, rolled, cpu)
                rows.append(row)
                print(json.dumps(row), flush=True)
                np.savez_compressed(
                    out / f"qpos_sqp{sqp}_admm{admm}.npz", shared_seed=gpu, rollout=rolled
                )
            write_csv(out / "accuracy.csv", rows)
    print(
        f"Saved {out}/accuracy.csv; CPU original GMR {metadata['cpu_frames_per_s']:.1f} frames/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
