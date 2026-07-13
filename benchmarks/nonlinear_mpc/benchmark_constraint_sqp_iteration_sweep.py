#!/usr/bin/env python3
"""SQP-iteration sweep for constrained nonlinear MPC closed-loop benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[2]
CARTPOLE_JAX_EVALUATOR = ROOT / "benchmarks" / "nonlinear_mpc" / "evaluate_cartpole_quadratic_mpc.py"
CARTPOLE_MPX_EVALUATOR = ROOT / "benchmarks" / "nonlinear_mpc" / "evaluate_mpx_cartpole_quadratic_mpc.py"
CRAZYFLIE_OBSTACLE_EVALUATOR = ROOT / "benchmarks" / "nonlinear_mpc" / "evaluate_crazyflie_obstacle_mpc.py"


ADDITIONAL_CONSTRAINT_FIELDS = [
    "rollout_constraint_satisfaction",
    "obstacle_rollout_constraint_satisfaction",
    "rail_rollout_constraint_satisfaction",
    "input_constraint_satisfaction",
    "input_rollout_constraint_satisfaction",
    "motor_rollout_constraint_satisfaction",
    "rollout_violation_rate",
    "obstacle_rollout_violation_rate",
    "rail_rollout_violation_rate",
    "input_violation_rate",
    "input_rollout_violation_rate",
    "motor_rollout_violation_rate",
    "combined_rollout_violation_severity_mean",
    "combined_rollout_violation_severity_p95",
    "combined_rollout_violation_severity_p99",
    "combined_rollout_violation_severity_max",
    "obstacle_violation_mean",
    "obstacle_violation_p95",
    "obstacle_violation_p99",
    "obstacle_violation_max",
    "obstacle_rollout_violation_severity_mean",
    "obstacle_rollout_violation_severity_p95",
    "obstacle_rollout_violation_severity_p99",
    "obstacle_rollout_violation_severity_max",
    "rail_violation_mean",
    "rail_violation_p95",
    "rail_violation_p99",
    "rail_violation_max",
    "rail_rollout_violation_severity_mean",
    "rail_rollout_violation_severity_p95",
    "rail_rollout_violation_severity_p99",
    "rail_rollout_violation_severity_max",
    "input_violation_mean",
    "input_violation_p95",
    "input_violation_p99",
    "input_violation_max",
    "input_rollout_violation_severity_mean",
    "input_rollout_violation_severity_p95",
    "input_rollout_violation_severity_p99",
    "input_rollout_violation_severity_max",
    "motor_violation_mean",
    "motor_violation_p95",
    "motor_violation_p99",
    "motor_rollout_violation_severity_mean",
    "motor_rollout_violation_severity_p95",
    "motor_rollout_violation_severity_p99",
    "motor_rollout_violation_severity_max",
]


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_int_csv(text: str) -> list[int]:
    return [int(item) for item in _parse_csv(text)]


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_float(summary: dict[str, object], *keys: str, default: float = float("nan")) -> float:
    for key in keys:
        value = _safe_float(summary.get(key))
        if not math.isnan(value):
            return value
    return default


def _nan_min(*values: float) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return min(valid) if valid else float("nan")


def _nan_max(*values: float) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return max(valid) if valid else float("nan")


def _copy_constraint_metrics(out: dict[str, object], summary: dict[str, object], prefix: str) -> None:
    out[f"{prefix}_violation_rate"] = _first_float(
        summary,
        f"{prefix}_violation_rate_gt_1e-6",
        f"{prefix}_violation_rate",
    )
    out[f"{prefix}_constraint_satisfaction"] = _first_float(
        summary,
        f"{prefix}_constraint_satisfaction_gt_1e-6",
    )
    out[f"{prefix}_rollout_violation_rate"] = _first_float(
        summary,
        f"{prefix}_rollout_violation_rate_gt_1e-6",
    )
    out[f"{prefix}_rollout_constraint_satisfaction"] = _first_float(
        summary,
        f"{prefix}_rollout_constraint_satisfaction_gt_1e-6",
    )
    for suffix in (
        "violation_mean",
        "violation_p95",
        "violation_p99",
        "violation_max",
        "rollout_violation_severity_mean",
        "rollout_violation_severity_p95",
        "rollout_violation_severity_p99",
        "rollout_violation_severity_max",
    ):
        key = f"{prefix}_{suffix}"
        if key in summary:
            out[key] = summary[key]


def _common_row(summary: dict[str, object], *, wall_s: float, status: str, error: str = "") -> dict[str, object]:
    system = str(summary.get("system", ""))
    batch_size = int(summary.get("batch_size", 0) or 0)
    steps = int(summary.get("rollout_steps", summary.get("sim_steps", 0)) or 0)
    sqp_iterations = int(summary.get("sqp_iterations", 0) or 0)
    elapsed_s = _safe_float(summary.get("elapsed_s"))
    optimization_s_per_sim_step = elapsed_s / steps if steps > 0 else float("nan")
    optimization_us_per_rollout_step = (
        1.0e6 * elapsed_s / (batch_size * steps) if batch_size > 0 and steps > 0 else float("nan")
    )
    rollouts_per_s = batch_size / elapsed_s if batch_size > 0 and elapsed_s > 0 else float("nan")
    reported_constraint_satisfaction = summary.get(
        "reported_constraint_satisfaction",
        summary.get("state_rollout_constraint_satisfaction", summary.get("rollout_constraint_satisfaction", "")),
    )
    reported_constraint_satisfaction_value = _safe_float(reported_constraint_satisfaction)
    success_rate_value = _safe_float(summary.get("success_rate"))
    row = {
        "system": system,
        "solver": summary.get("solver", summary.get("qp_solver", "")),
        "sqp_iterations": sqp_iterations,
        "osqp_max_iter": summary.get("max_iter", summary.get("osqp_max_iter", "")),
        "obstacle_constraint_scale": summary.get("obstacle_constraint_scale", ""),
        "osqp_scaling": summary.get("osqp_scaling", ""),
        "osqp_rho_is_vec": summary.get("osqp_rho_is_vec", ""),
        "line_search_step_min": summary.get("line_search_step_min", ""),
        "trajectory_initialization": summary.get("trajectory_initialization", ""),
        "batch_size": batch_size,
        "steps": steps,
        "status": status,
        "dtype": summary.get("dtype", ""),
        "success_rate": summary.get("success_rate", ""),
        "success_rate_percent": 100.0 * success_rate_value if not math.isnan(success_rate_value) else "",
        "reported_constraint_satisfaction": reported_constraint_satisfaction,
        "reported_constraint_satisfaction_percent": 100.0 * reported_constraint_satisfaction_value
        if not math.isnan(reported_constraint_satisfaction_value)
        else "",
        "state_constraint_satisfaction": summary.get("state_constraint_satisfaction", ""),
        "state_rollout_constraint_satisfaction": summary.get("state_rollout_constraint_satisfaction", ""),
        "constraint_satisfaction": summary.get("constraint_satisfaction", ""),
        "obstacle_constraint_satisfaction": summary.get("obstacle_constraint_satisfaction", ""),
        "rail_constraint_satisfaction": summary.get("rail_constraint_satisfaction", ""),
        "motor_constraint_satisfaction": summary.get("motor_constraint_satisfaction", ""),
        "violation_rate": summary.get("violation_rate", ""),
        "obstacle_violation_rate": summary.get("obstacle_violation_rate", ""),
        "rail_violation_rate": summary.get("rail_violation_rate", ""),
        "motor_violation_rate": summary.get("motor_violation_rate", ""),
        "max_violation": summary.get("max_violation", ""),
        "obstacle_penetration_max": summary.get("obstacle_penetration_max", ""),
        "motor_violation_max": summary.get("motor_violation_max", ""),
        "final_position_rms": summary.get("final_position_rms", ""),
        "final_state_cost_mean": summary.get("final_state_cost_mean", ""),
        "return_mean": summary.get("return_mean", ""),
        "finite_rate": summary.get("finite_rate", ""),
        "elapsed_s": elapsed_s,
        "optimization_s_per_sim_step": optimization_s_per_sim_step,
        "optimization_ms_per_sim_step": 1.0e3 * optimization_s_per_sim_step,
        "optimization_us_per_rollout_step": optimization_us_per_rollout_step,
        "rollouts_per_s": rollouts_per_s,
        "rollout_steps_per_s": summary.get("closed_loop_steps_per_s", summary.get("rti_steps_per_s", "")),
        "sqp_iterations_per_s": summary.get("sqp_iterations_per_s", ""),
        "closed_loop_steps_per_s": summary.get("closed_loop_steps_per_s", summary.get("rti_steps_per_s", "")),
        "wall_s": wall_s,
        "summary_json": summary.get("summary_json", ""),
        "log_path": summary.get("log_path", ""),
        "plot_path": summary.get("plot_path", ""),
        "position_plot_path": summary.get("position_plot_path", ""),
        "clearance_plot_path": summary.get("clearance_plot_path", ""),
        "error": error,
    }
    for key in ADDITIONAL_CONSTRAINT_FIELDS:
        row[key] = summary.get(key, "")
    return row


def _normalize_summary(system: str, solver: str, summary: dict[str, object]) -> dict[str, object]:
    out = dict(summary)
    out["system"] = system
    out["solver"] = solver
    if system == "cartpole":
        success = _safe_float(summary.get("rollout_success_rate"))
        _copy_constraint_metrics(out, summary, "rail")
        _copy_constraint_metrics(out, summary, "input")
        rail_violation = _first_float(summary, "rail_violation_rate_gt_1e-6", "rail_violation_rate")
        input_violation = _first_float(summary, "input_violation_rate_gt_1e-6", default=0.0)
        rail_sat = _first_float(
            summary,
            "rail_constraint_satisfaction_gt_1e-6",
            default=1.0 - rail_violation if not math.isnan(rail_violation) else float("nan"),
        )
        input_sat = _first_float(
            summary,
            "input_constraint_satisfaction_gt_1e-6",
            default=1.0 - input_violation if not math.isnan(input_violation) else float("nan"),
        )
        combined_violation = _first_float(
            summary,
            "combined_violation_rate_gt_1e-6",
            default=_nan_max(rail_violation, input_violation),
        )
        combined_sat = _first_float(
            summary,
            "combined_constraint_satisfaction_gt_1e-6",
            default=_nan_min(rail_sat, input_sat),
        )
        out["success_rate"] = success
        out["rail_violation_rate"] = rail_violation
        out["input_violation_rate"] = input_violation
        out["violation_rate"] = combined_violation
        out["rail_constraint_satisfaction"] = rail_sat
        out["input_constraint_satisfaction"] = input_sat
        out["constraint_satisfaction"] = combined_sat
        out["state_constraint_satisfaction"] = rail_sat
        out["state_rollout_constraint_satisfaction"] = _first_float(
            summary,
            "rail_rollout_constraint_satisfaction_gt_1e-6",
        )
        out["reported_constraint_satisfaction"] = out["state_rollout_constraint_satisfaction"]
        out["rollout_violation_rate"] = _first_float(summary, "combined_rollout_violation_rate_gt_1e-6")
        out["rollout_constraint_satisfaction"] = _first_float(
            summary,
            "combined_rollout_constraint_satisfaction_gt_1e-6",
        )
    elif system == "crazyflie_obstacle":
        success = _safe_float(summary.get("tracking_success_rate_10cm"))
        _copy_constraint_metrics(out, summary, "obstacle")
        _copy_constraint_metrics(out, summary, "motor")
        _copy_constraint_metrics(out, summary, "input")
        obstacle_sat = _first_float(summary, "obstacle_constraint_satisfaction_gt_1e-6")
        obstacle_violation = _first_float(summary, "obstacle_violation_rate_gt_1e-6")
        motor_violation = _first_float(summary, "motor_violation_rate_gt_1e-6")
        motor_sat = _first_float(
            summary,
            "motor_constraint_satisfaction_gt_1e-6",
            default=1.0 - motor_violation if not math.isnan(motor_violation) else float("nan"),
        )
        input_violation = _first_float(summary, "input_violation_rate_gt_1e-6", default=motor_violation)
        input_sat = _first_float(summary, "input_constraint_satisfaction_gt_1e-6", default=motor_sat)
        out["success_rate"] = success
        out["obstacle_constraint_satisfaction"] = obstacle_sat
        out["obstacle_violation_rate"] = obstacle_violation
        out["motor_violation_rate"] = motor_violation
        out["motor_constraint_satisfaction"] = motor_sat
        out["input_violation_rate"] = input_violation
        out["input_constraint_satisfaction"] = input_sat
        out["state_constraint_satisfaction"] = obstacle_sat
        out["state_rollout_constraint_satisfaction"] = _first_float(
            summary,
            "obstacle_rollout_constraint_satisfaction_gt_1e-6",
        )
        out["reported_constraint_satisfaction"] = out["state_rollout_constraint_satisfaction"]
        out["violation_rate"] = _first_float(
            summary,
            "combined_violation_rate_gt_1e-6",
            default=_nan_max(obstacle_violation, motor_violation),
        )
        out["constraint_satisfaction"] = _first_float(
            summary,
            "combined_constraint_satisfaction_gt_1e-6",
            default=_nan_min(obstacle_sat, motor_sat),
        )
        out["rollout_violation_rate"] = _first_float(summary, "combined_rollout_violation_rate_gt_1e-6")
        out["rollout_constraint_satisfaction"] = _first_float(
            summary,
            "combined_rollout_constraint_satisfaction_gt_1e-6",
        )
    return out


def _base_output_prefix(args: argparse.Namespace, system: str, solver: str, sqp_iterations: int) -> pathlib.Path:
    return args.output_dir / f"{system}_{solver}_sqp{sqp_iterations}_b{args.batch_size}_seed{args.seed}"


def _cartpole_cmd(args: argparse.Namespace, *, solver: str, sqp_iterations: int, prefix: pathlib.Path) -> list[str]:
    evaluator = CARTPOLE_MPX_EVALUATOR if solver == "mpx" else CARTPOLE_JAX_EVALUATOR
    cmd = [
        sys.executable,
        str(evaluator),
        "--batch-size",
        str(args.batch_size),
        "--horizon-steps",
        str(args.cartpole_horizon_steps),
        "--dt-start",
        str(args.cartpole_dt_start),
        "--dt-growth",
        str(args.cartpole_dt_growth),
        "--sim-time",
        str(args.cartpole_sim_time),
        "--control-dt",
        str(args.cartpole_control_dt),
        "--integrator-substeps",
        str(args.cartpole_integrator_substeps),
        "--sqp-iterations",
        str(sqp_iterations),
        "--dtype",
        args.dtype,
        "--enable-rail-constraint",
        "--noise-scale",
        "0",
        "--process-noise-scale",
        "0,0,0,0",
        "--input-noise-scale",
        "0",
        "--summary-json",
        str(prefix.with_suffix(".json")),
    ]
    if args.cartpole_rollout_steps is not None:
        cmd.extend(["--rollout-steps", str(args.cartpole_rollout_steps)])
    if solver == "jax_osqp":
        cmd.extend(
            [
                "--qp-solver",
                "jax_osqp",
                "--osqp-max-iter",
                str(args.osqp_max_iter),
                "--rho",
                str(args.rho),
                "--sigma",
                str(args.sigma),
                "--alpha",
                str(args.alpha),
                "--qdldl-backend",
                args.qdldl_backend,
                "--qdldl-factor-backend",
                args.qdldl_factor_backend,
                "--qdldl-solve-backend",
                args.qdldl_solve_backend,
                "--segment-budget",
                str(args.cartpole_segment_budget),
                "--segment-strategy",
                args.segment_strategy,
                "--level-scheduled-solve-threshold",
                str(args.level_scheduled_solve_threshold),
            ]
        )
        if args.level_scheduled_solve:
            cmd.append("--level-scheduled-solve")
        if not args.transpose_work:
            cmd.append("--no-transpose-work")
        if not args.segmented:
            cmd.append("--no-segmented")
        if not args.group_repeated_stages:
            cmd.append("--no-group-repeated-stages")
        if args.skip_plots:
            cmd.append("--skip-state-plot")
        else:
            cmd.extend(["--plot-path", str(prefix.with_name(prefix.name + "_states.png"))])
    else:
        cmd.extend(
            [
                "--mpx-solver-mode",
                args.mpx_solver_mode,
                "--mpx-equality-weight",
                str(args.mpx_equality_weight),
                "--mpx-barrier-alpha",
                str(args.cartpole_mpx_barrier_alpha),
                "--mpx-barrier-sigma",
                str(args.cartpole_mpx_barrier_sigma),
                "--mpx-num-alpha",
                str(args.mpx_num_alpha),
            ]
        )
        if not args.mpx_limited_memory:
            cmd.append("--no-mpx-limited-memory")
        if args.skip_plots:
            cmd.append("--skip-state-plot")
        else:
            cmd.extend(["--plot-path", str(prefix.with_name(prefix.name + "_states.png"))])
    return cmd


def _crazyflie_obstacle_cmd(
    args: argparse.Namespace,
    *,
    solver: str,
    sqp_iterations: int,
    prefix: pathlib.Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(CRAZYFLIE_OBSTACLE_EVALUATOR),
        "--solver",
        solver,
        "--batch-size",
        str(args.batch_size),
        "--horizon-steps",
        str(args.crazyflie_horizon_steps),
        "--sim-time",
        str(args.crazyflie_sim_time),
        "--control-dt",
        str(args.crazyflie_control_dt),
        "--sqp-iterations",
        str(sqp_iterations),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--trajectory-initialization",
        args.crazyflie_obstacle_trajectory_initialization,
        "--summary-json",
        str(prefix.with_suffix(".json")),
    ]
    if args.crazyflie_sim_steps is not None:
        cmd.extend(["--sim-steps", str(args.crazyflie_sim_steps)])
    if solver == "jax_osqp":
        cmd.extend(
            [
                "--max-iter",
                str(args.crazyflie_obstacle_osqp_max_iter),
                "--osqp-scaling",
                str(args.crazyflie_obstacle_osqp_scaling),
                "--line-search-step-min",
                str(args.crazyflie_obstacle_line_search_step_min),
                "--rho",
                str(args.rho),
                "--sigma",
                str(args.sigma),
                "--alpha",
                str(args.alpha),
                "--obstacle-constraint-scale",
                str(args.crazyflie_obstacle_constraint_scale),
                "--qdldl-backend",
                args.qdldl_backend,
                "--qdldl-factor-backend",
                args.qdldl_factor_backend,
                "--qdldl-solve-backend",
                args.qdldl_solve_backend,
                "--segment-budget",
                str(args.crazyflie_segment_budget),
                "--segment-strategy",
                args.segment_strategy,
                "--level-scheduled-solve-threshold",
                str(args.level_scheduled_solve_threshold),
            ]
        )
        if args.crazyflie_obstacle_osqp_rho_is_vec:
            cmd.append("--osqp-rho-is-vec")
        else:
            cmd.append("--no-osqp-rho-is-vec")
        if args.level_scheduled_solve:
            cmd.append("--level-scheduled-solve")
        if not args.transpose_work:
            cmd.append("--no-transpose-work")
        if not args.segmented:
            cmd.append("--no-segmented")
        if not args.group_repeated_stages:
            cmd.append("--no-group-repeated-stages")
    else:
        cmd.extend(
            [
                "--mpx-solver-mode",
                args.mpx_solver_mode,
                "--mpx-equality-weight",
                str(args.mpx_equality_weight),
                "--mpx-barrier-alpha",
                str(args.crazyflie_obstacle_mpx_barrier_alpha),
                "--mpx-barrier-sigma",
                str(args.crazyflie_obstacle_mpx_barrier_sigma),
                "--mpx-num-alpha",
                str(args.mpx_num_alpha),
            ]
        )
        if not args.mpx_limited_memory:
            cmd.append("--no-mpx-limited-memory")
    if args.write_npz:
        cmd.extend(["--output-npz", str(prefix.with_suffix(".npz"))])
    else:
        cmd.append("--skip-output-npz")
    if args.skip_plots:
        cmd.append("--skip-plots")
    else:
        cmd.extend(
            [
                "--plot-path",
                str(prefix.with_name(prefix.name + "_states.png")),
                "--position-plot-path",
                str(prefix.with_name(prefix.name + "_positions.png")),
                "--clearance-plot-path",
                str(prefix.with_name(prefix.name + "_clearance.png")),
            ]
        )
    return cmd


def _command(args: argparse.Namespace, system: str, solver: str, sqp_iterations: int, prefix: pathlib.Path) -> list[str]:
    if system == "cartpole":
        return _cartpole_cmd(args, solver=solver, sqp_iterations=sqp_iterations, prefix=prefix)
    if system == "crazyflie_obstacle":
        return _crazyflie_obstacle_cmd(args, solver=solver, sqp_iterations=sqp_iterations, prefix=prefix)
    raise ValueError(f"unsupported system: {system}")


def _sqp_values_for_solver(args: argparse.Namespace, system: str, solver: str) -> list[int]:
    common = _parse_int_csv(args.sqp_iterations)
    if system == "cartpole" and solver == "jax_osqp":
        specific = _parse_int_csv(args.cartpole_jax_osqp_sqp_iterations or args.jax_osqp_sqp_iterations)
    elif system == "cartpole" and solver == "mpx":
        specific = _parse_int_csv(args.cartpole_mpx_sqp_iterations or args.mpx_sqp_iterations)
    elif system == "crazyflie_obstacle" and solver == "jax_osqp":
        specific = _parse_int_csv(args.crazyflie_obstacle_jax_osqp_sqp_iterations or args.jax_osqp_sqp_iterations)
    elif system == "crazyflie_obstacle" and solver == "mpx":
        specific = _parse_int_csv(args.crazyflie_obstacle_mpx_sqp_iterations or args.mpx_sqp_iterations)
    elif solver == "jax_osqp":
        specific = _parse_int_csv(args.jax_osqp_sqp_iterations)
    elif solver == "mpx":
        specific = _parse_int_csv(args.mpx_sqp_iterations)
    else:
        specific = common
    common_set = set(common)
    return [value for value in specific if value in common_set]


def _write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system",
        "solver",
        "sqp_iterations",
        "osqp_max_iter",
        "obstacle_constraint_scale",
        "osqp_scaling",
        "osqp_rho_is_vec",
        "line_search_step_min",
        "trajectory_initialization",
        "batch_size",
        "steps",
        "status",
        "dtype",
        "success_rate",
        "success_rate_percent",
        "reported_constraint_satisfaction",
        "reported_constraint_satisfaction_percent",
        "state_constraint_satisfaction",
        "state_rollout_constraint_satisfaction",
        "constraint_satisfaction",
        "obstacle_constraint_satisfaction",
        "rail_constraint_satisfaction",
        "input_constraint_satisfaction",
        "motor_constraint_satisfaction",
        "rollout_constraint_satisfaction",
        "obstacle_rollout_constraint_satisfaction",
        "rail_rollout_constraint_satisfaction",
        "input_rollout_constraint_satisfaction",
        "motor_rollout_constraint_satisfaction",
        "violation_rate",
        "obstacle_violation_rate",
        "rail_violation_rate",
        "input_violation_rate",
        "motor_violation_rate",
        "rollout_violation_rate",
        "obstacle_rollout_violation_rate",
        "rail_rollout_violation_rate",
        "input_rollout_violation_rate",
        "motor_rollout_violation_rate",
        "max_violation",
        "obstacle_penetration_max",
        "motor_violation_max",
        "combined_rollout_violation_severity_mean",
        "combined_rollout_violation_severity_p95",
        "combined_rollout_violation_severity_p99",
        "combined_rollout_violation_severity_max",
        "obstacle_violation_mean",
        "obstacle_violation_p95",
        "obstacle_violation_p99",
        "obstacle_violation_max",
        "obstacle_rollout_violation_severity_mean",
        "obstacle_rollout_violation_severity_p95",
        "obstacle_rollout_violation_severity_p99",
        "obstacle_rollout_violation_severity_max",
        "rail_violation_mean",
        "rail_violation_p95",
        "rail_violation_p99",
        "rail_violation_max",
        "rail_rollout_violation_severity_mean",
        "rail_rollout_violation_severity_p95",
        "rail_rollout_violation_severity_p99",
        "rail_rollout_violation_severity_max",
        "input_violation_mean",
        "input_violation_p95",
        "input_violation_p99",
        "input_violation_max",
        "input_rollout_violation_severity_mean",
        "input_rollout_violation_severity_p95",
        "input_rollout_violation_severity_p99",
        "input_rollout_violation_severity_max",
        "motor_violation_mean",
        "motor_violation_p95",
        "motor_violation_p99",
        "motor_rollout_violation_severity_mean",
        "motor_rollout_violation_severity_p95",
        "motor_rollout_violation_severity_p99",
        "motor_rollout_violation_severity_max",
        "final_position_rms",
        "final_state_cost_mean",
        "return_mean",
        "finite_rate",
        "elapsed_s",
        "optimization_s_per_sim_step",
        "optimization_ms_per_sim_step",
        "optimization_us_per_rollout_step",
        "rollouts_per_s",
        "rollout_steps_per_s",
        "sqp_iterations_per_s",
        "closed_loop_steps_per_s",
        "wall_s",
        "summary_json",
        "log_path",
        "plot_path",
        "position_plot_path",
        "clearance_plot_path",
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_case(args: argparse.Namespace, system: str, solver: str, sqp_iterations: int) -> dict[str, object]:
    prefix = _base_output_prefix(args, system, solver, sqp_iterations)
    summary_json = prefix.with_suffix(".json")
    log_path = prefix.with_suffix(".log")
    if args.resume and summary_json.exists():
        summary = _normalize_summary(system, solver, json.loads(summary_json.read_text()))
        summary["summary_json"] = str(summary_json)
        summary["log_path"] = str(log_path)
        return _common_row(summary, wall_s=0.0, status="ok")
    cmd = _command(args, system, solver, sqp_iterations, prefix)
    print(f"Running {system} {solver} SQP={sqp_iterations}: {' '.join(cmd)}", flush=True)
    start = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        completed = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    wall_s = time.perf_counter() - start
    if completed.returncode != 0:
        return _common_row(
            {
                "system": system,
                "solver": solver,
                "sqp_iterations": sqp_iterations,
                "batch_size": args.batch_size,
                "summary_json": str(summary_json),
                "log_path": str(log_path),
            },
            wall_s=wall_s,
            status="failed",
            error=f"returncode={completed.returncode}",
        )
    summary = _normalize_summary(system, solver, json.loads(summary_json.read_text()))
    summary["summary_json"] = str(summary_json)
    summary["log_path"] = str(log_path)
    for suffix, key in (
        ("_states.png", "plot_path"),
        ("_positions.png", "position_plot_path"),
        ("_clearance.png", "clearance_plot_path"),
    ):
        path = prefix.with_name(prefix.name + suffix)
        if path.exists():
            summary[key] = str(path)
    return _common_row(summary, wall_s=wall_s, status="ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default="cartpole,crazyflie_obstacle")
    parser.add_argument("--solvers", default="jax_osqp,mpx")
    parser.add_argument("--sqp-iterations", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--jax-osqp-sqp-iterations", default="5")
    parser.add_argument("--mpx-sqp-iterations", default="5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--cartpole-jax-osqp-sqp-iterations", default="5")
    parser.add_argument("--cartpole-mpx-sqp-iterations", default="5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--crazyflie-obstacle-jax-osqp-sqp-iterations", default="1")
    parser.add_argument("--crazyflie-obstacle-mpx-sqp-iterations", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/nonlinear_mpc/constraint_sqp_sweep"))
    parser.add_argument("--csv-path", type=pathlib.Path, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write-npz", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--cartpole-horizon-steps", type=int, default=100)
    parser.add_argument("--cartpole-dt-start", type=float, default=0.1)
    parser.add_argument("--cartpole-dt-growth", type=float, default=1.0)
    parser.add_argument("--cartpole-sim-time", type=float, default=10.0)
    parser.add_argument("--cartpole-control-dt", type=float, default=0.1)
    parser.add_argument("--cartpole-rollout-steps", type=int, default=None)
    parser.add_argument("--cartpole-integrator-substeps", type=int, default=1)
    parser.add_argument("--cartpole-segment-budget", type=int, default=384)
    parser.add_argument("--cartpole-mpx-barrier-alpha", type=float, default=0.1)
    parser.add_argument("--cartpole-mpx-barrier-sigma", type=float, default=1.0)

    parser.add_argument("--crazyflie-horizon-steps", type=int, default=40)
    parser.add_argument("--crazyflie-sim-time", type=float, default=2.0)
    parser.add_argument("--crazyflie-control-dt", type=float, default=0.01)
    parser.add_argument("--crazyflie-sim-steps", type=int, default=None)
    parser.add_argument("--crazyflie-segment-budget", type=int, default=256)
    parser.add_argument("--crazyflie-obstacle-mpx-barrier-alpha", type=float, default=0.0007)
    parser.add_argument("--crazyflie-obstacle-mpx-barrier-sigma", type=float, default=0.5)
    parser.add_argument("--crazyflie-obstacle-osqp-max-iter", type=int, default=100)
    parser.add_argument("--crazyflie-obstacle-constraint-scale", type=float, default=20.0)
    parser.add_argument("--crazyflie-obstacle-osqp-scaling", type=int, default=10)
    parser.add_argument("--crazyflie-obstacle-line-search-step-min", type=float, default=0.01)
    parser.add_argument("--crazyflie-obstacle-osqp-rho-is-vec", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--crazyflie-obstacle-trajectory-initialization",
        choices=("linear", "initial_state"),
        default="initial_state",
    )

    parser.add_argument("--osqp-max-iter", type=int, default=25)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--qdldl-backend", choices=("jax", "warp"), default="warp")
    parser.add_argument("--qdldl-factor-backend", choices=("jax", "warp"), default="warp")
    parser.add_argument("--qdldl-solve-backend", choices=("jax", "warp"), default="warp")
    parser.add_argument("--transpose-work", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segmented", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segment-strategy", choices=("fixed", "greedy", "optimal"), default="optimal")
    parser.add_argument("--level-scheduled-solve", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--level-scheduled-solve-threshold", type=int, default=2)
    parser.add_argument("--group-repeated-stages", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--mpx-solver-mode", choices=("primal_dual",), default="primal_dual")
    parser.add_argument("--mpx-equality-weight", type=float, default=1.0e4)
    parser.add_argument("--mpx-num-alpha", type=int, default=11)
    parser.add_argument("--mpx-limited-memory", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    systems = _parse_csv(args.systems)
    solvers = _parse_csv(args.solvers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.csv_path is None:
        args.csv_path = args.output_dir / "summary.csv"
    rows: list[dict[str, object]] = []
    for system in systems:
        for solver in solvers:
            sqp_values = _sqp_values_for_solver(args, system, solver)
            for sqp_iterations in sqp_values:
                row = run_case(args, system, solver, sqp_iterations)
                rows.append(row)
                _write_csv(args.csv_path, rows)
                status = row["status"]
                success = row.get("success_rate", "")
                satisfaction = row.get("reported_constraint_satisfaction_percent", "")
                opt_ms = row.get("optimization_ms_per_sim_step", "")
                print(
                    f"{system} {solver} SQP={sqp_iterations}: status={status} "
                    f"success={success} state_rollout_satisfaction_pct={satisfaction} "
                    f"opt_ms_per_step={opt_ms}",
                    flush=True,
                )
    _write_csv(args.csv_path, rows)
    print(f"Wrote {args.csv_path}")


if __name__ == "__main__":
    main()
