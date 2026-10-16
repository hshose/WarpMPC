#!/usr/bin/env python3
"""Single-case closed-loop rollout for the MPPI/CEM nonlinear baselines."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
import pathlib
import platform
from statistics import median
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.jax_cache import configure_jax_compilation_cache

configure_jax_compilation_cache()

import jax
import jax.numpy as jnp

from benchmarks.nonlinear_mpc.sampling_mpc import (
    CHUNK_SIZES,
    REFERENCE_SQP_ITERATIONS,
    SIM_STEPS,
    Settings,
    default_settings,
    make_model,
    make_optimizer,
    make_rollout_functions,
    require_full_controls,
)


def source_hash() -> str:
    paths = (
        "benchmarks/nonlinear_mpc/sampling_mpc.py",
        "benchmarks/nonlinear_mpc/evaluate_sampling_mpc.py",
        "benchmarks/nonlinear_mpc/benchmark_sampling_mpc.py",
        "benchmarks/problems/crazyflie_sqp.py",
        "benchmarks/nonlinear_mpc/evaluate_crazyflie_sqp.py",
        "benchmarks/nonlinear_mpc/evaluate_cartpole_quadratic_mpc.py",
        "benchmarks/problems/cartpole_physical_quadratic.py",
        "examples/cartpole_tuning/tune_cartpole_physical_quadratic_mpc.py",
    )
    digest = hashlib.sha256()
    for name in paths:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def summarize(model, states, actions):
    states, actions = np.asarray(states, dtype=np.float64), np.asarray(actions, dtype=np.float64)
    finite = np.all(np.isfinite(states), axis=(0, 2)) & np.all(np.isfinite(actions), axis=(0, 2))
    state_cost = .5 * np.sum(states**2 * model.q, axis=-1)
    control_cost = .5 * np.sum(actions**2 * model.r, axis=-1)
    total = model.dt * np.sum(state_cost[:-1] + control_cost, axis=0)
    final = .5 * np.sum(states[-1]**2 * model.terminal_q, axis=-1)
    result = dict(finite_rate=float(np.mean(finite)), closed_loop_cost_mean=float(np.mean(total)),
                  final_state_cost_mean=float(np.mean(final)),
                  final_position_rms=float(np.sqrt(np.mean(states[-1, :, :3]**2))),
                  final_state_rms=float(np.sqrt(np.mean(states[-1]**2))))
    if model.name == "cartpole":
        # Same reward and success definition as evaluate_cartpole_quadratic_mpc.
        x, v, angle, omega = np.moveaxis(states, -1, 0)
        tail = max(1, actions.shape[0] // 2)
        rail = np.maximum(np.abs(x) - .35, 0.)
        rail_rate = np.mean(rail > 0, axis=0)
        stay = np.flip(np.cumprod(np.flip((np.abs(angle) <= np.deg2rad(10)).astype(float), axis=0), axis=0), axis=0)
        success = np.any(stay[:-1] > .5, axis=0) & finite
        returns = (250*np.any(np.abs(angle) < .35, axis=0)
                   + 80*np.max(.5*(1+np.cos(angle)), axis=0)
                   + 60*np.max(np.exp(-.5*(angle/.45)**2), axis=0)
                   + 50*np.mean(np.abs(angle) < .55, axis=0)
                   + 45*np.mean((np.abs(angle[tail:]) < .25) & (np.abs(omega[tail:]) < 1), axis=0)
                   + 25*np.mean(np.exp(-.5*(angle[tail:]/.35)**2)*np.exp(-.5*(x[tail:]/.15)**2), axis=0)
                   - 40*rail_rate - 40*np.max(rail, axis=0) - 150*np.mean(rail**2, axis=0)
                   - .10*np.mean(v[tail:]**2, axis=0) - .01*np.mean(omega[tail:]**2, axis=0)
                   - .002*np.mean(actions[..., 0]**2, axis=0))
        returns = np.where(finite, returns, -1e6)
        result.update(return_mean=float(np.mean(returns)), return_std=float(np.std(returns)),
                      return_p10=float(np.percentile(returns, 10)), return_median=float(np.median(returns)),
                      return_p90=float(np.percentile(returns, 90)), rollout_success_rate=float(np.mean(success)),
                      rail_violation_rate=float(np.mean(rail_rate)), rollout_rail_violation_mean=float(np.mean(rail)),
                      rollout_rail_violation_max=float(np.max(rail)),
                      max_action_violation=float(np.max(np.maximum(np.abs(actions)-9, 0))))
    return result



def _run_rollout(system, settings, batch_size=64, seed=0, chunk_size=256, repeats=1, save_path=None,
        sim_steps=None):
    if min(batch_size, chunk_size, repeats) < 1:
        raise ValueError("batch size, chunk size and repeats must be positive")
    model = make_model(system)
    if sim_steps is not None:
        if not isinstance(sim_steps, int) or sim_steps < 1:
            raise ValueError("sim_steps must be a positive integer")
        # Change only the evaluation duration, never the MPC prediction horizon.
        model.steps = sim_steps
    score, execute = make_rollout_functions(model, settings)
    optimize = make_optimizer(score, settings)
    chunk_size = min(batch_size, chunk_size)
    x0 = model.sample_initial(batch_size, seed)
    key = jax.random.PRNGKey(seed + 1_000_003)

    @jax.jit
    def rollout(x, ids):
        keys = jax.vmap(lambda idx: jax.random.fold_in(key, idx))(ids)
        mean = jnp.zeros((x.shape[0], settings.knots, model.nu), dtype=x.dtype)
        def step(carry, t):
            x, mean = carry
            step_keys = jax.vmap(lambda k: jax.random.fold_in(k, t))(keys)
            budget = jnp.where(t == 0, settings.initial_iterations or settings.iterations,
                               settings.iterations)
            noise_scale = jnp.where(t == 0, 1., settings.warm_sigma)
            mean = optimize(x, mean, step_keys, budget, noise_scale)
            xn, next_mean, u = execute(x, mean)
            return (xn, next_mean), (x, u)
        (xn, _), (xs, us) = jax.lax.scan(step, (x, mean), jnp.arange(model.steps))
        return jnp.concatenate((xs, xn[None]), axis=0), us

    inputs = []
    for start in range(0, batch_size, chunk_size):
        actual = min(chunk_size, batch_size-start)
        inputs.append((jnp.asarray(x0[start:start+actual]),
                       jnp.arange(start, start+actual, dtype=jnp.uint32), actual))
    jax.block_until_ready(inputs)
    compile_start = time.perf_counter()
    compiled = {}
    for x, ids, actual in inputs:
        if actual not in compiled:
            compiled[actual] = rollout.lower(x, ids).compile()
    compile_s = time.perf_counter() - compile_start
    warm_start = time.perf_counter()
    warmed = set()
    for x, ids, actual in inputs:
        if actual not in warmed:
            jax.block_until_ready(compiled[actual](x, ids))
            warmed.add(actual)
    warm_s = time.perf_counter()-warm_start
    times = []
    outputs = None
    for _ in range(repeats):
        outputs = []
        start = time.perf_counter()
        for x, ids, actual in inputs:
            # Synchronization and transfers are charged to the closed-loop time.
            states, actions = jax.device_get(compiled[actual](x, ids))
            outputs.append((states, actions))
        times.append(time.perf_counter()-start)
    states = np.concatenate([v[0] for v in outputs], axis=1)
    actions = np.concatenate([v[1] for v in outputs], axis=1)
    elapsed = float(np.median(times))
    result = dict(system=system, method=settings.method, qp_solver=settings.method, status="ok",
                  batch_size=batch_size, seed=seed, settings=asdict(settings), dtype="float32",
                  horizon_steps=model.horizon, sim_steps=model.steps, control_dt=model.dt,
                  compile_s=compile_s, warmup_s=warm_s, elapsed_s=elapsed, solve_s=elapsed,
                  repeat_times_s=times, closed_loop_steps_per_s=batch_size*model.steps/elapsed,
                  total_closed_loop_steps=batch_size*model.steps, chunk_size=chunk_size,
                  padded_instances=0,
                  first_action_candidate_rollouts=settings.samples*(settings.initial_iterations or settings.iterations),
                  later_action_candidate_rollouts=settings.samples*settings.iterations,
                  device=str(jax.devices()[0]), device_kind=jax.devices()[0].device_kind,
                  hostname=platform.node(), slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
                  jax_version=jax.__version__, source_hash=source_hash())
    result.update(summarize(model, states, actions))
    if save_path:
        np.savez_compressed(save_path, states=states, controls=actions, initial_states=x0)
    return result



def _configuration(args: argparse.Namespace) -> dict[str, object]:
    overrides = {
        name: getattr(args, name)
        for name in Settings.__dataclass_fields__
        if getattr(args, name) is not None
    }
    settings = replace(default_settings(args.system, args.method), **overrides)
    require_full_controls(args.system, settings)
    return {
        "system": args.system,
        "method": args.method,
        "settings": json.loads(json.dumps(asdict(settings))),
        "seed": args.seed,
        "sim_steps": args.sim_steps or SIM_STEPS[args.system],
        "chunk_size": min(args.batch_size, args.chunk_size or CHUNK_SIZES[args.system]),
        "reference_sqp_iterations": (
            args.reference_sqp_iterations or REFERENCE_SQP_ITERATIONS[args.system]
        ),
    }


def _can_resume(
    result: dict[str, object], args: argparse.Namespace, config: dict[str, object]
) -> bool:
    device = jax.devices()[0]
    if not (
        result.get("status") == "ok"
        and result.get("finite_rate") == 1.0
        and result.get("sampling_protocol") == "paper-v1"
        and result.get("source_hash") == source_hash()
        and result.get("device_kind") == device.device_kind
        and result.get("jax_version") == jax.__version__
        and result.get("batch_size") == args.batch_size
        and all(result.get(name) == value for name, value in config.items())
    ):
        return False
    times = result.get("repeat_times_s", [])
    if len(times) != args.repeats or not all(
        isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        for value in times
    ):
        return False
    elapsed = median(times)
    steps = args.batch_size * config["sim_steps"]
    return (
        result.get("total_closed_loop_steps") == steps
        and result.get("padded_instances") == 0
        and math.isclose(result.get("elapsed_s", 0), elapsed, rel_tol=1e-10)
        and math.isclose(result.get("closed_loop_steps_per_s", 0), steps / elapsed, rel_tol=1e-10)
        and (args.output_npz is None or args.output_npz.exists())
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    config = _configuration(args)
    device = jax.devices()[0]
    if (args.require_gpu or args.require_h100) and device.platform != "gpu":
        raise RuntimeError("Paper sampling timings require a GPU")
    if args.require_h100 and "H100" not in device.device_kind:
        raise RuntimeError("This paper run requires an H100")
    if args.resume and args.summary_json.exists():
        previous = json.loads(args.summary_json.read_text())
        if _can_resume(previous, args, config):
            print(f"Resuming verified {args.system}/{args.method} batch {args.batch_size}", flush=True)
            return previous
    if args.output_npz is not None:
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    result = _run_rollout(
        args.system,
        Settings(**config["settings"]),
        args.batch_size,
        args.seed,
        config["chunk_size"],
        args.repeats,
        args.output_npz,
        sim_steps=config["sim_steps"],
    )
    result.update(
        mode=args.method.upper(),
        sampling_protocol="paper-v1",
        rollout_steps=config["sim_steps"],
        reference_sqp_iterations=config["reference_sqp_iterations"],
        sqp_equivalent_iterations_per_s=(
            result["closed_loop_steps_per_s"] * config["reference_sqp_iterations"]
        ),
    )
    if result["finite_rate"] != 1.0:
        result.update(status="quality_failed", error="Nonfinite closed-loop trajectory")
    return result


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("cartpole", "crazyflie"), required=True)
    parser.add_argument("--method", choices=("mppi", "cem"), required=True)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=_positive_int)
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument("--sim-steps", type=_positive_int)
    parser.add_argument("--reference-sqp-iterations", type=_positive_int)
    for name in ("samples", "iterations", "knots"):
        parser.add_argument("--" + name, type=_positive_int)
    parser.add_argument("--initial-iterations", type=int)
    for name in (
        "sigma", "temperature", "elite-fraction", "smoothing", "min-std",
        "penalty", "noise-decay", "warm-sigma", "noise-correlation",
    ):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--control-scale", type=lambda text: tuple(map(float, text.split(","))))
    parser.add_argument("--feedback", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--shift", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--summary-json", type=pathlib.Path, required=True)
    parser.add_argument("--output-npz", type=pathlib.Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-h100", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = run(args)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if result["status"] != "ok":
        raise RuntimeError(result["error"])


if __name__ == "__main__":
    main()
