"""Benchmark numpysadi against jaxadi on synthetic dense SX functions.

The generated CasADi functions are intentionally long straight-line scalar SX
graphs.  They are not meant to model a particular physical system; they stress
translation and XLA compilation with many scalar instructions.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import time
import traceback
from typing import Any, Callable

import casadi as cs
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from warpmpc.numpysadi import generate_source, load_function, random_inputs_with_key


DEFAULT_SIZES = [100, 1_000, 10_000, 100_000, 1_000_000]
DEFAULT_BATCH_SIZE = 10_000
DEFAULT_COMPILE_TIMEOUT_SECONDS = 10 * 60
COMPILE_INPUT_SEED = 0
WARMUP_INPUT_SEED = 1
EVAL_INPUT_SEED = 2


def build_synthetic_function(target_instructions: int, width: int = 8) -> cs.Function:
    """Build a dense SX function with approximately ``target_instructions``."""

    rounds = max(1, int(np.ceil(target_instructions / 30.0)))
    fn = _build_synthetic_rounds(rounds, width, target_instructions)
    if fn.n_instructions() < target_instructions:
        scale = target_instructions / max(fn.n_instructions(), 1)
        rounds = max(rounds + 1, int(np.ceil(rounds * scale)))
        fn = _build_synthetic_rounds(rounds, width, target_instructions)
    return fn


def _build_synthetic_rounds(rounds: int, width: int, target: int) -> cs.Function:
    x = cs.SX.sym("x", width, 1)
    u = cs.SX.sym("u", width, 1)
    state = [x[i] for i in range(width)]

    for k in range(rounds):
        i = k % width
        j = (k * 5 + 1) % width
        m = (k * 7 + 3) % width
        a = state[i]
        b = state[j]
        c = u[m]

        t = cs.sin(0.1 * a + c) + cs.cos(0.2 * b - c)
        t = t + cs.tanh(0.01 * a * b)
        t = t + cs.sqrt(t * t + 1.0)
        t = t + cs.atan2(a + 1.0, c + 2.0)
        t = t + 0.01 * cs.fmin(a, b) + 0.02 * cs.fmax(b, c)
        state[i] = 0.91 * a + 0.07 * t + 0.02 * b

    return cs.Function(f"synthetic_{target}", [x, u], [cs.vertcat(*state)])


def convert_numpysadi(casadi_function: cs.Function) -> tuple[Callable[..., Any], dict[str, float]]:
    start = time.perf_counter()
    source, function_name = generate_source(casadi_function, return_name=True)
    codegen_seconds = time.perf_counter() - start

    start = time.perf_counter()
    jax_function = load_function(source, function_name)
    declare_seconds = time.perf_counter() - start

    return jax_function, {
        "translation_seconds": codegen_seconds,
        "declare_seconds": declare_seconds,
        "source_bytes": len(source.encode("utf-8")),
    }


def convert_jaxadi(casadi_function: cs.Function) -> tuple[Callable[..., Any], dict[str, float]]:
    from jaxadi import convert as jaxadi_convert

    start = time.perf_counter()
    jax_function = jaxadi_convert(casadi_function, compile=False)
    translation_seconds = time.perf_counter() - start
    return jax_function, {
        "translation_seconds": translation_seconds,
        "declare_seconds": 0.0,
        "source_bytes": 0,
    }


def benchmark_callable(
    jax_function: Callable[..., Any],
    casadi_function: cs.Function,
    *,
    batch_size: int,
    repeats: int,
) -> dict[str, float]:
    compile_inputs = random_inputs_with_key(
        casadi_function,
        batch_size,
        jax.random.PRNGKey(COMPILE_INPUT_SEED),
        low=-0.1,
        high=0.1,
    )
    vmapped = jax.jit(jax.vmap(jax_function))

    start = time.perf_counter()
    compiled = vmapped.lower(*compile_inputs).compile()
    compile_seconds = time.perf_counter() - start

    warmup_inputs = random_inputs_with_key(
        casadi_function,
        batch_size,
        jax.random.PRNGKey(WARMUP_INPUT_SEED),
        low=-0.1,
        high=0.1,
    )
    block_until_ready(compiled(*warmup_inputs))

    run_inputs = random_inputs_with_key(
        casadi_function,
        batch_size,
        jax.random.PRNGKey(EVAL_INPUT_SEED),
        low=-0.1,
        high=0.1,
    )

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        block_until_ready(compiled(*run_inputs))
        timings.append(time.perf_counter() - start)

    return {
        "jax_compile_seconds": compile_seconds,
        "execution_mean_seconds": float(np.mean(timings)),
        "execution_min_seconds": float(np.min(timings)),
        "execution_max_seconds": float(np.max(timings)),
        "compile_input_seed": COMPILE_INPUT_SEED,
        "warmup_input_seed": WARMUP_INPUT_SEED,
        "eval_input_seed": EVAL_INPUT_SEED,
    }


def block_until_ready(value: Any) -> None:
    for leaf in jax.tree.leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def clear_jax_caches() -> None:
    if hasattr(jax, "clear_caches"):
        jax.clear_caches()
    gc.collect()


def run_one(
    provider: str,
    casadi_function: cs.Function,
    *,
    target_size: int,
    batch_size: int,
    repeats: int,
    dtype: str,
) -> dict[str, Any]:
    converters = {
        "numpysadi": convert_numpysadi,
        "jaxadi": convert_jaxadi,
    }
    result: dict[str, Any] = {
        "provider": provider,
        "target_instructions": target_size,
        "instructions": casadi_function.n_instructions(),
        "batch_size": batch_size,
        "repeats": repeats,
        "dtype": dtype,
        "status": "ok",
    }
    try:
        clear_jax_caches()
        jax_function, conversion_metrics = converters[provider](casadi_function)
        result.update(conversion_metrics)
        result.update(
            benchmark_callable(
                jax_function,
                casadi_function,
                batch_size=batch_size,
                repeats=repeats,
            )
        )
    except Exception as exc:
        result["status"] = "error"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        clear_jax_caches()
    return result


def run_one_with_timeout(
    provider: str,
    casadi_function: cs.Function,
    *,
    target_size: int,
    batch_size: int,
    repeats: int,
    dtype: str,
    compile_timeout_seconds: float,
) -> dict[str, Any]:
    """Run one benchmark in an isolated process and kill it on timeout.

    The timeout is intentionally process-level: XLA compilation can spend a
    long time inside native code, where Python timers are not reliable enough
    for a benchmark harness.
    """

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_run_one_worker,
        args=(
            result_queue,
            provider,
            casadi_function,
            target_size,
            batch_size,
            repeats,
            dtype,
        ),
    )

    start = time.perf_counter()
    process.start()
    process.join(compile_timeout_seconds)
    elapsed = time.perf_counter() - start

    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=10)
        row = {
            "provider": provider,
            "target_instructions": target_size,
            "instructions": casadi_function.n_instructions(),
            "batch_size": batch_size,
            "repeats": repeats,
            "dtype": dtype,
            "compile_timeout_seconds": compile_timeout_seconds,
            "elapsed_seconds_before_timeout": elapsed,
            "status": "timeout",
            "error": f"compile exceeded {compile_timeout_seconds:g}s timeout",
        }
        return row

    try:
        row = result_queue.get_nowait()
    except queue.Empty:
        row = {
            "provider": provider,
            "target_instructions": target_size,
            "instructions": casadi_function.n_instructions(),
            "batch_size": batch_size,
            "repeats": repeats,
            "dtype": dtype,
            "status": "error",
            "error": f"worker exited with code {process.exitcode} without a result",
        }

    row["compile_timeout_seconds"] = compile_timeout_seconds
    return row


def _run_one_worker(
    result_queue: mp.Queue,
    provider: str,
    casadi_function: cs.Function,
    target_size: int,
    batch_size: int,
    repeats: int,
    dtype: str,
) -> None:
    jax.config.update("jax_enable_x64", dtype == "float64")
    row = run_one(
        provider,
        casadi_function,
        target_size=target_size,
        batch_size=batch_size,
        repeats=repeats,
        dtype=dtype,
    )
    result_queue.put(row)


def write_result(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def print_result(row: dict[str, Any]) -> None:
    if row["status"] == "skipped":
        print(
            f"{row['provider']:15s} target={row['target_instructions']:>8} "
            f"instr={row['instructions']:>8} SKIPPED {row['error']}",
            flush=True,
        )
        return
    if row["status"] == "timeout":
        print(
            f"{row['provider']:15s} target={row['target_instructions']:>8} "
            f"instr={row['instructions']:>8} TIMEOUT {row['error']}",
            flush=True,
        )
        return
    if row["status"] != "ok":
        print(
            f"{row['provider']:15s} target={row['target_instructions']:>8} "
            f"instr={row['instructions']:>8} ERROR {row['error']}",
            flush=True,
        )
        return
    print(
        f"{row['provider']:15s} target={row['target_instructions']:>8} "
        f"instr={row['instructions']:>8} translate={row['translation_seconds']:.3f}s "
        f"declare={row['declare_seconds']:.3f}s "
        f"compile={row['jax_compile_seconds']:.3f}s "
        f"exec_mean={row['execution_mean_seconds']:.6f}s",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["numpysadi", "jaxadi"],
        default=["numpysadi", "jaxadi"],
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("results/numpysadi_synthetic/benchmark_results.jsonl"))
    parser.add_argument("--compile-timeout-seconds", type=float, default=DEFAULT_COMPILE_TIMEOUT_SECONDS)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--enable-x64", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.enable_x64:
        args.dtype = "float64"
    jax.config.update("jax_enable_x64", args.dtype == "float64")

    print(f"JAX devices: {jax.devices()}")
    print(f"Writing JSONL results to {args.output}")

    provider_timeout_at: dict[str, int] = {}

    for size in args.sizes:
        build_start = time.perf_counter()
        casadi_function = build_synthetic_function(size, width=args.width)
        build_seconds = time.perf_counter() - build_start
        print(
            f"Built target={size} actual={casadi_function.n_instructions()} "
            f"in {build_seconds:.3f}s",
            flush=True,
        )

        for provider in args.providers:
            if provider in provider_timeout_at:
                row = {
                    "provider": provider,
                    "target_instructions": size,
                    "instructions": casadi_function.n_instructions(),
                    "batch_size": args.batch_size,
                    "repeats": args.repeats,
                    "dtype": args.dtype,
                    "compile_timeout_seconds": args.compile_timeout_seconds,
                    "status": "skipped",
                    "skipped_after_timeout_size": provider_timeout_at[provider],
                    "error": (
                        "previous timeout at target "
                        f"{provider_timeout_at[provider]}"
                    ),
                }
            else:
                row = run_one_with_timeout(
                    provider,
                    casadi_function,
                    target_size=size,
                    batch_size=args.batch_size,
                    repeats=args.repeats,
                    dtype=args.dtype,
                    compile_timeout_seconds=args.compile_timeout_seconds,
                )
                if row["status"] == "timeout":
                    provider_timeout_at[provider] = size
            row["build_seconds"] = build_seconds
            print_result(row)
            write_result(args.output, row)


if __name__ == "__main__":
    main()
