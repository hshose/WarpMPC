#!/usr/bin/env python3
"""Benchmark NVIDIA cuDSS on the same fixed-pattern MPC KKT batches."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import pathlib
import sys
import time
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.jax_cache import configure_jax_compilation_cache

configure_jax_compilation_cache()

import numpy as np
import qdldl
import scipy.sparse as scipy_sp

from benchmarks.problems import make_mpc_kkt, sample_kkt_values, structural_density
from warpmpc.jax_qdldl import (
    build_qdldl_plan,
)


@dataclass(frozen=True)
class CpuQdldlTiming:
    factor_s_per_system: float
    solve_s_per_system: float
    update_solve_s_per_system: float

    @property
    def factor_systems_per_s(self) -> float:
        return 1.0 / self.factor_s_per_system

    @property
    def solve_systems_per_s(self) -> float:
        return 1.0 / self.solve_s_per_system

    @property
    def update_solve_systems_per_s(self) -> float:
        return 1.0 / self.update_solve_s_per_system


@dataclass(frozen=True)
class CudssResult:
    dtype: str
    batch_size: int
    matrix_nnz: int
    estimated_operand_gib: float
    plan_s: float
    factor_s: float
    solve_s: float
    factor_solve_s: float
    max_abs_x: float | None
    max_rel_residual: float | None
    cpu_timing: CpuQdldlTiming | None

    @property
    def systems_per_s(self) -> float:
        return self.batch_size / self.factor_solve_s

    @property
    def factor_systems_per_s(self) -> float:
        return self.batch_size / self.factor_s

    @property
    def solve_systems_per_s(self) -> float:
        return self.batch_size / self.solve_s

    @property
    def cpu_throughput(self) -> float | None:
        return None if self.cpu_timing is None else self.cpu_timing.update_solve_systems_per_s


@dataclass(frozen=True)
class CudssOperands:
    a_batch: list
    b_batch: list
    data_arrays: list
    rhs_arrays: list


@dataclass(frozen=True)
class CudssSource:
    values_gpu: object
    rhs_gpu: object


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_dtypes(text: str) -> list[np.dtype]:
    return [np.dtype(part.strip()) for part in text.split(",") if part.strip()]


def _parse_dtype_batch_sizes(text: str) -> dict[str, list[int]]:
    overrides: dict[str, list[int]] = {}
    if not text.strip():
        return overrides

    for entry in text.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                "--dtype-batch-sizes entries must look like "
                "float64:512,2048,10000"
            )
        dtype_text, batches_text = entry.split(":", 1)
        dtype_name = np.dtype(dtype_text.strip()).name
        batches = _parse_ints(batches_text)
        if not batches:
            raise ValueError(f"No batch sizes provided for dtype {dtype_name}")
        overrides[dtype_name] = batches
    return overrides


def _sync(cp) -> None:
    cp.cuda.get_current_stream().synchronize()


def _estimate_operand_gib(batch: int, n: int, nnz: int, dtype: np.dtype) -> float:
    item = np.dtype(dtype).itemsize
    data_bytes = batch * (nnz + n) * item
    # CSR index arrays are shared between all samples in the benchmark operands.
    index_bytes = (nnz + n + 1) * np.dtype(np.int32).itemsize
    return (data_bytes + index_bytes) / 1024**3


def _lower_csr_pattern_from_upper_csc(
    upper_csc: scipy_sp.csc_matrix,
) -> tuple[scipy_sp.csr_matrix, np.ndarray]:
    """Return lower-triangular CSR pattern and map from upper CSC values."""

    upper_csc = scipy_sp.csc_matrix(upper_csc)
    if upper_csc.shape[0] != upper_csc.shape[1]:
        raise ValueError("KKT matrix must be square")

    rows: list[int] = []
    cols: list[int] = []
    upper_value_positions: list[int] = []
    for upper_col in range(upper_csc.shape[1]):
        start = upper_csc.indptr[upper_col]
        stop = upper_csc.indptr[upper_col + 1]
        for upper_pos in range(start, stop):
            upper_row = int(upper_csc.indices[upper_pos])
            rows.append(upper_col)
            cols.append(upper_row)
            upper_value_positions.append(upper_pos)

    order_tags = np.arange(len(upper_value_positions), dtype=np.int64)
    lower = scipy_sp.coo_matrix(
        (order_tags, (rows, cols)), shape=upper_csc.shape
    ).tocsr()
    lower.sort_indices()
    tag_to_upper_pos = np.asarray(upper_value_positions, dtype=np.int64)
    value_map = tag_to_upper_pos[np.asarray(lower.data, dtype=np.int64)]
    pattern = scipy_sp.csr_matrix(
        (np.ones_like(value_map, dtype=np.float64), lower.indices, lower.indptr),
        shape=upper_csc.shape,
    )
    pattern.sort_indices()
    return pattern, value_map


def _make_operands(cp, cupy_sp, pattern, value_map, values, rhs):
    indices_gpu = cp.asarray(pattern.indices.astype(np.int32, copy=False))
    indptr_gpu = cp.asarray(pattern.indptr.astype(np.int32, copy=False))
    source = _make_source(cp, value_map, values, rhs)
    return _make_operands_from_source(cp, cupy_sp, pattern, indices_gpu, indptr_gpu, source)


def _make_operands_from_source(cp, cupy_sp, pattern, indices_gpu, indptr_gpu, source):
    a_batch = []
    b_batch = []
    data_arrays = []
    rhs_arrays = []
    for i in range(source.values_gpu.shape[0]):
        # cuDSS explicit batching currently needs a compact 1D data buffer per
        # sample for the symmetric path; views into a 2D value batch gave wrong
        # answers in smoke tests on nvmath 0.9.0.
        data_i = cp.ascontiguousarray(source.values_gpu[i])
        rhs_i = cp.ascontiguousarray(source.rhs_gpu[i])
        a_batch.append(
            cupy_sp.csr_matrix((data_i, indices_gpu, indptr_gpu), shape=pattern.shape)
        )
        b_batch.append(rhs_i)
        data_arrays.append(data_i)
        rhs_arrays.append(rhs_i)

    _sync(cp)
    return CudssOperands(
        a_batch=a_batch,
        b_batch=b_batch,
        data_arrays=data_arrays,
        rhs_arrays=rhs_arrays,
    )


def _make_source(cp, value_map, values, rhs) -> CudssSource:
    return CudssSource(
        values_gpu=cp.asarray(values[:, value_map]),
        rhs_gpu=cp.asarray(rhs),
    )


def _copy_values_and_rhs(cp, operands: CudssOperands, source: CudssSource) -> None:
    for data_i, source_i in zip(operands.data_arrays, source.values_gpu, strict=True):
        data_i[...] = source_i
    for rhs_i, source_i in zip(operands.rhs_arrays, source.rhs_gpu, strict=True):
        rhs_i[...] = source_i
    _sync(cp)


def _copy_rhs(cp, operands: CudssOperands, source: CudssSource) -> None:
    for rhs_i, source_i in zip(operands.rhs_arrays, source.rhs_gpu, strict=True):
        rhs_i[...] = source_i
    _sync(cp)


def _free_cupy_blocks(cp) -> None:
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _benchmark_cpu_qdldl(plan, values: np.ndarray, rhs: np.ndarray, repeat: int) -> CpuQdldlTiming:
    matrices = [
        scipy_sp.csc_matrix((np.asarray(row), plan.a_indices, plan.a_indptr), shape=(plan.n, plan.n))
        for row in values
    ]
    solver = qdldl.Solver(matrices[0], upper=True)
    solver.solve(rhs[0])

    start = time.perf_counter()
    count = 0
    for _ in range(repeat):
        for matrix in matrices:
            solver.update(matrix, upper=True)
            count += 1
    factor_s_per_system = (time.perf_counter() - start) / count

    solvers = [qdldl.Solver(matrix, upper=True) for matrix in matrices]
    for solver_i, rhs_i in zip(solvers, rhs, strict=True):
        solver_i.solve(rhs_i)
    start = time.perf_counter()
    count = 0
    for _ in range(repeat):
        for solver_i, rhs_i in zip(solvers, rhs, strict=True):
            solver_i.solve(rhs_i)
            count += 1
    solve_s_per_system = (time.perf_counter() - start) / count

    solver = qdldl.Solver(matrices[0], upper=True)
    solver.solve(rhs[0])
    start = time.perf_counter()
    count = 0
    for _ in range(repeat):
        for i, matrix in enumerate(matrices):
            solver.update(matrix, upper=True)
            solver.solve(rhs[i])
            count += 1
    update_solve_s_per_system = (time.perf_counter() - start) / count
    return CpuQdldlTiming(
        factor_s_per_system=factor_s_per_system,
        solve_s_per_system=solve_s_per_system,
        update_solve_s_per_system=update_solve_s_per_system,
    )


def _check_solution(plan, values: np.ndarray, rhs: np.ndarray, x_batch) -> tuple[float, float]:
    solver = qdldl.Solver(
        scipy_sp.csc_matrix((values[0], plan.a_indices, plan.a_indptr), shape=(plan.n, plan.n)),
        upper=True,
    )
    max_abs_x = 0.0
    max_rel_residual = 0.0
    for i, x_gpu in enumerate(x_batch):
        matrix_upper = scipy_sp.csc_matrix(
            (values[i], plan.a_indices, plan.a_indptr), shape=(plan.n, plan.n)
        )
        solver.update(matrix_upper, upper=True)
        x_cpu = solver.solve(rhs[i])

        x = np.asarray(x_gpu.get()).reshape(-1)
        max_abs_x = max(max_abs_x, float(np.max(np.abs(x - x_cpu))))

        full = matrix_upper + matrix_upper.T - scipy_sp.diags(
            matrix_upper.diagonal(), format="csc"
        )
        residual = full @ x - rhs[i]
        denom = max(1.0, float(np.linalg.norm(rhs[i])))
        max_rel_residual = max(max_rel_residual, float(np.linalg.norm(residual) / denom))
    return max_abs_x, max_rel_residual


def _time_factor(cp, solver, operands: CudssOperands, sources: list[CudssSource], repeat: int) -> float:
    elapsed = 0.0
    for i in range(repeat):
        _copy_values_and_rhs(cp, operands, sources[i % len(sources)])
        start = time.perf_counter()
        solver.factorize()
        _sync(cp)
        elapsed += time.perf_counter() - start
    return elapsed / repeat


def _time_solve(cp, solver, operands: CudssOperands, sources: list[CudssSource], repeat: int) -> float:
    _copy_values_and_rhs(cp, operands, sources[0])
    solver.factorize()
    _sync(cp)

    elapsed = 0.0
    for i in range(repeat):
        _copy_rhs(cp, operands, sources[i % len(sources)])
        start = time.perf_counter()
        x = solver.solve()
        _sync(cp)
        del x
        elapsed += time.perf_counter() - start
    return elapsed / repeat


def _time_factor_solve(cp, solver, operands: CudssOperands, sources: list[CudssSource], repeat: int):
    last_x = None
    elapsed = 0.0
    for i in range(repeat):
        _copy_values_and_rhs(cp, operands, sources[i % len(sources)])
        start = time.perf_counter()
        solver.factorize()
        last_x = solver.solve()
        _sync(cp)
        elapsed += time.perf_counter() - start
    return elapsed / repeat, last_x


def _write_csv(path: pathlib.Path, rows: list[CudssResult]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dtype",
                "batch_size",
                "matrix_nnz",
                "estimated_operand_gib",
                "plan_s",
                "factor_s",
                "solve_s",
                "factor_solve_s",
                "factor_systems_per_s",
                "solve_systems_per_s",
                "systems_per_s",
                "max_abs_x",
                "max_rel_residual",
                "cpu_qdldl_factor_s_per_system",
                "cpu_qdldl_solve_s_per_system",
                "cpu_qdldl_update_solve_s_per_system",
                "cpu_qdldl_factor_systems_per_s",
                "cpu_qdldl_solve_systems_per_s",
                "cpu_qdldl_update_solve_systems_per_s",
            ],
        )
        writer.writeheader()
        for row in rows:
            cpu = row.cpu_timing
            writer.writerow(
                {
                    "dtype": row.dtype,
                    "batch_size": row.batch_size,
                    "matrix_nnz": row.matrix_nnz,
                    "estimated_operand_gib": row.estimated_operand_gib,
                    "plan_s": row.plan_s,
                    "factor_s": row.factor_s,
                    "solve_s": row.solve_s,
                    "factor_solve_s": row.factor_solve_s,
                    "factor_systems_per_s": row.factor_systems_per_s,
                    "solve_systems_per_s": row.solve_systems_per_s,
                    "systems_per_s": row.systems_per_s,
                    "max_abs_x": row.max_abs_x,
                    "max_rel_residual": row.max_rel_residual,
                    "cpu_qdldl_factor_s_per_system": None if cpu is None else cpu.factor_s_per_system,
                    "cpu_qdldl_solve_s_per_system": None if cpu is None else cpu.solve_s_per_system,
                    "cpu_qdldl_update_solve_s_per_system": None if cpu is None else cpu.update_solve_s_per_system,
                    "cpu_qdldl_factor_systems_per_s": None if cpu is None else cpu.factor_systems_per_s,
                    "cpu_qdldl_solve_systems_per_s": None if cpu is None else cpu.solve_systems_per_s,
                    "cpu_qdldl_update_solve_systems_per_s": row.cpu_throughput,
                }
            )


def _plot(path: pathlib.Path, rows: list[CudssResult], cpu_throughput: float | None) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"float32": "o", "float64": "s"}
    for dtype in sorted({row.dtype for row in rows}):
        dtype_rows = [row for row in rows if row.dtype == dtype]
        dtype_rows.sort(key=lambda row: row.batch_size)
        ax.plot(
            [row.batch_size for row in dtype_rows],
            [row.systems_per_s for row in dtype_rows],
            marker=markers.get(dtype, "o"),
            linewidth=2,
            label=f"cuDSS GPU {dtype}",
        )

    if cpu_throughput is not None:
        ax.axhline(
            cpu_throughput,
            linestyle="--",
            color="black",
            linewidth=1.8,
            label="CPU qdldl update+solve",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput [systems/s]")
    ax.set_title("cuDSS batched direct solve on fixed MPC KKT pattern")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--nu", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1e-6)
    parser.add_argument("--dtypes", default="float32,float64")
    parser.add_argument("--batch-sizes", default="512,2048,10000,20000")
    parser.add_argument(
        "--dtype-batch-sizes",
        default="",
        help=(
            "Optional semicolon-separated per-dtype batch overrides, e.g. "
            "'float64:512,2048,10000'. Dtypes without an override use "
            "--batch-sizes."
        ),
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cpu-samples", type=int, default=512)
    parser.add_argument("--cpu-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--variation", type=float, default=0.25)
    parser.add_argument("--check-samples", type=int, default=2)
    parser.add_argument("--max-device-gb", type=float, default=0.0)
    parser.add_argument("--plot-path", default="results/qdldl_factorization/throughput_mpc_cudss.png")
    parser.add_argument("--csv-path", default="results/qdldl_factorization/throughput_mpc_cudss.csv")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--no-cpu", action="store_true")
    args = parser.parse_args()

    import cupy as cp
    import cupyx.scipy.sparse as cupy_sp
    import nvmath
    from nvmath.sparse.advanced import (
        DirectSolver,
        DirectSolverMatrixType,
        DirectSolverMatrixViewType,
        DirectSolverOptions,
    )

    problem = make_mpc_kkt(
        nx=args.nx,
        nu=args.nu,
        horizon=args.horizon,
        rho=args.rho,
        sigma=args.sigma,
        extra_kkt_density=0.0,
        seed=args.seed,
    )
    kkt = problem.kkt_upper
    plan = build_qdldl_plan(kkt, upper=True)
    lower_pattern, value_map = _lower_csr_pattern_from_upper_csc(kkt)

    print("CuPy:", cp.__version__, "nvmath:", getattr(nvmath, "__version__", "?"))
    print(
        "KKT:",
        f"dim={kkt.shape[0]}",
        f"upper_nnz={kkt.nnz}",
        f"lower_nnz={lower_pattern.nnz}",
        f"upper_density={100 * structural_density(kkt):.2f}%",
        f"nnz_L={plan.nnz_l}",
    )
    print(
        "cuDSS options: symmetric lower CSR; matrix objects stay fixed and "
        "numeric GPU buffers are updated in place after planning."
    )

    cpu_timing = None
    if not args.no_cpu:
        cpu_values = sample_kkt_values(
            kkt,
            batch_size=args.cpu_samples,
            seed=args.seed + 50,
            variation=args.variation,
            dtype=np.float64,
        )
        cpu_rhs = np.random.default_rng(args.seed + 51).standard_normal(
            (args.cpu_samples, plan.n)
        )
        cpu_timing = _benchmark_cpu_qdldl(
            plan, cpu_values, cpu_rhs, repeat=args.cpu_repeat
        )
        print(
            "CPU qdldl:",
            f"samples={args.cpu_samples}",
            f"factor={cpu_timing.factor_systems_per_s:.2f}/s",
            f"solve={cpu_timing.solve_systems_per_s:.2f}/s",
            f"update+solve={cpu_timing.update_solve_systems_per_s:.2f}/s",
        )

    options = DirectSolverOptions(
        sparse_system_type=DirectSolverMatrixType.SYMMETRIC,
        sparse_system_view=DirectSolverMatrixViewType.LOWER,
    )
    rows: list[CudssResult] = []
    max_device_gib = args.max_device_gb
    default_batches = _parse_ints(args.batch_sizes)
    dtype_batch_sizes = _parse_dtype_batch_sizes(args.dtype_batch_sizes)
    for dtype in _parse_dtypes(args.dtypes):
        batches = dtype_batch_sizes.get(dtype.name, default_batches)
        print(
            f"Benchmark dtype={dtype.name}: fixed MPC structural pattern "
            f"batches={','.join(str(batch) for batch in batches)}"
        )
        for batch in batches:
            estimate = _estimate_operand_gib(batch, plan.n, lower_pattern.nnz, dtype)
            if max_device_gib > 0.0 and estimate > max_device_gib:
                print(f"  batch={batch}: skipped estimated operands {estimate:.2f} GiB")
                break

            try:
                sources: list[CudssSource] = []
                for set_id in range(2):
                    values = sample_kkt_values(
                        kkt,
                        batch_size=batch,
                        seed=args.seed + batch + 10000 * set_id,
                        variation=args.variation,
                        dtype=dtype,
                    )
                    rhs = np.random.default_rng(
                        args.seed + 1000 + batch + 10000 * set_id
                    ).standard_normal((batch, plan.n)).astype(dtype)
                    sources.append(_make_source(cp, value_map, values, rhs))
                indices_gpu = cp.asarray(
                    lower_pattern.indices.astype(np.int32, copy=False)
                )
                indptr_gpu = cp.asarray(
                    lower_pattern.indptr.astype(np.int32, copy=False)
                )
                operands = _make_operands_from_source(
                    cp,
                    cupy_sp,
                    lower_pattern,
                    indices_gpu,
                    indptr_gpu,
                    sources[0],
                )

                check_values = None
                check_rhs = None
                max_abs_x = None
                max_rel_residual = None
                if not args.no_check:
                    check_values = sample_kkt_values(
                        kkt,
                        batch_size=args.check_samples,
                        seed=args.seed + 10,
                        variation=args.variation,
                        dtype=dtype,
                    )
                    check_rhs = np.random.default_rng(args.seed + 11).standard_normal(
                        (args.check_samples, plan.n)
                    ).astype(dtype)
                    check_operands = _make_operands(
                        cp, cupy_sp, lower_pattern, value_map, check_values, check_rhs
                    )
                    with DirectSolver(
                        check_operands.a_batch,
                        check_operands.b_batch,
                        options=options,
                        execution="cuda",
                    ) as check_solver:
                        check_solver.plan()
                        check_solver.factorize()
                        check_x = check_solver.solve()
                        _sync(cp)
                    max_abs_x, max_rel_residual = _check_solution(
                        plan, check_values, check_rhs, check_x
                    )
                    print(
                        "  check:",
                        f"max_abs_x={max_abs_x:.3e}",
                        f"max_rel_residual={max_rel_residual:.3e}",
                    )
                    del check_operands, check_x

                with DirectSolver(
                    operands.a_batch,
                    operands.b_batch,
                    options=options,
                    execution="cuda",
                ) as solver:
                    start = time.perf_counter()
                    solver.plan()
                    _sync(cp)
                    plan_s = time.perf_counter() - start

                    # Untimed warmup: factor and solve once before any timing.
                    solver.factorize()
                    warm_x = solver.solve()
                    _sync(cp)
                    del warm_x

                    factor_s = _time_factor(cp, solver, operands, sources, args.repeat)
                    solve_s = _time_solve(cp, solver, operands, sources, args.repeat)
                    factor_solve_s, _ = _time_factor_solve(
                        cp, solver, operands, sources, args.repeat
                    )

                row = CudssResult(
                    dtype=dtype.name,
                    batch_size=batch,
                    matrix_nnz=lower_pattern.nnz,
                    estimated_operand_gib=estimate,
                    plan_s=plan_s,
                    factor_s=factor_s,
                    solve_s=solve_s,
                    factor_solve_s=factor_solve_s,
                    max_abs_x=max_abs_x,
                    max_rel_residual=max_rel_residual,
                    cpu_timing=cpu_timing,
                )
                rows.append(row)
                print(
                    f"  batch={batch}: plan={plan_s:.3f}s "
                    f"factor={factor_s:.6f}s solve={solve_s:.6f}s "
                    f"total={factor_solve_s:.6f}s throughput={row.systems_per_s:.2f}/s"
                )
            except Exception as exc:
                print(f"  batch={batch}: failed {type(exc).__name__}: {exc}")
                break
            finally:
                try:
                    del operands, sources
                except UnboundLocalError:
                    pass
                _free_cupy_blocks(cp)

    csv_path = pathlib.Path(args.csv_path)
    plot_path = pathlib.Path(args.plot_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, rows)
    _plot(plot_path, rows, None if cpu_timing is None else cpu_timing.update_solve_systems_per_s)
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
