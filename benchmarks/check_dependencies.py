#!/usr/bin/env python3
"""Check the Python and backend dependencies used by the benchmark suite."""

from __future__ import annotations

import argparse
import ctypes.util
import importlib
import importlib.util
import pathlib
import shutil
import sys
from dataclasses import dataclass


ROOT = pathlib.Path(__file__).resolve().parents[1]
RESOURCE_PATHS = (
    ROOT / "resources" / "MPAX",
    ROOT / "resources" / "mpx",
    ROOT / "resources" / "turbompc",
    ROOT / "resources" / "qpax",
    ROOT / "resources" / "jaxopt",
    ROOT / "resources" / "jaxproxqp" / "src",
)


@dataclass(frozen=True)
class Check:
    name: str
    modules: tuple[str, ...] = ()
    executables: tuple[str, ...] = ()
    paths: tuple[pathlib.Path, ...] = ()
    required: bool = False
    note: str = ""


CHECKS = (
    Check("core:numpy", ("numpy",), required=True),
    Check("core:scipy", ("scipy", "scipy.sparse"), required=True),
    Check("core:jax", ("jax", "jax.numpy"), required=True),
    Check("core:casadi", ("casadi",), required=True),
    Check("core:qdldl", ("qdldl",), required=True),
    Check("core:flax", ("flax",), required=True),
    Check("core:optax", ("optax",), required=True),
    Check("core:warp", ("warp",), required=True),
    Check(
        "core:warpmpc",
        (
            "warpmpc",
            "warpmpc.jax_ampc",
            "warpmpc.jax_osqp",
            "warpmpc.jax_qdldl",
            "warpmpc.jax_sqp",
            "warpmpc.numpysadi",
        ),
        required=True,
    ),
    Check("examples:matplotlib", ("matplotlib",), required=True),
    Check("benchmarks:osqp", ("osqp",), note="Linear MPC CPU references."),
    Check("benchmarks:jaxadi", ("jaxadi",), note="numpysadi comparison."),
    Check("benchmarks:jaxopt", ("jaxopt",), note="BoxOSQP and CG baselines."),
    Check("benchmarks:qpax", ("qpax",), note="QPAX linear MPC baselines."),
    Check("benchmarks:mpax", ("mpax",), note="MPAX linear/nonlinear MPC baselines."),
    Check("benchmarks:mpx", ("mpx", "trajax"), note="MPX nonlinear MPC baselines."),
    Check("benchmarks:turbompc", ("turbompc",), note="TurboMPC nonlinear MPC baselines."),
    Check("benchmarks:torch", ("torch",), note="Torch is used by selected external baselines."),
    Check("benchmarks:yaml", ("yaml",), note="PyYAML for solver configuration files."),
    Check("benchmarks:cmake", executables=("cmake",), note="Building selected external backends."),
    Check(
        "benchmarks:cudss",
        ("cupy", "cupyx.scipy.sparse", "nvmath.sparse.advanced"),
        note="cuDSS factorization benchmark.",
    ),
    Check(
        "embedded:resources",
        paths=(
            ROOT / "resources" / "modm",
            ROOT / "resources" / "CMSIS-NN",
            ROOT / "resources" / "CMSIS-DSP",
            ROOT / "resources" / "eigen",
        ),
        note="Only needed for embedded AMPC microcontroller benchmarks.",
    ),
)


def _add_resource_paths() -> list[pathlib.Path]:
    added: list[pathlib.Path] = []
    for path in RESOURCE_PATHS:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
            added.append(path)
    return added


def _import_module(name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - diagnostics should show import failures.
        return False, f"{type(exc).__name__}: {exc}"
    return True, "ok"


def _check_module_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _check_one(check: Check, strict: bool) -> bool:
    ok = True
    required = check.required or strict
    messages: list[str] = []

    for module in check.modules:
        present, detail = _import_module(module)
        ok &= present
        messages.append(f"{module}={detail}")

    for executable in check.executables:
        present = shutil.which(executable) is not None
        ok &= present
        messages.append(f"{executable}={'ok' if present else 'missing'}")

    for path in check.paths:
        present = path.exists()
        ok &= present
        messages.append(f"{path.relative_to(ROOT)}={'ok' if present else 'missing'}")

    status = "ok" if ok else ("missing" if required else "optional-missing")
    print(f"[{status}] {check.name}: {', '.join(messages)}")
    if check.note:
        print(f"        {check.note}")
    return ok or not required


def _report_jax(require_gpu: bool) -> bool:
    if not _check_module_spec("jax"):
        return not require_gpu

    import jax

    devices = jax.devices()
    platforms = sorted({device.platform for device in devices})
    gpu_devices = [device for device in devices if device.platform in {"gpu", "cuda"}]
    print(f"[info] jax devices: platforms={platforms}, count={len(devices)}")
    for device in devices:
        print(f"       - {device}")
    if require_gpu and not gpu_devices:
        print("[missing] jax gpu backend: no GPU devices visible to JAX")
        return False
    return True


def _report_warp(require_gpu: bool) -> bool:
    if not _check_module_spec("warp"):
        return not require_gpu

    import warp as wp

    try:
        wp.init()
    except Exception as exc:  # noqa: BLE001
        print(f"[missing] warp init: {type(exc).__name__}: {exc}")
        return not require_gpu

    devices = []
    if hasattr(wp, "get_cuda_devices"):
        try:
            devices = list(wp.get_cuda_devices())
        except Exception:
            devices = []
    if not devices and hasattr(wp, "get_devices"):
        try:
            devices = [device for device in wp.get_devices() if "cuda" in str(device).lower()]
        except Exception:
            devices = []
    print(f"[info] warp cuda devices: {devices if devices else 'none'}")
    if require_gpu and not devices:
        print("[missing] warp cuda backend: no CUDA devices visible to Warp")
        return False
    return True


def _report_osqp() -> None:
    if not _check_module_spec("osqp"):
        return

    import osqp

    available = getattr(osqp, "algebras_available", None)
    if callable(available):
        try:
            print(f"[info] osqp algebras: {available()}")
        except Exception as exc:  # noqa: BLE001
            print(f"[info] osqp algebras: unavailable ({type(exc).__name__}: {exc})")


def _report_cudss(strict: bool) -> bool:
    cudss_lib = ctypes.util.find_library("cudss")
    nvmath_present = _check_module_spec("nvmath.sparse.advanced")
    cupy_present = _check_module_spec("cupy")
    if cudss_lib or (nvmath_present and cupy_present):
        print(
            "[info] cuDSS backend: "
            f"libcudss={cudss_lib or 'not on library path'}, "
            f"nvmath={nvmath_present}, cupy={cupy_present}"
        )
        return True
    print("[optional-missing] cuDSS backend: install nvmath-python, cupy, and cuDSS libraries")
    return not strict


def _report_turbompc_backends(strict: bool) -> bool:
    if not _check_module_spec("turbompc"):
        return not strict

    backend_names = ("admm_fused_cudss", "direct_cudss_ffi")
    print(f"[info] turbompc requested backends: {', '.join(backend_names)}")
    print("       The final backend check happens when TurboMPC constructs the solver.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Treat optional benchmark deps as required.")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if JAX or Warp cannot see a GPU.")
    args = parser.parse_args()

    added = _add_resource_paths()
    if added:
        print("Using ignored local resource checkouts:")
        for path in added:
            print(f"  - {path.relative_to(ROOT)}")

    ok = True
    for check in CHECKS:
        ok &= _check_one(check, strict=args.strict)

    ok &= _report_jax(require_gpu=args.require_gpu)
    ok &= _report_warp(require_gpu=args.require_gpu)
    _report_osqp()
    ok &= _report_cudss(strict=args.strict)
    ok &= _report_turbompc_backends(strict=args.strict)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
