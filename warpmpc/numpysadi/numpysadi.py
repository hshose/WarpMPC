"""Generate JAX-compatible Python functions from dense CasADi SX functions.

The converter intentionally goes through CasADi's C code generator and
translates the straight-line scalar C body into Python/JAX source.  This keeps
the generated function close to what CasADi already knows how to emit, while
avoiding the graph-level Python translation cost that can dominate for very
large SX expressions.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import keyword
import os
from pathlib import Path
import re
import tempfile
from types import ModuleType
from typing import Any, Callable, Iterable, Sequence

import casadi as cs
import jax
import jax.numpy as jnp
import numpy as np


__all__ = [
    "ConversionResult",
    "UnsupportedCodegenError",
    "convert",
    "export",
    "generate_source",
    "generate_warp_source",
    "import_python_function",
    "load_function",
    "random_inputs",
    "random_inputs_with_key",
    "compare_casadi_and_python_function",
    "casadi_to_python_code_from_c",
]


class UnsupportedCodegenError(ValueError):
    """Raised when a generated C line cannot be translated safely."""


@dataclass(frozen=True)
class ConversionResult:
    """The generated source and live callable for a converted CasADi function."""

    function_name: str
    source: str
    function: Callable[..., Any]


_C_MATH_FUNCTIONS = {
    "casadi_sq": "jnp.square",
    "casadi_fabs": "jnp.abs",
    "fabs": "jnp.abs",
    "casadi_sign": "jnp.sign",
    "casadi_fmin": "jnp.fmin",
    "fmin": "jnp.fmin",
    "casadi_fmax": "jnp.fmax",
    "fmax": "jnp.fmax",
    "casadi_hypot": "jnp.hypot",
    "hypot": "jnp.hypot",
    "casadi_log1p": "jnp.log1p",
    "log1p": "jnp.log1p",
    "casadi_expm1": "jnp.expm1",
    "expm1": "jnp.expm1",
    "sqrt": "jnp.sqrt",
    "sin": "jnp.sin",
    "cos": "jnp.cos",
    "tan": "jnp.tan",
    "asin": "jnp.arcsin",
    "acos": "jnp.arccos",
    "atan": "jnp.arctan",
    "atan2": "jnp.arctan2",
    "sinh": "jnp.sinh",
    "cosh": "jnp.cosh",
    "tanh": "jnp.tanh",
    "asinh": "jnp.arcsinh",
    "acosh": "jnp.arccosh",
    "atanh": "jnp.arctanh",
    "exp": "jnp.exp",
    "log": "jnp.log",
    "floor": "jnp.floor",
    "ceil": "jnp.ceil",
    "fmod": "jnp.fmod",
    "remainder": "_c_remainder",
    "copysign": "jnp.copysign",
    "pow": "jnp.power",
    "erf": "jsp.erf",
    "erfinv": "jsp.erfinv",
}


def convert(
    casadi_function: cs.Function,
    *,
    function_name: str | None = None,
    backend: str = "jax",
    warp_dtype: str = "float32",
    warp_compile_profile: str = "runtime",
    jit: bool = False,
    return_result: bool = False,
) -> Callable[..., Any] | ConversionResult:
    """Convert a dense SX CasADi function to a Python callable.

    Parameters
    ----------
    casadi_function:
        A CasADi ``SXFunction`` with dense inputs and outputs.
    function_name:
        Optional Python name for the generated function.  By default the
        CasADi function name is used after Python identifier sanitization.
    backend:
        Code generation backend. ``"jax"`` emits pure JAX scalar code.
        ``"warp"`` emits a JAX frontend that calls a Warp GPU kernel through
        Warp's JAX interoperability layer.
    warp_dtype:
        Floating point dtype for the Warp kernel.  Currently only
        ``"float32"`` is supported.
    warp_compile_profile:
        Warp module option profile. ``"runtime"`` favors optimized kernels;
        ``"compile_time"`` lowers GPU compiler optimization for very large
        generated kernels.
    jit:
        If true, return ``jax.jit(generated_function)``.
    return_result:
        If true, return ``ConversionResult`` containing the source as well as
        the callable.
    """

    source, generated_name = generate_source(
        casadi_function,
        function_name=function_name,
        backend=backend,
        warp_dtype=warp_dtype,
        warp_compile_profile=warp_compile_profile,
        return_name=True,
    )
    fn = load_function(source, generated_name)
    if jit:
        fn = jax.jit(fn)
    if return_result:
        return ConversionResult(generated_name, source, fn)
    return fn


def generate_source(
    casadi_function: cs.Function,
    *,
    function_name: str | None = None,
    backend: str = "jax",
    warp_dtype: str = "float32",
    warp_compile_profile: str = "runtime",
    return_name: bool = False,
) -> str | tuple[str, str]:
    """Generate source code from a dense SX CasADi function."""

    _validate_function(casadi_function)
    if backend not in {"jax", "warp"}:
        raise ValueError("backend must be 'jax' or 'warp'")
    generated_name = _unique_identifier(function_name or casadi_function.name())
    input_names = _input_names(casadi_function)
    c_code = _generate_c_code(casadi_function)
    body = _extract_c_function_body(c_code)

    if backend == "warp":
        source = _generate_warp_source(
            casadi_function,
            generated_name=generated_name,
            input_names=input_names,
            body=body,
            warp_dtype=warp_dtype,
            warp_compile_profile=warp_compile_profile,
        )
        if return_name:
            return source, generated_name
        return source

    statements, output_values = _translate_c_body(body, casadi_function)

    lines = [
        "# This file was generated by numpysadi.",
        "import jax",
        "import jax.numpy as jnp",
        "import jax.scipy.special as jsp",
        "",
        "",
        "def _c_remainder(x, y):",
        "    return x - y * jnp.round(x / y)",
        "",
        f"def {generated_name}({', '.join(input_names)}):",
    ]

    if input_names:
        lines.append(f"    _dtype = jnp.result_type(0.0, {', '.join(input_names)})")
    else:
        lines.append("    _dtype = jnp.result_type(0.0)")

    for i, input_name in enumerate(input_names):
        lines.append(f"    arg{i}_flat = jnp.ravel(jnp.asarray({input_name}).T)")

    lines.extend(f"    {statement}" for statement in statements)
    lines.extend(
        f"    {statement}"
        for statement in _output_construction_lines(casadi_function, output_values)
    )
    lines.append(f"    return {_return_expression(casadi_function)}")
    lines.append("")

    source = "\n".join(lines)
    if return_name:
        return source, generated_name
    return source


def generate_warp_source(
    casadi_function: cs.Function,
    *,
    function_name: str | None = None,
    warp_dtype: str = "float32",
    warp_compile_profile: str = "runtime",
    return_name: bool = False,
) -> str | tuple[str, str]:
    """Generate a JAX/Warp-interoperability source file for a dense SX function."""

    return generate_source(
        casadi_function,
        function_name=function_name,
        backend="warp",
        warp_dtype=warp_dtype,
        warp_compile_profile=warp_compile_profile,
        return_name=return_name,
    )


def export(
    casadi_function: cs.Function,
    path: str | os.PathLike[str],
    *,
    function_name: str | None = None,
) -> Path:
    """Write generated Python/JAX source for ``casadi_function`` to ``path``."""

    source = generate_source(casadi_function, function_name=function_name)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8")
    return output_path


def load_function(source: str, function_name: str) -> Callable[..., Any]:
    """Declare a generated function from source code."""

    if "import warp as wp" in source:
        with tempfile.TemporaryDirectory(prefix="numpysadi_warp_import_") as tmpdir:
            path = Path(tmpdir) / f"{function_name}.py"
            path.write_text(source, encoding="utf-8")
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not import generated Warp module from {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, function_name)

    namespace: dict[str, Any] = {}
    exec(compile(source, f"<numpysadi:{function_name}>", "exec"), namespace)
    return namespace[function_name]


def import_python_function(
    python_function_path: str | os.PathLike[str],
    function_name: str,
) -> Callable[..., Any]:
    """Import a generated function from a Python file."""

    path = Path(python_function_path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import generated module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def compare_casadi_and_python_function(
    casadi_function: cs.Function,
    python_function_path: str | os.PathLike[str],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> None:
    """Compare a generated Python file against the source CasADi function."""

    python_function = import_python_function(python_function_path, casadi_function.name())
    inputs = random_inputs(casadi_function, rng=rng)
    casadi_outputs = _as_sequence(casadi_function(*inputs))
    python_outputs = _as_sequence(python_function(*inputs))

    if len(casadi_outputs) != len(python_outputs):
        raise ValueError(
            f"Output count mismatch: CasADi returned {len(casadi_outputs)}, "
            f"Python returned {len(python_outputs)}"
        )

    for i, (casadi_out, python_out) in enumerate(zip(casadi_outputs, python_outputs)):
        np.testing.assert_allclose(
            np.asarray(casadi_out),
            np.asarray(jax.device_get(python_out)),
            atol=atol,
            rtol=rtol,
            err_msg=f"Output {i} mismatch",
        )


def random_inputs(
    casadi_function: cs.Function,
    *,
    rng: np.random.Generator | None = None,
    low: float = -0.5,
    high: float = 0.5,
) -> list[np.ndarray]:
    """Generate NumPy inputs with shapes matching a CasADi function."""

    rng = rng or np.random.default_rng(0)
    return [
        rng.uniform(low, high, size=casadi_function.size_in(i))
        for i in range(casadi_function.n_in())
    ]


def random_inputs_with_key(
    casadi_function: cs.Function,
    batch_size: int,
    rng: jax.Array,
    *,
    low: float = -0.5,
    high: float = 0.5,
) -> list[jax.Array]:
    """Generate batched JAX inputs with shapes matching a CasADi function."""

    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    inputs = []
    for i in range(casadi_function.n_in()):
        rng, subkey = jax.random.split(rng)
        shape = (batch_size, *casadi_function.size_in(i))
        inputs.append(jax.random.uniform(subkey, shape, dtype=dtype, minval=low, maxval=high))
    return inputs


def casadi_to_python_code_from_c(
    casadi_function: cs.Function,
    output_file: str | os.PathLike[str],
    numpy_module: str = "jax.numpy",
) -> None:
    """Compatibility wrapper for the old prototype API.

    ``numpy_module`` is accepted for source compatibility.  The maintained
    converter always emits JAX code using ``jax.numpy``.
    """

    if numpy_module not in {"jax.numpy", "numpy"}:
        raise ValueError("Only 'jax.numpy' and legacy 'numpy' values are accepted")
    export(casadi_function, output_file, function_name=casadi_function.name())


def _validate_function(casadi_function: cs.Function) -> None:
    if not casadi_function.is_a("SXFunction"):
        raise TypeError(
            "numpysadi only supports CasADi SX functions. "
            f"Got {casadi_function.class_name()}."
        )
    for kind, count, sparsity_getter in (
        ("input", casadi_function.n_in(), casadi_function.sparsity_in),
        ("output", casadi_function.n_out(), casadi_function.sparsity_out),
    ):
        for i in range(count):
            sparsity = sparsity_getter(i)
            if not sparsity.is_dense():
                raise ValueError(
                    f"numpysadi only supports dense arrays; {kind} {i} has "
                    f"sparsity {sparsity}."
                )


def _generate_c_code(casadi_function: cs.Function) -> str:
    with tempfile.TemporaryDirectory(prefix="numpysadi_codegen_") as tmpdir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            c_name = f"{casadi_function.name()}.c"
            casadi_function.generate(c_name, {"with_header": False})
            return Path(c_name).read_text(encoding="utf-8")
        finally:
            os.chdir(old_cwd)


def _extract_c_function_body(c_code: str) -> str:
    match = re.search(
        r"static\s+int\s+casadi_f0\s*\([^)]*\)\s*\{",
        c_code,
        flags=re.MULTILINE,
    )
    if match is None:
        raise UnsupportedCodegenError("Could not find the generated casadi_f0 body")

    body_start = match.end()
    depth = 1
    i = body_start
    while i < len(c_code) and depth:
        if c_code[i] == "{":
            depth += 1
        elif c_code[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        raise UnsupportedCodegenError("Generated casadi_f0 body has unbalanced braces")
    return c_code[body_start : i - 1]


def _translate_c_body(
    body: str,
    casadi_function: cs.Function,
) -> tuple[list[str], dict[int, dict[int, str]]]:
    statements: list[str] = []
    output_values: dict[int, dict[int, str]] = {
        i: {} for i in range(casadi_function.n_out())
    }

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("/*"):
            continue
        if line.startswith("casadi_real ") or line.startswith("casadi_int "):
            continue
        if line == "return 0;":
            continue
        if line.endswith(";"):
            line = line[:-1].strip()

        input_match = re.fullmatch(
            r"([A-Za-z_]\w*)\s*=\s*arg\[(\d+)\]\s*\?\s*arg\[\2\]\[(\d+)\]\s*:\s*0",
            line,
        )
        if input_match:
            target, input_idx, flat_idx = input_match.groups()
            statements.append(f"{target} = arg{input_idx}_flat[{flat_idx}]")
            continue

        output_match = re.fullmatch(
            r"if\s*\(res\[(\d+)\]!=0\)\s*res\[\1\]\[(\d+)\]\s*=\s*(.+)",
            line,
        )
        if output_match:
            output_idx, flat_idx, expr = output_match.groups()
            name = _output_element_name(int(output_idx), int(flat_idx))
            statements.append(f"{name} = {_convert_c_expr(expr)}")
            output_values[int(output_idx)][int(flat_idx)] = name
            continue

        assignment_match = re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*(.+)", line)
        if assignment_match:
            target, expr = assignment_match.groups()
            statements.append(f"{target} = {_convert_c_expr(expr)}")
            continue

        raise UnsupportedCodegenError(f"Unsupported generated C line: {raw_line!r}")

    return statements, output_values


_WARP_MATH_FUNCTIONS = {
    "casadi_sq": "_numpysadi_warp_sq",
    "casadi_fabs": "wp.abs",
    "fabs": "wp.abs",
    "casadi_sign": "_numpysadi_warp_sign",
    "casadi_fmin": "wp.min",
    "fmin": "wp.min",
    "casadi_fmax": "wp.max",
    "fmax": "wp.max",
    "casadi_hypot": "_numpysadi_warp_hypot",
    "hypot": "_numpysadi_warp_hypot",
    "casadi_log1p": "_numpysadi_warp_log1p",
    "log1p": "_numpysadi_warp_log1p",
    "casadi_expm1": "_numpysadi_warp_expm1",
    "expm1": "_numpysadi_warp_expm1",
    "sqrt": "wp.sqrt",
    "sin": "wp.sin",
    "cos": "wp.cos",
    "tan": "wp.tan",
    "asin": "wp.asin",
    "acos": "wp.acos",
    "atan": "wp.atan",
    "atan2": "wp.atan2",
    "sinh": "wp.sinh",
    "cosh": "wp.cosh",
    "tanh": "wp.tanh",
    "asinh": "_numpysadi_warp_asinh",
    "acosh": "_numpysadi_warp_acosh",
    "atanh": "_numpysadi_warp_atanh",
    "exp": "wp.exp",
    "log": "wp.log",
    "floor": "wp.floor",
    "ceil": "wp.ceil",
    "fmod": "_numpysadi_warp_fmod",
    "remainder": "_numpysadi_warp_remainder",
    "copysign": "wp.copysign",
    "pow": "wp.pow",
    "erf": "wp.erf",
    "erfinv": "wp.erfinv",
}


def _warp_module_options(warp_compile_profile: str) -> dict[str, object]:
    base_options: dict[str, object] = {
        "mode": "release",
        "fast_math": True,
        "enable_backward": False,
        "block_dim": 256,
    }

    if warp_compile_profile == "runtime":
        return {
            **base_options,
            "optimization_level": 3,
            "fuse_fp": True,
            "cuda_output": "cubin",
        }

    if warp_compile_profile == "compile_time":
        return {
            **base_options,
            "optimization_level": 0,
            "fuse_fp": True,
            "cuda_output": "ptx",
        }

    raise ValueError(
        "warp_compile_profile must be 'runtime' or 'compile_time'"
    )


def _generate_warp_source(
    casadi_function: cs.Function,
    *,
    generated_name: str,
    input_names: Sequence[str],
    body: str,
    warp_dtype: str,
    warp_compile_profile: str,
) -> str:
    if warp_dtype != "float32":
        raise ValueError("The Warp backend currently supports only warp_dtype='float32'")

    statements, output_values = _translate_c_body_to_warp(body, casadi_function)
    kernel_name = f"_{generated_name}_warp_kernel"
    output_flat_sizes = [_flat_size(casadi_function.size_out(i)) for i in range(casadi_function.n_out())]
    output_dims = {f"out{i}_flat": size for i, size in enumerate(output_flat_sizes)}

    module_options = _warp_module_options(warp_compile_profile)

    lines = [
        "# This file was generated by numpysadi.",
        "import jax",
        "import jax.numpy as jnp",
        "import warp as wp",
        "",
        f"_numpysadi_warp_compile_profile = {warp_compile_profile!r}",
        f"_numpysadi_warp_module_options = {module_options!r}",
        "_numpysadi_warp_available_options = wp.get_module_options()",
        "wp.set_module_options(",
        "    {",
        "        key: value",
        "        for key, value in _numpysadi_warp_module_options.items()",
        "        if key in _numpysadi_warp_available_options",
        "    }",
        ")",
        "",
        "",
        "@wp.func",
        "def _numpysadi_warp_sq(x: float):",
        "    return x * x",
        "",
        "",
        "@wp.func",
        "def _numpysadi_warp_sign(x: float):",
        "    return wp.where(x > 0.0, 1.0, wp.where(x < 0.0, -1.0, 0.0))",
        "",
        "",
        "@wp.func",
        "def _numpysadi_warp_hypot(x: float, y: float):",
        "    return wp.sqrt(x * x + y * y)",
        "",
        "",
        "@wp.func",
        "def _numpysadi_warp_log1p(x: float):",
        "    return wp.log(1.0 + x)",
        "",
        "",
        "@wp.func",
        "def _numpysadi_warp_expm1(x: float):",
        "    return wp.exp(x) - 1.0",
        "",
        "",
        "@wp.func",
        "def _numpysadi_warp_asinh(x: float):",
        "    return wp.log(x + wp.sqrt(x * x + 1.0))",
        "",
        "",
        "@wp.func",
        "def _numpysadi_warp_acosh(x: float):",
        "    return wp.log(x + wp.sqrt(x * x - 1.0))",
        "",
        "",
        "@wp.func",
        "def _numpysadi_warp_atanh(x: float):",
        "    return 0.5 * wp.log((1.0 + x) / (1.0 - x))",
        "",
        "",
        "@wp.func",
        "def _numpysadi_warp_fmod(x: float, y: float):",
        "    return x - y * wp.trunc(x / y)",
        "",
        "",
        "@wp.func",
        "def _numpysadi_warp_remainder(x: float, y: float):",
        "    return x - y * wp.rint(x / y)",
        "",
        "",
        "@wp.kernel",
    ]

    kernel_args = [
        f"arg{i}_flat: wp.array(dtype=wp.float32)" for i in range(casadi_function.n_in())
    ]
    kernel_args.extend(
        f"out{i}_flat: wp.array(dtype=wp.float32)" for i in range(casadi_function.n_out())
    )
    if len(kernel_args) <= 2:
        lines.append(f"def {kernel_name}({', '.join(kernel_args)}):")
    else:
        lines.append(f"def {kernel_name}(")
        for arg in kernel_args:
            lines.append(f"    {arg},")
        lines.append("):")

    lines.append("    _sample = wp.tid()")
    for statement in statements:
        lines.append(f"    {statement}")
    for output_idx, flat_size in enumerate(output_flat_sizes):
        for flat_idx in range(flat_size):
            expr = output_values[output_idx].get(flat_idx, "0.0")
            lines.append(
                f"    out{output_idx}_flat[_sample * {flat_size} + {flat_idx}] = wp.float32({expr})"
            )

    lines.extend(
        [
            "",
            "",
            f"_numpysadi_warp_output_dims = {output_dims!r}",
            f"_numpysadi_warp_jax_kernel = wp.jax_kernel(",
            f"    {kernel_name},",
            f"    num_outputs={casadi_function.n_out()},",
            "    launch_dims=1,",
            "    output_dims=_numpysadi_warp_output_dims,",
            "    vmap_method=\"broadcast_all\",",
            ")",
            "",
            "",
            f"def {generated_name}({', '.join(input_names)}):",
        ]
    )

    if not input_names:
        lines.append("    _kernel_result = _numpysadi_warp_jax_kernel()")
    else:
        for i, input_name in enumerate(input_names):
            lines.append(
                f"    arg{i}_flat = jnp.ravel(jnp.asarray({input_name}, dtype=jnp.float32).T)"
            )
        lines.append(
            "    _kernel_result = _numpysadi_warp_jax_kernel("
            + ", ".join(f"arg{i}_flat" for i in range(casadi_function.n_in()))
            + ")"
        )

    if casadi_function.n_out() == 1:
        lines.append("    (out0_flat,) = _kernel_result")
    else:
        lines.append(
            "    "
            + ", ".join(f"out{i}_flat" for i in range(casadi_function.n_out()))
            + " = _kernel_result"
        )

    for i in range(casadi_function.n_out()):
        rows, cols = casadi_function.size_out(i)
        lines.append(f"    out{i} = jnp.reshape(out{i}_flat, ({cols}, {rows})).T")
    lines.append(f"    return {_return_expression(casadi_function)}")
    lines.append("")

    return "\n".join(lines)


def _translate_c_body_to_warp(
    body: str,
    casadi_function: cs.Function,
) -> tuple[list[str], dict[int, dict[int, str]]]:
    statements: list[str] = []
    output_values: dict[int, dict[int, str]] = {
        i: {} for i in range(casadi_function.n_out())
    }
    input_flat_sizes = [_flat_size(casadi_function.size_in(i)) for i in range(casadi_function.n_in())]

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("/*"):
            continue
        if line.startswith("casadi_real ") or line.startswith("casadi_int "):
            continue
        if line == "return 0;":
            continue
        if line.endswith(";"):
            line = line[:-1].strip()

        input_match = re.fullmatch(
            r"([A-Za-z_]\w*)\s*=\s*arg\[(\d+)\]\s*\?\s*arg\[\2\]\[(\d+)\]\s*:\s*0",
            line,
        )
        if input_match:
            target, input_idx, flat_idx = input_match.groups()
            input_idx_int = int(input_idx)
            input_flat_size = input_flat_sizes[input_idx_int]
            statements.append(
                f"{target} = arg{input_idx}_flat[_sample * {input_flat_size} + {flat_idx}]"
            )
            continue

        output_match = re.fullmatch(
            r"if\s*\(res\[(\d+)\]!=0\)\s*res\[\1\]\[(\d+)\]\s*=\s*(.+)",
            line,
        )
        if output_match:
            output_idx, flat_idx, expr = output_match.groups()
            name = _output_element_name(int(output_idx), int(flat_idx))
            statements.append(f"{name} = wp.float32({_convert_c_expr_warp(expr)})")
            output_values[int(output_idx)][int(flat_idx)] = name
            continue

        assignment_match = re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*(.+)", line)
        if assignment_match:
            target, expr = assignment_match.groups()
            statements.append(f"{target} = wp.float32({_convert_c_expr_warp(expr)})")
            continue

        raise UnsupportedCodegenError(f"Unsupported generated C line: {raw_line!r}")

    return statements, output_values


def _convert_c_expr_warp(expr: str) -> str:
    expr = _strip_outer_parentheses(expr.strip())

    ternary = _split_top_level_ternary(expr)
    if ternary is not None:
        condition, if_true, if_false = ternary
        converted_condition = _warp_truthy_expr(_convert_c_expr_warp(condition))
        return (
            "wp.where("
            f"{converted_condition}, "
            f"wp.float32({_convert_c_expr_warp(if_true)}), "
            f"wp.float32({_convert_c_expr_warp(if_false)})"
            ")"
        )

    for operator, replacement in (("||", "or"), ("&&", "and")):
        split = _split_top_level_binary(expr, operator)
        if split is not None:
            left, right = split
            return (
                f"({_warp_truthy_expr(_convert_c_expr_warp(left))} "
                f"{replacement} "
                f"{_warp_truthy_expr(_convert_c_expr_warp(right))})"
            )

    if expr.startswith("!"):
        return f"(not {_warp_truthy_expr(_convert_c_expr_warp(expr[1:]))})"

    expr = expr.replace("casadi_inf", "float('inf')")
    expr = expr.replace("casadi_nan", "float('nan')")
    expr = re.sub(r"\bINFINITY\b", "float('inf')", expr)
    expr = re.sub(r"\bNAN\b", "float('nan')", expr)

    for c_name in sorted(_WARP_MATH_FUNCTIONS, key=len, reverse=True):
        warp_name = _WARP_MATH_FUNCTIONS[c_name]
        expr = re.sub(rf"(?<![\w.]){re.escape(c_name)}\s*\(", f"{warp_name}(", expr)

    return expr


def _warp_truthy_expr(expr: str) -> str:
    expr = _strip_outer_parentheses(expr.strip())
    if re.search(r"(?:[<>]=?|[!=]=)|\band\b|\bor\b|\bnot\b", expr):
        return f"({expr})"
    return f"(({expr}) != 0.0)"


def _convert_c_expr(expr: str) -> str:
    expr = _strip_outer_parentheses(expr.strip())

    ternary = _split_top_level_ternary(expr)
    if ternary is not None:
        condition, if_true, if_false = ternary
        return (
            "jnp.where("
            f"{_convert_c_expr(condition)}, "
            f"{_convert_c_expr(if_true)}, "
            f"{_convert_c_expr(if_false)}"
            ")"
        )

    for operator, replacement in (("||", "jnp.logical_or"), ("&&", "jnp.logical_and")):
        split = _split_top_level_binary(expr, operator)
        if split is not None:
            left, right = split
            return f"{replacement}({_convert_c_expr(left)}, {_convert_c_expr(right)})"

    if expr.startswith("!"):
        return f"jnp.logical_not({_convert_c_expr(expr[1:])})"

    expr = expr.replace("casadi_inf", "jnp.inf")
    expr = expr.replace("casadi_nan", "jnp.nan")
    expr = re.sub(r"\bINFINITY\b", "jnp.inf", expr)
    expr = re.sub(r"\bNAN\b", "jnp.nan", expr)

    for c_name in sorted(_C_MATH_FUNCTIONS, key=len, reverse=True):
        py_name = _C_MATH_FUNCTIONS[c_name]
        expr = re.sub(rf"(?<![\w.]){re.escape(c_name)}\s*\(", f"{py_name}(", expr)

    return expr


def _split_top_level_ternary(expr: str) -> tuple[str, str, str] | None:
    depth = 0
    question_index: int | None = None
    nested_ternaries = 0
    for i, char in enumerate(expr):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif depth == 0 and char == "?":
            if question_index is None:
                question_index = i
            else:
                nested_ternaries += 1
        elif depth == 0 and char == ":" and question_index is not None:
            if nested_ternaries:
                nested_ternaries -= 1
            else:
                return (
                    expr[:question_index].strip(),
                    expr[question_index + 1 : i].strip(),
                    expr[i + 1 :].strip(),
                )
    return None


def _split_top_level_binary(expr: str, operator: str) -> tuple[str, str] | None:
    depth = 0
    i = len(expr) - len(operator)
    while i >= 0:
        char = expr[i]
        if char in ")]":
            depth += 1
        elif char in "([":
            depth -= 1
        elif depth == 0 and expr.startswith(operator, i):
            return expr[:i].strip(), expr[i + len(operator) :].strip()
        i -= 1
    return None


def _strip_outer_parentheses(expr: str) -> str:
    while expr.startswith("(") and expr.endswith(")"):
        depth = 0
        encloses_entire_expression = True
        for i, char in enumerate(expr):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and i != len(expr) - 1:
                    encloses_entire_expression = False
                    break
        if not encloses_entire_expression:
            break
        expr = expr[1:-1].strip()
    return expr


def _output_construction_lines(
    casadi_function: cs.Function,
    output_values: dict[int, dict[int, str]],
) -> list[str]:
    lines: list[str] = []
    for i in range(casadi_function.n_out()):
        rows, cols = casadi_function.size_out(i)
        flat_size = rows * cols
        elements = [
            output_values[i].get(j, "0.0")
            for j in range(flat_size)
        ]
        output_name = _output_name(i)
        if flat_size == 1:
            lines.append(f"{output_name}_flat = jnp.asarray([{elements[0]}], dtype=_dtype)")
        else:
            lines.append(
                f"{output_name}_flat = jnp.asarray([{', '.join(elements)}], dtype=_dtype)"
            )
        lines.append(f"{output_name} = jnp.reshape({output_name}_flat, ({cols}, {rows})).T")
    return lines


def _return_expression(casadi_function: cs.Function) -> str:
    output_names = [_output_name(i) for i in range(casadi_function.n_out())]
    if len(output_names) == 1:
        return output_names[0]
    return "(" + ", ".join(output_names) + ")"


def _input_names(casadi_function: cs.Function) -> list[str]:
    names = []
    used: set[str] = set()
    for i in range(casadi_function.n_in()):
        raw = casadi_function.name_in(i) or f"arg{i}"
        name = _unique_identifier(raw, used=used, fallback=f"arg{i}")
        names.append(name)
    return names


def _flat_size(size: tuple[int, int]) -> int:
    rows, cols = size
    return rows * cols


def _output_name(index: int) -> str:
    return f"out{index}"


def _output_element_name(output_idx: int, flat_idx: int) -> str:
    return f"_out{output_idx}_{flat_idx}"


def _unique_identifier(
    value: str,
    *,
    used: set[str] | None = None,
    fallback: str = "generated_function",
) -> str:
    used = used if used is not None else set()
    candidate = re.sub(r"\W", "_", value).strip("_") or fallback
    if candidate[0].isdigit():
        candidate = f"_{candidate}"
    if keyword.iskeyword(candidate):
        candidate = f"{candidate}_"

    base = candidate
    suffix = 1
    while candidate in used:
        suffix += 1
        candidate = f"{base}_{suffix}"
    used.add(candidate)
    return candidate


def _as_sequence(value: Any) -> list[Any]:
    if isinstance(value, (tuple, list)):
        return list(value)
    return [value]


def _block_until_ready(value: Any) -> None:
    leaves = jax.tree.leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def benchmark_function(
    f: Callable[..., Any],
    casadi_function: cs.Function,
    batch_sizes: Sequence[int],
    repetitions: int = 100,
) -> list[dict[str, float | int]]:
    """Small convenience benchmark for a generated function.

    For more complete numpysadi-vs-jaxadi measurements, use
    ``benchmarks/benchmark_synthetic.py``.
    """

    results: list[dict[str, float | int]] = []
    for batch_size in batch_sizes:
        inputs = random_inputs_with_key(casadi_function, batch_size, jax.random.PRNGKey(0))
        vmapped = jax.jit(jax.vmap(f))

        import time

        start = time.perf_counter()
        compiled = vmapped.lower(*inputs).compile()
        _block_until_ready(compiled(*inputs))
        compile_seconds = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(repetitions):
            _block_until_ready(compiled(*inputs))
        execution_seconds = (time.perf_counter() - start) / repetitions

        results.append(
            {
                "batch_size": batch_size,
                "compile_seconds": compile_seconds,
                "execution_seconds": execution_seconds,
                "instructions": casadi_function.n_instructions(),
            }
        )
    return results
