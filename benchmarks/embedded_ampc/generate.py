from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import shutil
import subprocess

import numpy as np

from warpmpc.jax_ampc import (
    CodeExportOptions,
    DenseLayerSpec,
    MLPExportSpec,
    QuantizationConfig,
    export_mlp_policy,
    make_test_cases,
)

from benchmarks.embedded_ampc.config import (
    BACKENDS,
    CMSIS_DSP_ROOT,
    CMSIS_NN_ROOT,
    CMSIS_VENDOR_ROOT,
    EIGEN_ROOT,
    INPUT_DIM,
    MODM_ROOT,
    OUTPUT_DIM,
    PRECISIONS,
    TARGETS,
    TEST_CASE_COUNT,
    BenchmarkCase,
    iter_cases,
    section_name,
)


DEFAULT_OUTPUT_ROOT = pathlib.Path(__file__).resolve().parent / "generated"
BENCHMARK_REPETITIONS = 10


def build_spec(case: BenchmarkCase) -> tuple[MLPExportSpec, np.ndarray]:
    seed = int.from_bytes(hashlib.sha256(case.case_id.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    layers: list[DenseLayerSpec] = []
    in_dim = INPUT_DIM
    negative_slope = 0.02 if case.activation == "leaky_relu" else 0.0
    for _ in range(case.hidden_layers):
        scale = 0.55 / math.sqrt(float(in_dim))
        layers.append(
            DenseLayerSpec(
                weight=rng.normal(scale=scale, size=(case.width, in_dim)),
                bias=rng.normal(scale=0.03, size=case.width),
                activation=case.activation,
                negative_slope=negative_slope,
            )
        )
        in_dim = case.width
    layers.append(
        DenseLayerSpec(
            weight=rng.normal(scale=0.45 / math.sqrt(float(in_dim)), size=(OUTPUT_DIM, in_dim)),
            bias=rng.normal(scale=0.03, size=OUTPUT_DIM),
            activation="linear",
        )
    )
    spec = MLPExportSpec(
        name=case.case_id,
        layers=tuple(layers),
        x_mean=np.zeros(INPUT_DIM, dtype=np.float64),
        x_std=np.ones(INPUT_DIM, dtype=np.float64),
        y_mean=np.zeros(OUTPUT_DIM, dtype=np.float64),
        y_std=np.ones(OUTPUT_DIM, dtype=np.float64),
        y_clip_low=np.full(OUTPUT_DIM, -10.0, dtype=np.float64),
        y_clip_high=np.full(OUTPUT_DIM, 10.0, dtype=np.float64),
        model_config={
            "hidden_sizes": [case.width] * case.hidden_layers,
            "output_dim": OUTPUT_DIM,
            "activation": case.activation,
            "negative_slope": negative_slope,
        },
        metadata={"benchmark_case": case.case_id},
    )
    inputs = rng.normal(size=(TEST_CASE_COUNT, INPUT_DIM))
    return spec, inputs


def generate_case(
    case: BenchmarkCase,
    *,
    output_root: pathlib.Path = DEFAULT_OUTPUT_ROOT,
    host_check: bool = True,
) -> pathlib.Path:
    if case.target not in TARGETS:
        raise ValueError(f"unknown target {case.target!r}")
    if case.backend not in BACKENDS:
        raise ValueError(f"unknown backend {case.backend!r}")
    if case.precision not in PRECISIONS:
        raise ValueError(f"unknown precision {case.precision!r}")

    project_dir = output_root / TARGETS[case.target].label / case.case_id
    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    spec, inputs = build_spec(case)
    test_cases = make_test_cases(spec, inputs)
    quantization = QuantizationConfig(qtype=case.precision) if case.is_quantized else None
    export_mlp_policy(
        spec,
        project_dir,
        test_cases=test_cases,
        options=CodeExportOptions(
            prefix=case.prefix,
            backend=case.backend,
            precision="float32_t",
            generate_example_main=True,
            test_tolerance=0.08 if case.is_quantized else 1e-4,
            quantization=quantization,
        ),
    )

    _patch_data_section(project_dir / f"{case.prefix}_data.c", case.section)
    if case.backend == "cmsis" and case.is_quantized:
        _copy_cmsis_nn_subset(project_dir, case.precision)
        _patch_cmsis_nn_includes(project_dir / f"{case.prefix}_forward.c")
    if case.is_quantized and case.backend in ("simple", "cmsis"):
        _refresh_quantized_test_outputs(project_dir, case, test_cases.inputs)

    _write_main_cpp(project_dir, case)
    _write_project_xml(project_dir, case)
    _write_case_manifest(project_dir, case)

    if host_check:
        run_host_check(project_dir, case)
    _hide_generated_example_main(project_dir, case)
    return project_dir


def run_host_check(project_dir: pathlib.Path, case: BenchmarkCase) -> None:
    compiler = shutil.which("g++" if case.backend == "eigen" else "cc")
    if compiler is None:
        raise RuntimeError("required host compiler is not available")
    source_suffix = "cpp" if case.backend == "eigen" else "c"
    example_main = project_dir / f"{case.prefix}_example_main.{source_suffix}"
    if not example_main.exists():
        _write_host_check_main(project_dir, case, example_main)
    exe = project_dir / f"{case.prefix}_host_check"
    cmd = [compiler, "-std=c++17" if case.backend == "eigen" else "-std=c99", "-Wall", "-Wextra", "-I", str(project_dir)]
    if case.backend == "eigen":
        cmd += ["-I", str(EIGEN_ROOT)]
    if case.backend == "cmsis":
        if case.is_quantized:
            cmd += [
                "-I",
                str(project_dir),
                "-I",
                str(CMSIS_VENDOR_ROOT / "DSP" / "Include"),
                "-I",
                str(CMSIS_VENDOR_ROOT / "Core" / "Include"),
            ]
        else:
            cmd += [
                "-I",
                str(CMSIS_DSP_ROOT / "Include"),
                "-I",
                str(CMSIS_DSP_ROOT / "PrivateInclude"),
                "-I",
                str(CMSIS_VENDOR_ROOT / "Core" / "Include"),
                "-I",
                str(CMSIS_DSP_ROOT / "Source" / "MatrixFunctions"),
            ]
    cmd += [
        str(project_dir / f"{case.prefix}_data.c"),
        str(project_dir / f"{case.prefix}_test_data.c"),
    ]
    if case.backend == "eigen":
        cmd.append(str(example_main))
    else:
        cmd += [str(project_dir / f"{case.prefix}_forward.c"), str(example_main)]
    if case.backend == "cmsis":
        if case.is_quantized:
            cmd += _cmsis_nn_host_sources(project_dir, case.precision)
        else:
            cmd += [
                str(CMSIS_DSP_ROOT / "Source" / "MatrixFunctions" / "arm_mat_init_f32.c"),
                str(CMSIS_DSP_ROOT / "Source" / "MatrixFunctions" / "arm_mat_vec_mult_f32.c"),
            ]
    cmd += ["-lm", "-o", str(exe)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    subprocess.run([str(exe)], check=True, capture_output=True, text=True)


def _refresh_quantized_test_outputs(
    project_dir: pathlib.Path,
    case: BenchmarkCase,
    inputs: np.ndarray,
) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("required host C compiler is not available")
    p = case.prefix
    u = p.upper()
    oracle_main = project_dir / f"{p}_oracle_main.c"
    oracle_exe = project_dir / f"{p}_oracle"
    oracle_main.write_text(
        f"""#include "{p}_forward.h"
#include "{p}_test_data.h"

#include <stdio.h>

int main(void)
{{
    {p}_workspace_t workspace;
    {p}_float_t output[{u}_OUTPUT_DIM];
    for (uint16_t tc = 0u; tc < {u}_TEST_CASE_COUNT; ++tc)
    {{
        if ({p}_forward({p}_test_cases[tc].input, output, &workspace) != 0)
        {{
            return 2;
        }}
        for (uint16_t i = 0u; i < {u}_OUTPUT_DIM; ++i)
        {{
            printf("%s%.17g", i == 0u ? "" : " ", (double)output[i]);
        }}
        printf("\\n");
    }}
    return 0;
}}
""",
        encoding="utf-8",
    )
    include_args = ["-I", str(project_dir)]
    if case.backend == "cmsis":
        include_args += [
            "-I",
            str(CMSIS_VENDOR_ROOT / "DSP" / "Include"),
            "-I",
            str(CMSIS_VENDOR_ROOT / "Core" / "Include"),
        ]
    source_args = [
        str(project_dir / f"{p}_data.c"),
        str(project_dir / f"{p}_test_data.c"),
        str(project_dir / f"{p}_forward.c"),
    ]
    if case.backend == "cmsis":
        source_args += _cmsis_nn_host_sources(project_dir, case.precision)
    source_args.append(str(oracle_main))

    cmd = [
        compiler,
        "-std=c99",
        "-Wall",
        "-Wextra",
        *include_args,
        *source_args,
        "-lm",
        "-o",
        str(oracle_exe),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        completed = subprocess.run([str(oracle_exe)], check=True, capture_output=True, text=True)
    finally:
        oracle_main.unlink(missing_ok=True)
    outputs = np.asarray(
        [[float(value) for value in line.split()] for line in completed.stdout.splitlines() if line.strip()],
        dtype=np.float64,
    )
    if outputs.shape != (inputs.shape[0], OUTPUT_DIM):
        raise RuntimeError(f"CMSIS oracle produced shape {outputs.shape}, expected {(inputs.shape[0], OUTPUT_DIM)}")
    _rewrite_test_data_source(project_dir, case, inputs, outputs)


def _rewrite_test_data_source(
    project_dir: pathlib.Path,
    case: BenchmarkCase,
    inputs: np.ndarray,
    outputs: np.ndarray,
) -> None:
    p = case.prefix
    u = p.upper()
    source = f"""/* Auto-generated AMPC policy test data. */
#include "{p}_test_data.h"

#ifdef __cplusplus
extern "C" {{
#endif

const {p}_test_case_t {p}_test_cases[{u}_TEST_CASE_COUNT] = {{
"""
    for i in range(inputs.shape[0]):
        comma = "," if i < inputs.shape[0] - 1 else ""
        source += f"    /* test case {i} */\n"
        source += "    {\n"
        source += f"        .input = {{{_format_c_float_array(inputs[i])}}},\n"
        source += f"        .output = {{{_format_c_float_array(outputs[i])}}}\n"
        source += f"    }}{comma}\n"
    source += "};\n\n#ifdef __cplusplus\n}\n#endif\n"
    (project_dir / f"{p}_test_data.c").write_text(source, encoding="utf-8")


def _format_c_float_array(values: np.ndarray) -> str:
    return ", ".join(_format_c_float(float(value)) for value in np.asarray(values).reshape(-1))


def _format_c_float(value: float) -> str:
    text = f"{value:.9g}"
    if "e" not in text and "E" not in text and "." not in text:
        text += ".0"
    return f"{text}f"


def _write_host_check_main(project_dir: pathlib.Path, case: BenchmarkCase, path: pathlib.Path) -> None:
    if case.backend == "eigen":
        raise RuntimeError(f"missing generated Eigen example main at {path}")
    tol = 0.08 if case.is_quantized else 1e-4
    p = case.prefix
    u = p.upper()
    text = f"""#include "{p}_forward.h"
#include "{p}_test_data.h"

#include <math.h>
#include <stdio.h>

int main(void)
{{
    {p}_workspace_t workspace;
    {p}_float_t output[{u}_OUTPUT_DIM];
    double max_abs_error = 0.0;

    for (uint16_t tc = 0u; tc < {u}_TEST_CASE_COUNT; ++tc)
    {{
        if ({p}_forward({p}_test_cases[tc].input, output, &workspace) != 0)
        {{
            return 2;
        }}
        for (uint16_t i = 0u; i < {u}_OUTPUT_DIM; ++i)
        {{
            const double err = fabs((double)output[i] - (double){p}_test_cases[tc].output[i]);
            if (err > max_abs_error)
            {{
                max_abs_error = err;
            }}
        }}
    }}

    printf("AMPC policy tests: %u cases, max_abs_error=%.9g\\n", (unsigned){u}_TEST_CASE_COUNT, max_abs_error);
    return max_abs_error <= {tol:.9g} ? 0 : 1;
}}
"""
    path.write_text(text, encoding="utf-8")


def _cmsis_nn_host_sources(project_dir: pathlib.Path, precision: str) -> list[str]:
    sources = [
        project_dir / ("arm_relu_q15.c" if precision == "q15" else "arm_relu_q7.c"),
        project_dir / "arm_nn_activation_s16.c",
        project_dir / "arm_nntables.c",
    ]
    if precision == "q4":
        sources += [project_dir / "arm_fully_connected_s4.c", project_dir / "arm_nn_vec_mat_mult_t_s4.c"]
    if precision == "q7":
        sources += [
            project_dir / "arm_fully_connected_s8.c",
            project_dir / "arm_nn_vec_mat_mult_t_s8.c",
        ]
    if precision == "q15":
        sources += [project_dir / "arm_fully_connected_s16.c", project_dir / "arm_nn_vec_mat_mult_t_s16.c"]
    return [str(path) for path in sources]


def _hide_generated_example_main(project_dir: pathlib.Path, case: BenchmarkCase) -> None:
    suffix = "cpp" if case.backend == "eigen" else "c"
    path = project_dir / f"{case.prefix}_example_main.{suffix}"
    if path.exists():
        path.rename(project_dir / f"{case.prefix}_example_main.{suffix}.hostcheck")


def _copy_cmsis_nn_subset(project_dir: pathlib.Path, precision: str) -> None:
    include_dir = CMSIS_NN_ROOT / "Include"
    for header in include_dir.rglob("*.h"):
        destination = project_dir / header.relative_to(include_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(header, destination)
    source_dir = CMSIS_NN_ROOT / "Source"
    shutil.copy2(source_dir / "ActivationFunctions" / ("arm_relu_q15.c" if precision == "q15" else "arm_relu_q7.c"), project_dir)
    shutil.copy2(source_dir / "ActivationFunctions" / "arm_nn_activation_s16.c", project_dir)
    shutil.copy2(source_dir / "NNSupportFunctions" / "arm_nntables.c", project_dir)
    if precision == "q4":
        shutil.copy2(source_dir / "FullyConnectedFunctions" / "arm_fully_connected_s4.c", project_dir)
        shutil.copy2(source_dir / "NNSupportFunctions" / "arm_nn_vec_mat_mult_t_s4.c", project_dir)
    if precision == "q7":
        shutil.copy2(source_dir / "FullyConnectedFunctions" / "arm_fully_connected_s8.c", project_dir)
        shutil.copy2(source_dir / "NNSupportFunctions" / "arm_nn_vec_mat_mult_t_s8.c", project_dir)
    if precision == "q15":
        shutil.copy2(source_dir / "FullyConnectedFunctions" / "arm_fully_connected_s16.c", project_dir)
        shutil.copy2(source_dir / "NNSupportFunctions" / "arm_nn_vec_mat_mult_t_s16.c", project_dir)


def _patch_cmsis_nn_includes(path: pathlib.Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace("#include <arm_nnfunctions.h>", '#include "arm_nnfunctions.h"')
    path.write_text(text, encoding="utf-8")


def _patch_data_section(data_source: pathlib.Path, section: str) -> None:
    actual_section = section_name(section)
    if actual_section is None:
        return
    text = data_source.read_text(encoding="utf-8")
    define = (
        f'#ifndef AMPC_BENCH_DATA_SECTION\n'
        f'#define AMPC_BENCH_DATA_SECTION __attribute__((section("{actual_section}")))\n'
        f'#endif\n\n'
    )
    text = text.replace("#include", define + "#include", 1)
    text = re.sub(r"^static const ", "static AMPC_BENCH_DATA_SECTION const ", text, flags=re.MULTILINE)
    text = re.sub(r"^const ", "AMPC_BENCH_DATA_SECTION const ", text, flags=re.MULTILINE)
    data_source.write_text(text, encoding="utf-8")


def _write_main_cpp(project_dir: pathlib.Path, case: BenchmarkCase) -> None:
    calls = _forward_call_snippet(case)
    include_forward = (
        f'#include "{case.prefix}_forward.hpp"\n'
        if case.backend == "eigen"
        else f'extern "C" {{\n#include "{case.prefix}_forward.h"\n}}\n'
    )
    test_include = f'extern "C" {{\n#include "{case.prefix}_test_data.h"\n}}\n'
    text = f"""#define MODM_LOG_LEVEL modm::log::INFO

#include <modm/board.hpp>
#include <modm/driver/time/cycle_counter.hpp>

#include <cmath>
#include <cstdint>

{include_forward}{test_include}

using namespace Board;

modm_fastdata modm::CycleCounter counter;
static volatile {case.prefix}_float_t output_sink[{case.prefix.upper()}_OUTPUT_DIM];

{_workspace_decl(case)}

static int
run_forward(const {case.prefix}_float_t *input, {case.prefix}_float_t *output)
{{
{calls}
}}

static double
check_outputs()
{{
    {case.prefix}_float_t output[{case.prefix.upper()}_OUTPUT_DIM];
    double max_abs_error = 0.0;
    for (uint16_t tc = 0; tc < {case.prefix.upper()}_TEST_CASE_COUNT; ++tc)
    {{
        if (run_forward({case.prefix}_test_cases[tc].input, output) != 0)
        {{
            return -1.0;
        }}
        for (uint16_t i = 0; i < {case.prefix.upper()}_OUTPUT_DIM; ++i)
        {{
            const double err = std::fabs((double)output[i] - (double){case.prefix}_test_cases[tc].output[i]);
            if (err > max_abs_error) max_abs_error = err;
        }}
    }}
    return max_abs_error;
}}

static void
print_benchmark_failure(double max_abs_error)
{{
    MODM_LOG_INFO << "BENCH_FAIL case={case.case_id} max_abs_error=" << max_abs_error << modm::endl;
    MODM_LOG_INFO.flush();
}}

static void
print_benchmark_result(uint32_t repetitions, uint32_t cycles, uint32_t cycles_per, double us_per_inference, double max_abs_error)
{{
    MODM_LOG_INFO
        << "BENCH_RESULT {{"
        << "\\"case_id\\":\\"{case.case_id}\\","
        << "\\"target\\":\\"{case.target}\\","
        << "\\"backend\\":\\"{case.backend}\\","
        << "\\"precision\\":\\"{case.precision}\\","
        << "\\"activation\\":\\"{case.activation}\\","
        << "\\"hidden_layers\\":" << {case.hidden_layers} << ","
        << "\\"width\\":" << {case.width} << ","
        << "\\"section\\":\\"{case.section}\\","
        << "\\"repetitions\\":" << repetitions << ","
        << "\\"cycles_total\\":" << cycles << ","
        << "\\"cycles_per_inference\\":" << cycles_per << ","
        << "\\"us_per_inference\\":" << us_per_inference << ","
        << "\\"max_abs_error\\":" << max_abs_error
        << "}}" << modm::endl;
    MODM_LOG_INFO << "BENCH_DONE" << modm::endl;
    MODM_LOG_INFO.flush();
}}

int
main()
{{
    Board::initialize();
    counter.initialize();
    modm::delay(1500ms);

    const double max_abs_error = check_outputs();
    if (max_abs_error < 0.0 || max_abs_error > {(0.08 if case.is_quantized else 1e-4):.9g})
    {{
        while (true)
        {{
            print_benchmark_failure(max_abs_error);
            modm::delay(1s);
        }}
    }}

    {case.prefix}_float_t output[{case.prefix.upper()}_OUTPUT_DIM];
    const uint32_t repetitions = {BENCHMARK_REPETITIONS}u;
    counter.start();
    for (uint32_t rep = 0; rep < repetitions; ++rep)
    {{
        const uint16_t tc = rep % {case.prefix.upper()}_TEST_CASE_COUNT;
        run_forward({case.prefix}_test_cases[tc].input, output);
        output_sink[rep % {case.prefix.upper()}_OUTPUT_DIM] = output[rep % {case.prefix.upper()}_OUTPUT_DIM];
    }}
    counter.stop();
    const uint32_t cycles = counter.cycles();
    const uint32_t cycles_per = cycles / repetitions;
    const double us_per_inference = ((double)cycles_per * 1000000.0) / (double)Board::SystemClock::Frequency;

    while (true)
    {{
        print_benchmark_result(repetitions, cycles, cycles_per, us_per_inference, max_abs_error);
        modm::delay(1s);
    }}
}}
"""
    (project_dir / "main.cpp").write_text(text, encoding="utf-8")


def _workspace_decl(case: BenchmarkCase) -> str:
    if case.backend == "eigen":
        return ""
    return f"static {case.prefix}_workspace_t workspace;"


def _forward_call_snippet(case: BenchmarkCase) -> str:
    if case.backend == "eigen":
        return f"    return {case.prefix}::forward(input, output);"
    return f"    return {case.prefix}_forward(input, output, &workspace);"


def _write_project_xml(project_dir: pathlib.Path, case: BenchmarkCase) -> None:
    target = TARGETS[case.target]
    repo_path = pathlib.Path(MODM_ROOT / "repo.lb").relative_to(project_dir, walk_up=True)
    modules = ["modm:build:scons", "modm:driver:cycle_counter"]
    if case.backend == "eigen":
        modules.append("modm:eigen")
    if case.backend == "cmsis":
        modules.append("modm:cmsis:dsp:matrix" if case.precision == "f32" else "modm:cmsis:dsp:fast_math")
    options = [("modm:build:build.path", "build"), *target.cache_options]
    text = ["<library>", "  <repositories>", "    <repository>", f"      <path>{repo_path}</path>", "    </repository>", "  </repositories>", "", f"  <extends>{target.modm_extends}</extends>", "  <options>"]
    for name, value in options:
        text.append(f'    <option name="{name}">{value}</option>')
    text += ["  </options>", "  <collectors>", '    <collect name="modm:build:path.include">.</collect>', "  </collectors>", "  <modules>"]
    for module in modules:
        text.append(f"    <module>{module}</module>")
    text += ["  </modules>", "</library>", ""]
    (project_dir / "project.xml").write_text("\n".join(text), encoding="utf-8")


def _write_case_manifest(project_dir: pathlib.Path, case: BenchmarkCase) -> None:
    (project_dir / "benchmark_case.json").write_text(
        json.dumps(
            {
                "case_id": case.case_id,
                "target": case.target,
                "hidden_layers": case.hidden_layers,
                "width": case.width,
                "activation": case.activation,
                "backend": case.backend,
                "precision": case.precision,
                "section": case.section,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--hidden-layers", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--activation")
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--precision", choices=PRECISIONS)
    parser.add_argument("--section")
    parser.add_argument("--case-id")
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-host-check", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cases = iter_cases(
        target=args.target,
        hidden_layers=args.hidden_layers,
        width=args.width,
        activation=args.activation,
        backend=args.backend,
        precision=args.precision,
        section=args.section,
    )
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
    if args.list:
        for case in cases:
            print(case.case_id)
        return
    for case in cases:
        project_dir = generate_case(case, output_root=args.output_root, host_check=not args.no_host_check)
        print(project_dir)


if __name__ == "__main__":
    main()
