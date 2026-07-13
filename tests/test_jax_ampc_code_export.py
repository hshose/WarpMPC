from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization

from warpmpc.jax_ampc import (
    CodeExportOptions,
    MLP,
    QuantizationConfig,
    export_mlp_policy,
    load_mlp_checkpoint,
    make_test_cases,
    quantize_mlp_policy,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CMSIS_DSP_ROOT = REPO_ROOT / "resources" / "CMSIS-DSP"
CMSIS_CORE_INCLUDE = (
    REPO_ROOT
    / "examples"
    / "crazyflie_ampc"
    / "crazyflie-firmware"
    / "vendor"
    / "CMSIS"
    / "CMSIS"
    / "Core"
    / "Include"
)
CMSIS_NN_ROOT = REPO_ROOT / "resources" / "CMSIS-NN"
CMSIS_VENDOR_DSP_INCLUDE = (
    REPO_ROOT
    / "examples"
    / "crazyflie_ampc"
    / "crazyflie-firmware"
    / "vendor"
    / "CMSIS"
    / "CMSIS"
    / "DSP"
    / "Include"
)
EIGEN_ROOT = REPO_ROOT / "resources" / "eigen"


def _write_tiny_checkpoint(tmp_path: pathlib.Path) -> tuple[pathlib.Path, MLP, object, dict[str, np.ndarray]]:
    ckpt_dir = tmp_path / "checkpoints" / "iter_00"
    ckpt_dir.mkdir(parents=True)

    model = MLP(hidden_sizes=(5, 4), output_dim=2, activation="leaky_relu", negative_slope=0.02)
    params = model.init(jax.random.PRNGKey(7), jnp.zeros((1, 3), dtype=jnp.float32))["params"]

    norm = {
        "x_mean": np.array([0.5, -0.25, 1.0], dtype=np.float32),
        "x_std": np.array([2.0, 0.5, 1.5], dtype=np.float32),
        "y_mean": np.array([0.2, -0.3], dtype=np.float32),
        "y_std": np.array([0.75, 1.25], dtype=np.float32),
        "y_clip_low": np.array([-0.5, -1.5], dtype=np.float32),
        "y_clip_high": np.array([0.7, 1.25], dtype=np.float32),
    }

    (ckpt_dir / "params.msgpack").write_bytes(serialization.to_bytes(params))
    np.savez(ckpt_dir / "normalization.npz", **norm)
    (ckpt_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "iteration": 0,
                "model_config": {
                    "hidden_sizes": [5, 4],
                    "output_dim": 2,
                    "prediction_target": "first_action",
                    "activation": "leaky_relu",
                    "negative_slope": 0.02,
                },
                "args": {},
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    return ckpt_dir, model, params, norm


def _tiny_test_inputs() -> np.ndarray:
    return np.array(
        [
            [0.1, -0.2, 0.3],
            [1.5, 0.4, -0.8],
            [-2.0, 1.0, 3.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def _export_backend(
    tmp_path: pathlib.Path,
    backend: str,
    *,
    prefix: str,
    example_main: bool,
    quantization: QuantizationConfig | None = None,
    test_tolerance: float = 2e-5,
):
    ckpt_dir, _, _, _ = _write_tiny_checkpoint(tmp_path)
    spec = load_mlp_checkpoint(ckpt_dir)
    cases = make_test_cases(spec, _tiny_test_inputs())
    suffix = "" if quantization is None else f"_{quantization.qtype}"
    out_dir = tmp_path / f"export_{backend}{suffix}"
    export_mlp_policy(
        spec,
        out_dir,
        test_cases=cases,
        options=CodeExportOptions(
            prefix=prefix,
            backend=backend,
            precision="float32_t",
            generate_example_main=example_main,
            test_tolerance=test_tolerance,
            quantization=quantization,
        ),
    )
    return out_dir, spec, cases


def test_load_mlp_checkpoint_matches_flax_policy(tmp_path: pathlib.Path) -> None:
    ckpt_dir, model, params, norm = _write_tiny_checkpoint(tmp_path)
    spec = load_mlp_checkpoint(ckpt_dir)

    x = np.array(
        [
            [0.1, -0.2, 0.3],
            [1.5, 0.4, -0.8],
            [-2.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )
    x_norm = (x - norm["x_mean"][None, :]) / norm["x_std"][None, :]
    y_norm = np.asarray(model.apply({"params": params}, jnp.asarray(x_norm)))
    expected = np.clip(
        y_norm * norm["y_std"][None, :] + norm["y_mean"][None, :],
        norm["y_clip_low"][None, :],
        norm["y_clip_high"][None, :],
    )

    np.testing.assert_allclose(spec.forward(x), expected, rtol=2e-6, atol=2e-6)


def test_simple_c_export_compiles_and_runs_generated_example(tmp_path: pathlib.Path) -> None:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("no C compiler available")

    out_dir, _, _ = _export_backend(tmp_path, "simple", prefix="tiny_ampc", example_main=True)

    exe = out_dir / "tiny_ampc_example"
    cmd = [
        cc,
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-I",
        str(out_dir),
        str(out_dir / "tiny_ampc_data.c"),
        str(out_dir / "tiny_ampc_test_data.c"),
        str(out_dir / "tiny_ampc_forward.c"),
        str(out_dir / "tiny_ampc_example_main.c"),
        "-lm",
        "-o",
        str(exe),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    completed = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    assert "AMPC policy tests: 4 cases" in completed.stdout


def test_cmsis_export_compiles_and_runs_against_cmsis_dsp_host_sources(tmp_path: pathlib.Path) -> None:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("no C compiler available")
    if not CMSIS_DSP_ROOT.exists():
        pytest.skip("resources/CMSIS-DSP checkout is not available")
    if not (CMSIS_CORE_INCLUDE / "cmsis_compiler.h").exists():
        pytest.skip("CMSIS Core include headers are not available")

    out_dir, _, _ = _export_backend(tmp_path, "cmsis", prefix="tiny_cmsis", example_main=False)
    main_c = out_dir / "tiny_cmsis_host_main.c"
    main_c.write_text(
        """
#include "tiny_cmsis_forward.h"
#include "tiny_cmsis_test_data.h"

#include <math.h>
#include <stdio.h>

int main(void) {
    tiny_cmsis_workspace_t workspace;
    tiny_cmsis_float_t output[TINY_CMSIS_OUTPUT_DIM];
    double max_abs_error = 0.0;

    for (uint16_t tc = 0u; tc < TINY_CMSIS_TEST_CASE_COUNT; ++tc) {
        if (tiny_cmsis_forward(tiny_cmsis_test_cases[tc].input, output, &workspace) != 0) {
            return 2;
        }
        for (uint16_t i = 0u; i < TINY_CMSIS_OUTPUT_DIM; ++i) {
            const double err = fabs((double)output[i] - (double)tiny_cmsis_test_cases[tc].output[i]);
            if (err > max_abs_error) {
                max_abs_error = err;
            }
        }
    }

    printf("CMSIS AMPC policy tests: %u cases, max_abs_error=%.9g\\n",
           (unsigned)TINY_CMSIS_TEST_CASE_COUNT, max_abs_error);
    return max_abs_error <= 2e-5 ? 0 : 1;
}
""",
        encoding="utf-8",
    )

    exe = out_dir / "tiny_cmsis_example"
    cmd = [
        cc,
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-I",
        str(out_dir),
        "-I",
        str(CMSIS_DSP_ROOT / "Include"),
        "-I",
        str(CMSIS_DSP_ROOT / "PrivateInclude"),
        "-I",
        str(CMSIS_CORE_INCLUDE),
        "-I",
        str(CMSIS_DSP_ROOT / "Source" / "MatrixFunctions"),
        str(out_dir / "tiny_cmsis_data.c"),
        str(out_dir / "tiny_cmsis_test_data.c"),
        str(out_dir / "tiny_cmsis_forward.c"),
        str(CMSIS_DSP_ROOT / "Source" / "MatrixFunctions" / "arm_mat_init_f32.c"),
        str(CMSIS_DSP_ROOT / "Source" / "MatrixFunctions" / "arm_mat_vec_mult_f32.c"),
        str(main_c),
        "-lm",
        "-o",
        str(exe),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    completed = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    assert "CMSIS AMPC policy tests: 4 cases" in completed.stdout


def test_eigen_export_compiles_and_runs_generated_example(tmp_path: pathlib.Path) -> None:
    cxx = shutil.which("g++") or shutil.which("c++")
    if cxx is None:
        pytest.skip("no C++ compiler available")
    if not EIGEN_ROOT.exists():
        pytest.skip("resources/eigen checkout is not available")

    out_dir, _, _ = _export_backend(tmp_path, "eigen", prefix="tiny_eigen", example_main=True)
    exe = out_dir / "tiny_eigen_example"
    cmd = [
        cxx,
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-I",
        str(out_dir),
        "-I",
        str(EIGEN_ROOT),
        str(out_dir / "tiny_eigen_data.c"),
        str(out_dir / "tiny_eigen_test_data.c"),
        str(out_dir / "tiny_eigen_example_main.cpp"),
        "-lm",
        "-o",
        str(exe),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    completed = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    assert "AMPC policy tests: 4 cases" in completed.stdout


def test_quantized_test_cases_are_exported_against_quantized_policy(tmp_path: pathlib.Path) -> None:
    cfg = QuantizationConfig(qtype="q7")
    out_dir, spec, float_cases = _export_backend(
        tmp_path,
        "none",
        prefix="tiny_qcases",
        example_main=False,
        quantization=cfg,
    )

    qspec = quantize_mlp_policy(spec, float_cases.inputs, cfg)
    with np.load(out_dir / "tiny_qcases_test_cases.npz") as data:
        np.testing.assert_allclose(data["inputs"], float_cases.inputs)
        np.testing.assert_allclose(data["outputs"], qspec.forward(float_cases.inputs), rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(data["float_outputs"], float_cases.outputs, rtol=0.0, atol=1e-12)

    manifest = json.loads((out_dir / "tiny_qcases_manifest.json").read_text(encoding="utf-8"))
    assert manifest["quantization"]["qtype"] == "q7"
    assert manifest["quantization"]["test_outputs"] == "quantized_policy"


@pytest.mark.parametrize("qtype", ["q4", "q7", "q15", "q31"])
def test_quantized_simple_c_export_compiles_and_runs_generated_example(tmp_path: pathlib.Path, qtype: str) -> None:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("no C compiler available")

    prefix = f"tiny_simple_{qtype}"
    out_dir, _, _ = _export_backend(
        tmp_path,
        "simple",
        prefix=prefix,
        example_main=True,
        quantization=QuantizationConfig(qtype=qtype),
        test_tolerance=1e-3,
    )

    exe = out_dir / f"{prefix}_example"
    cmd = [
        cc,
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-I",
        str(out_dir),
        str(out_dir / f"{prefix}_data.c"),
        str(out_dir / f"{prefix}_test_data.c"),
        str(out_dir / f"{prefix}_forward.c"),
        str(out_dir / f"{prefix}_example_main.c"),
        "-lm",
        "-o",
        str(exe),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    completed = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    assert "AMPC policy tests: 4 cases" in completed.stdout


@pytest.mark.parametrize("qtype", ["q4", "q7", "q15"])
def test_quantized_cmsis_export_compiles_and_runs_against_cmsis_nn_host_sources(
    tmp_path: pathlib.Path,
    qtype: str,
) -> None:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("no C compiler available")
    if not CMSIS_NN_ROOT.exists():
        pytest.skip("CMSIS-NN vendor checkout is not available")
    if not (CMSIS_CORE_INCLUDE / "cmsis_compiler.h").exists():
        pytest.skip("CMSIS Core include headers are not available")

    prefix = f"tiny_cmsis_{qtype}"
    out_dir, _, _ = _export_backend(
        tmp_path,
        "cmsis",
        prefix=prefix,
        example_main=True,
        quantization=QuantizationConfig(qtype=qtype),
        test_tolerance=1e-3,
    )

    nn_source = CMSIS_NN_ROOT / "Source"
    cmsis_nn_sources = [
        nn_source / "ActivationFunctions" / ("arm_relu_q15.c" if qtype == "q15" else "arm_relu_q7.c"),
        nn_source / "ActivationFunctions" / "arm_nn_activation_s16.c",
        nn_source / "NNSupportFunctions" / "arm_nntables.c",
    ]
    if qtype == "q4":
        cmsis_nn_sources.extend(
            [
                nn_source / "FullyConnectedFunctions" / "arm_fully_connected_s4.c",
                nn_source / "NNSupportFunctions" / "arm_nn_vec_mat_mult_t_s4.c",
            ]
        )
    if qtype == "q7":
        cmsis_nn_sources.extend(
            [
                nn_source / "FullyConnectedFunctions" / "arm_fully_connected_s8.c",
                nn_source / "NNSupportFunctions" / "arm_nn_vec_mat_mult_t_s8.c",
            ]
        )
    if qtype == "q15":
        cmsis_nn_sources.extend(
            [
                nn_source / "FullyConnectedFunctions" / "arm_fully_connected_s16.c",
                nn_source / "NNSupportFunctions" / "arm_nn_vec_mat_mult_t_s16.c",
            ]
        )

    exe = out_dir / f"{prefix}_example"
    cmd = [
        cc,
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-I",
        str(out_dir),
        "-I",
        str(CMSIS_NN_ROOT / "Include"),
        "-I",
        str(CMSIS_CORE_INCLUDE),
        str(out_dir / f"{prefix}_data.c"),
        str(out_dir / f"{prefix}_test_data.c"),
        str(out_dir / f"{prefix}_forward.c"),
        *(str(path) for path in cmsis_nn_sources),
        str(out_dir / f"{prefix}_example_main.c"),
        "-lm",
        "-o",
        str(exe),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    completed = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    assert "AMPC policy tests: 4 cases" in completed.stdout
@pytest.mark.parametrize("qtype", ["q4", "q7", "q15", "q31"])
def test_quantized_eigen_export_compiles_and_runs_generated_example(tmp_path: pathlib.Path, qtype: str) -> None:
    cxx = shutil.which("g++") or shutil.which("c++")
    if cxx is None:
        pytest.skip("no C++ compiler available")
    if not EIGEN_ROOT.exists():
        pytest.skip("resources/eigen checkout is not available")

    prefix = f"tiny_eigen_{qtype}"
    out_dir, _, _ = _export_backend(
        tmp_path,
        "eigen",
        prefix=prefix,
        example_main=True,
        quantization=QuantizationConfig(qtype=qtype),
        test_tolerance=1e-3,
    )
    exe = out_dir / f"{prefix}_example"
    cmd = [
        cxx,
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-I",
        str(out_dir),
        "-I",
        str(EIGEN_ROOT),
        str(out_dir / f"{prefix}_data.c"),
        str(out_dir / f"{prefix}_test_data.c"),
        str(out_dir / f"{prefix}_example_main.cpp"),
        "-lm",
        "-o",
        str(exe),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    completed = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    assert "AMPC policy tests: 4 cases" in completed.stdout
