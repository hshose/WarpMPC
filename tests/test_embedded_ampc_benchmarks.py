from __future__ import annotations

import pathlib

import pytest

from benchmarks.embedded_ampc.config import BenchmarkCase, iter_cases
from benchmarks.embedded_ampc.config import CMSIS_NN_ROOT
from benchmarks.embedded_ampc.generate import generate_case


def test_embedded_ampc_case_matrix_contains_requested_extremes() -> None:
    cases = iter_cases(target="g474")
    assert any(case.hidden_layers == 2 and case.width == 16 for case in cases)
    assert any(case.hidden_layers == 5 and case.width == 128 for case in cases)
    assert {case.activation for case in cases} >= {"leaky_relu", "tanh", "elu"}
    assert {case.backend for case in cases} >= {"simple", "cmsis", "eigen"}
    assert {case.precision for case in cases} >= {"f32", "q4", "q7", "q15"}
    assert any(case.backend == "cmsis" and case.precision == "q4" for case in cases)


def test_embedded_ampc_generator_host_checks_small_cmsis_case(tmp_path: pathlib.Path) -> None:
    if not CMSIS_NN_ROOT.exists():
        pytest.skip("resources/CMSIS-NN checkout is not available")

    case = BenchmarkCase(
        target="g474",
        hidden_layers=2,
        width=16,
        activation="leaky_relu",
        backend="cmsis",
        precision="q7",
        section="flash",
    )
    project_dir = generate_case(case, output_root=tmp_path, host_check=True)
    assert (project_dir / "main.cpp").exists()
    assert (project_dir / "project.xml").exists()
    assert (project_dir / "arm_fully_connected_s8.c").exists()
    assert (project_dir / "arm_nn_activation_s16.c").exists()
    assert (project_dir / "arm_nntables.c").exists()
