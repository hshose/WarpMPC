from __future__ import annotations

from dataclasses import dataclass
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODM_ROOT = REPO_ROOT / "resources" / "modm"
CMSIS_NN_ROOT = REPO_ROOT / "resources" / "CMSIS-NN"
CMSIS_VENDOR_ROOT = (
    REPO_ROOT
    / "examples"
    / "crazyflie_ampc"
    / "crazyflie-firmware"
    / "vendor"
    / "CMSIS"
    / "CMSIS"
)
CMSIS_DSP_ROOT = REPO_ROOT / "resources" / "CMSIS-DSP"
EIGEN_ROOT = REPO_ROOT / "resources" / "eigen"


@dataclass(frozen=True)
class TargetConfig:
    key: str
    label: str
    modm_extends: str
    default_sections: tuple[str, ...]
    cache_options: tuple[tuple[str, str], ...] = ()


TARGETS: dict[str, TargetConfig] = {
    "g474": TargetConfig(
        key="g474",
        label="nucleo_g474re",
        modm_extends="modm:nucleo-g474re",
        default_sections=("flash", "data", "fastdata", "data_ccm", "data_sram1", "data_sram2"),
    ),
    "h723": TargetConfig(
        key="h723",
        label="nucleo_h723zg",
        modm_extends="modm:nucleo-h723zg",
        default_sections=(
            "flash",
            "data",
            "fastdata",
            "data_itcm",
            "data_dtcm",
            "data_d1_sram",
            "data_d2_sram1",
            "data_d2_sram2",
            "data_d3_sram",
        ),
        cache_options=(
            ("modm:platform:cortex-m:enable_icache", "true"),
            ("modm:platform:cortex-m:enable_dcache", "true"),
        ),
    ),
}


NETWORK_SIZES: tuple[tuple[int, int], ...] = (
    (2, 16),
    (3, 64),
    (5, 128),
)
ACTIVATIONS = ("relu", "leaky_relu", "tanh", "elu")
BACKENDS = ("simple", "cmsis", "eigen")
PRECISIONS = ("f32", "q4", "q7", "q15")

INPUT_DIM = 16
OUTPUT_DIM = 4
TEST_CASE_COUNT = 16


@dataclass(frozen=True)
class BenchmarkCase:
    target: str
    hidden_layers: int
    width: int
    activation: str
    backend: str
    precision: str
    section: str

    @property
    def case_id(self) -> str:
        activation = self.activation.replace("_", "")
        return (
            f"{self.target}_l{self.hidden_layers}_w{self.width}_{activation}_"
            f"{self.backend}_{self.precision}_{self.section}"
        )

    @property
    def prefix(self) -> str:
        return f"ampc_{self.case_id}"

    @property
    def is_quantized(self) -> bool:
        return self.precision in ("q4", "q7", "q15")


def iter_cases(
    *,
    target: str,
    hidden_layers: int | None = None,
    width: int | None = None,
    activation: str | None = None,
    backend: str | None = None,
    precision: str | None = None,
    section: str | None = None,
) -> list[BenchmarkCase]:
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {sorted(TARGETS)}")
    target_cfg = TARGETS[target]
    sizes = NETWORK_SIZES
    if hidden_layers is not None:
        sizes = tuple(item for item in sizes if item[0] == hidden_layers)
    if width is not None:
        sizes = tuple(item for item in sizes if item[1] == width)
    activations = (activation,) if activation is not None else ACTIVATIONS
    backends = (backend,) if backend is not None else BACKENDS
    precisions = (precision,) if precision is not None else PRECISIONS
    sections = (section,) if section is not None else target_cfg.default_sections

    cases: list[BenchmarkCase] = []
    for layers_value, width_value in sizes:
        for activation_value in activations:
            for backend_value in backends:
                for precision_value in precisions:
                    for section_value in sections:
                        if backend_value == "cmsis" and precision_value not in ("f32", "q4", "q7", "q15"):
                            continue
                        cases.append(
                            BenchmarkCase(
                                target=target,
                                hidden_layers=layers_value,
                                width=width_value,
                                activation=activation_value,
                                backend=backend_value,
                                precision=precision_value,
                                section=section_value,
                            )
                        )
    return cases


def section_name(section: str) -> str | None:
    if section == "flash":
        return None
    if section == "data":
        return ".data"
    if section == "fastdata":
        return ".fastdata"
    if section.startswith("."):
        return section
    return f".{section}"
