"""Standalone C/C++ export helpers for AMPC neural policies.

The exporter intentionally separates policy data from inference backends.  The
``*_data`` files contain only dimensions, normalization constants, layer
metadata, weights, and biases.  Optional forward-pass files can then consume the
same data using plain C loops, CMSIS-DSP, Eigen, or future backends.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import pathlib
import re
from typing import Literal, Mapping, Sequence

import numpy as np
from flax import serialization


ForwardBackend = Literal["none", "simple", "cmsis", "eigen"]
QuantizedDType = Literal["q31", "q15", "q7", "q4"]


@dataclass(frozen=True)
class DenseLayerSpec:
    """One dense layer with row-major weights shaped ``(output_dim, input_dim)``."""

    weight: np.ndarray
    bias: np.ndarray
    activation: str
    negative_slope: float = 0.0

    @property
    def input_dim(self) -> int:
        return int(self.weight.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.weight.shape[0])


@dataclass(frozen=True)
class MLPExportSpec:
    """Library-independent MLP policy description."""

    name: str
    layers: tuple[DenseLayerSpec, ...]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    y_clip_low: np.ndarray
    y_clip_high: np.ndarray
    model_config: Mapping[str, object]
    metadata: Mapping[str, object]

    @property
    def input_dim(self) -> int:
        return int(self.x_mean.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.y_mean.shape[0])

    @property
    def hidden_activation(self) -> str:
        if not self.layers:
            return "linear"
        for layer in self.layers[:-1]:
            if layer.activation != "linear":
                return layer.activation
        return "linear"

    @property
    def max_layer_size(self) -> int:
        return max(layer.output_dim for layer in self.layers)

    def forward(self, x: np.ndarray, *, clip: bool = True) -> np.ndarray:
        """Evaluate the exported policy with NumPy."""

        x_arr = np.asarray(x, dtype=np.float64)
        single = x_arr.ndim == 1
        if single:
            x_arr = x_arr[None, :]
        if x_arr.ndim != 2 or x_arr.shape[1] != self.input_dim:
            raise ValueError(f"expected input shape (*, {self.input_dim}), got {x_arr.shape}")

        h = (x_arr - self.x_mean[None, :]) / self.x_std[None, :]
        for layer in self.layers:
            h = h @ layer.weight.T + layer.bias[None, :]
            h = _apply_activation_np(h, layer.activation, layer.negative_slope)
        y = h * self.y_std[None, :] + self.y_mean[None, :]
        if clip:
            y = np.clip(y, self.y_clip_low[None, :], self.y_clip_high[None, :])
        return y[0] if single else y


@dataclass(frozen=True)
class QuantizationConfig:
    """Post-training quantization settings for an exported MLP.

    This implements the same core representation used by Qwix ``QArray``:
    symmetric quantized values with explicit floating-point scales and no zero
    point.  We use per-output-channel scales for weights and per-tensor static
    scales for activations.
    """

    qtype: QuantizedDType = "q7"
    weight_qtype: QuantizedDType | None = None
    calibration_method: str = "absmax"
    eps: float = 1e-12


@dataclass(frozen=True)
class QuantizedDenseLayerSpec:
    weight_q: np.ndarray
    weight_scale: np.ndarray
    bias: np.ndarray
    activation: str
    negative_slope: float = 0.0

    @property
    def input_dim(self) -> int:
        return int(self.weight_q.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.weight_q.shape[0])


@dataclass(frozen=True)
class QuantizedMLPExportSpec:
    name: str
    layers: tuple[QuantizedDenseLayerSpec, ...]
    activation_scales: np.ndarray
    layer_output_scales: np.ndarray
    qtype: QuantizedDType
    weight_qtype: QuantizedDType
    qmin: int
    qmax: int
    weight_qmin: int
    weight_qmax: int
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    y_clip_low: np.ndarray
    y_clip_high: np.ndarray
    model_config: Mapping[str, object]
    metadata: Mapping[str, object]

    @property
    def input_dim(self) -> int:
        return int(self.x_mean.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.y_mean.shape[0])

    @property
    def max_layer_size(self) -> int:
        return max(layer.output_dim for layer in self.layers)

    def forward(self, x: np.ndarray, *, clip: bool = True) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        single = x_arr.ndim == 1
        if single:
            x_arr = x_arr[None, :]
        if x_arr.ndim != 2 or x_arr.shape[1] != self.input_dim:
            raise ValueError(f"expected input shape (*, {self.input_dim}), got {x_arr.shape}")

        h_float = (x_arr - self.x_mean[None, :]) / self.x_std[None, :]
        h_q = _quantize_symmetric(h_float, self.activation_scales[0], self.qmin, self.qmax)
        y_norm = None
        for layer_idx, layer in enumerate(self.layers):
            raw = h_q.astype(np.float64) @ layer.weight_q.astype(np.float64).T
            h_float = raw * self.activation_scales[layer_idx] * layer.weight_scale[None, :]
            h_float = h_float + layer.bias[None, :]
            h_float = _apply_activation_np(h_float, layer.activation, layer.negative_slope)
            if layer_idx == len(self.layers) - 1:
                y_norm = h_float
            else:
                h_q = _quantize_symmetric(
                    h_float,
                    self.activation_scales[layer_idx + 1],
                    self.qmin,
                    self.qmax,
                )
        if y_norm is None:
            raise ValueError("quantized policy has no layers")
        y = y_norm * self.y_std[None, :] + self.y_mean[None, :]
        if clip:
            y = np.clip(y, self.y_clip_low[None, :], self.y_clip_high[None, :])
        return y[0] if single else y


@dataclass(frozen=True)
class TestCases:
    inputs: np.ndarray
    outputs: np.ndarray

    @property
    def count(self) -> int:
        return int(self.inputs.shape[0])


@dataclass(frozen=True)
class CodeExportOptions:
    prefix: str = "ampc_policy"
    backend: ForwardBackend = "none"
    precision: str = "float32_t"
    generate_example_main: bool = False
    test_tolerance: float = 1e-4
    quantization: QuantizationConfig | None = None


def load_mlp_checkpoint(checkpoint_dir: pathlib.Path | str, *, name: str = "ampc_policy") -> MLPExportSpec:
    """Load a saved ``warpmpc.jax_ampc`` Flax MLP checkpoint into an export spec."""

    ckpt_dir = pathlib.Path(checkpoint_dir)
    checkpoint_json = ckpt_dir / "checkpoint.json"
    params_path = ckpt_dir / "params.msgpack"
    norm_path = ckpt_dir / "normalization.npz"

    if not checkpoint_json.exists():
        raise FileNotFoundError(f"missing checkpoint metadata: {checkpoint_json}")
    if not params_path.exists():
        raise FileNotFoundError(f"missing checkpoint params: {params_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"missing checkpoint normalization: {norm_path}")

    metadata = json.loads(checkpoint_json.read_text(encoding="utf-8"))
    model_config = dict(metadata.get("model_config", {}))
    activation = str(model_config.get("activation", "leaky_relu"))
    negative_slope = float(model_config.get("negative_slope", 0.01))

    params = serialization.msgpack_restore(params_path.read_bytes())
    if "params" in params and isinstance(params["params"], Mapping):
        params = params["params"]

    layers = _layers_from_flax_params(
        params,
        hidden_activation=activation,
        negative_slope=negative_slope,
    )
    if not layers:
        raise ValueError(f"no Dense_* layers found in {params_path}")

    norm = np.load(norm_path)
    x_mean = _load_norm_array(norm, "x_mean")
    x_std = _load_norm_array(norm, "x_std")
    y_mean = _load_norm_array(norm, "y_mean", fallback="u_mean")
    y_std = _load_norm_array(norm, "y_std", fallback="u_std")
    y_clip_low = _load_norm_array(norm, "y_clip_low", fallback="u_clip_low")
    y_clip_high = _load_norm_array(norm, "y_clip_high", fallback="u_clip_high")

    _validate_spec_arrays(layers, x_mean, x_std, y_mean, y_std, y_clip_low, y_clip_high)

    return MLPExportSpec(
        name=name,
        layers=tuple(layers),
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        y_clip_low=y_clip_low,
        y_clip_high=y_clip_high,
        model_config=model_config,
        metadata=metadata,
    )


def make_random_test_cases(
    spec: MLPExportSpec,
    *,
    count: int,
    seed: int = 0,
    scale: float = 1.0,
) -> TestCases:
    """Generate deterministic random inputs around the training normalization."""

    if count < 1:
        raise ValueError("test case count must be >= 1")
    rng = np.random.default_rng(seed)
    x_std = np.where(np.isfinite(spec.x_std) & (spec.x_std > 0.0), spec.x_std, 1.0)
    inputs = spec.x_mean[None, :] + scale * rng.normal(size=(count, spec.input_dim)) * x_std[None, :]
    outputs = spec.forward(inputs)
    return TestCases(inputs=inputs.astype(np.float64), outputs=outputs.astype(np.float64))


def make_test_cases(spec: MLPExportSpec, inputs: np.ndarray) -> TestCases:
    """Pair caller-provided inputs with correct exported-policy outputs."""

    x = np.asarray(inputs, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != spec.input_dim:
        raise ValueError(f"expected test inputs shape (*, {spec.input_dim}), got {x.shape}")
    return TestCases(inputs=x, outputs=spec.forward(x).astype(np.float64))


def quantize_mlp_policy(
    spec: MLPExportSpec,
    calibration_inputs: np.ndarray,
    config: QuantizationConfig | None = None,
) -> QuantizedMLPExportSpec:
    """Post-training quantize an MLP export spec.

    The representation follows Qwix's symmetric ``QArray`` convention:
    ``dequantized ~= qvalue * scale`` with no zero point.  Weights use
    per-output-channel scales, while activations use static per-tensor scales
    collected from a calibration batch.
    """

    cfg = config or QuantizationConfig()
    if cfg.calibration_method != "absmax":
        raise ValueError("only absmax post-training quantization is currently supported")
    if not spec.layers:
        raise ValueError("cannot quantize an MLP without layers")

    dtype_info = _quant_dtype_info(cfg.qtype)
    weight_qtype = cfg.weight_qtype or cfg.qtype
    weight_dtype_info = _quant_dtype_info(weight_qtype)
    x = np.asarray(calibration_inputs, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != spec.input_dim:
        raise ValueError(f"expected calibration inputs shape (*, {spec.input_dim}), got {x.shape}")
    if x.shape[0] < 1:
        raise ValueError("calibration input batch must contain at least one sample")

    activation_scales = _collect_activation_scales(spec, x, dtype_info.qmax, cfg.eps)
    layer_output_scales = _collect_layer_output_scales(spec, x, dtype_info.qmax, cfg.eps)
    layers: list[QuantizedDenseLayerSpec] = []
    for layer in spec.layers:
        row_absmax = np.max(np.abs(layer.weight), axis=1)
        weight_scale = np.maximum(row_absmax / float(weight_dtype_info.qmax), cfg.eps)
        weight_q = _quantize_symmetric(layer.weight, weight_scale[:, None], weight_dtype_info.qmin, weight_dtype_info.qmax)
        layers.append(
            QuantizedDenseLayerSpec(
                weight_q=weight_q.astype(weight_dtype_info.numpy_dtype, copy=False),
                weight_scale=weight_scale.astype(np.float64),
                bias=layer.bias.astype(np.float64, copy=True),
                activation=layer.activation,
                negative_slope=layer.negative_slope,
            )
        )

    metadata = dict(spec.metadata)
    metadata["quantization"] = {
        "method": "post_training",
        "representation": "symmetric_scale_no_zero_point",
        "qtype": cfg.qtype,
        "weight_qtype": weight_qtype,
        "calibration_method": cfg.calibration_method,
        "calibration_sample_count": int(x.shape[0]),
    }
    return QuantizedMLPExportSpec(
        name=spec.name,
        layers=tuple(layers),
        activation_scales=activation_scales.astype(np.float64),
        layer_output_scales=layer_output_scales.astype(np.float64),
        qtype=cfg.qtype,
        weight_qtype=weight_qtype,
        qmin=dtype_info.qmin,
        qmax=dtype_info.qmax,
        weight_qmin=weight_dtype_info.qmin,
        weight_qmax=weight_dtype_info.qmax,
        x_mean=spec.x_mean,
        x_std=spec.x_std,
        y_mean=spec.y_mean,
        y_std=spec.y_std,
        y_clip_low=spec.y_clip_low,
        y_clip_high=spec.y_clip_high,
        model_config=spec.model_config,
        metadata=metadata,
    )


def export_mlp_policy(
    spec: MLPExportSpec,
    output_dir: pathlib.Path | str,
    *,
    test_cases: TestCases | None = None,
    options: CodeExportOptions | None = None,
) -> dict[str, pathlib.Path]:
    """Export policy data and optional forward-pass backend files."""

    opts = options or CodeExportOptions()
    prefix = _sanitize_identifier(opts.prefix)
    if opts.backend not in ("none", "simple", "cmsis", "eigen"):
        raise ValueError(f"unsupported forward backend {opts.backend!r}")
    if opts.backend == "cmsis" and opts.quantization is None and _float_type(opts.precision) != "float":
        raise ValueError("CMSIS-DSP export currently supports only float32 precision")
    if not spec.layers:
        raise ValueError("cannot export an MLP without layers")
    if opts.quantization is not None:
        return _export_quantized_mlp_policy(spec, output_dir, test_cases=test_cases, options=opts, prefix=prefix)

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: dict[str, pathlib.Path] = {}
    context = _ExportContext(spec=spec, options=opts, prefix=prefix)

    data_h, data_c = _generate_data_files(context)
    written["data_header"] = _write_text(out / f"{prefix}_data.h", data_h)
    written["data_source"] = _write_text(out / f"{prefix}_data.c", data_c)

    if test_cases is not None:
        _validate_test_cases(spec, test_cases)
        test_h, test_c = _generate_test_data_files(context, test_cases)
        written["test_header"] = _write_text(out / f"{prefix}_test_data.h", test_h)
        written["test_source"] = _write_text(out / f"{prefix}_test_data.c", test_c)
        np.savez(out / f"{prefix}_test_cases.npz", inputs=test_cases.inputs, outputs=test_cases.outputs)
        written["test_npz"] = out / f"{prefix}_test_cases.npz"

    if opts.backend == "simple":
        forward_h, forward_c = _generate_simple_forward_files(context)
        written["forward_header"] = _write_text(out / f"{prefix}_forward.h", forward_h)
        written["forward_source"] = _write_text(out / f"{prefix}_forward.c", forward_c)
        if opts.generate_example_main:
            if test_cases is None:
                raise ValueError("example main generation requires test cases")
            written["example_main"] = _write_text(
                out / f"{prefix}_example_main.c",
                _generate_c_example_main(context),
            )
    elif opts.backend == "cmsis":
        forward_h, forward_c = _generate_simple_forward_files(context, cmsis=True)
        written["forward_header"] = _write_text(out / f"{prefix}_forward.h", forward_h)
        written["forward_source"] = _write_text(out / f"{prefix}_forward.c", forward_c)
    elif opts.backend == "eigen":
        written["forward_header"] = _write_text(
            out / f"{prefix}_forward.hpp",
            _generate_eigen_forward_header(context),
        )
        if opts.generate_example_main:
            if test_cases is None:
                raise ValueError("example main generation requires test cases")
            written["example_main"] = _write_text(
                out / f"{prefix}_example_main.cpp",
                _generate_eigen_example_main(context),
            )

    manifest = _manifest(context, test_cases, written)
    manifest_path = out / f"{prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written["manifest"] = manifest_path
    return written


def export_checkpoint(
    checkpoint_dir: pathlib.Path | str,
    output_dir: pathlib.Path | str,
    *,
    test_inputs: np.ndarray | None = None,
    test_count: int = 16,
    test_seed: int = 0,
    test_scale: float = 1.0,
    options: CodeExportOptions | None = None,
    name: str = "ampc_policy",
) -> dict[str, pathlib.Path]:
    """Convenience wrapper for loading a checkpoint and exporting it."""

    spec = load_mlp_checkpoint(checkpoint_dir, name=name)
    if test_inputs is None:
        tests = make_random_test_cases(spec, count=test_count, seed=test_seed, scale=test_scale)
    else:
        tests = make_test_cases(spec, test_inputs)
    return export_mlp_policy(spec, output_dir, test_cases=tests, options=options)


def _export_quantized_mlp_policy(
    spec: MLPExportSpec,
    output_dir: pathlib.Path | str,
    *,
    test_cases: TestCases | None,
    options: CodeExportOptions,
    prefix: str,
) -> dict[str, pathlib.Path]:
    if test_cases is None:
        raise ValueError("quantized export requires test cases to calibrate activations")

    _validate_test_cases(spec, test_cases)
    quantization_config = options.quantization
    if options.backend == "cmsis" and quantization_config is not None and quantization_config.qtype == "q15":
        quantization_config = QuantizationConfig(
            qtype=quantization_config.qtype,
            weight_qtype="q7",
            calibration_method=quantization_config.calibration_method,
            eps=quantization_config.eps,
        )
    qspec = quantize_mlp_policy(spec, test_cases.inputs, quantization_config)
    if options.backend == "cmsis" and qspec.qtype not in ("q4", "q7", "q15"):
        raise ValueError("CMSIS-NN quantized export supports only q4/int4, q7/int8, and q15/int16 policies")
    quantized_outputs = (
        _forward_quantized_cmsis_nn(qspec, test_cases.inputs)
        if options.backend == "cmsis" and qspec.qtype in ("q4", "q7", "q15")
        else qspec.forward(test_cases.inputs)
    )
    quantized_cases = TestCases(
        inputs=np.asarray(test_cases.inputs, dtype=np.float64),
        outputs=np.asarray(quantized_outputs, dtype=np.float64),
    )

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: dict[str, pathlib.Path] = {}
    context = _QuantizedExportContext(spec=qspec, options=options, prefix=prefix)

    data_h, data_c = _generate_quantized_data_files(context)
    written["data_header"] = _write_text(out / f"{prefix}_data.h", data_h)
    written["data_source"] = _write_text(out / f"{prefix}_data.c", data_c)

    test_h, test_c = _generate_test_data_files(context, quantized_cases)
    written["test_header"] = _write_text(out / f"{prefix}_test_data.h", test_h)
    written["test_source"] = _write_text(out / f"{prefix}_test_data.c", test_c)
    np.savez(
        out / f"{prefix}_test_cases.npz",
        inputs=quantized_cases.inputs,
        outputs=quantized_cases.outputs,
        float_outputs=test_cases.outputs,
    )
    written["test_npz"] = out / f"{prefix}_test_cases.npz"

    if options.backend == "simple":
        forward_h, forward_c = _generate_quantized_simple_forward_files(context)
        written["forward_header"] = _write_text(out / f"{prefix}_forward.h", forward_h)
        written["forward_source"] = _write_text(out / f"{prefix}_forward.c", forward_c)
        if options.generate_example_main:
            written["example_main"] = _write_text(out / f"{prefix}_example_main.c", _generate_c_example_main(context))
    elif options.backend == "cmsis":
        forward_h, forward_c = _generate_quantized_simple_forward_files(context, cmsis=True)
        written["forward_header"] = _write_text(out / f"{prefix}_forward.h", forward_h)
        written["forward_source"] = _write_text(out / f"{prefix}_forward.c", forward_c)
        if options.generate_example_main:
            written["example_main"] = _write_text(out / f"{prefix}_example_main.c", _generate_c_example_main(context))
    elif options.backend == "eigen":
        written["forward_header"] = _write_text(out / f"{prefix}_forward.hpp", _generate_quantized_eigen_forward_header(context))
        if options.generate_example_main:
            written["example_main"] = _write_text(out / f"{prefix}_example_main.cpp", _generate_eigen_example_main(context))

    manifest = _manifest(context, quantized_cases, written)
    manifest_path = out / f"{prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written["manifest"] = manifest_path
    return written


def _layers_from_flax_params(
    params: Mapping[str, object],
    *,
    hidden_activation: str,
    negative_slope: float,
) -> list[DenseLayerSpec]:
    keys = sorted((key for key in params.keys() if _dense_key_index(str(key)) is not None), key=lambda k: _dense_key_index(str(k)))
    layers: list[DenseLayerSpec] = []
    for index, key in enumerate(keys):
        layer_params = params[key]
        if not isinstance(layer_params, Mapping):
            raise ValueError(f"layer {key!r} params are not a mapping")
        kernel = np.asarray(layer_params["kernel"], dtype=np.float64)
        bias = np.asarray(layer_params["bias"], dtype=np.float64)
        if kernel.ndim != 2:
            raise ValueError(f"layer {key!r} kernel must be 2D, got {kernel.shape}")
        if bias.ndim != 1:
            raise ValueError(f"layer {key!r} bias must be 1D, got {bias.shape}")
        if kernel.shape[1] != bias.shape[0]:
            raise ValueError(f"layer {key!r} kernel/bias mismatch: {kernel.shape} vs {bias.shape}")
        activation = hidden_activation if index < len(keys) - 1 else "linear"
        slope = negative_slope if activation == "leaky_relu" else 0.0
        layers.append(
            DenseLayerSpec(
                weight=kernel.T.copy(),
                bias=bias.copy(),
                activation=activation,
                negative_slope=slope,
            )
        )
    return layers


def _dense_key_index(key: str) -> int | None:
    match = re.fullmatch(r"Dense_(\d+)", key)
    if match is None:
        return None
    return int(match.group(1))


def _load_norm_array(norm, key: str, *, fallback: str | None = None) -> np.ndarray:
    if key in norm:
        return np.asarray(norm[key], dtype=np.float64)
    if fallback is not None and fallback in norm:
        return np.asarray(norm[fallback], dtype=np.float64)
    raise KeyError(f"normalization.npz is missing {key!r}")


def _validate_spec_arrays(
    layers: Sequence[DenseLayerSpec],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    y_clip_low: np.ndarray,
    y_clip_high: np.ndarray,
) -> None:
    if x_mean.ndim != 1 or x_std.shape != x_mean.shape:
        raise ValueError("x_mean/x_std must be matching 1D arrays")
    if y_mean.ndim != 1 or y_std.shape != y_mean.shape:
        raise ValueError("y_mean/y_std must be matching 1D arrays")
    if y_clip_low.shape != y_mean.shape or y_clip_high.shape != y_mean.shape:
        raise ValueError("y clipping arrays must match y_mean")
    if layers[0].input_dim != x_mean.shape[0]:
        raise ValueError(f"first layer input_dim={layers[0].input_dim} but x_dim={x_mean.shape[0]}")
    if layers[-1].output_dim != y_mean.shape[0]:
        raise ValueError(f"last layer output_dim={layers[-1].output_dim} but y_dim={y_mean.shape[0]}")
    for prev, nxt in zip(layers[:-1], layers[1:]):
        if prev.output_dim != nxt.input_dim:
            raise ValueError(f"layer dimension mismatch: {prev.output_dim} -> {nxt.input_dim}")
    if np.any(~np.isfinite(x_std)) or np.any(x_std <= 0.0):
        raise ValueError("x_std must be finite and positive")
    if np.any(~np.isfinite(y_std)) or np.any(y_std <= 0.0):
        raise ValueError("y_std must be finite and positive")


def _validate_test_cases(spec: MLPExportSpec, cases: TestCases) -> None:
    if cases.inputs.ndim != 2 or cases.inputs.shape[1] != spec.input_dim:
        raise ValueError(f"test input shape must be (*, {spec.input_dim}), got {cases.inputs.shape}")
    if cases.outputs.ndim != 2 or cases.outputs.shape != (cases.inputs.shape[0], spec.output_dim):
        raise ValueError(
            f"test output shape must be ({cases.inputs.shape[0]}, {spec.output_dim}), got {cases.outputs.shape}"
        )


def _apply_activation_np(x: np.ndarray, activation: str, negative_slope: float) -> np.ndarray:
    if activation in ("linear", "none", None):
        return x
    if activation == "leaky_relu":
        return np.where(x >= 0.0, x, negative_slope * x)
    if activation == "relu":
        return np.maximum(x, 0.0)
    if activation == "tanh":
        return np.tanh(x)
    if activation == "gelu":
        erf_vec = np.vectorize(math.erf)
        return 0.5 * x * (1.0 + erf_vec(x / math.sqrt(2.0)))
    if activation == "elu":
        return np.where(x >= 0.0, x, np.expm1(x))
    if activation in ("silu", "swish"):
        return x / (1.0 + np.exp(-x))
    raise ValueError(f"unsupported activation {activation!r}")


@dataclass(frozen=True)
class _QuantDTypeInfo:
    qtype: QuantizedDType
    numpy_dtype: np.dtype
    c_type: str
    cmsis_type: str
    qmin: int
    qmax: int


def _quant_dtype_info(qtype: QuantizedDType) -> _QuantDTypeInfo:
    if qtype == "q4":
        return _QuantDTypeInfo(qtype=qtype, numpy_dtype=np.dtype(np.int8), c_type="int8_t", cmsis_type="q7_t", qmin=-7, qmax=7)
    if qtype == "q7":
        return _QuantDTypeInfo(qtype=qtype, numpy_dtype=np.dtype(np.int8), c_type="int8_t", cmsis_type="q7_t", qmin=-127, qmax=127)
    if qtype == "q15":
        return _QuantDTypeInfo(
            qtype=qtype,
            numpy_dtype=np.dtype(np.int16),
            c_type="int16_t",
            cmsis_type="q15_t",
            qmin=-32767,
            qmax=32767,
        )
    if qtype == "q31":
        return _QuantDTypeInfo(
            qtype=qtype,
            numpy_dtype=np.dtype(np.int32),
            c_type="int32_t",
            cmsis_type="q31_t",
            qmin=-2147483647,
            qmax=2147483647,
        )
    raise ValueError(f"unsupported quantized dtype {qtype!r}")


def _q4_packed_size(value_count: int) -> int:
    return (int(value_count) + 1) // 2


def _pack_q4_signed(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    if np.any(flat < -8) or np.any(flat > 7):
        raise ValueError("q4 values must be in signed 4-bit range")
    encoded = (flat.astype(np.int16) & 0x0F).astype(np.uint8)
    packed = np.zeros(_q4_packed_size(encoded.size), dtype=np.uint8)
    packed[: (encoded.size + 1) // 2] = encoded[0::2]
    if encoded.size > 1:
        packed[: encoded.size // 2] |= (encoded[1::2] << 4)
    return packed


def _packed_q4_weight_matrix(layer: QuantizedDenseLayerSpec) -> np.ndarray:
    rows = [_pack_q4_signed(layer.weight_q[row]) for row in range(layer.output_dim)]
    return np.concatenate(rows).astype(np.uint8, copy=False)


def _weight_storage_c_type(spec: QuantizedMLPExportSpec) -> str:
    return "uint8_t" if spec.weight_qtype == "q4" else _quant_dtype_info(spec.weight_qtype).c_type


def _weight_storage_stride(layer: QuantizedDenseLayerSpec, weight_qtype: QuantizedDType) -> int:
    return _q4_packed_size(layer.input_dim) if weight_qtype == "q4" else layer.input_dim


def _bias_q_c_type(ctx: "_QuantizedExportContext") -> str:
    if ctx.options.backend == "cmsis" and ctx.spec.qtype in ("q4", "q7"):
        return "int32_t"
    return "int64_t"


def _bias_q_numpy_dtype(ctx: "_QuantizedExportContext") -> np.dtype:
    return np.dtype(np.int32 if _bias_q_c_type(ctx) == "int32_t" else np.int64)


def _collect_activation_scales(
    spec: MLPExportSpec,
    calibration_inputs: np.ndarray,
    qmax: int,
    eps: float,
) -> np.ndarray:
    h = (np.asarray(calibration_inputs, dtype=np.float64) - spec.x_mean[None, :]) / spec.x_std[None, :]
    scales = [_safe_absmax_scale(h, qmax, eps)]
    for layer in spec.layers[:-1]:
        h = h @ layer.weight.T + layer.bias[None, :]
        h = _apply_activation_np(h, layer.activation, layer.negative_slope)
        scales.append(_safe_absmax_scale(h, qmax, eps))
    return np.asarray(scales, dtype=np.float64)


def _collect_layer_output_scales(
    spec: MLPExportSpec,
    calibration_inputs: np.ndarray,
    qmax: int,
    eps: float,
) -> np.ndarray:
    h = (np.asarray(calibration_inputs, dtype=np.float64) - spec.x_mean[None, :]) / spec.x_std[None, :]
    scales = []
    for layer in spec.layers:
        h = h @ layer.weight.T + layer.bias[None, :]
        h = _apply_activation_np(h, layer.activation, layer.negative_slope)
        scales.append(_safe_absmax_scale(h, qmax, eps))
    return np.asarray(scales, dtype=np.float64)


def _safe_absmax_scale(values: np.ndarray, qmax: int, eps: float) -> float:
    absmax = float(np.max(np.abs(np.asarray(values, dtype=np.float64))))
    if not math.isfinite(absmax):
        raise ValueError("calibration data produced non-finite activation values")
    return max(absmax / float(qmax), eps)


def _quantize_symmetric(values: np.ndarray, scale: np.ndarray | float, qmin: int, qmax: int) -> np.ndarray:
    scale_arr = np.asarray(scale, dtype=np.float64)
    if np.any(~np.isfinite(scale_arr)) or np.any(scale_arr <= 0.0):
        raise ValueError("quantization scales must be finite and positive")
    scaled = np.asarray(values, dtype=np.float64) / scale_arr
    q = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    return np.clip(q, qmin, qmax).astype(np.int64)


def _forward_quantized_cmsis_nn(
    spec: QuantizedMLPExportSpec,
    x: np.ndarray,
    *,
    clip: bool = True,
) -> np.ndarray:
    """Evaluate the generated CMSIS-NN fixed-point forward path with NumPy."""

    x_arr = np.asarray(x, dtype=np.float64)
    single = x_arr.ndim == 1
    if single:
        x_arr = x_arr[None, :]
    if x_arr.ndim != 2 or x_arr.shape[1] != spec.input_dim:
        raise ValueError(f"expected input shape (*, {spec.input_dim}), got {x_arr.shape}")

    h_float = (x_arr - spec.x_mean[None, :]) / spec.x_std[None, :]
    h_q = _quantize_symmetric(h_float, spec.activation_scales[0], spec.qmin, spec.qmax)
    for layer_idx, layer in enumerate(spec.layers):
        params = _cmsis_nn_layer_quant_params(spec, layer_idx)
        out_q = np.empty((h_q.shape[0], layer.output_dim), dtype=np.int64)
        for row in range(layer.output_dim):
            raw = h_q.astype(np.int64) @ layer.weight_q[row].astype(np.int64)
            raw = raw + int(params["bias_q"][row])
            out_q[:, row] = _cmsis_requantize_array(
                raw,
                int(params["output_multipliers"][row]),
                int(params["output_shifts"][row]),
                spec.qmin,
                spec.qmax,
            )
        output_scale = float(spec.layer_output_scales[layer_idx])
        h_q = _apply_quantized_activation_np(
            out_q,
            output_scale,
            spec.qmin,
            spec.qmax,
            layer.activation,
            layer.negative_slope,
        )

    y_norm = h_q.astype(np.float64) * spec.layer_output_scales[-1]
    y = y_norm * spec.y_std[None, :] + spec.y_mean[None, :]
    if clip:
        y = np.clip(y, spec.y_clip_low[None, :], spec.y_clip_high[None, :])
    return y[0] if single else y


def _apply_quantized_activation_np(
    values_q: np.ndarray,
    scale: float,
    qmin: int,
    qmax: int,
    activation: str,
    negative_slope: float,
) -> np.ndarray:
    if activation in ("linear", "none", None):
        return np.asarray(values_q, dtype=np.int64)
    if activation == "relu":
        return np.maximum(values_q, 0).astype(np.int64)
    if activation == "tanh":
        params = _cmsis_nn_activation_rescale_params(float(scale))
        cmsis_input = _cmsis_requantize_array(
            values_q,
            params["activation_input_multiplier"],
            params["activation_input_shift"],
            -32768,
            32767,
        )
        cmsis_output = _cmsis_activation_s16_tanh_array(cmsis_input)
        return _cmsis_requantize_array(
            cmsis_output,
            params["activation_output_multiplier"],
            params["activation_output_shift"],
            qmin,
            qmax,
        )
    values = np.asarray(values_q, dtype=np.float64) * float(scale)
    activated = _apply_activation_np(values, activation, negative_slope)
    return _quantize_symmetric(activated, scale, qmin, qmax)


_CMSIS_SIGMOID_TABLE_UINT16 = np.asarray(
    [
        32768, 33451, 34133, 34813, 35493, 36169, 36843, 37513, 38180, 38841, 39498, 40149, 40794, 41432, 42064, 42688,
        43304, 43912, 44511, 45102, 45683, 46255, 46817, 47369, 47911, 48443, 48964, 49475, 49975, 50464, 50942, 51409,
        51865, 52311, 52745, 53169, 53581, 53983, 54374, 54755, 55125, 55485, 55834, 56174, 56503, 56823, 57133, 57433,
        57724, 58007, 58280, 58544, 58800, 59048, 59288, 59519, 59743, 59959, 60168, 60370, 60565, 60753, 60935, 61110,
        61279, 61441, 61599, 61750, 61896, 62036, 62172, 62302, 62428, 62549, 62666, 62778, 62886, 62990, 63090, 63186,
        63279, 63368, 63454, 63536, 63615, 63691, 63765, 63835, 63903, 63968, 64030, 64090, 64148, 64204, 64257, 64308,
        64357, 64405, 64450, 64494, 64536, 64576, 64614, 64652, 64687, 64721, 64754, 64786, 64816, 64845, 64873, 64900,
        64926, 64950, 64974, 64997, 65019, 65039, 65060, 65079, 65097, 65115, 65132, 65149, 65164, 65179, 65194, 65208,
        65221, 65234, 65246, 65258, 65269, 65280, 65291, 65301, 65310, 65319, 65328, 65337, 65345, 65352, 65360, 65367,
        65374, 65381, 65387, 65393, 65399, 65404, 65410, 65415, 65420, 65425, 65429, 65433, 65438, 65442, 65445, 65449,
        65453, 65456, 65459, 65462, 65465, 65468, 65471, 65474, 65476, 65479, 65481, 65483, 65485, 65488, 65489, 65491,
        65493, 65495, 65497, 65498, 65500, 65501, 65503, 65504, 65505, 65507, 65508, 65509, 65510, 65511, 65512, 65513,
        65514, 65515, 65516, 65517, 65517, 65518, 65519, 65520, 65520, 65521, 65522, 65522, 65523, 65523, 65524, 65524,
        65525, 65525, 65526, 65526, 65526, 65527, 65527, 65528, 65528, 65528, 65529, 65529, 65529, 65529, 65530, 65530,
        65530, 65530, 65531, 65531, 65531, 65531, 65531, 65532, 65532, 65532, 65532, 65532, 65532, 65533, 65533, 65533,
        65533, 65533, 65533, 65533, 65533, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65535,
    ],
    dtype=np.int64,
)


def _cmsis_activation_s16_tanh_array(values: np.ndarray) -> np.ndarray:
    input_data = np.asarray(values, dtype=np.int64) * 3
    abs_input_data = np.abs(input_data)
    uh = abs_input_data >> 8
    ut = abs_input_data & 0x0FF
    result = np.full(input_data.shape, 0xFFFF << 8, dtype=np.int64)
    table_mask = uh < 255
    if np.any(table_mask):
        indices = uh[table_mask]
        ua = _CMSIS_SIGMOID_TABLE_UINT16[indices]
        ub = _CMSIS_SIGMOID_TABLE_UINT16[indices + 1]
        result[table_mask] = (ua << 8) + ut[table_mask] * (ub - ua)
    positive = input_data >= 0
    output = np.empty(input_data.shape, dtype=np.int64)
    output[positive] = (result[positive] - (1 << 23) + (1 << 7)) >> 8
    output[~positive] = (-result[~positive] + (1 << 23) + (1 << 7) - 1) >> 8
    return np.clip(output, -32768, 32767).astype(np.int64)


def _cmsis_requantize_array(
    values: np.ndarray,
    multiplier: int,
    shift: int,
    qmin: int,
    qmax: int,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.int64)
    left_shift = max(int(shift), 0)
    right_shift = max(-int(shift), 0)
    x = x * (1 << left_shift)
    x = _cmsis_doubling_high_mult_no_sat_array(x, int(multiplier))
    x = _cmsis_divide_by_power_of_two_array(x, right_shift)
    return np.clip(x, qmin, qmax).astype(np.int64)


def _cmsis_doubling_high_mult_no_sat_array(values: np.ndarray, multiplier: int) -> np.ndarray:
    return ((np.asarray(values, dtype=np.int64) * int(multiplier) + (1 << 30)) >> 31).astype(np.int64)


def _cmsis_divide_by_power_of_two_array(values: np.ndarray, exponent: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.int64)
    if exponent == 0:
        return x
    remainder_mask = (1 << int(exponent)) - 1
    remainder = x & remainder_mask
    result = x >> int(exponent)
    threshold = np.full(x.shape, remainder_mask >> 1, dtype=np.int64)
    threshold = np.where(result < 0, threshold + 1, threshold)
    return np.where(remainder > threshold, result + 1, result).astype(np.int64)


def _cmsis_real_multiplier(multiplier: int, shift: int) -> float:
    return math.ldexp(float(multiplier) / float(1 << 31), int(shift))


def _quantize_cmsis_multiplier(real_multiplier: float) -> tuple[int, int]:
    if not math.isfinite(real_multiplier) or real_multiplier <= 0.0:
        raise ValueError(f"invalid CMSIS-NN real multiplier {real_multiplier!r}")
    significand, shift = math.frexp(real_multiplier)
    quantized = int(round(significand * float(1 << 31)))
    if quantized == (1 << 31):
        quantized //= 2
        shift += 1
    if quantized <= 0 or quantized > 0x7FFFFFFF:
        raise ValueError(f"could not quantize CMSIS-NN multiplier {real_multiplier!r}")
    return quantized, shift


def _cmsis_nn_layer_quant_params(spec: QuantizedMLPExportSpec, layer_idx: int) -> dict[str, np.ndarray]:
    layer = spec.layers[layer_idx]
    input_scale = float(spec.activation_scales[layer_idx])
    output_scale = float(spec.layer_output_scales[layer_idx])
    if output_scale <= 0.0:
        raise ValueError("CMSIS-NN output scales must be positive")

    bias_dtype = np.int64 if spec.qtype in ("q15", "q31") else np.int32
    bias_q = np.empty(layer.output_dim, dtype=bias_dtype)
    output_multipliers = np.empty(layer.output_dim, dtype=np.int32)
    output_shifts = np.empty(layer.output_dim, dtype=np.int32)
    for row in range(layer.output_dim):
        effective_weight_scale = float(layer.weight_scale[row])
        accumulator_scale = input_scale * effective_weight_scale
        if accumulator_scale <= 0.0:
            raise ValueError("CMSIS-NN accumulator scales must be positive")
        if bias_dtype == np.int32:
            bias_q[row] = int(_quantize_symmetric(layer.bias[row], accumulator_scale, -(2**31), 2**31 - 1))
        else:
            bias_q[row] = int(_quantize_symmetric(layer.bias[row], accumulator_scale, -(2**62), 2**62 - 1))
        multiplier, shift = _quantize_cmsis_multiplier(accumulator_scale / output_scale)
        output_multipliers[row] = multiplier
        output_shifts[row] = shift
    return {
        "bias_q": bias_q,
        "output_multipliers": output_multipliers,
        "output_shifts": output_shifts,
    }


def _cmsis_nn_activation_rescale_params(output_scale: float) -> dict[str, int]:
    if output_scale <= 0.0:
        raise ValueError("CMSIS-NN activation scales must be positive")
    input_multiplier, input_shift = _quantize_cmsis_multiplier(output_scale * 4096.0)
    output_multiplier, output_shift = _quantize_cmsis_multiplier(1.0 / (32768.0 * output_scale))
    return {
        "activation_input_multiplier": input_multiplier,
        "activation_input_shift": input_shift,
        "activation_output_multiplier": output_multiplier,
        "activation_output_shift": output_shift,
    }


@dataclass(frozen=True)
class _ExportContext:
    spec: MLPExportSpec
    options: CodeExportOptions
    prefix: str

    @property
    def upper(self) -> str:
        return self.prefix.upper()

    @property
    def float_type(self) -> str:
        return _float_type(self.options.precision)

    @property
    def default_c_type(self) -> str:
        return "double" if self.float_type == "double" else "float"


@dataclass(frozen=True)
class _QuantizedExportContext:
    spec: QuantizedMLPExportSpec
    options: CodeExportOptions
    prefix: str

    @property
    def upper(self) -> str:
        return self.prefix.upper()

    @property
    def float_type(self) -> str:
        return _float_type(self.options.precision)

    @property
    def default_c_type(self) -> str:
        return "double" if self.float_type == "double" else "float"

    @property
    def dtype_info(self) -> _QuantDTypeInfo:
        return _quant_dtype_info(self.spec.qtype)


def _generate_data_files(ctx: _ExportContext) -> tuple[str, str]:
    p = ctx.prefix
    u = ctx.upper
    spec = ctx.spec
    guard = f"{u}_DATA_H_"
    header = f"""/* Auto-generated AMPC policy data. */
#ifndef {guard}
#define {guard}

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

#ifndef {u}_FLOAT_TYPE
#define {u}_FLOAT_TYPE {ctx.default_c_type}
#endif
typedef {u}_FLOAT_TYPE {p}_float_t;

#define {u}_INPUT_DIM {spec.input_dim}u
#define {u}_OUTPUT_DIM {spec.output_dim}u
#define {u}_NUM_LAYERS {len(spec.layers)}u
#define {u}_MAX_LAYER_SIZE {spec.max_layer_size}u
#define {u}_MAX_INPUT_SIZE {max(spec.input_dim, spec.max_layer_size)}u

typedef enum {{
    {u}_ACTIVATION_LINEAR = 0,
    {u}_ACTIVATION_RELU = 1,
    {u}_ACTIVATION_LEAKY_RELU = 2,
    {u}_ACTIVATION_TANH = 3,
    {u}_ACTIVATION_GELU = 4,
    {u}_ACTIVATION_SILU = 5,
    {u}_ACTIVATION_ELU = 6
}} {p}_activation_t;

typedef struct {{
    uint16_t input_dim;
    uint16_t output_dim;
    {p}_activation_t activation;
    {p}_float_t negative_slope;
    const {p}_float_t *weights;  /* row-major: output_dim x input_dim */
    const {p}_float_t *biases;   /* output_dim */
}} {p}_layer_t;

extern const {p}_float_t {p}_x_mean[{u}_INPUT_DIM];
extern const {p}_float_t {p}_x_std[{u}_INPUT_DIM];
extern const {p}_float_t {p}_y_mean[{u}_OUTPUT_DIM];
extern const {p}_float_t {p}_y_std[{u}_OUTPUT_DIM];
extern const {p}_float_t {p}_y_clip_low[{u}_OUTPUT_DIM];
extern const {p}_float_t {p}_y_clip_high[{u}_OUTPUT_DIM];
extern const {p}_layer_t {p}_layers[{u}_NUM_LAYERS];

#ifdef __cplusplus
}}
#endif

#endif /* {guard} */
"""

    source = f"""/* Auto-generated AMPC policy data. */
#include "{p}_data.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {{
#endif

const {p}_float_t {p}_x_mean[{u}_INPUT_DIM] = {{
{_format_array(spec.x_mean, ctx)}
}};

const {p}_float_t {p}_x_std[{u}_INPUT_DIM] = {{
{_format_array(spec.x_std, ctx)}
}};

const {p}_float_t {p}_y_mean[{u}_OUTPUT_DIM] = {{
{_format_array(spec.y_mean, ctx)}
}};

const {p}_float_t {p}_y_std[{u}_OUTPUT_DIM] = {{
{_format_array(spec.y_std, ctx)}
}};

const {p}_float_t {p}_y_clip_low[{u}_OUTPUT_DIM] = {{
{_format_array(spec.y_clip_low, ctx)}
}};

const {p}_float_t {p}_y_clip_high[{u}_OUTPUT_DIM] = {{
{_format_array(spec.y_clip_high, ctx)}
}};

"""
    for i, layer in enumerate(spec.layers):
        source += f"static const {p}_float_t {p}_layer_{i}_weights[{layer.output_dim * layer.input_dim}u] = {{\n"
        source += _format_matrix(layer.weight, ctx)
        source += "};\n\n"
        source += f"static const {p}_float_t {p}_layer_{i}_biases[{layer.output_dim}u] = {{\n"
        source += _format_array(layer.bias, ctx)
        source += "};\n\n"

    source += f"const {p}_layer_t {p}_layers[{u}_NUM_LAYERS] = {{\n"
    for i, layer in enumerate(spec.layers):
        comma = "," if i < len(spec.layers) - 1 else ""
        source += (
            "    {\n"
            f"        .input_dim = {layer.input_dim}u,\n"
            f"        .output_dim = {layer.output_dim}u,\n"
            f"        .activation = {_activation_enum(ctx, layer.activation)},\n"
            f"        .negative_slope = {_c_number(layer.negative_slope, ctx)},\n"
            f"        .weights = {p}_layer_{i}_weights,\n"
            f"        .biases = {p}_layer_{i}_biases\n"
            f"    }}{comma}\n"
        )
    source += "};\n"
    source += "\n#ifdef __cplusplus\n}\n#endif\n"
    return header, source


def _generate_quantized_data_files(ctx: _QuantizedExportContext) -> tuple[str, str]:
    p = ctx.prefix
    u = ctx.upper
    spec = ctx.spec
    dtype_info = ctx.dtype_info
    weight_storage_c_type = _weight_storage_c_type(spec)
    bias_q_c_type = _bias_q_c_type(ctx)
    guard = f"{u}_DATA_H_"
    header = f"""/* Auto-generated quantized AMPC policy data. */
#ifndef {guard}
#define {guard}

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

#ifndef {u}_FLOAT_TYPE
#define {u}_FLOAT_TYPE {ctx.default_c_type}
#endif
typedef {u}_FLOAT_TYPE {p}_float_t;
typedef {dtype_info.c_type} {p}_q_t;
typedef {weight_storage_c_type} {p}_weight_storage_t;
typedef {bias_q_c_type} {p}_bias_q_t;

#define {u}_INPUT_DIM {spec.input_dim}u
#define {u}_OUTPUT_DIM {spec.output_dim}u
#define {u}_NUM_LAYERS {len(spec.layers)}u
#define {u}_MAX_LAYER_SIZE {spec.max_layer_size}u
#define {u}_MAX_INPUT_SIZE {max(spec.input_dim, spec.max_layer_size)}u
#define {u}_QUANTIZED 1u
#define {u}_QMIN {spec.qmin}
#define {u}_QMAX {spec.qmax}
#define {u}_QTYPE_{spec.qtype.upper()} 1u
#define {u}_WEIGHT_QTYPE_{spec.weight_qtype.upper()} 1u

typedef enum {{
    {u}_ACTIVATION_LINEAR = 0,
    {u}_ACTIVATION_RELU = 1,
    {u}_ACTIVATION_LEAKY_RELU = 2,
    {u}_ACTIVATION_TANH = 3,
    {u}_ACTIVATION_GELU = 4,
    {u}_ACTIVATION_SILU = 5,
    {u}_ACTIVATION_ELU = 6
}} {p}_activation_t;

typedef struct {{
    uint16_t input_dim;
    uint16_t output_dim;
    uint16_t weight_stride;
    {p}_activation_t activation;
    {p}_float_t negative_slope;
    const {p}_weight_storage_t *weights; /* row-major; q4 stores two signed weights per byte */
    const {p}_float_t *weight_scales;   /* output_dim */
    const {p}_float_t *biases;          /* output_dim */
    const {p}_bias_q_t *biases_q;       /* output_dim, accumulator scale */
    const int32_t *output_multipliers;  /* output_dim, CMSIS-NN requantization */
    const int32_t *output_shifts;       /* output_dim, CMSIS-NN requantization */
    int32_t activation_input_multiplier;  /* rescale calibrated q to CMSIS-NN s16 activation input */
    int32_t activation_input_shift;
    int32_t activation_output_multiplier; /* rescale CMSIS-NN s16 tanh output back to calibrated q */
    int32_t activation_output_shift;
}} {p}_layer_t;

extern const {p}_float_t {p}_x_mean[{u}_INPUT_DIM];
extern const {p}_float_t {p}_x_std[{u}_INPUT_DIM];
extern const {p}_float_t {p}_y_mean[{u}_OUTPUT_DIM];
extern const {p}_float_t {p}_y_std[{u}_OUTPUT_DIM];
extern const {p}_float_t {p}_y_clip_low[{u}_OUTPUT_DIM];
extern const {p}_float_t {p}_y_clip_high[{u}_OUTPUT_DIM];
extern const {p}_float_t {p}_activation_scales[{u}_NUM_LAYERS];
extern const {p}_float_t {p}_layer_output_scales[{u}_NUM_LAYERS];
extern const {p}_layer_t {p}_layers[{u}_NUM_LAYERS];

#ifdef __cplusplus
}}
#endif

#endif /* {guard} */
"""

    source = f"""/* Auto-generated quantized AMPC policy data. */
#include "{p}_data.h"

#ifdef __cplusplus
extern "C" {{
#endif

const {p}_float_t {p}_x_mean[{u}_INPUT_DIM] = {{
{_format_array(spec.x_mean, ctx)}
}};

const {p}_float_t {p}_x_std[{u}_INPUT_DIM] = {{
{_format_array(spec.x_std, ctx)}
}};

const {p}_float_t {p}_y_mean[{u}_OUTPUT_DIM] = {{
{_format_array(spec.y_mean, ctx)}
}};

const {p}_float_t {p}_y_std[{u}_OUTPUT_DIM] = {{
{_format_array(spec.y_std, ctx)}
}};

const {p}_float_t {p}_y_clip_low[{u}_OUTPUT_DIM] = {{
{_format_array(spec.y_clip_low, ctx)}
}};

const {p}_float_t {p}_y_clip_high[{u}_OUTPUT_DIM] = {{
{_format_array(spec.y_clip_high, ctx)}
}};

const {p}_float_t {p}_activation_scales[{u}_NUM_LAYERS] = {{
{_format_array(spec.activation_scales, ctx)}
}};

const {p}_float_t {p}_layer_output_scales[{u}_NUM_LAYERS] = {{
{_format_array(spec.layer_output_scales, ctx)}
}};

"""
    for i, layer in enumerate(spec.layers):
        cmsis_params = _cmsis_nn_layer_quant_params(spec, i)
        activation_params = _cmsis_nn_activation_rescale_params(float(spec.layer_output_scales[i]))
        weight_stride = _weight_storage_stride(layer, spec.weight_qtype)
        if spec.weight_qtype == "q4":
            packed_weights = _packed_q4_weight_matrix(layer)
            source += f"static const {p}_weight_storage_t {p}_layer_{i}_weights[{layer.output_dim * weight_stride}u] = {{\n"
            source += _format_integer_array(packed_weights)
        else:
            source += f"static const {p}_weight_storage_t {p}_layer_{i}_weights[{layer.output_dim * weight_stride}u] = {{\n"
            source += _format_integer_matrix(layer.weight_q)
        source += "};\n\n"
        source += f"static const {p}_float_t {p}_layer_{i}_weight_scales[{layer.output_dim}u] = {{\n"
        source += _format_array(layer.weight_scale, ctx)
        source += "};\n\n"
        source += f"static const {p}_float_t {p}_layer_{i}_biases[{layer.output_dim}u] = {{\n"
        source += _format_array(layer.bias, ctx)
        source += "};\n\n"
        source += f"static const {p}_bias_q_t {p}_layer_{i}_biases_q[{layer.output_dim}u] = {{\n"
        source += _format_integer_array(cmsis_params["bias_q"])
        source += "};\n\n"
        source += f"static const int32_t {p}_layer_{i}_output_multipliers[{layer.output_dim}u] = {{\n"
        source += _format_integer_array(cmsis_params["output_multipliers"])
        source += "};\n\n"
        source += f"static const int32_t {p}_layer_{i}_output_shifts[{layer.output_dim}u] = {{\n"
        source += _format_integer_array(cmsis_params["output_shifts"])
        source += "};\n\n"

    source += f"const {p}_layer_t {p}_layers[{u}_NUM_LAYERS] = {{\n"
    for i, layer in enumerate(spec.layers):
        comma = "," if i < len(spec.layers) - 1 else ""
        source += (
            "    {\n"
            f"        .input_dim = {layer.input_dim}u,\n"
            f"        .output_dim = {layer.output_dim}u,\n"
            f"        .weight_stride = {_weight_storage_stride(layer, spec.weight_qtype)}u,\n"
            f"        .activation = {_activation_enum(ctx, layer.activation)},\n"
            f"        .negative_slope = {_c_number(layer.negative_slope, ctx)},\n"
            f"        .weights = {p}_layer_{i}_weights,\n"
            f"        .weight_scales = {p}_layer_{i}_weight_scales,\n"
            f"        .biases = {p}_layer_{i}_biases,\n"
            f"        .biases_q = {p}_layer_{i}_biases_q,\n"
            f"        .output_multipliers = {p}_layer_{i}_output_multipliers,\n"
            f"        .output_shifts = {p}_layer_{i}_output_shifts,\n"
            f"        .activation_input_multiplier = {activation_params['activation_input_multiplier']},\n"
            f"        .activation_input_shift = {activation_params['activation_input_shift']},\n"
            f"        .activation_output_multiplier = {activation_params['activation_output_multiplier']},\n"
            f"        .activation_output_shift = {activation_params['activation_output_shift']}\n"
            f"    }}{comma}\n"
        )
    source += "};\n"
    source += "\n#ifdef __cplusplus\n}\n#endif\n"
    return header, source


def _generate_test_data_files(ctx: _ExportContext, cases: TestCases) -> tuple[str, str]:
    p = ctx.prefix
    u = ctx.upper
    guard = f"{u}_TEST_DATA_H_"
    header = f"""/* Auto-generated AMPC policy test data. */
#ifndef {guard}
#define {guard}

#include "{p}_data.h"

#ifdef __cplusplus
extern "C" {{
#endif

#define {u}_TEST_CASE_COUNT {cases.count}u

typedef struct {{
    {p}_float_t input[{u}_INPUT_DIM];
    {p}_float_t output[{u}_OUTPUT_DIM];
}} {p}_test_case_t;

extern const {p}_test_case_t {p}_test_cases[{u}_TEST_CASE_COUNT];

#ifdef __cplusplus
}}
#endif

#endif /* {guard} */
"""
    source = f"""/* Auto-generated AMPC policy test data. */
#include "{p}_test_data.h"

#ifdef __cplusplus
extern "C" {{
#endif

const {p}_test_case_t {p}_test_cases[{u}_TEST_CASE_COUNT] = {{
"""
    for i in range(cases.count):
        comma = "," if i < cases.count - 1 else ""
        source += f"    /* test case {i} */\n"
        source += "    {\n"
        source += f"        .input = {{{_format_inline_array(cases.inputs[i], ctx)}}},\n"
        source += f"        .output = {{{_format_inline_array(cases.outputs[i], ctx)}}}\n"
        source += f"    }}{comma}\n"
    source += "};\n\n#ifdef __cplusplus\n}\n#endif\n"
    return header, source


def _generate_simple_forward_files(ctx: _ExportContext, *, cmsis: bool = False) -> tuple[str, str]:
    p = ctx.prefix
    u = ctx.upper
    guard = f"{u}_FORWARD_H_"
    header = f"""/* Auto-generated AMPC policy forward pass. */
#ifndef {guard}
#define {guard}

#include "{p}_data.h"

#ifdef __cplusplus
extern "C" {{
#endif

typedef struct {{
    {p}_float_t input_scaled[{u}_INPUT_DIM];
    {p}_float_t layer_a[{u}_MAX_LAYER_SIZE];
    {p}_float_t layer_b[{u}_MAX_LAYER_SIZE];
}} {p}_workspace_t;

int {p}_forward(
    const {p}_float_t input[{u}_INPUT_DIM],
    {p}_float_t output[{u}_OUTPUT_DIM],
    {p}_workspace_t *workspace
);

#ifdef __cplusplus
}}
#endif

#endif /* {guard} */
"""
    if cmsis:
        source = _generate_cmsis_forward_source(ctx)
    else:
        source = _generate_simple_forward_source(ctx)
    return header, source


def _generate_quantized_simple_forward_files(ctx: _QuantizedExportContext, *, cmsis: bool = False) -> tuple[str, str]:
    p = ctx.prefix
    u = ctx.upper
    guard = f"{u}_FORWARD_H_"
    cmsis_include = "\n#include <arm_math.h>\n" if cmsis else ""
    cmsis_workspace = (
        f"    {p}_q_t layer_linear[{u}_MAX_LAYER_SIZE];\n"
        f"    int32_t cmsis_buffer[{u}_MAX_LAYER_SIZE];\n"
        if cmsis
        else ""
    )
    header = f"""/* Auto-generated quantized AMPC policy forward pass. */
#ifndef {guard}
#define {guard}

#include "{p}_data.h"
{cmsis_include}

#ifdef __cplusplus
extern "C" {{
#endif

typedef struct {{
    {p}_q_t input_q[{u}_INPUT_DIM];
    {p}_q_t layer_a[{u}_MAX_LAYER_SIZE];
    {p}_q_t layer_b[{u}_MAX_LAYER_SIZE];
{cmsis_workspace}    {p}_float_t layer_float[{u}_MAX_LAYER_SIZE];
}} {p}_workspace_t;

int {p}_forward(
    const {p}_float_t input[{u}_INPUT_DIM],
    {p}_float_t output[{u}_OUTPUT_DIM],
    {p}_workspace_t *workspace
);

#ifdef __cplusplus
}}
#endif

#endif /* {guard} */
"""
    source = _generate_quantized_cmsis_forward_source(ctx) if cmsis else _generate_quantized_simple_forward_source(ctx)
    return header, source


def _generate_simple_forward_source(ctx: _ExportContext) -> str:
    p = ctx.prefix
    u = ctx.upper
    return f"""/* Auto-generated plain-C AMPC policy forward pass. */
#include "{p}_forward.h"

#include <math.h>
#include <stddef.h>

static {p}_float_t {p}_activate({p}_float_t x, {p}_activation_t activation, {p}_float_t negative_slope) {{
    switch (activation) {{
        case {u}_ACTIVATION_RELU:
            return x > ({p}_float_t)0 ? x : ({p}_float_t)0;
        case {u}_ACTIVATION_LEAKY_RELU:
            return x >= ({p}_float_t)0 ? x : negative_slope * x;
        case {u}_ACTIVATION_TANH:
            return ({p}_float_t)tanh((double)x);
        case {u}_ACTIVATION_GELU:
            return ({p}_float_t)(0.5 * (double)x * (1.0 + erf((double)x * 0.70710678118654752440)));
        case {u}_ACTIVATION_ELU:
            return x >= ({p}_float_t)0 ? x : ({p}_float_t)expm1((double)x);
        case {u}_ACTIVATION_SILU:
            return ({p}_float_t)((double)x / (1.0 + exp(-(double)x)));
        case {u}_ACTIVATION_LINEAR:
        default:
            return x;
    }}
}}

int {p}_forward(
    const {p}_float_t input[{u}_INPUT_DIM],
    {p}_float_t output[{u}_OUTPUT_DIM],
    {p}_workspace_t *workspace
) {{
    if (input == NULL || output == NULL || workspace == NULL) {{
        return -1;
    }}

    for (uint16_t i = 0u; i < {u}_INPUT_DIM; ++i) {{
        workspace->input_scaled[i] = (input[i] - {p}_x_mean[i]) / {p}_x_std[i];
    }}

    const {p}_float_t *layer_input = workspace->input_scaled;
    {p}_float_t *buffers[2] = {{workspace->layer_a, workspace->layer_b}};
    uint16_t buffer_index = 0u;

    for (uint16_t layer_index = 0u; layer_index < {u}_NUM_LAYERS; ++layer_index) {{
        const {p}_layer_t *layer = &{p}_layers[layer_index];
        {p}_float_t *layer_output = buffers[buffer_index];

        for (uint16_t row = 0u; row < layer->output_dim; ++row) {{
            {p}_float_t acc = layer->biases[row];
            const {p}_float_t *weight_row = &layer->weights[(uint32_t)row * (uint32_t)layer->input_dim];
            for (uint16_t col = 0u; col < layer->input_dim; ++col) {{
                acc += weight_row[col] * layer_input[col];
            }}
            layer_output[row] = {p}_activate(acc, layer->activation, layer->negative_slope);
        }}

        layer_input = layer_output;
        buffer_index = (uint16_t)(1u - buffer_index);
    }}

    for (uint16_t i = 0u; i < {u}_OUTPUT_DIM; ++i) {{
        {p}_float_t y = layer_input[i] * {p}_y_std[i] + {p}_y_mean[i];
        if (y < {p}_y_clip_low[i]) {{
            y = {p}_y_clip_low[i];
        }}
        if (y > {p}_y_clip_high[i]) {{
            y = {p}_y_clip_high[i];
        }}
        output[i] = y;
    }}

    return 0;
}}
"""


def _generate_cmsis_forward_source(ctx: _ExportContext) -> str:
    p = ctx.prefix
    u = ctx.upper
    return f"""/* Auto-generated CMSIS-DSP AMPC policy forward pass. */
#include "{p}_forward.h"

#include <arm_math.h>
#include <math.h>
#include <stddef.h>

static {p}_float_t {p}_activate({p}_float_t x, {p}_activation_t activation, {p}_float_t negative_slope) {{
    switch (activation) {{
        case {u}_ACTIVATION_RELU:
            return x > ({p}_float_t)0 ? x : ({p}_float_t)0;
        case {u}_ACTIVATION_LEAKY_RELU:
            return x >= ({p}_float_t)0 ? x : negative_slope * x;
        case {u}_ACTIVATION_TANH:
            return ({p}_float_t)tanhf((float32_t)x);
        case {u}_ACTIVATION_GELU:
            return ({p}_float_t)(0.5f * (float32_t)x * (1.0f + erff((float32_t)x * 0.70710678118654752440f)));
        case {u}_ACTIVATION_ELU:
            return x >= ({p}_float_t)0 ? x : ({p}_float_t)expm1f((float32_t)x);
        case {u}_ACTIVATION_SILU:
            return ({p}_float_t)((float32_t)x / (1.0f + expf(-(float32_t)x)));
        case {u}_ACTIVATION_LINEAR:
        default:
            return x;
    }}
}}

int {p}_forward(
    const {p}_float_t input[{u}_INPUT_DIM],
    {p}_float_t output[{u}_OUTPUT_DIM],
    {p}_workspace_t *workspace
) {{
    if (input == NULL || output == NULL || workspace == NULL) {{
        return -1;
    }}

    for (uint16_t i = 0u; i < {u}_INPUT_DIM; ++i) {{
        workspace->input_scaled[i] = (input[i] - {p}_x_mean[i]) / {p}_x_std[i];
    }}

    const {p}_float_t *layer_input = workspace->input_scaled;
    {p}_float_t *buffers[2] = {{workspace->layer_a, workspace->layer_b}};
    uint16_t buffer_index = 0u;

    for (uint16_t layer_index = 0u; layer_index < {u}_NUM_LAYERS; ++layer_index) {{
        const {p}_layer_t *layer = &{p}_layers[layer_index];
        {p}_float_t *layer_output = buffers[buffer_index];
        arm_matrix_instance_f32 weight_mat;
        arm_mat_init_f32(
            &weight_mat,
            layer->output_dim,
            layer->input_dim,
            (float32_t *)layer->weights
        );
        arm_mat_vec_mult_f32(&weight_mat, (float32_t *)layer_input, (float32_t *)layer_output);

        for (uint16_t row = 0u; row < layer->output_dim; ++row) {{
            const {p}_float_t acc = layer_output[row] + layer->biases[row];
            layer_output[row] = {p}_activate(acc, layer->activation, layer->negative_slope);
        }}

        layer_input = layer_output;
        buffer_index = (uint16_t)(1u - buffer_index);
    }}

    for (uint16_t i = 0u; i < {u}_OUTPUT_DIM; ++i) {{
        {p}_float_t y = layer_input[i] * {p}_y_std[i] + {p}_y_mean[i];
        if (y < {p}_y_clip_low[i]) {{
            y = {p}_y_clip_low[i];
        }}
        if (y > {p}_y_clip_high[i]) {{
            y = {p}_y_clip_high[i];
        }}
        output[i] = y;
    }}

    return 0;
}}
"""


def _quantized_weight_helpers_source(ctx: _QuantizedExportContext) -> str:
    p = ctx.prefix
    if ctx.spec.weight_qtype == "q4":
        return f"""static int8_t {p}_decode_q4(uint8_t value) {{
    const uint8_t nibble = (uint8_t)(value & 0x0Fu);
    return (int8_t)(((nibble & 0x08u) != 0u) ? ((int16_t)nibble - 16) : (int16_t)nibble);
}}

static int32_t {p}_weight_value(const {p}_layer_t *layer, uint16_t row, uint16_t col) {{
    const {p}_weight_storage_t *weight_row = &layer->weights[(uint32_t)row * (uint32_t)layer->weight_stride];
    const uint8_t packed = weight_row[col >> 1u];
    const uint8_t nibble = ((col & 1u) == 0u) ? (uint8_t)(packed & 0x0Fu) : (uint8_t)(packed >> 4);
    return (int32_t){p}_decode_q4(nibble);
}}"""
    return f"""static int32_t {p}_weight_value(const {p}_layer_t *layer, uint16_t row, uint16_t col) {{
    const {p}_q_t *weight_row = (const {p}_q_t *)&layer->weights[(uint32_t)row * (uint32_t)layer->weight_stride];
    return (int32_t)weight_row[col];
}}"""


def _generate_quantized_simple_forward_source(ctx: _QuantizedExportContext) -> str:
    p = ctx.prefix
    u = ctx.upper
    weight_helpers = _quantized_weight_helpers_source(ctx)
    return f"""/* Auto-generated plain-C quantized AMPC policy forward pass. */
#include "{p}_forward.h"

#include <math.h>
#include <stddef.h>

static {p}_float_t {p}_activate({p}_float_t x, {p}_activation_t activation, {p}_float_t negative_slope) {{
    switch (activation) {{
        case {u}_ACTIVATION_RELU:
            return x > ({p}_float_t)0 ? x : ({p}_float_t)0;
        case {u}_ACTIVATION_LEAKY_RELU:
            return x >= ({p}_float_t)0 ? x : negative_slope * x;
        case {u}_ACTIVATION_TANH:
            return ({p}_float_t)tanh((double)x);
        case {u}_ACTIVATION_GELU:
            return ({p}_float_t)(0.5 * (double)x * (1.0 + erf((double)x * 0.70710678118654752440)));
        case {u}_ACTIVATION_ELU:
            return x >= ({p}_float_t)0 ? x : ({p}_float_t)expm1((double)x);
        case {u}_ACTIVATION_SILU:
            return ({p}_float_t)((double)x / (1.0 + exp(-(double)x)));
        case {u}_ACTIVATION_LINEAR:
        default:
            return x;
    }}
}}

static {p}_q_t {p}_quantize_scalar({p}_float_t x, {p}_float_t scale) {{
    double q = round((double)x / (double)scale);
    if (q < (double){u}_QMIN) {{
        q = (double){u}_QMIN;
    }}
    if (q > (double){u}_QMAX) {{
        q = (double){u}_QMAX;
    }}
    return ({p}_q_t)q;
}}

{weight_helpers}

static {p}_float_t {p}_dot_row(
    const {p}_q_t *input,
    const {p}_layer_t *layer,
    uint16_t row,
    {p}_float_t input_scale,
    {p}_float_t weight_scale
) {{
    long double acc = 0.0L;
    for (uint16_t col = 0u; col < layer->input_dim; ++col) {{
        acc += (long double)input[col] * (long double){p}_weight_value(layer, row, col);
    }}
    return ({p}_float_t)(acc * (long double)input_scale * (long double)weight_scale);
}}

int {p}_forward(
    const {p}_float_t input[{u}_INPUT_DIM],
    {p}_float_t output[{u}_OUTPUT_DIM],
    {p}_workspace_t *workspace
) {{
    if (input == NULL || output == NULL || workspace == NULL) {{
        return -1;
    }}

    for (uint16_t i = 0u; i < {u}_INPUT_DIM; ++i) {{
        const {p}_float_t scaled = (input[i] - {p}_x_mean[i]) / {p}_x_std[i];
        workspace->input_q[i] = {p}_quantize_scalar(scaled, {p}_activation_scales[0]);
    }}

    const {p}_q_t *layer_input = workspace->input_q;
    {p}_q_t *q_buffers[2] = {{workspace->layer_a, workspace->layer_b}};
    uint16_t buffer_index = 0u;

    for (uint16_t layer_index = 0u; layer_index < {u}_NUM_LAYERS; ++layer_index) {{
        const {p}_layer_t *layer = &{p}_layers[layer_index];
        const {p}_float_t input_scale = {p}_activation_scales[layer_index];

        for (uint16_t row = 0u; row < layer->output_dim; ++row) {{
            const {p}_float_t dot = {p}_dot_row(
                layer_input,
                layer,
                row,
                input_scale,
                layer->weight_scales[row]
            );
            const {p}_float_t acc = dot + layer->biases[row];
            workspace->layer_float[row] = {p}_activate(acc, layer->activation, layer->negative_slope);
        }}

        if (layer_index + 1u < {u}_NUM_LAYERS) {{
            {p}_q_t *layer_output_q = q_buffers[buffer_index];
            const {p}_float_t output_scale = {p}_activation_scales[layer_index + 1u];
            for (uint16_t row = 0u; row < layer->output_dim; ++row) {{
                layer_output_q[row] = {p}_quantize_scalar(workspace->layer_float[row], output_scale);
            }}
            layer_input = layer_output_q;
            buffer_index = (uint16_t)(1u - buffer_index);
        }}
    }}

    for (uint16_t i = 0u; i < {u}_OUTPUT_DIM; ++i) {{
        {p}_float_t y = workspace->layer_float[i] * {p}_y_std[i] + {p}_y_mean[i];
        if (y < {p}_y_clip_low[i]) {{
            y = {p}_y_clip_low[i];
        }}
        if (y > {p}_y_clip_high[i]) {{
            y = {p}_y_clip_high[i];
        }}
        output[i] = y;
    }}

    return 0;
}}
"""


def _generate_quantized_cmsis_forward_source(ctx: _QuantizedExportContext) -> str:
    p = ctx.prefix
    u = ctx.upper
    fc_call = _cmsis_nn_fully_connected_call_source(ctx)
    return f"""/* Auto-generated CMSIS-NN quantized AMPC policy forward pass. */
#include "{p}_forward.h"

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
#include <math.h>
#include <stddef.h>

static {p}_q_t {p}_quantize_scalar({p}_float_t x, {p}_float_t scale) {{
    double q = round((double)x / (double)scale);
    if (q < (double){u}_QMIN) {{
        q = (double){u}_QMIN;
    }}
    if (q > (double){u}_QMAX) {{
        q = (double){u}_QMAX;
    }}
    return ({p}_q_t)q;
}}

static {p}_float_t {p}_dequantize_scalar({p}_q_t x, {p}_float_t scale) {{
    return ({p}_float_t)((double)x * (double)scale);
}}

static {p}_q_t {p}_clamp_q(int32_t q) {{
    if (q < (int32_t){u}_QMIN) {{
        q = (int32_t){u}_QMIN;
    }}
    if (q > (int32_t){u}_QMAX) {{
        q = (int32_t){u}_QMAX;
    }}
    return ({p}_q_t)q;
}}

static int16_t {p}_clamp_s16(int32_t q) {{
    if (q < -32768) {{
        q = -32768;
    }}
    if (q > 32767) {{
        q = 32767;
    }}
    return (int16_t)q;
}}

static int {p}_cmsis_tanh_q(
    {p}_q_t *data,
    uint16_t size,
    const {p}_layer_t *layer,
    int16_t *activation_buffer
) {{
    for (uint16_t i = 0u; i < size; ++i) {{
        activation_buffer[i] = {p}_clamp_s16(arm_nn_requantize(
            (int32_t)data[i],
            layer->activation_input_multiplier,
            layer->activation_input_shift
        ));
    }}
    const arm_cmsis_nn_status status = arm_nn_activation_s16(
        activation_buffer,
        activation_buffer,
        (int32_t)size,
        0,
        ARM_TANH
    );
    if (status != ARM_CMSIS_NN_SUCCESS) {{
        return -3;
    }}
    for (uint16_t i = 0u; i < size; ++i) {{
        data[i] = {p}_clamp_q(arm_nn_requantize(
            (int32_t)activation_buffer[i],
            layer->activation_output_multiplier,
            layer->activation_output_shift
        ));
    }}
    return 0;
}}

static int {p}_activate_q(
    {p}_q_t *data,
    uint16_t size,
    {p}_activation_t activation,
    {p}_float_t negative_slope,
    {p}_float_t scale,
    const {p}_layer_t *layer,
    int16_t *activation_buffer
) {{
    switch (activation) {{
        case {u}_ACTIVATION_RELU:
            {_cmsis_nn_relu_call(ctx)}
            return 0;
        case {u}_ACTIVATION_LEAKY_RELU:
            for (uint16_t i = 0u; i < size; ++i) {{
                if (data[i] < ({p}_q_t)0) {{
                    const {p}_float_t x = {p}_dequantize_scalar(data[i], scale);
                    data[i] = {p}_quantize_scalar(negative_slope * x, scale);
                }}
            }}
            return 0;
        case {u}_ACTIVATION_TANH:
            return {p}_cmsis_tanh_q(data, size, layer, activation_buffer);
        case {u}_ACTIVATION_GELU:
            for (uint16_t i = 0u; i < size; ++i) {{
                const {p}_float_t x = {p}_dequantize_scalar(data[i], scale);
                const {p}_float_t y = ({p}_float_t)(0.5 * (double)x * (1.0 + erf((double)x * 0.70710678118654752440)));
                data[i] = {p}_quantize_scalar(y, scale);
            }}
            return 0;
        case {u}_ACTIVATION_ELU:
            for (uint16_t i = 0u; i < size; ++i) {{
                const {p}_float_t x = {p}_dequantize_scalar(data[i], scale);
                const {p}_float_t y = x >= ({p}_float_t)0 ? x : ({p}_float_t)expm1((double)x);
                data[i] = {p}_quantize_scalar(y, scale);
            }}
            return 0;
        case {u}_ACTIVATION_SILU:
            for (uint16_t i = 0u; i < size; ++i) {{
                const {p}_float_t x = {p}_dequantize_scalar(data[i], scale);
                const {p}_float_t y = ({p}_float_t)((double)x / (1.0 + exp(-(double)x)));
                data[i] = {p}_quantize_scalar(y, scale);
            }}
            return 0;
        case {u}_ACTIVATION_LINEAR:
        default:
            return 0;
    }}
}}

int {p}_forward(
    const {p}_float_t input[{u}_INPUT_DIM],
    {p}_float_t output[{u}_OUTPUT_DIM],
    {p}_workspace_t *workspace
) {{
    if (input == NULL || output == NULL || workspace == NULL) {{
        return -1;
    }}

    for (uint16_t i = 0u; i < {u}_INPUT_DIM; ++i) {{
        const {p}_float_t scaled = (input[i] - {p}_x_mean[i]) / {p}_x_std[i];
        workspace->input_q[i] = {p}_quantize_scalar(scaled, {p}_activation_scales[0]);
    }}

    const {p}_q_t *layer_input = workspace->input_q;
    {p}_q_t *q_buffers[2] = {{workspace->layer_a, workspace->layer_b}};
    uint16_t buffer_index = 0u;

    for (uint16_t layer_index = 0u; layer_index < {u}_NUM_LAYERS; ++layer_index) {{
        const {p}_layer_t *layer = &{p}_layers[layer_index];
        {p}_q_t *layer_output = q_buffers[buffer_index];
        const {p}_float_t output_scale = {p}_layer_output_scales[layer_index];

{fc_call}

        if ({p}_activate_q(
                layer_output,
                layer->output_dim,
                layer->activation,
                layer->negative_slope,
                output_scale,
                layer,
                (int16_t *)workspace->cmsis_buffer) != 0) {{
            return -3;
        }}
        layer_input = layer_output;
        buffer_index = (uint16_t)(1u - buffer_index);
    }}

    for (uint16_t i = 0u; i < {u}_OUTPUT_DIM; ++i) {{
        {p}_float_t y = {p}_dequantize_scalar(layer_input[i], {p}_layer_output_scales[{u}_NUM_LAYERS - 1u]);
        y = y * {p}_y_std[i] + {p}_y_mean[i];
        if (y < {p}_y_clip_low[i]) {{
            y = {p}_y_clip_low[i];
        }}
        if (y > {p}_y_clip_high[i]) {{
            y = {p}_y_clip_high[i];
        }}
        output[i] = y;
    }}

    return 0;
}}
"""


def _cmsis_nn_relu_call(ctx: _QuantizedExportContext) -> str:
    if ctx.spec.qtype in ("q4", "q7"):
        return "arm_relu_q7((int8_t *)data, size);"
    if ctx.spec.qtype == "q15":
        return "arm_relu_q15((int16_t *)data, size);"
    raise ValueError(f"unsupported CMSIS-NN activation dtype {ctx.spec.qtype!r}")


def _cmsis_nn_fully_connected_call_source(ctx: _QuantizedExportContext) -> str:
    u = ctx.upper
    if ctx.spec.qtype == "q7":
        return f"""        const cmsis_nn_fc_params fc_params = {{
            .input_offset = 0,
            .filter_offset = 0,
            .output_offset = 0,
            .activation = {{.min = {u}_QMIN, .max = {u}_QMAX}}
        }};
        const cmsis_nn_dims input_dims = {{.n = 1, .h = 1, .w = 1, .c = layer->input_dim}};
        const cmsis_nn_dims filter_dims = {{.n = layer->input_dim, .h = 1, .w = 1, .c = 1}};
        const cmsis_nn_dims bias_dims = {{.n = 1, .h = 1, .w = 1, .c = 1}};
        const cmsis_nn_dims output_dims = {{.n = 1, .h = 1, .w = 1, .c = 1}};
        for (uint16_t row = 0u; row < layer->output_dim; ++row) {{
            cmsis_nn_context cmsis_ctx = {{.buf = NULL, .size = 0}};
            const cmsis_nn_per_tensor_quant_params quant_params = {{
                .multiplier = layer->output_multipliers[row],
                .shift = layer->output_shifts[row]
            }};
            const arm_cmsis_nn_status status = arm_fully_connected_s8(
                &cmsis_ctx,
                &fc_params,
                &quant_params,
                &input_dims,
                (const int8_t *)layer_input,
                &filter_dims,
                (const int8_t *)&layer->weights[(uint32_t)row * (uint32_t)layer->weight_stride],
                &bias_dims,
                (const int32_t *)&layer->biases_q[row],
                &output_dims,
                (int8_t *)&layer_output[row]
            );
            if (status != ARM_CMSIS_NN_SUCCESS) {{
                return -2;
            }}
        }}"""
    if ctx.spec.qtype == "q4":
        return f"""        const cmsis_nn_fc_params fc_params = {{
            .input_offset = 0,
            .filter_offset = 0,
            .output_offset = 0,
            .activation = {{.min = {u}_QMIN, .max = {u}_QMAX}}
        }};
        const cmsis_nn_dims input_dims = {{.n = 1, .h = 1, .w = 1, .c = layer->input_dim}};
        const cmsis_nn_dims filter_dims = {{.n = layer->input_dim, .h = 1, .w = 1, .c = 1}};
        const cmsis_nn_dims bias_dims = {{.n = 1, .h = 1, .w = 1, .c = 1}};
        const cmsis_nn_dims output_dims = {{.n = 1, .h = 1, .w = 1, .c = 1}};
        for (uint16_t row = 0u; row < layer->output_dim; ++row) {{
            cmsis_nn_context cmsis_ctx = {{.buf = NULL, .size = 0}};
            const cmsis_nn_per_tensor_quant_params quant_params = {{
                .multiplier = layer->output_multipliers[row],
                .shift = layer->output_shifts[row]
            }};
            const arm_cmsis_nn_status status = arm_fully_connected_s4(
                &cmsis_ctx,
                &fc_params,
                &quant_params,
                &input_dims,
                (const int8_t *)layer_input,
                &filter_dims,
                (const int8_t *)&layer->weights[(uint32_t)row * (uint32_t)layer->weight_stride],
                &bias_dims,
                (const int32_t *)&layer->biases_q[row],
                &output_dims,
                (int8_t *)&layer_output[row]
            );
            if (status != ARM_CMSIS_NN_SUCCESS) {{
                return -2;
            }}
        }}"""
    if ctx.spec.qtype == "q15":
        return f"""        const cmsis_nn_fc_params fc_params = {{
            .input_offset = 0,
            .filter_offset = 0,
            .output_offset = 0,
            .activation = {{.min = {u}_QMIN, .max = {u}_QMAX}}
        }};
        const cmsis_nn_dims input_dims = {{.n = 1, .h = 1, .w = 1, .c = layer->input_dim}};
        const cmsis_nn_dims filter_dims = {{.n = layer->input_dim, .h = 1, .w = 1, .c = 1}};
        const cmsis_nn_dims bias_dims = {{.n = 1, .h = 1, .w = 1, .c = 1}};
        const cmsis_nn_dims output_dims = {{.n = 1, .h = 1, .w = 1, .c = 1}};
        for (uint16_t row = 0u; row < layer->output_dim; ++row) {{
            cmsis_nn_context cmsis_ctx = {{.buf = NULL, .size = 0}};
            const cmsis_nn_per_tensor_quant_params quant_params = {{
                .multiplier = layer->output_multipliers[row],
                .shift = layer->output_shifts[row]
            }};
            const arm_cmsis_nn_status status = arm_fully_connected_s16(
                &cmsis_ctx,
                &fc_params,
                &quant_params,
                &input_dims,
                (const int16_t *)layer_input,
                &filter_dims,
                (const int8_t *)&layer->weights[(uint32_t)row * (uint32_t)layer->weight_stride],
                &bias_dims,
                (const int64_t *)&layer->biases_q[row],
                &output_dims,
                (int16_t *)&layer_output[row]
            );
            if (status != ARM_CMSIS_NN_SUCCESS) {{
                return -2;
            }}
        }}"""
    raise ValueError(f"unsupported CMSIS-NN quantized dtype {ctx.spec.qtype!r}")


def _cmsis_quantized_dot_source(ctx: _QuantizedExportContext) -> str:
    p = ctx.prefix
    if ctx.spec.qtype == "q7":
        return f"""static {p}_float_t {p}_dot_row(
    const {p}_q_t *input,
    const {p}_q_t *weights,
    uint16_t input_dim,
    {p}_float_t input_scale,
    {p}_float_t weight_scale
) {{
    q31_t result = 0;
    arm_dot_prod_q7((const q7_t *)input, (const q7_t *)weights, (uint32_t)input_dim, &result);
    return ({p}_float_t)((double)result * (double)input_scale * (double)weight_scale);
}}"""
    if ctx.spec.qtype == "q15":
        return f"""static {p}_float_t {p}_dot_row(
    const {p}_q_t *input,
    const {p}_q_t *weights,
    uint16_t input_dim,
    {p}_float_t input_scale,
    {p}_float_t weight_scale
) {{
    q63_t result = 0;
    arm_dot_prod_q15((const q15_t *)input, (const q15_t *)weights, (uint32_t)input_dim, &result);
    return ({p}_float_t)((double)result * (double)input_scale * (double)weight_scale);
}}"""
    if ctx.spec.qtype == "q31":
        return f"""static {p}_float_t {p}_dot_row(
    const {p}_q_t *input,
    const {p}_q_t *weights,
    uint16_t input_dim,
    {p}_float_t input_scale,
    {p}_float_t weight_scale
) {{
    q63_t result = 0;
    arm_dot_prod_q31((const q31_t *)input, (const q31_t *)weights, (uint32_t)input_dim, &result);
    return ({p}_float_t)((double)result * 16384.0 * (double)input_scale * (double)weight_scale);
}}"""
    raise ValueError(f"unsupported CMSIS quantized dtype {ctx.spec.qtype!r}")


def _generate_eigen_forward_header(ctx: _ExportContext) -> str:
    p = ctx.prefix
    u = ctx.upper
    return f"""/* Auto-generated Eigen AMPC policy forward pass. */
#pragma once

#include "{p}_data.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace {p} {{

inline {p}_float_t activate({p}_float_t x, {p}_activation_t activation, {p}_float_t negative_slope) {{
    switch (activation) {{
        case {u}_ACTIVATION_RELU:
            return std::max(({p}_float_t)0, x);
        case {u}_ACTIVATION_LEAKY_RELU:
            return x >= ({p}_float_t)0 ? x : negative_slope * x;
        case {u}_ACTIVATION_TANH:
            return ({p}_float_t)std::tanh((double)x);
        case {u}_ACTIVATION_GELU:
            return ({p}_float_t)(0.5 * (double)x * (1.0 + std::erf((double)x * 0.70710678118654752440)));
        case {u}_ACTIVATION_ELU:
            return x >= ({p}_float_t)0 ? x : ({p}_float_t)std::expm1((double)x);
        case {u}_ACTIVATION_SILU:
            return ({p}_float_t)((double)x / (1.0 + std::exp(-(double)x)));
        case {u}_ACTIVATION_LINEAR:
        default:
            return x;
    }}
}}

inline int forward(const {p}_float_t input[{u}_INPUT_DIM], {p}_float_t output[{u}_OUTPUT_DIM]) {{
    if (input == nullptr || output == nullptr) {{
        return -1;
    }}

    using Vec = Eigen::Matrix<{p}_float_t, Eigen::Dynamic, 1>;
    using RowMajorMat = Eigen::Matrix<{p}_float_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    Vec h({u}_INPUT_DIM);
    for (int i = 0; i < (int){u}_INPUT_DIM; ++i) {{
        h(i) = (input[i] - {p}_x_mean[i]) / {p}_x_std[i];
    }}

    for (uint16_t layer_index = 0u; layer_index < {u}_NUM_LAYERS; ++layer_index) {{
        const {p}_layer_t &layer = {p}_layers[layer_index];
        Eigen::Map<const RowMajorMat> W(layer.weights, layer.output_dim, layer.input_dim);
        Eigen::Map<const Vec> b(layer.biases, layer.output_dim);
        Vec next = W * h + b;
        for (int i = 0; i < next.size(); ++i) {{
            next(i) = activate(next(i), layer.activation, layer.negative_slope);
        }}
        h = next;
    }}

    for (uint16_t i = 0u; i < {u}_OUTPUT_DIM; ++i) {{
        {p}_float_t y = h(i) * {p}_y_std[i] + {p}_y_mean[i];
        y = std::max(y, {p}_y_clip_low[i]);
        y = std::min(y, {p}_y_clip_high[i]);
        output[i] = y;
    }}
    return 0;
}}

}}  // namespace {p}
"""


def _generate_quantized_eigen_forward_header(ctx: _QuantizedExportContext) -> str:
    p = ctx.prefix
    u = ctx.upper
    weight_helpers = _quantized_eigen_weight_helpers_source(ctx)
    raw_source = _quantized_eigen_raw_source(ctx)
    return f"""/* Auto-generated Eigen quantized AMPC policy forward pass. */
#pragma once

#include "{p}_data.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace {p} {{

inline {p}_float_t activate({p}_float_t x, {p}_activation_t activation, {p}_float_t negative_slope) {{
    switch (activation) {{
        case {u}_ACTIVATION_RELU:
            return std::max(({p}_float_t)0, x);
        case {u}_ACTIVATION_LEAKY_RELU:
            return x >= ({p}_float_t)0 ? x : negative_slope * x;
        case {u}_ACTIVATION_TANH:
            return ({p}_float_t)std::tanh((double)x);
        case {u}_ACTIVATION_GELU:
            return ({p}_float_t)(0.5 * (double)x * (1.0 + std::erf((double)x * 0.70710678118654752440)));
        case {u}_ACTIVATION_ELU:
            return x >= ({p}_float_t)0 ? x : ({p}_float_t)std::expm1((double)x);
        case {u}_ACTIVATION_SILU:
            return ({p}_float_t)((double)x / (1.0 + std::exp(-(double)x)));
        case {u}_ACTIVATION_LINEAR:
        default:
            return x;
    }}
}}

inline {p}_q_t quantize_scalar({p}_float_t x, {p}_float_t scale) {{
    double q = std::round((double)x / (double)scale);
    q = std::max(q, (double){u}_QMIN);
    q = std::min(q, (double){u}_QMAX);
    return ({p}_q_t)q;
}}

{weight_helpers}

inline int forward(const {p}_float_t input[{u}_INPUT_DIM], {p}_float_t output[{u}_OUTPUT_DIM]) {{
    if (input == nullptr || output == nullptr) {{
        return -1;
    }}

    using QVec = Eigen::Matrix<{p}_q_t, Eigen::Dynamic, 1>;
    using FVec = Eigen::Matrix<{p}_float_t, Eigen::Dynamic, 1>;
    using DVec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using RowMajorQMat = Eigen::Matrix<{p}_q_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    QVec h_q({u}_INPUT_DIM);
    for (int i = 0; i < (int){u}_INPUT_DIM; ++i) {{
        const {p}_float_t scaled = (input[i] - {p}_x_mean[i]) / {p}_x_std[i];
        h_q(i) = quantize_scalar(scaled, {p}_activation_scales[0]);
    }}

    FVec final_output({u}_OUTPUT_DIM);
    for (uint16_t layer_index = 0u; layer_index < {u}_NUM_LAYERS; ++layer_index) {{
        const {p}_layer_t &layer = {p}_layers[layer_index];
{raw_source}
        FVec next(layer.output_dim);
        for (int row = 0; row < (int)layer.output_dim; ++row) {{
            const double dequantized = raw(row) * (double){p}_activation_scales[layer_index] * (double)layer.weight_scales[row];
            const {p}_float_t acc = ({p}_float_t)dequantized + layer.biases[row];
            next(row) = activate(acc, layer.activation, layer.negative_slope);
        }}

        if (layer_index + 1u < {u}_NUM_LAYERS) {{
            h_q.resize(layer.output_dim);
            for (int row = 0; row < (int)layer.output_dim; ++row) {{
                h_q(row) = quantize_scalar(next(row), {p}_activation_scales[layer_index + 1u]);
            }}
        }} else {{
            final_output = next;
        }}
    }}

    for (uint16_t i = 0u; i < {u}_OUTPUT_DIM; ++i) {{
        {p}_float_t y = final_output(i) * {p}_y_std[i] + {p}_y_mean[i];
        y = std::max(y, {p}_y_clip_low[i]);
        y = std::min(y, {p}_y_clip_high[i]);
        output[i] = y;
    }}
    return 0;
}}

}}  // namespace {p}
"""


def _quantized_eigen_weight_helpers_source(ctx: _QuantizedExportContext) -> str:
    p = ctx.prefix
    if ctx.spec.weight_qtype != "q4":
        return ""
    return f"""inline int8_t decode_q4(uint8_t value) {{
    const uint8_t nibble = (uint8_t)(value & 0x0Fu);
    return (int8_t)(((nibble & 0x08u) != 0u) ? ((int16_t)nibble - 16) : (int16_t)nibble);
}}

inline int32_t weight_value(const {p}_layer_t &layer, int row, int col) {{
    const {p}_weight_storage_t *weight_row = &layer.weights[(uint32_t)row * (uint32_t)layer.weight_stride];
    const uint8_t packed = weight_row[(uint32_t)col >> 1u];
    const uint8_t nibble = ((col & 1) == 0) ? (uint8_t)(packed & 0x0Fu) : (uint8_t)(packed >> 4);
    return (int32_t)decode_q4(nibble);
}}"""


def _quantized_eigen_raw_source(ctx: _QuantizedExportContext) -> str:
    p = ctx.prefix
    if ctx.spec.weight_qtype == "q4":
        return f"""        DVec raw(layer.output_dim);
        for (int row = 0; row < (int)layer.output_dim; ++row) {{
            double acc = 0.0;
            for (int col = 0; col < (int)layer.input_dim; ++col) {{
                acc += (double)weight_value(layer, row, col) * (double)h_q(col);
            }}
            raw(row) = acc;
        }}"""
    return f"""        Eigen::Map<const RowMajorQMat> W((const {p}_q_t *)layer.weights, layer.output_dim, layer.input_dim);
        const DVec raw = W.template cast<double>() * h_q.template cast<double>();"""


def _generate_c_example_main(ctx: _ExportContext) -> str:
    p = ctx.prefix
    u = ctx.upper
    tol = _c_number(ctx.options.test_tolerance, ctx)
    return f"""/* Auto-generated AMPC policy example/test main. */
#include "{p}_forward.h"
#include "{p}_test_data.h"

#include <math.h>
#include <stdio.h>

int main(void) {{
    {p}_workspace_t workspace;
    {p}_float_t output[{u}_OUTPUT_DIM];
    double max_abs_error = 0.0;

    for (uint16_t tc = 0u; tc < {u}_TEST_CASE_COUNT; ++tc) {{
        if ({p}_forward({p}_test_cases[tc].input, output, &workspace) != 0) {{
            printf("forward failed for test case %u\\n", (unsigned)tc);
            return 2;
        }}
        for (uint16_t i = 0u; i < {u}_OUTPUT_DIM; ++i) {{
            const double err = fabs((double)output[i] - (double){p}_test_cases[tc].output[i]);
            if (err > max_abs_error) {{
                max_abs_error = err;
            }}
        }}
    }}

    printf("AMPC policy tests: %u cases, max_abs_error=%.9g\\n", (unsigned){u}_TEST_CASE_COUNT, max_abs_error);
    return max_abs_error <= (double){tol} ? 0 : 1;
}}
"""


def _generate_eigen_example_main(ctx: _ExportContext) -> str:
    p = ctx.prefix
    u = ctx.upper
    tol = _c_number(ctx.options.test_tolerance, ctx)
    return f"""/* Auto-generated AMPC policy Eigen example/test main. */
#include "{p}_forward.hpp"
#include "{p}_test_data.h"

#include <cmath>
#include <cstdio>

int main() {{
    {p}_float_t output[{u}_OUTPUT_DIM];
    double max_abs_error = 0.0;

    for (uint16_t tc = 0u; tc < {u}_TEST_CASE_COUNT; ++tc) {{
        if ({p}::forward({p}_test_cases[tc].input, output) != 0) {{
            std::printf("forward failed for test case %u\\n", (unsigned)tc);
            return 2;
        }}
        for (uint16_t i = 0u; i < {u}_OUTPUT_DIM; ++i) {{
            const double err = std::fabs((double)output[i] - (double){p}_test_cases[tc].output[i]);
            if (err > max_abs_error) {{
                max_abs_error = err;
            }}
        }}
    }}

    std::printf("AMPC policy tests: %u cases, max_abs_error=%.9g\\n", (unsigned){u}_TEST_CASE_COUNT, max_abs_error);
    return max_abs_error <= (double){tol} ? 0 : 1;
}}
"""


def _manifest(
    ctx: _ExportContext | _QuantizedExportContext,
    test_cases: TestCases | None,
    written: Mapping[str, pathlib.Path],
) -> dict[str, object]:
    spec = ctx.spec
    manifest = {
        "schema_version": 1,
        "name": spec.name,
        "prefix": ctx.prefix,
        "backend": ctx.options.backend,
        "precision": ctx.options.precision,
        "input_dim": spec.input_dim,
        "output_dim": spec.output_dim,
        "architecture": {
            "type": "mlp",
            "layers": [
                {
                    "input_dim": layer.input_dim,
                    "output_dim": layer.output_dim,
                    "activation": layer.activation,
                    "negative_slope": layer.negative_slope,
                }
                for layer in spec.layers
            ],
        },
        "model_config": dict(spec.model_config),
        "metadata_iteration": spec.metadata.get("iteration"),
        "prediction_target": spec.model_config.get("prediction_target"),
        "test_cases": None
        if test_cases is None
        else {
            "count": test_cases.count,
            "tolerance": ctx.options.test_tolerance,
        },
        "files": {key: pathlib.Path(path).name for key, path in written.items()},
    }
    if isinstance(spec, QuantizedMLPExportSpec):
        quantization_metadata = spec.metadata.get("quantization", {})
        if not isinstance(quantization_metadata, Mapping):
            quantization_metadata = {}
        manifest["quantization"] = {
            "method": "post_training",
            "representation": "symmetric_scale_no_zero_point",
            "qtype": spec.qtype,
            "weight_qtype": spec.weight_qtype,
            "qmin": spec.qmin,
            "qmax": spec.qmax,
            "weight_qmin": spec.weight_qmin,
            "weight_qmax": spec.weight_qmax,
            "calibration_method": quantization_metadata.get("calibration_method", "absmax"),
            "activation_scale_count": int(spec.activation_scales.shape[0]),
            "weight_scale_granularity": "per_output_channel",
            "weight_storage": "packed_int4_low_nibble_first" if spec.weight_qtype == "q4" else "scalar",
            "bias_dtype": "float",
            "test_outputs": "quantized_policy",
        }
    return manifest


def _sanitize_identifier(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError("prefix must contain at least one identifier character")
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _float_type(precision: str) -> str:
    normalized = precision.strip().lower()
    if normalized in ("float", "float32", "float32_t", "single"):
        return "float"
    if normalized in ("double", "float64", "float64_t"):
        return "double"
    raise ValueError("precision must be one of float32, float32_t, float, double, or float64")


def _activation_enum(ctx: _ExportContext | _QuantizedExportContext, activation: str) -> str:
    activation = "linear" if activation in ("none", None) else activation
    mapping = {
        "linear": f"{ctx.upper}_ACTIVATION_LINEAR",
        "relu": f"{ctx.upper}_ACTIVATION_RELU",
        "leaky_relu": f"{ctx.upper}_ACTIVATION_LEAKY_RELU",
        "tanh": f"{ctx.upper}_ACTIVATION_TANH",
        "gelu": f"{ctx.upper}_ACTIVATION_GELU",
        "elu": f"{ctx.upper}_ACTIVATION_ELU",
        "silu": f"{ctx.upper}_ACTIVATION_SILU",
        "swish": f"{ctx.upper}_ACTIVATION_SILU",
    }
    if activation not in mapping:
        raise ValueError(f"unsupported activation {activation!r}")
    return mapping[activation]


def _format_array(values: np.ndarray, ctx: _ExportContext | _QuantizedExportContext, *, indent: str = "    ") -> str:
    arr = np.asarray(values).reshape(-1)
    lines = []
    for start in range(0, arr.size, 8):
        chunk = arr[start : start + 8]
        suffix = "," if start + 8 < arr.size else ""
        lines.append(indent + ", ".join(_c_number(float(v), ctx) for v in chunk) + suffix)
    return "\n".join(lines) + "\n"


def _format_matrix(values: np.ndarray, ctx: _ExportContext) -> str:
    mat = np.asarray(values)
    lines = []
    for row in range(mat.shape[0]):
        suffix = "," if row < mat.shape[0] - 1 else ""
        lines.append("    " + ", ".join(_c_number(float(v), ctx) for v in mat[row]) + suffix)
    return "\n".join(lines) + "\n"


def _format_integer_matrix(values: np.ndarray) -> str:
    mat = np.asarray(values)
    lines = []
    for row in range(mat.shape[0]):
        suffix = "," if row < mat.shape[0] - 1 else ""
        lines.append("    " + ", ".join(_c_integer(int(v)) for v in mat[row]) + suffix)
    return "\n".join(lines) + "\n"


def _format_integer_array(values: np.ndarray, *, indent: str = "    ") -> str:
    arr = np.asarray(values).reshape(-1)
    lines = []
    for start in range(0, arr.size, 8):
        chunk = arr[start : start + 8]
        suffix = "," if start + 8 < arr.size else ""
        lines.append(indent + ", ".join(_c_integer(int(v)) for v in chunk) + suffix)
    return "\n".join(lines) + "\n"


def _format_inline_array(values: np.ndarray, ctx: _ExportContext | _QuantizedExportContext) -> str:
    return ", ".join(_c_number(float(v), ctx) for v in np.asarray(values).reshape(-1))


def _c_number(value: float, ctx: _ExportContext | _QuantizedExportContext) -> str:
    if math.isnan(value):
        return "NAN"
    if math.isinf(value):
        return "INFINITY" if value > 0 else "-INFINITY"
    if ctx.float_type == "double":
        return _ensure_decimal(f"{value:.17g}")
    return f"{_ensure_decimal(f'{np.float32(value):.9g}')}f"


def _c_integer(value: int) -> str:
    return str(int(value))


def _ensure_decimal(text: str) -> str:
    if any(marker in text for marker in (".", "e", "E")):
        return text
    return f"{text}.0"


def _write_text(path: pathlib.Path, content: str) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
