import tempfile
import unittest
from pathlib import Path

import casadi as cs
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from warpmpc.numpysadi import convert, export, generate_source, import_python_function


def _as_list(value):
    if isinstance(value, (tuple, list)):
        return list(value)
    return [value]


def _assert_matches(testcase, casadi_function, inputs, atol=1e-9, rtol=1e-9):
    jax_function = convert(casadi_function)
    casadi_outputs = _as_list(casadi_function(*inputs))
    jax_outputs = _as_list(jax_function(*inputs))
    testcase.assertEqual(len(casadi_outputs), len(jax_outputs))

    for casadi_out, jax_out in zip(casadi_outputs, jax_outputs):
        np.testing.assert_allclose(
            np.asarray(casadi_out),
            np.asarray(jax.device_get(jax_out)),
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )

    compiled_outputs = _as_list(jax.jit(jax_function)(*inputs))
    for casadi_out, jax_out in zip(casadi_outputs, compiled_outputs):
        np.testing.assert_allclose(
            np.asarray(casadi_out),
            np.asarray(jax.device_get(jax_out)),
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )


class ConverterTests(unittest.TestCase):
    def test_all_common_scalar_sx_operations(self):
        x = cs.SX.sym("x", 3, 2)
        y = cs.SX.sym("y", 3, 2)
        z = cs.SX.sym("z", 3, 2)
        xp = x + 2.0
        yp = y + 2.0

        expressions = [
            x + y,
            x - y,
            x * y,
            x / yp,
            -x,
            x**2,
            xp ** (y / 10.0 + 2.0),
            2.0 ** (x / 10.0),
            cs.sqrt(xp),
            cs.sin(x),
            cs.cos(x),
            cs.tan(x / 5.0),
            cs.asin(x / 10.0),
            cs.acos(x / 10.0),
            cs.atan(x),
            cs.atan2(x, yp),
            cs.sinh(x / 5.0),
            cs.cosh(x / 5.0),
            cs.tanh(x),
            cs.asinh(x),
            cs.acosh(xp),
            cs.atanh(x / 10.0),
            cs.exp(x / 5.0),
            cs.expm1(x / 5.0),
            cs.log(xp),
            cs.log1p(xp),
            cs.floor(x),
            cs.ceil(x),
            cs.fabs(x),
            cs.sign(x),
            cs.fmin(x, y),
            cs.fmax(x, y),
            cs.fmod(xp, yp),
            cs.remainder(xp, yp),
            cs.copysign(x, y),
            cs.hypot(x, y),
            cs.erf(x),
            cs.erfinv(x / 10.0),
            x < y,
            x <= y,
            x == y,
            x != y,
            cs.logic_not(x),
            cs.logic_and(x, y),
            cs.logic_or(x, y),
            cs.if_else(x > 0.0, y, z),
            cs.if_else(x > 0.0, y, 0.0),
        ]
        out = cs.vertcat(*[cs.reshape(expr, -1, 1) for expr in expressions])
        casadi_function = cs.Function("all_ops", [x, y, z], [out])

        inputs = [
            np.linspace(-0.4, 0.5, 6).reshape(3, 2),
            np.linspace(0.2, -0.3, 6).reshape(3, 2),
            np.linspace(0.1, 0.6, 6).reshape(3, 2),
        ]

        _assert_matches(self, casadi_function, inputs)

    def test_multiple_inputs_and_outputs_with_vectors_and_matrices(self):
        x = cs.SX.sym("x", 2, 3)
        y = cs.SX.sym("y", 3, 2)
        v = cs.SX.sym("v", 3, 1)

        matrix_out = x @ y + cs.repmat(v[:2], 1, 2)
        vector_out = y @ x @ v
        scalar_out = cs.dot(v, v) + cs.det(x @ y + cs.SX.eye(2))
        casadi_function = cs.Function(
            "multi_io",
            [x, y, v],
            [matrix_out, vector_out, scalar_out],
            ["state", "gain", "vec"],
            ["matrix", "vector", "scalar"],
        )

        inputs = [
            np.arange(1.0, 7.0).reshape(2, 3) / 10.0,
            np.arange(1.0, 7.0).reshape(3, 2) / 7.0,
            np.array([[0.1], [0.2], [0.3]]),
        ]

        _assert_matches(self, casadi_function, inputs)

    def test_dense_matrix_algebra_scalarizes_through_c_codegen(self):
        a = cs.SX.sym("a", 2, 2)
        b = cs.SX.sym("b", 2, 1)
        shifted = a + cs.SX.eye(2)
        casadi_function = cs.Function(
            "matrix_algebra",
            [a, b],
            [
                cs.det(shifted),
                cs.inv(shifted),
                cs.solve(shifted, b),
                shifted.T,
                shifted @ shifted,
                cs.norm_1(shifted),
                cs.norm_2(b),
                cs.norm_inf(shifted),
                cs.norm_fro(shifted),
            ],
        )

        inputs = [
            np.array([[0.5, 0.2], [0.1, 0.7]]),
            np.array([[0.3], [0.4]]),
        ]

        _assert_matches(self, casadi_function, inputs)

    def test_exported_file_can_be_imported(self):
        x = cs.SX.sym("x", 4, 1)
        casadi_function = cs.Function("export_me", [x], [cs.sin(x) + x * x])
        inputs = [np.linspace(-0.2, 0.3, 4).reshape(4, 1)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "generated.py"
            export(casadi_function, output_path)
            jax_function = import_python_function(output_path, "export_me")
            np.testing.assert_allclose(
                np.asarray(casadi_function(*inputs)),
                np.asarray(jax_function(*inputs)),
                atol=1e-9,
                rtol=1e-9,
            )

    def test_source_contains_jax_imports_and_is_jitable(self):
        x = cs.SX.sym("x", 2, 1)
        casadi_function = cs.Function("source_test", [x], [cs.sqrt(x * x + 1.0)])
        source = generate_source(casadi_function)
        self.assertIn("import jax.numpy as jnp", source)
        self.assertIn("def source_test", source)

        jax_function = convert(casadi_function, jit=True)
        out = jax_function(jnp.array([[0.1], [0.2]]))
        np.testing.assert_allclose(np.asarray(out), np.sqrt(np.array([[1.01], [1.04]])))

    def test_rejects_mx_functions(self):
        x = cs.MX.sym("x", 1, 1)
        casadi_function = cs.Function("mx_function", [x], [x + 1.0])
        with self.assertRaises(TypeError):
            convert(casadi_function)

    def test_rejects_sparse_inputs_or_outputs(self):
        sparsity = cs.Sparsity.diag(3)
        x = cs.SX.sym("x", sparsity)
        casadi_function = cs.Function("sparse_input", [x], [cs.densify(x)])
        with self.assertRaises(ValueError):
            convert(casadi_function)


if __name__ == "__main__":
    unittest.main()
