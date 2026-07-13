"""Convert and JIT a small dense CasADi SX function."""

from pathlib import Path
import sys

import casadi as cs
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from warpmpc.numpysadi import convert


def main() -> None:
    x = cs.SX.sym("x", 2, 3)
    y = cs.SX.sym("y", 3, 2)
    casadi_fn = cs.Function("dense_example", [x, y], [x @ y + 1.0, x.T])

    jax_fn = convert(casadi_fn)
    compiled = jax.jit(jax_fn)

    x_value = jnp.arange(6.0).reshape(2, 3)
    y_value = jnp.ones((3, 2))
    product, transpose = compiled(x_value, y_value)

    print(product)
    print(transpose)


if __name__ == "__main__":
    main()
