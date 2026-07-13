"""Export generated JAX source to a Python file."""

from pathlib import Path
import sys

import casadi as cs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from warpmpc.numpysadi import export


def main() -> None:
    x = cs.SX.sym("x", 4, 1)
    y = cs.sin(x) + cs.sqrt(x * x + 1.0)
    casadi_fn = cs.Function("smooth_vector", [x], [y])

    output_path = Path(__file__).with_name("generated_smooth_vector.py")
    export(casadi_fn, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
