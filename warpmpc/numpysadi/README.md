# warpmpc.numpysadi

`warpmpc.numpysadi` converts dense CasADi `SXFunction` objects into JAX-compatible
Python functions.  It deliberately goes through CasADi's generated C code and
translates the straight-line scalar body into Python/JAX source, which avoids
the graph-level Python translation cost that can become large for huge SX
functions.

```python
import casadi as cs
import jax
import jax.numpy as jnp

from warpmpc.numpysadi import convert, export

x = cs.SX.sym("x", 2, 3)
y = cs.SX.sym("y", 3, 2)
casadi_fn = cs.Function("matmul_plus", [x, y], [x @ y + 1.0])

jax_fn = convert(casadi_fn)
compiled = jax.jit(jax_fn)

result = compiled(jnp.ones((2, 3)), jnp.ones((3, 2)))
export(casadi_fn, "generated_matmul_plus.py")
```

## Scope

The library currently supports:

- CasADi `SXFunction` inputs only.
- Dense vector and matrix inputs/outputs.
- Straight-line scalar code emitted by CasADi's C code generator, including
  arithmetic, powers, trigonometric functions, hyperbolic functions,
  comparisons, boolean operations, `if_else`, matrix multiplication, dense
  inverse/solve/determinant expressions after SX scalarization, and common C99
  math helpers.

MX functions and sparse inputs/outputs are rejected explicitly.

## Development

Run the tests from the repository root:

```bash
python -m pytest
```

From the repository root, run a small benchmark locally.  The default batch size
is `10_000`, and each provider/size run is killed and recorded as a timeout if
it exceeds 10 minutes.

```bash
python benchmarks/numpysadi_synthetic/benchmark_numpysadi_synthetic.py --sizes 100 1000 --repeats 3
```

The benchmark can compare against `jaxadi` when it is installed:

```bash
python benchmarks/numpysadi_synthetic/benchmark_numpysadi_synthetic.py --sizes 100 1000 10000 --providers numpysadi jaxadi
```

The Slurm runner used by the full benchmark submitter is
`benchmarks/numpysadi_synthetic/run_numpysadi_synthetic.sh`; it writes JSONL,
plot, and log outputs under the repository-level `results/` directory.
