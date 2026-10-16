# Fixed exponential shooting-grid experiment

Run, for example:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/franka-matplotlib \
  .venv/bin/python -m optimization.mesh.run \
  --name mesh24_wind20 --horizon 24 --seconds 20
```

The horizon remains **1.2 seconds**, the first interval remains **30 ms**,
the nominal controller update remains **20 ms**, Newton runs at **0.5 ms**,
and the controller receives the same delayed noisy position-only ball observer.
There is one SQP iteration and exactly 50 OSQP iterations. The simulation clock
pauses while the controller computes.

`dt[k] = 0.03 * growth**k`, where `growth` is solved so the interval sum is 1.2.
The final node's otherwise unused duration variable is fixed to the last interval.
Every stage retains all baseline constraints and all node bounds.

The cost is discretization-aware:

- Running residuals 0 through 50 receive `sqrt(dt[k]/0.03)` weighting.
- Torque-difference residuals 30 through 36 additionally receive `0.03/dt[k]`.
  This retains the original weight on the integrated squared torque derivative.
- Contact complementarity, restitution and impulse residuals 51 through 53 keep
  their original weights because they penalize a discrete impact event.
- Terminal residuals and the numerical Gauss–Newton proximal term are unchanged.

The warmstart evaluates the old joint and ball arcs at absolute shifted times,
interpolates torque on the original mesh, re-bins impacts using destination
interval lengths, then regenerates the physical ball trajectory on the new
joint arcs. Duals map by source interval, with contact and initial-boundary
multipliers reset as in the baseline. These are initial guesses for the QP,
not an exact transformation of a converged KKT solution.

`check.py` confirms that the 40-node uniform special case reproduces the
baseline initial seed, all SQP data, shifted seed and shifted duals to floating
point precision. `run.py` executes the shared Newton runner with only the
controller imports, source archive list and experiment-log destination replaced.
It does not modify the baseline files. Python timings are screening evidence;
they do not establish optimized C++ computing time.

## Closed-loop results

The full 20-second trials all completed 26 bounces, with zero torque clipping,
no unintended contacts, and all measured joint-motion limits satisfied. The
archived-trace audits independently reproduce the observer from position
packets, the wind and torque ramps, simulation clocks and iteration counts.

| Grid | Case | Apex range (m) | Apex RMSE (mm) | Maximum lateral displacement (mm) |
|---|---|---|---:|---:|
| 20 nodes | Nominal | 0.649–0.784 | 34.71 | 16.26 |
| 20 nodes | Wind | 0.698–0.790 | 27.64 | 48.83 |
| 20 nodes | Wind + damping | 0.704–0.790 | 26.56 | 42.58 |
| 24 nodes | Nominal | 0.704–0.793 | 28.56 | 14.05 |
| 24 nodes | Wind | 0.699–0.802 | 27.53 | 45.38 |
| 24 nodes | Wind + damping | 0.689–0.773 | 27.29 | 40.38 |
| 28 nodes | Wind | 0.700–0.781 | 26.20 | 45.39 |

The 24-node grid is the conservative candidate. Its apex RMSE improves over
the matched prior C++ 40-node results in each case (31.51, 28.58 and 28.56 mm
for nominal, wind and damped). Maximum lateral displacement rises by 2.79 mm
nominally and 2.54 mm with wind, and falls by 2.27 mm with damping. Twenty nodes
still pass the task, but include a lower nominal apex and slightly worse nominal
RMSE. These comparisons establish bounded performance on the tested cases;
they do not establish robustness to every initial condition or model mismatch.

See [results.json](results.json) for exact run paths and
[report.pdf](report.pdf) for the comparison. All original trace, prediction,
observer and contact PDFs are retained in those run folders. `summarize.py`
regenerates their arm-report captions to state the nonuniform grid correctly.
