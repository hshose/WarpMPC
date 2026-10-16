# Constraint pruning experiments

The conservative candidate removes `workspace`, `tilt`, `upward`,
`paddle_velocity`, and `jerk` after the first five shooting intervals. The saved
baseline trajectories never approached these five bounds. It also removes
duplicated endpoint-acceleration rows in the tail: retain the current-node
endpoint at each stage, plus the next-node endpoint in the final stage. Every
shooting node still has its endpoint acceleration constrained. Contact geometry,
contact approach, dynamics, clock, interval acceleration, torque rate, and all
simple variable bounds remain.

The initial uniform mesh uses 40 intervals of 30 ms. The combined candidate uses
24 intervals, `h_k = 0.03 * 1.0418034191927987**k`, totaling 1.2 s; its final
interval is 76.947 ms and the full-constraint prefix spans 163.076 ms.
Control updates remain 20 ms, with the existing short impact hold, and Newton
advances by 0.5 ms. Simulation pauses during every solve. All tests use the
timestamped position-only observer, 1 SQP step, and exactly 50 OSQP iterations.

Running residuals receive `sqrt(h_k / 0.03)` quadrature. The torque-difference
residual additionally receives `0.03 / h_k` so its physical torque-rate penalty
is unchanged. Discrete contact-event residuals and the terminal cost retain
their weights. These are fixed stage weights when computing derivatives.

## Closed-loop results

The conservative 24-node combined candidate passed all three 20-second tests:

| Case | Impacts | Apex range [m] | Apex RMSE about 0.75 m [mm] | Max acceleration / limit | Max torque rate / limit | Max jerk / limit |
|---|---:|---:|---:|---:|---:|---:|
| Nominal | 26 | 0.6842–0.7953 | 30.58 | 0.9696 | 0.5060 | 0.1994 |
| Smooth wind | 26 | 0.7016–0.7915 | 22.51 | 0.9705 | 0.9736 | 0.4542 |
| Wind + damping 0.02 | 26 | 0.7085–0.8010 | 24.75 | 0.9753 | 0.7403 | 0.2431 |

There were no unintended contacts, torque clipping, or measured research-limit
violations. Baseline C++ apex RMSE for the corresponding cases was 31.51, 28.58,
and 28.56 mm. The combined candidate improves this average-error metric, while
the nominal apex range is slightly wider than the baseline 0.6971–0.7740 m.

The 40-node conservative candidate also passed nominal, wind, and a stronger
synthetic damping mismatch of 0.08 N·m·s/rad, with 26 impacts in each test.
These damping values are synthetic mismatch cases, not identified hardware
friction parameters.

An exploratory candidate removed all endpoint-acceleration constraints after
eight uniform intervals. Its measured closed loops passed, but its predicted
tail acceleration reached 1.335 times the nominal acceleration limit. Increasing
the acceleration residual amplitude from 0.4 to 0.5 and torque-rate amplitude
from 0.15 to 0.18 reduced this predicted excess and improved the wind apex RMSE
from 29.1 to 23.3 mm; it still permitted a predicted ratio of 1.063. Those
amplitude changes multiply the squared cost weights by 1.5625 and 1.44.
**Full endpoint omission is exploratory; the recommended combined candidate
retains all node endpoint bounds and uses the original cost weights.**

The omitted five groups remained comfortably inside their original limits in
every saved prediction of the combined candidate. Its largest tail endpoint
residual was 0.00433 in normalized units, comparable to the baseline's finite
SQP/OSQP feasibility residual. Predictions are sampled every tenth control solve;
these samples establish observed behavior, not a proof over all possible states.

## Evidence and reproduction

- [Complete trial statistics](results.json), [one-page comparison](report.pdf).
- [Uniform sparse-row equivalence](checks.json): retained Jacobians, Hessian,
  gradient, constraints and cost agree exactly with the original formulation.
- [Combined mesh equivalence](combined_checks.json): exact agreement with the
  full 24-node mesh formulation on saved trajectories, with 1,330 nonlinear
  rows instead of 1,760. Every endpoint node 0 through 24 remains covered.
- Each run contains `pruning.json`, `config.json`, source snapshots, predictions,
  observer inputs, iteration logs, physical traces, and controller/contact/
  observer PDFs. `pruned_constraint_audit.json` independently evaluates the full
  original factor library, including rows omitted by that trial.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m optimization.pruning.combined_run -- \
  --name prun_mesh24_wind --seconds 20
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m optimization.pruning.combined_run -- \
  --name prun_mesh24_nominal --seconds 20 --wind none
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m optimization.pruning.combined_run -- \
  --name prun_mesh24_damped --seconds 20 --damping .02

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m optimization.pruning.run \
  --variant dedup --prefix 5 -- --name prun_dedup5_wind --seconds 20
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m optimization.pruning.run \
  --variant drop_endpoints --prefix 8 --accel-weight .5 --rate-weight .18 -- \
  --name prun_dropend8_reg_wind --seconds 20

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m optimization.pruning.check
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m optimization.pruning.combined_check
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m optimization.pruning.summarize
```

The Python wrappers change only the controller/formulation selection and their
own evidence destination. The existing Newton plant, observer, ramp execution,
metrics, and physical audits are shared. Python prototype timings in the report
are not optimized native timings and must not be used to establish a 10 ms
deployment budget. The native implementation and its workstation benchmarks
are validated separately.
