# Crazyflie AMPC DAgger Example

This example trains a small Flax MLP to imitate a selected target from the
jitted Crazyflie sparse SQP MPC controller.  By default it predicts the first
input, but `--prediction-target action_sequence` trains on the full open-loop
input sequence and `--prediction-target primal_solution` trains on the full MPC
primal vector.  It collects DAgger data from noisy closed-loop rollouts, filters
out-of-distribution SQP solves using bounds from a noise-free expert calibration
rollout, normalizes state and target data, and saves checkpoints after every
training round.

Reusable pieces live in `warpmpc.jax_ampc`: normalization,
dataset buffers and random transition dropping, supervised MLP training,
DAgger schedules, composable state/cost filters, prediction target selectors,
standard MLPs, checkpoint helpers, and modular noise samplers.  The Crazyflie
script keeps only the problem-specific SQP setup, dynamics, costs, plotting,
and experiment orchestration.

DAgger uses convex action mixing during data collection:
`u_apply = (1 - beta) * u_mpc + beta * u_policy`, while the dataset label is
always the expert MPC first action at the visited state.

The SQP rollout starts from the repository Crazyflie helper's linear
state-to-zero initialization with zero controls.  During closed-loop RTI
rollouts, the previous primal MPC solution is reused without shifting, and the
OSQP ADMM state `(x, z, y)` is carried across solves.

Default full run:

```bash
python examples/crazyflie_ampc/train_ampc.py
```

Useful local smoke test:

```bash
python examples/crazyflie_ampc/train_ampc.py \
  --output-dir /tmp/crazyflie_ampc_smoke \
  --iterations 2 \
  --dagger-mixing 0,0.1 \
  --collect-batch-size 4 \
  --dataset-keep-fraction 0.5 \
  --eval-batch-size 4 \
  --filter-batch-size 4 \
  --rollout-steps 2 \
  --horizon-steps 2 \
  --max-iter 2 \
  --train-epochs 1 \
  --train-batch-size 4 \
  --segment-budget 2 \
  --segment-strategy optimal \
  --integrator-substeps 1
```

SQP rollout diagnostics only:

```bash
python examples/crazyflie_ampc/train_ampc.py \
  --sqp-rollout-diagnostics-only \
  --output-dir /tmp/crazyflie_ampc_sqp_diag \
  --collect-batch-size 4 \
  --filter-batch-size 4 \
  --rollout-steps 2 \
  --horizon-steps 2 \
  --max-iter 2 \
  --segment-budget 2 \
  --segment-strategy optimal \
  --integrator-substeps 1
```

The retained H100 smoke entrypoint is `run_h100_ampc_one_iteration.sh`; it runs
one DAgger collection/training iteration with H100-scale solver settings.

There is also a tiny library-only smoke example:

```bash
python examples/jax_ampc_minimal.py \
  --prediction-target action_sequence
```

Outputs are written below `results/crazyflie_ampc` by default:

- `summary.json` contains collection, training, and evaluation metrics.
- `sqp_rollout_diagnostics_summary.json` is written by diagnostics-only runs.
- `outlier_bounds.npz` contains the noise-free calibration bounds.
- `--dataset-keep-fraction` controls random per-timestep transition
  subsampling before appending to the persistent training dataset.
- `--filter-bounds-npz previous/outlier_bounds.npz` reuses hard-coded filter
  bounds and skips calibration; `--disable-filtering` skips both calibration
  and outlier filtering.
- `--prediction-target` selects whether the supervised target is the first
  action, the open-loop action sequence, or the full primal solution vector.
- `plots/*.pdf` contains paired per-state distribution plots for calibration,
  DAgger collection, and policy evaluations: `_all.pdf` includes every
  trajectory, while `_valid.pdf` includes only samples accepted by the current
  validity mask.  The folder also contains a noise-free calibration state/cost
  histogram PDF.
- `checkpoints/iter_XX/params.msgpack` and `normalization.npz` contain the
  policy checkpoint and the state/target normalization needed to reload it.
- Dataset increment NPZ files are written only when `--save-dataset-increments`
  is passed.

Extra example dependencies beyond the base project are listed in
`requirements.txt`.
