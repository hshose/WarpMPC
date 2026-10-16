# Crazyflie AMPC speedrun

Synthesize one fresh expert dataset and train a policy using the saved paper
configuration. Calibration bounds and the cost threshold are loaded from the
bundled, checksum-verified assets. Filtering and normalization use the fresh
data. The policy and normalization are returned in memory.

Defaults: 100,000 trajectories of 200 steps, MPC horizon 40, one SQP iteration,
25 OSQP iterations, Warp factorization and solve, and 100 training epochs with
batch size 100,000 and a 32–32–32 leaky-ReLU policy. The original training loop,
including its per-epoch training/test losses, is retained.

Calibration rollouts, policy evaluation rollouts, plots, checkpoints, dataset
dumps, and C export are omitted. The collector is lowered and compiled before
the dataset-synthesis timer. Printed stage times include any compilation that
occurs inside that stage; they are elapsed times, not pure execution timings.

## Run

The most stripped down version runs directly on a GPU machine with:

```bash
bash examples/crazyflie_ampc/run_h100_ampc_speedrun.sh
```

Set `VENV=/path/to/venv` or `PYTHON=/path/to/python` to select its interpreter.
The equivalent Python entrypoint is:

```bash
python examples/crazyflie_ampc/speedrun.py --cache-dir /path/to/stable/private/cache
```

The default cache root is `~/.cache/warpmpc/ampc_speedrun`, overridable with
`AMPC_CACHE_DIR` or `--cache-dir`. JAX, Warp, CUDA, and XDG caches use separate
subdirectories. Reuse the same absolute path on later runs. The shared cache
defaults also persist sub-second JAX compilations. Native JAX threshold settings
remain overridable through their environment variables.

Other benchmarks and examples use `~/.cache/warpmpc/{jax,warp}` by default
(respecting `XDG_CACHE_HOME`). `WARPMPC_CACHE_DIR` changes their common root;
`JAX_COMPILATION_CACHE_DIR` and `WARP_CACHE_PATH` override individual directories.
The AMPC/cartpole launchers use the same user cache, independent of the
source-copy location.
Fresh processes still trace and restore executables. Rebuilding live Warp
solvers in a different call order can still change their FFI cache keys.

To retain the trained policy in a caller:

```python
from examples.crazyflie_ampc import speedrun as speedrun

speedrun.configure_cache('/path/to/stable/private/cache')
result = speedrun.run()
params = result['params']
normalization = result['normalization']
model = result['model']
```

The CLI discards weights on exit. `--collect-batch-size`, `--rollout-steps`,
`--train-epochs`, `--train-batch-size`, and `--seed` allow small diagnostic runs.
The assets preserve the original paper configuration and calibration provenance;
recalibrate separately if the formulation or sampling distribution changes.

## H100

On a cluster, submit the launcher through Slurm:

```bash
sbatch examples/crazyflie_ampc/run_h100_ampc_speedrun.sh \
  /path/to/repository run \
  /path/to/results \
  /path/to/venv/bin/python
```

Arguments: `[SOURCE_ROOT] [run|smoke|validate] [WORK_ROOT] [PYTHON] [CACHE_ROOT]`.
Native execution locates the repository from the script; Slurm defaults to its
submission directory. Pass the repository explicitly when submitting elsewhere.
Supply these positionally because CLAIX uses `--export=NONE`. Configure your account and partition with normal `sbatch` options. Each job
copies its source into `WORK_ROOT/job-JOBID/code` and runs once. The default
`WORK_ROOT/cache` persists between jobs; use the fifth argument for a private
cache when independent cold runs are required. Logs contain stage/epoch timings;
there is no compilation-study harness or automated cold/warm experiment.

## Validation

```bash
JAX_PLATFORMS=cpu python examples/crazyflie_ampc/validate_speedrun.py
```

This compares sampling, physical noisy dynamics, filtering, masking, and training
against the original Crazyflie example. Slurm mode `validate` also compares the
real Warp collector on H100; mode `smoke` runs a small synthesis/training workload.
GPU output comparisons use float32 tolerances of `1e-3` and identical validity
masks. Validation is separate from the speedrun.
