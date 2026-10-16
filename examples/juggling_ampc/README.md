# Franka juggling AMPC

Train a **40 → 256 → 256 → 256 → 7** tanh policy from 60,000 simultaneous
Newton worlds. The default collection duration is **5 seconds**; held-out
neural-policy evaluation lasts **30 seconds** on 1,024 worlds. Collection,
100 training epochs, Eigen export, and evaluation run in one GPU job.
Optimization-MPC evaluation is disabled unless `--compare-expert` is requested.

The controller uses the tested `245fb5c` full-dynamics formulation: 20 intervals
over 1.2 seconds, 20 ms feedback, two SQP / 40 QP iterations, the native scaling,
penalties, warm starts and merit backtracking. Warp QDLDL runs in float32 with
transposed work, optimal segments and level scheduling. Ball reaction on the
arm is neglected and the midpoint preconditioner is constant.

## Dependencies

Install WarpMPC and the example dependencies into a CUDA-enabled JAX environment:

```bash
python -m pip install -e .
python -m pip install -r examples/juggling_ampc/requirements.txt
```

The bundled reference environment used Warp 1.17.0 and the Newton revision pinned
in `requirements.txt`. Older Warp releases may not provide the kernel options
used here. Eigen headers and CMake are needed for exported C++ inference.
The full reference environment is recorded in `reference/requirements.lock`.

For the staged H100 workflow below, create a dependency directory with your
`venv/`, the pinned Newton checkout under `external/newton/`, and Eigen headers
under `eigen/include/eigen3/`. The helper copies external sources into each job.

## H100 submission

From this repository, submit one 5-second run:

```bash
python3 examples/juggling_ampc/submit_h100_ampc.py \
  --dependencies /path/to/dependencies \
  --account <your-slurm-account> --partition <your-gpu-partition>
```

The dependency directory contains `venv/bin/python`, `external/newton` and
`eigen/include/eigen3`. Installed packages are shared; each job receives a
private copy of the committed repository and external sources. Commit changes
before submitting: the default archive is `git archive HEAD`. An existing
source archive can instead be passed with `--archive`.

`run_h100_ampc.sh` is the Slurm job script; `run_h100_ampc_one_iteration.sh`
provides the same single-training-pass entry. The submission helper supplies
absolute paths as positional arguments, including on CLAIX installations that
strip exported environment variables. No jobs are submitted by merging this
example, and the helper never cancels existing jobs.

Defaults are one H100, 24 CPUs, 120 GiB host RAM and 24 hours. Account and
partition come from `--account` / `--partition`, `SBATCH_ACCOUNT` /
`SBATCH_PARTITION`, or the site defaults.
`--prepare-only` stages sources and records commands without submitting.

By default each job writes to:

```
<repository>/results/juggling_ampc/<timestamp>/arm15_5s/results/
```

Use `--root PATH` to select the enclosing study directory. Each private case
contains `source/`, `gpu_factors/`, `cache/`, and `results/`. Outputs include
`policy/policy.npz`, `policy/training.json`, `policy/eigen/ampc_policy.hpp`,
`evaluation_student/rollout.json`, `pipeline_timings.jsonl`, and the Slurm log.
GPU factors are generated from the bundled formulation unless matching factors
are supplied with `--factors PATH`.

For the calibrated protocol used by the deployed 5 s and 10 s policies:

```bash
python3 examples/juggling_ampc/submit_h100_ampc.py \
  --dependencies /path/to/dependencies --protocol calibrated
```

For an explicit 5/10/20-second sweep, add `--seconds 5 10 20`. Each duration is a
separate job. `submit_claix23_instant.py` remains available for reproducing the
two-protocol study; it also defaults to 5 seconds, with longer durations opt-in.
Older `run_claix23_*` scripts provide alternative workflows; use the portable
H100 entry above for new submissions.

## Collection and training

The default `arm15` protocol keeps measured ball-flight, contact/rebound,
sensing and delivery uncertainty calibrated; only arm servo lag/dead-time
variations are inflated 1.5×. `calibrated` uses 1.0× throughout. Evaluation is
fully calibrated in both cases. The empirical plant and provenance are in
`calibration/`; the native controller snapshot is in `reference/`.

Collection uses true simulator state plus empirical position and correlated
timestamp/delivery residuals, with no observer. Simulation pauses for each
full-batch MPC update. Only the seven applied joint accelerations are retained
as labels. Ball-loss censoring permanently excludes future data from that world.
A 3 GiB GPU reservoir samples individual datums within trajectories; no dataset
shards are written to disk. Whole trajectories are split for training,
validation and testing; normalization uses training data only.

JAX preallocation is disabled. The on-demand `cuda_async` allocator and
`TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC=0` avoid the reproduced BFC fragmentation
failure at 60,000 worlds. The bounded dataset buffer is separate from JAX arena
preallocation.

For direct use, `run_gpu_pipeline.py` defaults to 5 seconds and writes under
`<repository>/results/juggling_ampc/<timestamp>/`. Pass `--factors`, `--source`,
`--calibration` and `--eigen`; `--output` overrides the result directory.
There are no prerequisite C++ parity replays or expert-evaluation jobs.

## Hardware export

`export_policy.py` produces fixed-size Eigen inference.
`build_hardware.sh EXPORT PROFILE [CHECKPOINT]` builds the copied neural
controller, supervisor, Franka session and Vicon application. Set
`FRANKA_SOURCE` to a separate SDK checkout and `BASE_PROFILE` to a matching
native factor profile. The build checks the compiled neural mapping offline;
it neither runs an optimization-MPC evaluation nor connects to the robot.
The generated `PROFILE/run.sh --seconds 30` starts the hardware experiment.

Ball acquisition aligns Vicon to the robot base while holding the current pose,
then searches above the allowed global XY workspace. The ball bottom must be
**at least 1 m above the strike plane** and stationary for three seconds.
The existing IK envelope and robot limits remain active.

Hardware endpoints are explicit: pass `--robot <robot-address>` and
`--vicon-host <server-address:port>` when starting a physical experiment.
Set `vicon_host` in your local timing calibration to the same server after
calibrating that setup. The bundled timing file has no site address.

Historical job logs and deployment records are omitted from this publication.
New training outputs go to `results/`. The source manifest records hashes of the
bundled reference files; calibration provenance records the bundled fit assets.
