# WarpMPC Benchmarks

The benchmark scripts are intentionally separate from the default WarpMPC
install. Install the library first:

```bash
python -m pip install -e .
python -m pip install -e ".[benchmarks]"
```

The optional extra covers common Python packages used by the benchmark suite
(`osqp`, `jaxadi`, `jaxopt`, `torch`, `pyyaml`, `cmake`, and `nvidia-ml-py`). It does not
install research baselines or backend stacks that usually need source checkouts
or cluster-specific CUDA setup.

Additional benchmark dependencies:

- QPAX for the QPAX linear MPC baseline.
- MPAX for MPAX linear and nonlinear MPC baselines. MPAX support in
  `warpmpc.jax_sqp` is still experimental.
- MPX and `trajax` for MPX nonlinear MPC baselines.
- TurboMPC for TurboMPC nonlinear MPC baselines. The H100 scripts default to
  `admm_fused_cudss` and `direct_cudss_ffi`.
- CuPy, NVIDIA `nvmath-python`, and cuDSS libraries for the cuDSS
  factorization benchmark.
- `resources/modm`, `resources/CMSIS-NN`, `resources/CMSIS-DSP`, and
  `resources/eigen` only for embedded AMPC microcontroller benchmarks.

Ignored local source checkouts can be placed under `resources/`, for example
`resources/MPAX`, `resources/mpx`, `resources/turbompc`, and `resources/qpax`.
Those folders are ignored by git.

Check an environment before launching the full H100 sweep:

```bash
python benchmarks/check_dependencies.py --strict --require-gpu
```

Run the H100 paper benchmark submitter from the repository root:

```bash
SBATCH_ACCOUNT=<your-slurm-account> SBATCH_PARTITION=<your-gpu-partition> bash benchmarks/run_all_paper_benchmarks_h100.sh
```

The submitter writes results to `results/h100_paper_benchmarks_<timestamp>/`
and Slurm logs to `hpclogs/%A.log`. To use a specific environment, pass
`PYTHON=/path/to/python` or `VENV=/path/to/venv` in the same command.

For a single benchmark script, pass the account directly to `sbatch`, for
example:

```bash
sbatch --account=<your-slurm-account> --partition=<your-gpu-partition> benchmarks/nonlinear_mpc/run_cartpole_quadratic_mpc.sh
```

The full and sequential launchers include the 1,000-sample finite-difference
sensitivity study; its figures stay inside the experiment results directory.
See [accuracy study](linear_mpc_gradients/README_accuracy.md) for configuration.
External VRAM sampling is enabled by default; see
[VRAM monitoring](VRAM_MONITORING.md) for measurement limits and artifacts.
Set `MONITOR_VRAM=0` to disable sampling. Cluster accounts, partitions, CUDA
library paths and environment activation belong in your submission environment.

MPAX correctness fixes are applied to a temporary cached copy of a compatible
MPAX package, without modifying its installation. The patch supports revision
`ca1c669fea422c2509fae4bc30e1d79e6ca8977c` and requires `git` on PATH.
The nonlinear MPAX launcher uses main's 500-iteration configuration and `1e-4`
tolerances; explicit `MPAX_*` environment variables can override them. The other
nonlinear QP backends retain the publication branch's `1e-5` tolerances.

The nonlinear suite includes MPPI and CEM sampling baselines for cartpole and
quadrotor, alongside the SQP baselines. Humanoid TurboMPC defaults to two Ruiz
scaling passes (`TURBOMPC_HUMANOID_SCALING=2`). The sampling jobs in the full
paper runners require an H100; individual sampling scripts can run on other
CUDA GPUs when `SAMPLING_REQUIRE_H100=0`.

Preview either complete sweep without submitting jobs or running benchmarks:

```bash
bash benchmarks/run_all_paper_benchmarks_h100.sh --dry-run
bash benchmarks/run_all_paper_benchmarks_h100_sequential.sh --dry-run
```

Both runners launch numerical result jobs independently of tests and dependency
checks. Run those checks separately when setting up the environment.

Cartpole sampling uses the publication grid (`dt=0.1`, 100 prediction stages,
20 closed-loop steps), matching the two-second SQP benchmark. Sampling settings
were carried over from main; quality and H100 throughput need remeasurement on
this grid.
