# WarpMPC Benchmarks

The benchmark scripts are intentionally separate from the default WarpMPC
install. Install the library first:

```bash
python -m pip install -e .
python -m pip install -e ".[benchmarks]"
```

The optional extra covers common Python packages used by the benchmark suite
(`osqp`, `jaxadi`, `jaxopt`, `torch`, `pyyaml`, and `cmake`). It does not
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
SBATCH_ACCOUNT=<your-slurm-account> bash benchmarks/run_all_paper_benchmarks_h100.sh
```

The submitter writes results to `results/h100_paper_benchmarks_<timestamp>/`
and Slurm logs to `hpclogs/%A.log`. To use a specific environment, pass
`PYTHON=/path/to/python` or `VENV=/path/to/venv` in the same command.

For a single benchmark script, pass the account directly to `sbatch`, for
example:

```bash
sbatch --account=<your-slurm-account> benchmarks/nonlinear_mpc/run_cartpole_quadratic_mpc.sh
```
