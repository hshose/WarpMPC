# Default H100 finite-difference comparison

The full launcher now submits **32 jobs**, including `run_sensitivity_accuracy.sh`.
All ordinary solver benchmarks use the production free-row fix by default:
`OSQPSettings.rho_free=None` assigns `rho` (normally 0.1) to doubly unbounded
rows when vector rho is enabled. Set `rho_free=1e-6` only for explicit legacy
controls. Equality, bounded-row and scalar-rho penalties are unchanged.

The accuracy job generates 1,000 horizon-40 linear MPC QPs and converged float64
CPU finite differences of the 4x12 first-action Jacobian. It compares float32 and
float64 using identical float32-rounded inputs, **both with the production fix**.
Forward counts are **25, 100, 200, 800, 2000**. Refinement counts are
**1, 5, 10, 20, 25, 50, 100, 200, 250, 400, 500**. The paper's 25-forward /
100-refinement setting stays highlighted. Forward states are reused across
refinement counts. No benchmark artifacts are exported to `paper/`.

Each Jacobian entry is normalized by its maximum absolute finite difference
over the full 1,000 samples. Curves report median, mean, p90, p95, p99 and max
over all 48,000 entries. All-sample and stable-active-set figures use the same
scales; filtered figures state the exclusion percentage measured in this run.
Histograms cover all 48 entries with both logarithmic and uniform bins.

Given the all-benchmarks results root `RUN_ROOT`, outputs are:

```
RUN_ROOT/linear_mpc_gradients/sensitivity_accuracy/
  reference/                 # inputs, FD references, active-set audit
  precision_grid/
    metadata.json
    float32_free_rho/         # forward states, Jacobians and summaries
    float64_free_rho/
    figures/
      INDEX.md
      precision_comparison_{all,stable}.{pdf,png}
      {float32_free_rho,float64_free_rho}/{all,stable}/
        # aggregate curves, individual panels, log/linear histograms and CSVs
```

Standalone run (argument is the same parent results root used by the suite):

```bash
ROOT=/path/to/checkout PYTHON=/path/to/gpu/python \
  bash benchmarks/linear_mpc_gradients/run_sensitivity_accuracy.sh /path/to/run_results
```

Configuration uses `SENSITIVITY_BATCH_SIZE` (1000), `SENSITIVITY_HORIZON` (40),
`SENSITIVITY_FORWARD_ITERATIONS` and `SENSITIVITY_REFINEMENTS` (comma-separated),
`SENSITIVITY_WORKERS` (Slurm CPUs or 8), and `SENSITIVITY_PLOT_WORKERS` (4).
`SENSITIVITY_INCLUDE_LEGACY=1` adds legacy-penalty cases explicitly.
`SENSITIVITY_USE_TEX=1` enables LaTeX; default cluster plotting needs no TeX.
`SENSITIVITY_REQUIRE_GPU=0` allows small CPU smoke tests only.
Keep forward 25 and refinement 100 in custom grids for the paper-setting overlays.

Regenerate all comparison figures without rerunning solves:

```bash
python benchmarks/linear_mpc_gradients/plot_precision_grid.py \
  --results-dir RUN_ROOT/linear_mpc_gradients/sensitivity_accuracy/precision_grid --no-tex
```

The plotters default to a `figures/` subdirectory of their input experiment
(`figures/histograms/` for the standalone histogram plotter). An explicit
`--output-dir` remains available for later, deliberate paper export.

