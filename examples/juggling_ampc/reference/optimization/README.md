# Horizon and constraint optimization

The recommended experimental controller is `specs/mesh24_prune.json`: 24
geometrically spaced intervals over 1.2 s, full constraints for the first five
intervals, and conservative tail pruning. It retains one SQP / 50 OSQP,
20 ms nominal updates, 0.5 ms Newton steps, and the position-only ball observer.
Use four evaluator workers on the target Ryzen 9950X workstation. OSQP remains
single-threaded. The original 40-interval controller remains available in `cpp/`.

See [the integrated results](../docs/horizon_optimization.md),
[constraint activity](../docs/constraint_activity.md),
[mesh trials](mesh/README.md), and [pruning trials](pruning/README.md).

Build the original optimized factors and source-built OSQP first, following
[the native build guide](../cpp/README.md). Then, from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m optimization.native_export \
  optimization/specs/mesh24_prune.json build/variants/mesh24_prune/factors
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m optimization.build_native \
  build/variants/mesh24_prune/factors build/variants/mesh24_prune/controller --jobs 32
JUGGLING_EVAL_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m optimization.check_native \
  build/variants/mesh24_prune/factors \
  build/variants/mesh24_prune/controller/libjuggling_controller.so \
  --output build/variants/mesh24_prune/check.json
JUGGLING_EVAL_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m optimization.run_native \
  --factors build/variants/mesh24_prune/factors \
  --library build/variants/mesh24_prune/controller/libjuggling_controller.so \
  --name native_mesh24_prune_wind --seconds 20
```

For nominal or damping tests append `--wind none` or `--damping .02 --vicon-seed 19`.
Newton stops its simulated clock during computation. The thin Python adapter
selects the variant while sharing the existing plant, observer, reporting and
audit machinery. Native exported model parameters are fixed; plant mismatch
arguments do not silently alter the compiled MPC.

The build requires modern Clang (`CC` and `CXX` can select explicit executables).
Build scripts default to 32 concurrent compile jobs; `--jobs` or
`CPP_BUILD_JOBS` can override this. This build parallelism is independent of
the four MPC evaluation workers used at runtime.
Newly compiled generated factors and C++ use `-O3 -DNDEBUG -march=native`. Byte-identical
baseline factor sources reuse the baseline library; changed functions compile
individually and are cached by source, compiler and CPU identity. Cold builds
can take several minutes. Rebuild on the target CPU instead of copying native
machine code between dissimilar processors.

Native timing and asynchronous mock commands:

```bash
JUGGLING_EVAL_THREADS=4 OPENBLAS_NUM_THREADS=1 taskset -c 8-13 \
  build/variants/mesh24_prune/controller/juggling_benchmark \
  build/variants/mesh24_prune/factors build/benchmark_inputs/wind_replay.csv \
  build/variants/mesh24_prune/steps.csv 3
JUGGLING_EVAL_THREADS=4 OPENBLAS_NUM_THREADS=1 taskset -c 8-13 \
  build/variants/mesh24_prune/controller/juggling_hil_mock \
  build/variants/mesh24_prune/factors build/variants/mesh24_prune/mock 20
```

CPU affinity above is the tested workstation placement, not a portable CPU
selection. The mock uses the shared asynchronous C++ runtime and a simulated
arm; it does not command hardware. Its reference interpolation follows the
nonuniform timestamps. The optional `JUGGLING_BUILD_FRANKA=ON` target now adds
the physical-arm adapter and read-only probe to this optimized build. See
[hardware/homing instructions](../cpp/README.md#optimized-hardware-build-and-homing)
for configured-load preservation, the explicit non-realtime option, and the
slow move to HOME before starting MPC.

The hardware-specific `specs/mesh24_prune_cad.json` keeps these settings and
substitutes the user-provided 0.48 kg assembly, full COM and inertia at the
flange. It replaces both the original fixture and paddle inertias; the face
geometry stays unchanged. Export/build it with the same commands into
`build/variants/mesh24_prune_cad/`. `cpp.check_assembly` compares its dynamics
against Pinocchio and checks that the total mass changes by exactly 0.23 kg.
The hardware-specific variant is tested with the asynchronous mock; historical
Newton results above still refer to the original 0.25 kg assembly. Newton's
plant is not silently changed by this export option.

`mesh/` and `pruning/` contain the independent Python experiments, including
the rejected aggressive endpoint omission and cost-regularization trials.
`check_native*.json` records actual sparse-data, cost, warmstart and fixed-budget
solution comparisons. `native_prediction_audit.py` evaluates omitted rows
against the original full formulation on saved predicted plans.

The 2026-10-02 physical-drop update is `specs/mesh24_identified_servo.json`:
restitution 0.54, provisional linear drag 0.34/s, stronger energy/contact costs,
and 10% speed/acceleration reserves. Its Newton test passes, but the offline
hardware velocity-servo mock still aborts on reference acceleration. It is an
experimental default, **not ready for another physical ball release**. See the
[identification and validation report](../runs/20261002_model_update/report.txt).
