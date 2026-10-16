# GMR AMASS motion retargeting

Retarget AMASS motion to the Unitree G1 with WarpMPC, using GMR's task mapping,
two IK stages, SE(3) errors, joint limits, and floating-base integration. The
reference uses the unmodified GMR retargeter with DAQP. GPU trajectories advance
independently through frames and SQP stages while sharing batched QP execution.

## Dependencies and motion data

Install WarpMPC and the optional example dependencies:

```bash
python -m pip install -e .
python -m pip install -r examples/gmr_retargeting/requirements.txt
```

Provide a GMR checkout in `resources/GMR`, or pass `--gmr-root` to the relevant
commands. AMASS motion archives and the corresponding licensed SMPL-X or SMPL-H
body models must be obtained separately. They are not bundled with this example.
Body models may be NPZ or trusted local pickle files in the layouts accepted by
`prepare.py`. Use `--help` on each command for its complete options.

```bash
python examples/gmr_retargeting/prepare.py \
  --amass /path/to/ACCAD.tar.bz2 --body-models /path/to/body_models \
  --clips 2 --frames-per-clip 2048 --output results/gmr/motion.npz
```

Preparation streams archives and computes skeletal targets without rendering or
skinning vertices. `inventory.py` audits available clips; `prepare_pool.py`
creates a diverse pool of nonoverlapping chunks. Chunk boundaries reset the
robot pose. Optional `check_preprocessing.py` compares targets to SMPL-X and GMR
and requires `smplx` and CPU `torch`.

## Accuracy and throughput

```bash
python examples/gmr_retargeting/validate.py \
  --data results/gmr/motion.npz --frames 4096 --chunk-size 2048 \
  --initial-sqp-iterations 11 --sqp-iterations 11 --admm-iterations 50 \
  --adaptive-stopping --scheduler independent --rho 0.1 \
  --output-dir results/gmr/reference
python examples/gmr_retargeting/benchmark.py \
  --reference-dir results/gmr/reference --chunk-size 2048 \
  --initial-sqp-iterations 11 --sqp-iterations 11 --admm-iterations 50 \
  --adaptive-stopping --scheduler independent --rho 0.1 \
  --output-dir results/gmr/throughput
python examples/gmr_retargeting/plot.py \
  --reference-dir results/gmr/reference --results-dir results/gmr/throughput \
  --output results/gmr/throughput.pdf --no-tex
```

SQP limits are per IK stage. Check pose errors on your motion data: the fixed
50-ADMM budget can yield large differences on difficult frames. Throughput
excludes compilation, warmup, file loading and transfers. The original GMR CPU
baseline must be measured on the same host. Oversized batches repeat the input
chunk pool; inspect replication and lane utilization in the output metadata.
`compare_schedulers.py` checks independent versus synchronous scheduling;
`check_batch.py` checks full-batch poses separately from timing.

## H100

From the repository root, create the log directory and submit:

```bash
mkdir -p hpclogs
sbatch \
  --account=<your-slurm-account> --partition=<your-gpu-partition> \
  examples/gmr_retargeting/run_h100.sh \
  "$PWD" "$PWD/results/gmr/motion.npz" /path/to/venv/bin/python
```

The positional arguments are repository root, prepared data, and Python; they
survive sites that strip exported environment variables.
Set `VENV=/path/to/venv` or `PYTHON=/path/to/python` if needed. `ROOT` overrides
repository discovery. `RESULTS_DIR`, `FRAMES`, `CHUNK_SIZE`, `BATCH_SIZES`,
`SQP_ITERATIONS`, `ADMM_ITERATIONS`, and `RHO` override the study settings.
`SYNTHETIC=1` explicitly selects synthetic fixtures instead of AMASS data.
`PLOT=1` enables plotting on the compute node; plotting separately is also
supported. Run with `bash` inside an existing H100 allocation for native use.

## Checks

```bash
JAX_PLATFORMS=cpu python -m pytest tests/test_gmr_retargeting_*.py
```

Model parity checks require GMR, MuJoCo, Mink and DAQP. No licensed motion assets
are required for the synthetic model checks.
