# Sampled VRAM for the H100 benchmark suite

The full H100 launcher enables `MONITOR_VRAM=1` by default. It runs each existing
Slurm script through a separate NVML sampler, preserving its resource directives,
solver settings, allocator behavior and timed code. Install `nvidia-ml-py` in
the benchmark environment. `VRAM_SAMPLE_INTERVAL=1` sets the default one-second
polling interval; `MONITOR_VRAM=0` disables the sampler and boundary recording.
The sampler lowers its own CPU priority after spawning the original benchmark.
No CUDA synchronizations, JAX memory queries or allocator resets are added.
Case-label file writes occur outside timed sections. Monitoring has small,
nonzero overhead; zero runtime impact is not guaranteed.

Artifacts are stored at:

```
<run>/vram/<study>/<benchmark>/<job_id>/
  metadata.json       # interval, command, measurement definition, exit/errors
  samples.csv         # timestamps, GPU UUID, case ID, resident bytes, GPU PIDs
  cases/*.jsonl      # batch/dtype/variant/solver boundary labels and timestamps
  case_peaks.csv     # sampled peak bytes/MiB and sample count per case/GPU
```

`case_peaks.csv` includes `batch_size`, dtype and variant/solver when relevant.
For nonlinear MPC the sampler follows GPU child PIDs to the labeled parent case.
Sensitivity grid rows distinguish forward and adjoint phases and iteration counts.
Synthetic rows also label provider and instruction count. Multiple benchmark
processes on the same GPU in a case are summed; different GPUs get separate rows.
Unrelated jobs' processes are excluded. The raw `unlabeled` rows describe setup
or transitions, and are not silently assigned to a batch size.

Peaks are sampled lower bounds on the true transient peak. A case with no
samples has an empty peak, **not zero**. Memory includes CUDA context overhead,
JAX reserved pools, cached allocations and previous-case retained buffers.
Keeping preallocation unchanged can produce flat VRAM curves. These are
resident-memory measurements, not live-array bytes or minimum required VRAM.

Join case labels with the benchmark's result CSV before plotting: a boundary
`end` only means Python left the scope normally; a benchmark may catch OOMs or
skip oversized batches inside that scope. `error` means an exception escaped.
A missing end event indicates interruption. Failed/OOM cases must be distinguished
from successful cases. Host-memory OOMs are not measurable through GPU NVML.
For killed jobs with only raw artifacts, regenerate the peak table with:

```bash
python benchmarks/monitor_vram.py --output-dir <job_vram_directory> --summarize-only
```
