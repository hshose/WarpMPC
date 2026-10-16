"""Bounded full-world collector memory diagnosis; never trains or starts hardware.

Run inside a private source copy on the same GPU as the failed collection.
Records the allocator/physical memory at each stage and the compiled MPC's
buffer assignment, then executes only the requested short simulation duration.
"""
import bootstrap
import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import jax
import warp as wp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=60000)
    parser.add_argument('--slots', type=int, default=285)
    parser.add_argument('--seconds', type=float, default=.02)
    parser.add_argument('--release-warp-pool', action='store_true')
    parser.add_argument('--stop-before-mpc', action='store_true')
    parser.add_argument('--original-dispatch', action='store_true',
                        help='Keep the original asynchronous JIT dispatch order, with no stage barriers.')
    parser.add_argument('--jit-with-barriers', action='store_true',
                        help='Use the ordinary JIT call with stage barriers, without explicit AOT compilation.')
    parser.add_argument('--aot-without-barriers', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wp.config.track_memory = True
    wp.init()
    start = time.monotonic()
    log = (args.output/'memory.jsonl').open('w', buffering=1)
    driver = ctypes.CDLL('libcuda.so.1')
    pool = ctypes.c_void_p()
    pool_status = driver.cuDeviceGetDefaultMemPool(ctypes.byref(pool), 0)

    def pool_reserved():
        value = ctypes.c_uint64()
        # CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 5 (CUDA driver API).
        status = driver.cuMemPoolGetAttribute(pool, 5, ctypes.byref(value)) if pool_status == 0 else pool_status
        return value.value if status == 0 else None

    def graph_memory(attribute):
        value = ctypes.c_uint64()
        status = driver.cuDeviceGetGraphMemAttribute(0, attribute, ctypes.byref(value))
        return value.value if status == 0 else None

    def snapshot(stage, **extra):
        if not (args.original_dispatch or args.aot_without_barriers):
            wp.synchronize()
        device = wp.get_device('cuda:0')
        row = dict(stage=stage, wall_s=time.monotonic()-start,
                   cuda_total=device.total_memory, cuda_free=device.free_memory,
                   warp_live=wp.get_mempool_used_mem_current(device),
                   cuda_default_pool_reserved=pool_reserved(),
                   cuda_graph_live=graph_memory(0),
                   cuda_graph_reserved=graph_memory(2),
                   warp_peak=wp.get_mempool_used_mem_high(device),
                   warp_release_threshold=wp.get_mempool_release_threshold(device),
                   jax=jax.devices()[0].memory_stats(), **extra)
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row)+'\n')

    import fast_rollout as rollout
    from gpu_dataset import GPUDataset
    snapshot('startup', environment={k: os.environ.get(k) for k in (
        'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_CLIENT_MEM_FRACTION', 'XLA_PYTHON_CLIENT_MEM_FRACTION',
        'XLA_PYTHON_CLIENT_ALLOCATOR', 'TF_GPU_ALLOCATOR')})
    store = GPUDataset(args.batch, args.slots)
    snapshot('dataset_allocated')

    def wrap_method(cls, name, label):
        original = getattr(cls, name)
        def wrapped(self, *pos, **kw):
            result = original(self, *pos, **kw)
            if not (args.original_dispatch or args.aot_without_barriers):
                jax.block_until_ready(result)
            snapshot(label)
            return result
        setattr(cls, name, wrapped)

    original_ik = rollout.FixtureIK
    class DiagnosticFixtureIK(original_ik):
        def __init__(self, *pos, **kw):
            super().__init__(*pos, **kw)
            original = self.solve
            def solve(*inputs):
                result = original(*inputs)
                if not (args.original_dispatch or args.aot_without_barriers):
                    jax.block_until_ready(result)
                snapshot('initial_ik_complete')
                return result
            self.solve = solve
    rollout.FixtureIK = DiagnosticFixtureIK
    wrap_method(rollout.EmpiricalNewton, '__init__', 'plant_initialized')
    wrap_method(rollout.EmpiricalNewton, 'enable_graph', 'plant_graphs_created')
    original_plant_step = rollout.EmpiricalNewton.step
    step_count = 0
    def plant_step(self, *inputs):
        nonlocal step_count
        step_count += 1
        if step_count <= 2:
            snapshot('before_plant_step', physics_tick=step_count)
        result = original_plant_step(self, *inputs)
        if step_count <= 2:
            snapshot('after_plant_step', physics_tick=step_count)
        return result
    rollout.EmpiricalNewton.step = plant_step
    wrap_method(rollout.GPUProjectors, '__call__', 'projectors_complete')
    original_warm = rollout.WarmStart

    class DiagnosticWarmStart(original_warm):
        def __init__(self, *pos, **kw):
            super().__init__(*pos, **kw)
            original = self.initialize
            def initialize(*inputs):
                result = original(*inputs)
                if not (args.original_dispatch or args.aot_without_barriers):
                    jax.block_until_ready(result)
                snapshot('warm_initialize_complete')
                return result
            self.initialize = initialize
    rollout.WarmStart = DiagnosticWarmStart
    original_mpc = rollout.MPC

    class DiagnosticMPC(original_mpc):
        def __init__(self, *pos, **kw):
            super().__init__(*pos, **kw)
            jit_solve = self.solve
            compiled = None
            def solve(*inputs):
                nonlocal compiled
                if not (args.original_dispatch or args.aot_without_barriers):
                    jax.block_until_ready(inputs)
                snapshot('before_mpc_compile')
                if args.stop_before_mpc:
                    with (args.output/'warp_allocations.txt').open('w') as stream:
                        wp.print_memory_report(file=stream, max_items=30)
                    raise SystemExit(0)
                if args.original_dispatch or args.jit_with_barriers:
                    result = jit_solve(*inputs)
                    if args.jit_with_barriers:
                        jax.block_until_ready(result)
                    snapshot('mpc_dispatched')
                    return result
                if compiled is None:
                    compiled = jit_solve.lower(*inputs).compile()
                    stat = compiled.memory_analysis()
                    fields = ('argument_size_in_bytes', 'output_size_in_bytes',
                              'temp_size_in_bytes', 'alias_size_in_bytes')
                    memory = {key: int(getattr(stat,key)) for key in fields}
                    (args.output/'mpc_memory.json').write_text(json.dumps(memory, indent=2)+'\n')
                    snapshot('mpc_compiled', compiled_memory=memory)
                if args.release_warp_pool:
                    gc.collect()
                    wp.set_mempool_release_threshold('cuda:0', 0)
                    wp.synchronize()
                    snapshot('warp_pool_released')
                try:
                    result = compiled(*inputs)
                    if not args.aot_without_barriers:
                        jax.block_until_ready(result)
                except Exception as error:
                    snapshot('mpc_failed', error=repr(error))
                    raise
                snapshot('mpc_dispatched' if args.aot_without_barriers else 'mpc_completed')
                with (args.output/'warp_allocations.txt').open('w') as stream:
                    wp.print_memory_report(file=stream, max_items=30)
                return result
            self.solve = solve
    rollout.MPC = DiagnosticMPC
    # JIT compilation and dispatch happen in rollout. Snapshot immediately
    # after its output becomes ready by wrapping the sink insertion instead.
    original_append = store.append
    def append(*inputs):
        if not (args.original_dispatch or args.aot_without_barriers):
            jax.block_until_ready(inputs)
        snapshot('action_dispatched' if (args.original_dispatch or args.aot_without_barriers) else 'action_ready')
        return original_append(*inputs)
    store.append = append
    root=args.root
    params = SimpleNamespace(factors=root/'results/juggling_reference/gpu_factors',
        source=root/'live_source', calibration=root/'live_source/runs/20261006_sim_quality/model.json',
        output=args.output/'collection', batch=args.batch, seconds=args.seconds, seed=2026,
        multiplier=1., arm_multiplier=1.5, policy=None)
    rollout.run(params, sink=store)
    snapshot('collection_complete', dataset=store.summary())


if __name__ == '__main__':
    main()
