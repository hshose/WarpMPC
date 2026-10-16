"""Case labels for the external VRAM sampler; never query/synchronize the GPU.

Only enter at Python benchmark boundaries, outside compiled/timed solver calls.
Disabled unless VRAM_CASE_DIR is supplied by monitor_vram.py.
"""
from contextlib import nullcontext
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import sys
import time
import uuid

FIELDS = 'batch_size batch dtype variant mode solver qp_solver system qdldl_variant qdldl_factor_backend qdldl_solve_backend level_scheduled_solve segmented transpose_work repeat_index repeat horizon horizon_steps horizon_nodes sqp_iterations max_iter iteration_limit forward refinement name provider target_instructions target size'.split()

def labels(namespace):
    args = namespace.get('args')
    source = dict(vars(args)) if hasattr(args, '__dict__') else {}
    source.update({k: namespace[k] for k in FIELDS if k in namespace})
    result = {'benchmark': Path(sys.argv[0]).name}
    for k in FIELDS:
        if k in source and source[k] is not None:
            v = source[k]
            if k == 'dtype': v = str(v)
            if isinstance(v, (str, int, float, bool)): result[k] = v
    if 'batch' in result: result['batch_size'] = result.pop('batch')
    return result

class _Case:
    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.data = dict(case_id=uuid.uuid4().hex, pid=os.getpid(), labels=metadata)
        self.active = self.directory / f'{os.getpid()}.json'
    def event(self, event):
        with (self.directory / f'{os.getpid()}.jsonl').open('a') as f:
            f.write(json.dumps(dict(self.data, event=event, time_ns=time.time_ns()))+'\n')
    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.previous = self.active.read_bytes() if self.active.exists() else None
        self.event('start')
        tmp = self.active.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data)); tmp.replace(self.active)
        return self
    def __exit__(self, kind, value, traceback):
        self.event('error' if kind else 'end')
        if self.previous is None:
            self.active.unlink(missing_ok=True)
        else:
            tmp=self.active.with_suffix('.tmp');tmp.write_bytes(self.previous);tmp.replace(self.active)
        return False

def vram_case(batch_size=None, metadata=None):
    directory = os.environ.get('VRAM_CASE_DIR')
    if not directory: return nullcontext()
    # Extract only scalar labels now. Do NOT retain locals()/GPU array references.
    info = labels(metadata or {})
    if batch_size is not None: info['batch_size'] = int(batch_size)
    return _Case(directory, info)

def record_vram_case(fn):
    signature = inspect.signature(fn)
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not os.environ.get('VRAM_CASE_DIR'): return fn(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        with vram_case(metadata=bound.arguments):
            return fn(*args, **kwargs)
    return wrapped
