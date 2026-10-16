"""Use the checked-in reference copy, never implicit imports from a live checkout."""
import sys
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC", "0")
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REFERENCE=HERE/'reference'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(REFERENCE))
