"""Persistent JAX compilation cache configuration for benchmark scripts."""

from __future__ import annotations

import os
from pathlib import Path

import jax


_CACHE_ROOT = Path(os.environ.get(
    "WARPMPC_CACHE_DIR", str(Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")) / "warpmpc")
)).expanduser().resolve()
JAX_COMPILATION_CACHE_DIR = str(_CACHE_ROOT / "jax")
JAX_XLA_CACHE_KIND = "xla_gpu_per_fusion_autotune_cache_dir"


def configure_jax_compilation_cache(cache_dir: str | os.PathLike | None = None) -> None:
    """Use stable user caches and persist short compilations as well.

    An explicit path takes precedence over JAX_COMPILATION_CACHE_DIR. Otherwise
    WARPMPC_CACHE_DIR selects the common JAX/Warp root (default ~/.cache/warpmpc,
    respecting XDG_CACHE_HOME). Configure before initializing either backend.
    Native JAX threshold/cache-kind overrides and WARP_CACHE_PATH are respected.
    """

    selected = cache_dir if cache_dir is not None else os.environ.get(
        "JAX_COMPILATION_CACHE_DIR", JAX_COMPILATION_CACHE_DIR)
    jax.config.update("jax_compilation_cache_dir", str(Path(selected).expanduser().resolve()))
    jax.config.update("jax_persistent_cache_enable_xla_caches", os.environ.get(
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", JAX_XLA_CACHE_KIND))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", float(os.environ.get(
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", int(os.environ.get(
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")))
    # Warp reads this lazily at initialization; JAX-only examples need no Warp import.
    os.environ.setdefault("WARP_CACHE_PATH", str(_CACHE_ROOT / "warp"))


configure_jax_compilation_cache()
