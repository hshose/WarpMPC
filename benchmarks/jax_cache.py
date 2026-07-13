"""Persistent JAX compilation cache configuration for benchmark scripts."""

from __future__ import annotations

import jax


JAX_COMPILATION_CACHE_DIR = "/tmp/jax_cache"
JAX_XLA_CACHE_KIND = "xla_gpu_per_fusion_autotune_cache_dir"


def configure_jax_compilation_cache() -> None:
    """Enable persistent compilation caches used by the GPU benchmarks."""

    jax.config.update("jax_compilation_cache_dir", JAX_COMPILATION_CACHE_DIR)
    jax.config.update("jax_persistent_cache_enable_xla_caches", JAX_XLA_CACHE_KIND)


configure_jax_compilation_cache()
