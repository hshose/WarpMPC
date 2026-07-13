"""Benchmark scripts and benchmark-only problem fixtures."""

from .jax_cache import configure_jax_compilation_cache

configure_jax_compilation_cache()

__all__ = ["configure_jax_compilation_cache"]
