"""Cache configuration and real reuse across independent Python processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CACHE_ENV = (
    "WARPMPC_CACHE_DIR", "JAX_COMPILATION_CACHE_DIR", "WARP_CACHE_PATH",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "JAX_ENABLE_COMPILATION_CACHE",
)


class CompilationCacheTests(unittest.TestCase):
    def run_python(self, source, directory, **overrides):
        env = {k: v for k, v in os.environ.items() if k not in CACHE_ENV}
        env.update(JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1",
                   XDG_CACHE_HOME=str(Path(directory) / "xdg"), **overrides)
        result = subprocess.run([sys.executable, "-c", source], cwd=ROOT,
                                env=env, text=True, capture_output=True, check=True)
        return json.loads(result.stdout.splitlines()[-1])

    def test_stable_user_default_and_small_compilations(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_python('''
import json, os, jax
import benchmarks.jax_cache
print(json.dumps(dict(path=jax.config.jax_compilation_cache_dir,
    warp=os.environ['WARP_CACHE_PATH'],
    minimum_time=jax.config.jax_persistent_cache_min_compile_time_secs,
    minimum_size=jax.config.jax_persistent_cache_min_entry_size_bytes)))
''', directory)
            root = Path(directory) / "xdg/warpmpc"
            self.assertEqual(result, dict(path=str(root / "jax"), warp=str(root / "warp"),
                                          minimum_time=0, minimum_size=-1))

    def test_all_examples_preserve_environment_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            selected = str(Path(directory) / "explicit-jax")
            warp = str(Path(directory) / "explicit-warp")
            result = self.run_python('''
import importlib, json, os, jax
modules = (
    'benchmarks.linear_mpc.benchmark_jax_osqp_linear_mpc',
    'benchmarks.nonlinear_mpc.evaluate_cartpole_quadratic_mpc',
    'benchmarks.nonlinear_mpc.evaluate_crazyflie_sqp',
    'benchmarks.nonlinear_mpc.evaluate_humanoid_mpc',
    'examples.crazyflie_ampc.train_ampc',
    'examples.crazyflie_ampc.speedrun',
)
for module in modules:
    importlib.import_module(module)
    assert jax.config.jax_compilation_cache_dir == os.environ['JAX_COMPILATION_CACHE_DIR']
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 2.5
    assert jax.config.jax_persistent_cache_min_entry_size_bytes == 2048
    assert jax.config.jax_persistent_cache_enable_xla_caches == 'none'
    assert not jax.config.jax_enable_compilation_cache
print(json.dumps(dict(modules=len(modules), warp=os.environ['WARP_CACHE_PATH'])))
''', directory, JAX_COMPILATION_CACHE_DIR=selected, WARP_CACHE_PATH=warp,
                JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="2.5",
                JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="2048",
                JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="none",
                JAX_ENABLE_COMPILATION_CACHE="false")
            self.assertEqual(result, dict(modules=6, warp=warp))

    def test_common_root_and_late_explicit_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "common"
            result = self.run_python('''
import json, os, jax
from pathlib import Path
from benchmarks.jax_cache import configure_jax_compilation_cache
root = Path(os.environ['WARPMPC_CACHE_DIR'])
assert jax.config.jax_compilation_cache_dir == str(root / 'jax')
assert os.environ['WARP_CACHE_PATH'] == str(root / 'warp')
os.environ['JAX_COMPILATION_CACHE_DIR'] = str(root / 'late')
configure_jax_compilation_cache()
assert jax.config.jax_compilation_cache_dir == str(root / 'late')
configure_jax_compilation_cache(root / 'explicit/../canonical')
assert jax.config.jax_compilation_cache_dir == str(root / 'canonical')
from examples.crazyflie_ampc import speedrun as run
run.configure_cache(root / 'speedrun')
from examples.crazyflie_ampc import train_ampc
assert jax.config.jax_compilation_cache_dir == str(root / 'speedrun/jax')
assert os.environ['WARP_CACHE_PATH'] == str(root / 'speedrun/warp')
assert jax.config.jax_persistent_cache_min_compile_time_secs == 0
print(json.dumps('passed'))
''', directory, WARPMPC_CACHE_DIR=str(root))
            self.assertEqual(result, "passed")

    def test_short_executable_hits_cache_in_fresh_process(self):
        source = '''
import json, jax, jax.numpy as jnp, numpy as np
import benchmarks.jax_cache
events = []
jax.monitoring.register_event_listener(lambda event, **kw: events.append(event))
@jax.jit
def short_kernel(x):
    return jnp.sin(x) * 2
compiled = short_kernel.lower(jax.ShapeDtypeStruct((32,), np.float32)).compile()
result = np.asarray(compiled(np.linspace(-1, 1, 32, dtype=np.float32)))
print(json.dumps(dict(hits=events.count('/jax/compilation_cache/cache_hits'),
                      values=result.tolist())))
'''
        with tempfile.TemporaryDirectory() as directory:
            cold = self.run_python(source, directory)
            warm = self.run_python(source, directory)
            self.assertEqual(cold["hits"], 0)
            self.assertGreater(warm["hits"], 0)
            self.assertEqual(cold["values"], warm["values"])


if __name__ == "__main__":
    unittest.main()
