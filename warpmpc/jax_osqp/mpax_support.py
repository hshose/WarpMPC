"""Load the audited MPAX implementation without modifying its installation.

The local resources/MPAX checkout is preferred over an installed package.
An unpatched compatible package is copied to a content-addressed temporary
cache and patched there. Concurrent benchmark processes share an atomic,
locked cache. Unsupported source revisions fail instead of silently running
without the QP correctness fixes.
"""
from __future__ import annotations

import fcntl
from functools import lru_cache
import hashlib
import importlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PATCH = Path(__file__).with_name("mpax_qp_correctness.patch")


def patched_package(source: Path, *, cache_root: Path | None = None) -> Path:
    """Return a package directory with the committed patch applied."""
    source = source.resolve()
    command = ["git", "apply", "--reverse", "--check", str(PATCH)]
    if subprocess.run(command, cwd=source.parent, capture_output=True).returncode == 0:
        return source

    digest = hashlib.sha256(PATCH.read_bytes())
    for path in sorted(source.rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    cache_root = cache_root or Path(tempfile.gettempdir()) / "warpmpc-mpax"
    cache_root.mkdir(parents=True, exist_ok=True)
    key = digest.hexdigest()
    destination = cache_root / key
    with (cache_root / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            temporary = Path(tempfile.mkdtemp(prefix=f"{key}.", dir=cache_root))
            try:
                shutil.copytree(source, temporary / "mpax",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                result = subprocess.run(["git", "apply", str(PATCH)],
                    cwd=temporary, capture_output=True, text=True)
                if result.returncode:
                    raise ImportError(
                        "MPAX source is incompatible with the committed correctness patch. "
                        "Use MPAX revision ca1c669fea422c2509fae4bc30e1d79e6ca8977c.\n"
                        + result.stderr
                    )
                temporary.rename(destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
    return destination / "mpax"


@lru_cache(maxsize=1)
def load_mpax():
    """Return create_qp and raPDHG with the same fixes for every MPC backend."""
    loaded = sys.modules.get("mpax")
    local = Path(__file__).resolve().parents[2] / "resources/MPAX/mpax"
    if loaded is not None:
        source = Path(loaded.__file__).parent
    elif local.is_dir():
        source = local
    else:
        spec = importlib.util.find_spec("mpax")
        if spec is None or spec.origin is None:
            raise ImportError("MPAX requires resources/MPAX or an installed MPAX package.")
        source = Path(spec.origin).parent
    package = patched_package(source)
    if loaded is not None and package.resolve() != source.resolve():
        raise ImportError("Unpatched MPAX was imported first; restart and use load_mpax() before importing MPAX.")
    sys.path.insert(0, str(package.parent))
    module = importlib.import_module("mpax")
    return module.create_qp, module.raPDHG
