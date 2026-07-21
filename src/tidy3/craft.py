"""CRAFT remote seeding — ship the local tidy3 source to the remote kernel.

The gpudev/CRAFT remote kernel has its own namespace *and* filesystem;
nothing on the GPU box has tidy3 unless we put it there. This module tars
the local package (~10 KB gzipped), embeds it in a bootstrap snippet, and
runs that through CRAFT's ``remote_run_``. The bootstrap extracts to
``~/.tidy3-src`` (persistent container home), ensures polars, loads the
Jupyter extension (API names + ``>>`` transformer + magics), and restyles
bare polars frames. A content stamp makes re-seeding a no-op and re-pushes
automatically after local edits.
"""

from __future__ import annotations

import base64
import hashlib
import io
import tarfile
import time
from pathlib import Path
from typing import Callable

__all__ = ["build_payload", "bootstrap_code", "seed"]

_OK_PREFIX = "tidy3 remote: OK"


def _pkg_files() -> list[tuple[str, bytes]]:
    pkg = Path(__file__).resolve().parent
    return sorted((f"tidy3/{p.name}", p.read_bytes()) for p in pkg.glob("*.py"))


def build_payload() -> tuple[str, str]:
    """Return (base64 tar.gz of the package, content stamp)."""
    from tidy3 import __version__

    files = _pkg_files()
    h = hashlib.sha256()
    for name, data in files:
        h.update(name.encode())
        h.update(b"\0")
        h.update(data)
        h.update(b"\0")
    stamp = f"{__version__}-{h.hexdigest()[:16]}"

    buf = io.BytesIO()
    now = int(time.time())
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            # fresh mtime, NOT 0: a same-size edit (e.g. version bump) with a
            # constant mtime would validate stale remote __pycache__ bytecode
            ti.mtime = now
            tar.addfile(ti, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode("ascii"), stamp


# Runs on the remote kernel. No f-strings/%-signs (filled via %-format);
# printed lines surface in the user's first %gpu cell, so keep them terse.
_BOOTSTRAP = r'''
import base64 as _b64, io as _io, sys as _sys, tarfile as _tarfile
from pathlib import Path as _Path
_root = _Path.home() / ".tidy3-src"
_stampf = _root / ".stamp"
_stamp = "%(stamp)s"
try:
    _fresh = _stampf.read_text().strip() == _stamp
except Exception:
    _fresh = False
if not _fresh:
    import shutil as _shutil
    _root.mkdir(parents=True, exist_ok=True)
    _shutil.rmtree(_root / "tidy3", ignore_errors=True)
    _buf = _io.BytesIO(_b64.b64decode("%(payload)s"))
    with _tarfile.open(fileobj=_buf, mode="r:gz") as _tar:
        try:
            _tar.extractall(_root, filter="data")
        except TypeError:
            _tar.extractall(_root)
    _stampf.write_text(_stamp)
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
try:
    import polars as _pl
except Exception:
    import subprocess as _sp
    print("tidy3 remote: installing polars (first time only)...", flush=True)
    _r = _sp.run(["uv", "pip", "install", "polars"], capture_output=True, text=True)
    if _r.returncode != 0:
        _r = _sp.run([_sys.executable, "-m", "pip", "install", "polars"],
                     capture_output=True, text=True)
    if _r.returncode != 0:
        print((_r.stdout or "")[-600:])
        print((_r.stderr or "")[-600:])
        raise RuntimeError("tidy3 seed: polars install failed")
    import importlib as _il
    _il.invalidate_caches()
    import polars as _pl
if not _fresh and "tidy3" in _sys.modules:
    for _k in [_m for _m in list(_sys.modules)
               if _m == "tidy3" or _m.startswith("tidy3.")]:
        del _sys.modules[_k]
import tidy3 as _t3
from IPython import get_ipython as _gi
_ip = _gi()
if _ip is not None:
    _em = _ip.extension_manager
    _loaded = getattr(_em, "loaded", set())
    if "tidy3.jupyter" in _loaded and "tidy3.jupyter" not in _sys.modules:
        _loaded.discard("tidy3.jupyter")
    if "tidy3.jupyter" in _loaded:
        if not _fresh:
            _em.reload_extension("tidy3.jupyter")
    else:
        _em.load_extension("tidy3.jupyter")
    # Always force-inject API: remote kernels often have datar/pipda ``mean``
    # already in user_ns; without force, inject_api would leave datar's mean.
    try:
        from tidy3.jupyter import ensure_ipython_integration as _ensure
        from tidy3.jupyter import inject_api as _inject
        from tidy3.jupyter import enable_r_style as _rstyle

        _ensure(quiet=True)
        _inject(_ip, force=True)
        _rstyle(_ip)
    except Exception as _e:
        print("tidy3 remote: inject_api warning: " + repr(_e), flush=True)
    if %(style)s:
        from tidy3.display import register_polars_formatter as _rpf
        _rpf(_ip)
    # Hint when competing grammars are present
    _clash = [m for m in ("datar", "pipda", "datar_numpy") if m in _sys.modules]
    if _clash:
        print("tidy3 remote: note — also loaded: " + ", ".join(_clash)
              + " (tidy3 names forced into user_ns)", flush=True)
print("tidy3 remote: OK v" + _t3.__version__ + " (" + _stamp + ")")
'''


def bootstrap_code(payload: str, stamp: str, *, style_polars: bool = True) -> str:
    return _BOOTSTRAP % {
        "payload": payload,
        "stamp": stamp,
        "style": "True" if style_polars else "False",
    }


def seed(
    remote_run: Callable[..., str],
    *,
    payload: str | None = None,
    stamp: str | None = None,
    style_polars: bool = True,
    max_chars: int = 8000,
) -> tuple[bool, str]:
    """Run the bootstrap via CRAFT's ``remote_run_``. Returns (ok, message)."""
    if payload is None or stamp is None:
        payload, stamp = build_payload()
    code = bootstrap_code(payload, stamp, style_polars=style_polars)
    try:
        out = remote_run(code, max_chars=max_chars) or ""
    except Exception as e:
        return False, f"remote bootstrap did not run: {e}"
    out = out.strip()
    for line in out.splitlines():
        if line.startswith(_OK_PREFIX):
            return True, line.strip()
    return False, (out[-1500:] if out else "no output from remote bootstrap")
