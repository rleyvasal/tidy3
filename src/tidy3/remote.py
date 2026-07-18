"""CRAFT remote execution — large data stays on the GPU host.

Under SolveIt you often cannot ship multi‑GB files to the notebook VM.
With CRAFT connected (``%gpu`` at least once so ``remote_run_`` exists),
these helpers run tidy3 **on the remote kernel**, keep frames there, and
only pull a small Polars preview back for display.

Typical flow
------------
::

    %local
    %run .../CRAFT.py
    %run .../addons/tidy3.py
    %gpu   # connect remote kernel

    # path is ON THE REMOTE MACHINE
    remote(\"\"\"
    scan_parquet("/home/gpudev/data/huge.parquet")
    >> filter(col("year") >= 2020)
    >> group_by("region")
    >> summarise(n=n(), avg=mean("value"))
    \"\"\")
"""

from __future__ import annotations

import base64
import io
import re
import textwrap
from typing import Any

import polars as pl

from tidy3.partial_run import normalize_pipe_source


_MARKER = "TIDY3_IPC64:"
_LAST_NAME = "_tidy3_last"


def _ipython_ns() -> dict[str, Any] | None:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None and getattr(ip, "user_ns", None) is not None:
            return ip.user_ns
    except Exception:
        pass
    return None


def _remote_run():
    ns = _ipython_ns() or {}
    rr = ns.get("remote_run_")
    if not callable(rr):
        raise RuntimeError(
            "remote_run_ not found — load CRAFT and run %gpu first "
            "(then stay on %local for remote() helpers, or run pipes under %gpu)."
        )
    return rr


def _remote_bootstrap() -> str:
    """Code to ensure tidy3 is importable on the remote kernel."""
    return textwrap.dedent(
        """
        def __tidy3_bootstrap():
            import sys
            from pathlib import Path
            cands = [
                Path.home() / "tidy3" / "src",
                Path("/home/gpudev/tidy3/src"),
                Path("/app/data/gpudevd/tidy3/src"),
                Path("/app/data/tidy3/src"),
            ]
            for p in cands:
                if (p / "tidy3").is_dir() and str(p) not in sys.path:
                    sys.path.insert(0, str(p))
                    break
            import tidy3  # noqa: F401
            return True
        __tidy3_bootstrap()
        """
    ).strip()


def remote(
    source: str,
    *,
    n: int | None = None,
    bind: str | None = None,
    max_chars: int = 8_000_000,
) -> pl.DataFrame:
    """Run a tidy3 pipe **on the CRAFT remote** and return a Polars preview.

    Parameters
    ----------
    source:
        Pipe text (multi-line ``>>`` ok; no outer parens required), e.g.::

            scan_parquet("/data/big.parquet")
            >> filter(col("x") > 0)
            >> summarise(n=n())

        Or continue from the last remote result::

            _tidy3_last
            >> mutate(y=col("x") * 2)

    n:
        Preview rows to pull back (default: tidy3 ``options.preview_rows``).
    bind:
        If set, also store the remote TidyFrame under this name in remote
        ``globals()`` for later pipes.
    """
    from tidy3.options import get_options

    if n is None:
        n = get_options().preview_rows

    # Validate/normalize locally so user gets fast errors
    normalize_pipe_source(source)

    bind_stmt = ""
    if bind:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", bind):
            raise ValueError(f"bind must be a simple identifier, got {bind!r}")
        bind_stmt = f"globals()[{bind!r}] = _tf"

    code = f"""
{_remote_bootstrap()}
from tidy3 import partial_run
from tidy3.partial_run import normalize_pipe_source
import base64, io

_src = {source!r}
_tf = partial_run(_src, namespace=globals())
globals()[{_LAST_NAME!r}] = _tf
{bind_stmt}
_prev = _tf.preview({int(n)})
_buf = io.BytesIO()
_prev.write_ipc(_buf)
print({_MARKER!r} + base64.b64encode(_buf.getvalue()).decode("ascii"))
print(f"tidy3 remote: preview {{_prev.height}} rows × {{_prev.width}} cols"
      f" (full plan left on remote as {_LAST_NAME}"
      + {f' and {bind!r}' if bind else '""'} + ")")
"""
    out = _remote_run()(code, max_chars=max_chars)
    return _parse_preview(out)


def remote_bind(name: str, source: str) -> str:
    """Create/update a named TidyFrame on the remote; return *name*.

    Example::

        remote_bind("trips", 'scan_parquet("/data/trips/*.parquet")')
        remote(\"\"\"
        trips
        >> filter(col("fare") > 10)
        >> count("borough")
        \"\"\")
    """
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"name must be a simple identifier, got {name!r}")
    normalize_pipe_source(source)
    code = f"""
{_remote_bootstrap()}
from tidy3 import partial_run
_tf = partial_run({source!r}, namespace=globals())
globals()[{name!r}] = _tf
globals()[{_LAST_NAME!r}] = _tf
print(f"tidy3 remote: bound {{ {name!r} }} (lazy plan on remote)")
"""
    print(_remote_run()(code, max_chars=4000))
    return name


def remote_collect(
    source: str | None = None,
    *,
    as_: str = "polars",
    max_rows: int = 100_000,
    max_chars: int = 50_000_000,
) -> Any:
    """Collect a remote pipe (or last frame) and bring **up to max_rows** back.

    Prefer :func:`remote` for exploration — full collect can be huge.
    """
    if source is None:
        source = _LAST_NAME
    normalize_pipe_source(source)
    code = f"""
{_remote_bootstrap()}
from tidy3 import partial_run
import base64, io
_tf = partial_run({source!r}, namespace=globals())
_df = _tf.head({int(max_rows)}).collect()
_buf = io.BytesIO()
_df.write_ipc(_buf)
print({_MARKER!r} + base64.b64encode(_buf.getvalue()).decode("ascii"))
print(f"tidy3 remote collect: {{_df.height}} rows (capped at {int(max_rows)})")
"""
    out = _remote_run()(code, max_chars=max_chars)
    df = _parse_preview(out)
    if as_ == "pandas":
        return df.to_pandas()
    if as_ == "polars":
        return df
    raise ValueError("as_ must be 'polars' or 'pandas'")


def _parse_preview(output: str) -> pl.DataFrame:
    if not output:
        raise RuntimeError("remote returned empty output")
    # Surface remote errors
    if "Error" in output and _MARKER not in output:
        raise RuntimeError(f"remote tidy3 failed:\n{output[-2000:]}")
    lines = [ln for ln in output.splitlines() if ln.startswith(_MARKER)]
    if not lines:
        raise RuntimeError(
            "no preview payload from remote — is tidy3/polars installed there?\n"
            f"remote output (tail):\n{output[-1500:]}"
        )
    b64 = lines[-1][len(_MARKER) :]
    raw = base64.b64decode(b64)
    return pl.read_ipc(io.BytesIO(raw))


def remote_status() -> str:
    """Ping remote and report tidy3/polars availability."""
    code = f"""
{_remote_bootstrap()}
import tidy3, polars as pl
print(f"tidy3={{tidy3.__version__}} polars={{pl.__version__}}")
print(f"last_frame={{ {_LAST_NAME!r} in globals() }}")
"""
    return _remote_run()(code, max_chars=2000)
