"""Standalone SolveIt / Jupyter loader — one command for local and CRAFT/GPU.

Usage::

    %run /app/data/gpudevd/tidy3/tidy3.py
    %run /path/to/tidy3/tidy3.py
    %run /path/to/tidy3/load.py          # same loader

What this does (always):

1. Puts ``src/`` on ``sys.path`` (editable checkout, no pip install needed)
2. Fresh-imports tidy3 so a ``git pull`` takes effect
3. Injects the public API into the IPython user namespace
4. Enables multi-line ``>>`` pipes + R-style bare names / backticks / ``~``
   (optional ``!`` sugar only inside tidy3 verb calls — never shell ``!pip``)

What this does **only when CRAFT is present** (auto-detected):

5. Registers remote seed hooks so ``%gpu`` cells get tidy3 without a second command
6. Exposes ``seed_tidy3_remote(force=True)`` for kernel restarts

You do **not** need different commands for local vs GPU. Load once; CRAFT is
optional and is detected from the environment.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

print("tidy3: loader starting…", flush=True)

# %run must execute this file as a script, not import it as package "tidy3".
if __name__ == "tidy3":  # pragma: no cover
    raise ImportError(
        "tidy3.py was imported as module 'tidy3' (sys.path shadowing). "
        "Load it with %run /path/to/tidy3/tidy3.py — the package lives in src/tidy3/."
    )

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
_PKG_PARENT = _SRC if (_SRC / "tidy3").is_dir() else _ROOT

_pkg_dir = str(_PKG_PARENT.resolve())
while _pkg_dir in sys.path:
    sys.path.remove(_pkg_dir)
sys.path.insert(0, _pkg_dir)
print(f"tidy3: root={_ROOT}  pkg_path={_pkg_dir}", flush=True)

try:
    from IPython import get_ipython
except Exception:  # pragma: no cover
    get_ipython = None

# Drop stale modules so re-%run / git pull is picked up without kernel restart.
for _m in [m for m in list(sys.modules) if m == "tidy3" or m.startswith("tidy3.")]:
    del sys.modules[_m]

try:
    import tidy3
except Exception:
    print("tidy3: FAILED to import package:", flush=True)
    traceback.print_exc()
    raise

print(
    f"tidy3: imported v{getattr(tidy3, '__version__', '?')} "
    f"from {Path(tidy3.__file__).resolve()}",
    flush=True,
)

_PUBLIC = {"tidy3": tidy3}
for _name in getattr(tidy3, "__all__", []):
    if _name.startswith("_"):
        continue
    try:
        _PUBLIC[_name] = getattr(tidy3, _name)
    except AttributeError:
        pass
for _name in ("tidy", "TidyFrame", "col", "filter", "select", "mutate", "arrange"):
    if hasattr(tidy3, _name):
        _PUBLIC[_name] = getattr(tidy3, _name)

# ── CRAFT / GPU detection (optional) ─────────────────────────────────────────

_SEED_STATE = {"stamp": None, "kc_id": None, "ok": False}


def _craft_status(ip=None) -> str:
    """Return ``connected`` | ``present`` | ``absent``."""
    if ip is None:
        try:
            ip = get_ipython() if get_ipython else None
        except Exception:
            ip = None
    ns = (getattr(ip, "user_ns", None) or {}) if ip is not None else {}
    rr = ns.get("remote_run_")
    mgr = ns.get("_exec_mgr")
    if callable(rr) and mgr is not None:
        return "connected"
    if "remote_run_" in ns or "register_local_magic" in ns:
        return "present"
    try:
        import gpudev_craft  # noqa: F401

        return "present"
    except Exception:
        pass
    for key in ("_craft_cfg", "CRAFT", "remote_run", "gpu_mode"):
        if key in ns:
            return "present"
    return "absent"


def seed_remote(
    *, force: bool = False, quiet: bool = False, style_polars: bool = True
) -> bool:
    """Ship tidy3 to the CRAFT remote kernel (no-op without CRAFT)."""
    try:
        from tidy3 import craft
    except ImportError:
        if not quiet:
            print(
                "tidy3: remote seed unavailable (no craft module in this build)",
                flush=True,
            )
        return False

    ip = get_ipython() if get_ipython else None
    if ip is None:
        return False
    ns = ip.user_ns or {}
    rr = ns.get("remote_run_")
    mgr = ns.get("_exec_mgr")
    if not callable(rr) or mgr is None:
        if not quiet:
            print(
                "tidy3: CRAFT not connected yet — local only "
                "(will seed automatically on first %gpu cell)",
                flush=True,
            )
        return False

    payload, stamp = craft.build_payload()
    kc_id = id(getattr(mgr, "remote_kc", None))
    if (
        not force
        and _SEED_STATE["stamp"] == stamp
        and _SEED_STATE["kc_id"] == kc_id
    ):
        return _SEED_STATE["ok"]

    ok, msg = craft.seed(
        rr, payload=payload, stamp=stamp, style_polars=style_polars
    )
    _SEED_STATE.update(stamp=stamp, kc_id=kc_id, ok=ok)
    if ok:
        if not quiet:
            print(f"tidy3: {msg}", flush=True)
    else:
        print(
            "tidy3: remote seed FAILED — %gpu cells won't know tidy3.\n"
            + msg
            + "\nRetry with seed_tidy3_remote(force=True)",
            flush=True,
        )
    return ok


def _maybe_seed_on_cell(_info=None):
    try:
        import gpudev_craft.core as _core

        router = getattr(_core, "ROUTER", None)
        py_be = getattr(_core, "PY_BACKEND", None)
        if router is None or py_be is None or router.backend is not py_be:
            return
    except Exception:
        return
    seed_remote(quiet=True)


ip = get_ipython() if get_ipython else None
_craft = _craft_status(ip)
print(f"tidy3: environment = {_craft} (local setup always runs)", flush=True)

if ip is None:
    print(
        "tidy3: WARNING — get_ipython() is None; run this with %run inside "
        "IPython / SolveIt / Jupyter to get pipes + magics.\n"
        "  Plain import still works: from tidy3 import tidy, select, …",
        flush=True,
    )
elif getattr(ip, "user_ns", None) is None:
    print("tidy3: WARNING — no user_ns on IPython shell", flush=True)
else:
    # Force-overwrite (datar/pipda often already own mean/sum/filter).
    ip.user_ns.update(_PUBLIC)
    ip.user_ns["seed_tidy3_remote"] = seed_remote
    print(
        f"tidy3: injected {len(_PUBLIC)} names into user_ns",
        flush=True,
    )

    try:
        import tidy3.jupyter as _tj

        if hasattr(_tj, "ensure_ipython_integration"):
            _tj.ensure_ipython_integration(quiet=False)
        else:
            _em = ip.extension_manager
            _loaded = getattr(_em, "loaded", set())
            _loaded.discard("tidy3.jupyter")
            _em.load_extension("tidy3.jupyter")

        if hasattr(_tj, "inject_api"):
            _tj.inject_api(ip, force=True)
        if hasattr(_tj, "enable_r_style"):
            _tj.enable_r_style(ip)
        elif hasattr(_tj, "enable_pipe_transform"):
            _tj.enable_pipe_transform(ip)

        _lines = getattr(ip.magics_manager, "magics", {}).get("line", {})
        print(
            f"tidy3: %tidy3_pipes registered: {'tidy3_pipes' in _lines}",
            flush=True,
        )
        _cleanup = getattr(ip, "input_transformers_cleanup", []) or []
        _is_pipe = getattr(_tj, "_is_pipe_transformer", None)
        if callable(_is_pipe):
            _pos = next((i for i, t in enumerate(_cleanup) if _is_pipe(t)), None)
        else:
            _pos = next(
                (
                    i
                    for i, t in enumerate(_cleanup)
                    if getattr(t, "__name__", "") == "tidy3_input_transformer"
                ),
                None,
            )
        print(
            f"tidy3: pipe transformer "
            f"{'ON at cleanup[' + str(_pos) + ']' if _pos is not None else 'OFF'}",
            flush=True,
        )
        if _pos is None:
            print(
                "  Multi-line >> without parentheses will SyntaxError until "
                "transformer is ON.\n"
                "  Workaround: ( tidy(cars) >> filter(...) >> ... )",
                flush=True,
            )
    except Exception as _ext_err:
        print(f"tidy3: jupyter setup FAILED: {_ext_err}", flush=True)
        traceback.print_exc()
        print(
            "  Workaround: wrap multi-line pipes in parentheses:\n"
            "    ( tidy(cars) >> filter(...) >> summarise(...) )",
            flush=True,
        )

    if _craft != "absent":
        try:
            prev = ip.user_ns.get("_tidy3_seed_cb")
            if prev is not None:
                try:
                    ip.events.unregister("pre_run_cell", prev)
                except Exception:
                    pass
            ip.events.register("pre_run_cell", _maybe_seed_on_cell)
            ip.user_ns["_tidy3_seed_cb"] = _maybe_seed_on_cell
        except Exception:
            pass
        seed_remote(quiet=False)
    else:
        print(
            "tidy3: no CRAFT detected — local only "
            "(fine for VS Code / SolveIt without %gpu)",
            flush=True,
        )

print(
    f"tidy3 {tidy3.__version__} ready "
    f"from {Path(tidy3.__file__).resolve().parent}",
    flush=True,
)
print(
    "  multi-line >> auto-rewritten when pipe transformer is ON\n"
    "  bare names / backticks / ~  (R-style masking; ! sugar only in tidy3 verbs)\n"
    "  %tidy3_pipes on|off|status    %tidy3_mask on|off|status\n"
    "  GPU: same %run; seeds remote when CRAFT is connected "
    "(seed_tidy3_remote(force=True) after kernel restart)\n"
    "  fallback: ( tidy(df) >> filter(...) >> ... )",
    flush=True,
)
