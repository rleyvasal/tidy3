"""IPython/Jupyter/SolveIt integration — kernel-side.

Highlight + run
---------------
If the frontend sends *selected text* to the kernel (JupyterLab "Run Selected
Text", some SolveIt builds, etc.), the input transformer rewrites multi-line
``>>`` prefixes so they parse.

If the UI has **no** run-selection command, use ``%%tidy3_run``, a separate
cell for the prefix, or ``%tidy3_run`` for a one-liner.

Large data (CRAFT)
------------------
Load tidy3 like every other addon (``%local`` + ``%run addons/tidy3.py``), then
``%gpu``. Subsequent cells run on the remote kernel — use paths that exist on
the GPU host with normal ``scan_parquet`` / pipes. No special remote API.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Callable

_TRANSFORMER: Callable[[list[str]], list[str]] | None = None


def _is_tidy3_owned(value: Any) -> bool:
    """Return whether *value* came from a previous tidy3 import.

    Remote re-seeding removes tidy3 from ``sys.modules``, but objects already
    injected into the notebook namespace keep their old module globals and
    class identities.  Provenance is intentionally narrow so a user's own
    variable with the same public name is left alone.
    """
    if isinstance(value, ModuleType):
        module = getattr(value, "__name__", "")
    else:
        module = getattr(value, "__module__", "")
    return module == "tidy3" or module.startswith("tidy3.")


def _is_pipe_transformer(value: Any) -> bool:
    """Match current or stale copies of the tidy3 input transformer."""
    return (
        getattr(value, "__module__", "") == "tidy3.jupyter"
        and getattr(value, "__name__", "") == "tidy3_input_transformer"
    )


def _lines_to_text(lines: list[str]) -> str:
    return "".join(lines)


def _text_to_lines(text: str) -> list[str]:
    if not text:
        return []
    if text.endswith("\n"):
        parts = text.splitlines(keepends=True)
    else:
        parts = [ln + "\n" for ln in text.splitlines()]
        if not parts:
            parts = [text]
    return parts


def tidy3_input_transformer(lines: list[str]) -> list[str]:
    """Rewrite multi-line tidy3 ``>>`` pipes (cells *and* run-selection)."""
    if not lines:
        return lines
    from tidy3.partial_run import maybe_rewrite_cell

    text = _lines_to_text(lines)
    rewritten = maybe_rewrite_cell(text)
    if rewritten is None:
        return lines
    return _text_to_lines(rewritten)


def enable_pipe_transform(ipython: Any | None = None) -> bool:
    """Install the multi-line ``>>`` rewriter.

    Inserted at the **front** of ``input_transformers_cleanup`` so it runs
    *before* CRAFT's mode router. Under ``%gpu`` the router captures the cell
    as ``pending`` and only the transformed (parenthesized) source is valid
    Python on the remote kernel.
    """
    global _TRANSFORMER
    if ipython is None:
        try:
            from IPython import get_ipython

            ipython = get_ipython()
        except Exception:
            ipython = None
    if ipython is None:
        return False

    transformers = getattr(ipython, "input_transformers_cleanup", None)
    if transformers is None:
        return False
    transformers[:] = [t for t in transformers if not _is_pipe_transformer(t)]
    # Front of list: before CRAFT ``_router_transform`` (appended on %gpu).
    transformers.insert(0, tidy3_input_transformer)
    post = getattr(ipython, "input_transformers_post", None)
    if isinstance(post, list):
        post[:] = [t for t in post if not _is_pipe_transformer(t)]
        post.insert(0, tidy3_input_transformer)
    _TRANSFORMER = tidy3_input_transformer
    return True


def disable_pipe_transform(ipython: Any | None = None) -> None:
    global _TRANSFORMER
    if ipython is None:
        try:
            from IPython import get_ipython

            ipython = get_ipython()
        except Exception:
            ipython = None
    if ipython is None:
        return
    for attr in ("input_transformers_cleanup", "input_transformers_post"):
        transformers = getattr(ipython, attr, None)
        if isinstance(transformers, list):
            transformers[:] = [t for t in transformers if not _is_pipe_transformer(t)]
    _TRANSFORMER = None


# Names that other data grammars (datar/pipda, pandas, etc.) often put in
# user_ns. When loading tidy3 we force-claim these so ``mean("mpg")`` is not
# datar's numpy mean.
_FORCE_NS_NAMES = frozenset(
    {
        "all",
        "any",
        "arrange",
        "col",
        "collect",
        "count",
        "desc",
        "distinct",
        "drop",
        "filter",
        "first",
        "group_by",
        "head",
        "last",
        "max",
        "mean",
        "median",
        "min",
        "mutate",
        "n",
        "pull",
        "rename",
        "sample_frac",
        "sample_n",
        "select",
        "slice",
        "slice_head",
        "slice_max",
        "slice_min",
        "slice_tail",
        "std",
        "sum",
        "summarise",
        "summarize",
        "tally",
        "tidy",
        "transmute",
        "ungroup",
        "var",
    }
)


def inject_api(ipython: Any | None = None, *, force: bool = True) -> None:
    """Put tidy3 public names into the IPython user namespace.

    ``force=True`` (default) overwrites conflicting symbols such as datar's
    ``mean`` / ``sum`` so tidy3 pipes use tidy3 expressions on remote kernels.
    """
    if ipython is None:
        try:
            from IPython import get_ipython

            ipython = get_ipython()
        except Exception:
            return
    if ipython is None:
        return
    user_ns = getattr(ipython, "user_ns", None)
    if user_ns is None:
        return
    import tidy3 as t3

    replaced: list[str] = []
    for name in t3.__all__:
        if name.startswith("_"):
            continue
        try:
            value = getattr(t3, name)
        except AttributeError:
            continue
        existing = user_ns.get(name, None)
        claim = (
            force
            or name in _FORCE_NS_NAMES
            or existing is None
            or _is_tidy3_owned(existing)
        )
        if not claim:
            continue
        if (
            existing is not None
            and not _is_tidy3_owned(existing)
            and existing is not value
        ):
            replaced.append(name)
        user_ns[name] = value
    if "tidy3" not in user_ns or force or _is_tidy3_owned(user_ns.get("tidy3")):
        user_ns["tidy3"] = t3
    if replaced:
        sample = ", ".join(replaced[:12])
        more = f" (+{len(replaced) - 12} more)" if len(replaced) > 12 else ""
        print(
            f"tidy3: reclaimed namespace from other libs: {sample}{more}",
            flush=True,
        )


_MAGICS_REGISTERED = False
_PRE_RUN_HOOK = None


def _line_magic_registered(ipython: Any, name: str) -> bool:
    try:
        lines = ipython.magics_manager.magics.get("line", {})
        return name in lines
    except Exception:
        return False


def _ensure_transform_before_cell(_info=None) -> None:
    """Re-assert tidy3 transformer at front (CRAFT %gpu re-appends its router)."""
    try:
        enable_pipe_transform()
    except Exception:
        pass


def load_ipython_extension(ipython: Any) -> None:
    global _MAGICS_REGISTERED, _PRE_RUN_HOOK
    from IPython.core.magic import Magics, cell_magic, line_magic, magics_class

    from tidy3.partial_run import partial_run

    enable_pipe_transform(ipython)
    inject_api(ipython)

    # Keep tidy3 ahead of CRAFT's router after every %gpu / %local switch.
    try:
        if _PRE_RUN_HOOK is not None:
            try:
                ipython.events.unregister("pre_run_cell", _PRE_RUN_HOOK)
            except Exception:
                pass
        ipython.events.register("pre_run_cell", _ensure_transform_before_cell)
        _PRE_RUN_HOOK = _ensure_transform_before_cell
    except Exception:
        pass

    if _MAGICS_REGISTERED and _line_magic_registered(ipython, "tidy3_pipes"):
        return

    @magics_class
    class Tidy3Magics(Magics):
        @line_magic("tidy3_run")
        def tidy3_run_line(self, line: str = ""):
            """Run a one-line pipe prefix."""
            line = (line or "").strip()
            if not line:
                print(
                    "Partial run (highlight):\n"
                    "  If SolveIt can 'Run Selected Text', select a pipe prefix and run it.\n"
                    "  Else: paste into %%tidy3_run, or put the prefix in its own cell.\n"
                    "Large data: load tidy3 under %local, then %gpu + host paths."
                )
                return None
            return partial_run(line, namespace=self.shell.user_ns)

        @cell_magic("tidy3_run")
        def tidy3_run_cell(self, line: str = "", cell: str = ""):
            """Run a multi-line pipe prefix; show Polars-style preview."""
            source = cell if cell.strip() else line
            return partial_run(source, namespace=self.shell.user_ns)

        @line_magic("tidy3_pipes")
        def tidy3_pipes(self, line: str = ""):
            arg = (line or "status").strip().lower()
            if arg in ("on", "1", "true", "enable"):
                enable_pipe_transform(self.shell)
                print("tidy3: pipe input transformer ON (before CRAFT router)")
            elif arg in ("off", "0", "false", "disable"):
                disable_pipe_transform(self.shell)
                print("tidy3: pipe input transformer OFF")
            else:
                cleanup = getattr(self.shell, "input_transformers_cleanup", []) or []
                on = any(_is_pipe_transformer(t) for t in cleanup)
                pos = next(
                    (i for i, t in enumerate(cleanup) if _is_pipe_transformer(t)),
                    None,
                )
                print(
                    f"tidy3: pipe input transformer {'ON' if on else 'OFF'}"
                    + (f" at cleanup[{pos}]" if pos is not None else "")
                )

    ipython.register_magics(Tidy3Magics)
    _MAGICS_REGISTERED = True


def ensure_ipython_integration(*, quiet: bool = True) -> bool:
    """Enable multi-line ``>>`` rewrite if running inside IPython/SolveIt.

    Safe to call multiple times. Used by ``import tidy3`` and ``tidy()`` so
    bare imports still get pipe rewriting (not only ``%load_ext`` / the CRAFT
    addon).
    """
    try:
        from IPython import get_ipython
    except Exception:
        return False
    ip = get_ipython()
    if ip is None:
        return False
    try:
        load_ipython_extension(ip)
    except Exception as e:
        if not quiet:
            print(f"tidy3: could not enable IPython integration: {e}")
        return False
    return True


def unload_ipython_extension(ipython: Any) -> None:
    global _MAGICS_REGISTERED, _PRE_RUN_HOOK
    disable_pipe_transform(ipython)
    try:
        if _PRE_RUN_HOOK is not None:
            ipython.events.unregister("pre_run_cell", _PRE_RUN_HOOK)
    except Exception:
        pass
    _PRE_RUN_HOOK = None
    _MAGICS_REGISTERED = False
