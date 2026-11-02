"""IPython/Jupyter/SolveIt integration — kernel-side.

Highlight + run
---------------
If the frontend sends *selected text* to the kernel (JupyterLab "Run Selected
Text", some SolveIt builds, etc.), the input transformer rewrites multi-line
``>>`` prefixes so they parse.

If the UI has **no** run-selection command, use ``%%tidy3_run``, a separate
cell for the prefix, or ``%tidy3_run`` for a one-liner.

R-style bare names / backticks (Jupyter only)
---------------------------------------------
A source preparser turns `` `hp new` `` into ``__tidy3_bt__("hp new")``, then
an AST transformer resolves bare names and sentinels by verb context::

    filter(mpg > 20)              # → filter(col("mpg") > 20)
    select(mpg, cyl)              # → select("mpg", "cyl")
    mutate(x = `hp new` / cyl)    # → mutate(x = col("hp new") / col("cyl"))

Negation: prefer ``~`` (Python-native). Optional ``!`` sugar is rewritten to
``~`` only inside tidy3 verb calls (``select(!starts_with(...))``); notebook
shell cells (``!pip install …``) are never rewritten.

Plain ``.py`` files keep explicit ``col("x")`` / string selectors.

Large data (CRAFT)
------------------
Load tidy3 like every other addon (``%local`` + ``%run addons/tidy3.py``), then
``%gpu``. Subsequent cells run on the remote kernel — use paths that exist on
the GPU host with normal ``scan_parquet`` / pipes. No special remote API.
"""

from __future__ import annotations

import dataclasses
from types import ModuleType
from typing import Any, Callable

import tidy3syntax.ipython

from tidy3.masking import (
    ASSIGN_NAME,
    BT_NAME,
    COL_NAME,
    DIALECT,
    Tidy3MaskTransformer,
    is_backtick_transformer,
    is_mask_transformer,
    has_formula_call,
    make_named_assign,
    tidy3_backtick_transform,
)

_TRANSFORMER: Callable[[list[str]], list[str]] | None = None
_R_STYLE_ON = True
_CRAFT_TRANSFORM_NAME = "tidy3"
# tidy3's syntax with R-style masking off: multi-line pipes only.
_PIPES_ONLY = dataclasses.replace(
    DIALECT, negation_calls=(), ordered_calls=(), settings=(), rule_calls=(), rule_after_column=()
)


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


def _is_tidy3_source_transformer(value: Any) -> bool:
    return _is_pipe_transformer(value) or is_backtick_transformer(value)


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
    from tidy3.partial_run import join_broken_pipe, maybe_rewrite_cell

    text = _lines_to_text(lines)
    rewritten = maybe_rewrite_cell(text)
    if rewritten is None:
        # A broken pipe: join it anyway so Python's error names the real
        # mistake, not an IndentationError at the first ``>>``.
        rewritten = join_broken_pipe(text)
    if rewritten is None:
        return lines
    return _text_to_lines(rewritten)


def tidy3_craft_transform(source: str) -> str:
    """CRAFT package transform: tidy3 source sugar for remote-routed cells.

    Registered with CRAFT via :func:`_craft_register_transform` so the core
    router never hard-codes tidy3. Prefer ``~`` for negation; ``!`` is only
    rewritten inside tidy3 verb calls (never shell ``!pip`` / ``!whoami``).
    """
    if not source:
        return source
    from tidy3.masking import is_jupyter_shell_cell

    # Shell cells must reach the remote as ``!cmd``, not ``~cmd``.
    if is_jupyter_shell_cell(source):
        non_empty = [ln for ln in source.splitlines() if ln.strip()]
        if non_empty and all(
            ln.lstrip().startswith("!") or ln.lstrip().startswith("#")
            for ln in non_empty
        ):
            return source
    text = source
    # Match IPython path: bang/backticks only while R-style masking is on.
    if _R_STYLE_ON and ("`" in text or "!" in text or has_formula_call(text)):
        from tidy3.masking import rewrite_backticks

        text = rewrite_backticks(text)
    from tidy3.partial_run import join_broken_pipe, maybe_rewrite_cell

    rewritten = maybe_rewrite_cell(text)
    if rewritten is None:
        rewritten = join_broken_pipe(text)
    return rewritten if rewritten is not None else text


def _craft_register_transform(*, active: bool | None = None) -> None:
    """Register/refresh tidy3 on CRAFT's transform registry when present."""
    if active is None:
        active = _R_STYLE_ON
    try:
        from IPython import get_ipython

        ip = get_ipython()
    except Exception:
        return
    if ip is None:
        return
    ns = getattr(ip, "user_ns", None) or {}
    reg = ns.get("register_transform")
    if not callable(reg):
        # CRAFT core not loaded — IPython transformers alone are enough.
        return
    try:
        reg(_CRAFT_TRANSFORM_NAME, tidy3_craft_transform, active=bool(active))
    except Exception:
        pass


def _craft_unregister_transform() -> None:
    try:
        from IPython import get_ipython

        ip = get_ipython()
    except Exception:
        return
    if ip is None:
        return
    ns = getattr(ip, "user_ns", None) or {}
    unreg = ns.get("unregister_transform")
    if callable(unreg):
        try:
            unreg(_CRAFT_TRANSFORM_NAME)
        except Exception:
            pass
    else:
        set_active = ns.get("set_transform_active")
        if callable(set_active):
            try:
                set_active(_CRAFT_TRANSFORM_NAME, False)
            except Exception:
                pass


def _shell(ipython: Any | None) -> Any | None:
    if ipython is not None:
        return ipython
    try:
        from IPython import get_ipython

        return get_ipython()
    except Exception:
        return None


def _remove_own(ipython: Any) -> None:
    """An earlier tidy3's own transformers, from before tidy3syntax's coordinator."""
    for attr in ("input_transformers_cleanup", "input_transformers_post"):
        transformers = getattr(ipython, attr, None)
        if isinstance(transformers, list):
            transformers[:] = [t for t in transformers if not _is_tidy3_source_transformer(t)]
    transformers = getattr(ipython, "ast_transformers", None)
    if isinstance(transformers, list):
        transformers[:] = [t for t in transformers if not is_mask_transformer(t)]


def _register(ipython: Any) -> None:
    """tidy3's syntax in tidy3syntax's coordinator: all of it with R-style masking on, the pipes alone with it off."""
    _remove_own(ipython)
    if _R_STYLE_ON:
        tidy3syntax.ipython.register(ipython, DIALECT, ast_transformer=Tidy3MaskTransformer())
    else:
        tidy3syntax.ipython.register(ipython, _PIPES_ONLY)


def enable_pipe_transform(ipython: Any | None = None) -> bool:
    """Turn on multi-line ``>>`` pipes (and, with R-style masking on, the rest of tidy3's syntax).

    The rewrite runs in tidy3syntax's coordinator, one input transformer
    shared with tidy3a, plot3 and stats3, kept ahead of CRAFT's router
    before each cell, so ``%gpu`` sends the remote kernel valid Python.
    """
    global _TRANSFORMER
    ipython = _shell(ipython)
    if ipython is None or not isinstance(getattr(ipython, "input_transformers_cleanup", None), list):
        return False
    _register(ipython)
    _TRANSFORMER = tidy3_input_transformer
    # CRAFT path: same package owns the transform; core does not hard-code tidy3.
    _craft_register_transform(active=True)
    return True


def disable_pipe_transform(ipython: Any | None = None) -> None:
    global _TRANSFORMER
    ipython = _shell(ipython)
    if ipython is None:
        return
    tidy3syntax.ipython.unregister(ipython, "tidy3")
    _remove_own(ipython)
    _TRANSFORMER = None
    _craft_unregister_transform()


def enable_r_style(ipython: Any | None = None) -> bool:
    """Enable bare-name + backtick masking (AST + source rewrites)."""
    global _R_STYLE_ON
    _R_STYLE_ON = True
    ipython = _shell(ipython)
    if ipython is None:
        return False
    from tidy3.expr import col

    ns = getattr(ipython, "user_ns", None)
    if ns is not None:
        ns[COL_NAME] = col
        # Fallback if a sentinel escapes AST handling → treat as column expr.
        ns[BT_NAME] = col
        # Runtime carrier for `new col` = expr → NamedAssign; the AST pass
        # renames tidy3syntax's __r_assign__ to this name in tidy3's calls.
        ns[ASSIGN_NAME] = make_named_assign
        # Bare notebook names that may be columns (dplyr's data mask).
        from tidy3.expr import name_ref
        from tidy3.masking import NAME_REF

        ns[NAME_REF] = name_ref
        import tidy3
        from tidy3.masking import API_NAME

        ns[API_NAME] = tidy3
    return enable_pipe_transform(ipython)


def disable_r_style(ipython: Any | None = None) -> None:
    """Disable bare-name / backtick masking; keep pipe ``>>`` rewriter."""
    global _R_STYLE_ON
    _R_STYLE_ON = False
    ipython = _shell(ipython)
    if ipython is None:
        return
    enable_pipe_transform(ipython)


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
        "compute",
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
        # force=True: always claim. force=False: only fill gaps / refresh stale tidy3.
        if force:
            claim = True
        else:
            claim = existing is None or _is_tidy3_owned(existing)
        if not claim:
            continue
        if (
            force
            and existing is not None
            and not _is_tidy3_owned(existing)
            and existing is not value
            and name in _FORCE_NS_NAMES
        ):
            replaced.append(name)
        user_ns[name] = value
    if force or "tidy3" not in user_ns or _is_tidy3_owned(user_ns.get("tidy3")):
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
    """tidy3 0.15's hook, kept so an old registration can be removed; the
    coordinator now keeps its transformer ahead of CRAFT's router itself."""


def load_ipython_extension(ipython: Any) -> None:
    global _MAGICS_REGISTERED, _PRE_RUN_HOOK
    from IPython.core.magic import Magics, cell_magic, line_magic, magics_class

    from tidy3.partial_run import partial_run

    inject_api(ipython)
    enable_r_style(ipython)  # pipes + backticks + AST masking

    # tidy3syntax's coordinator keeps the transformer ahead of CRAFT's router
    # before every cell; drop the hook an earlier tidy3 registered for that.
    if _PRE_RUN_HOOK is not None:
        try:
            ipython.events.unregister("pre_run_cell", _PRE_RUN_HOOK)
        except Exception:
            pass
        _PRE_RUN_HOOK = None

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
                on = "tidy3" in tidy3syntax.ipython.registered(self.shell)
                print(f"tidy3: pipe input transformer {'ON' if on else 'OFF'} (tidy3syntax's coordinator)")

        @line_magic("tidy3_mask")
        def tidy3_mask(self, line: str = ""):
            """Toggle R-style bare-name / backtick masking: on|off|status."""
            arg = (line or "status").strip().lower()
            if arg in ("on", "1", "true", "enable"):
                enable_r_style(self.shell)
                print("tidy3: R-style bare-name/backtick masking ON")
            elif arg in ("off", "0", "false", "disable"):
                disable_r_style(self.shell)
                print("tidy3: R-style bare-name/backtick masking OFF")
            else:
                ast_t = getattr(self.shell, "ast_transformers", []) or []
                on = any(is_mask_transformer(t) for t in ast_t)
                print(f"tidy3: R-style masking {'ON' if on else 'OFF'}")

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
    _craft_unregister_transform()
    try:
        if _PRE_RUN_HOOK is not None:
            ipython.events.unregister("pre_run_cell", _PRE_RUN_HOOK)
    except Exception:
        pass
    _PRE_RUN_HOOK = None
    _MAGICS_REGISTERED = False
