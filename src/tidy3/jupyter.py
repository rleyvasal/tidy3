"""IPython/Jupyter/SolveIt integration — kernel-side, no VS Code required.

When loaded (``%load_ext tidy3.jupyter`` or the gpudev addon):

1. **Input transformer** — multi-line ``>>`` pipes and partial prefixes become
   valid Python before the kernel parses them. Works for whole cells **and**
   for frontends that send *selected text* to the kernel (SolveIt, Jupyter
   "Run Selected Text", etc.).
2. **Magics** — ``%tidy3_run`` / ``%%tidy3_run`` for explicit partial runs.
3. **API inject** — optional helper to push tidy3 names into ``user_ns``.
"""

from __future__ import annotations

from typing import Any, Callable

# Transformer callable registered on the shell (for unload)
_TRANSFORMER: Callable[[list[str]], list[str]] | None = None


def _lines_to_text(lines: list[str]) -> str:
    return "".join(lines)


def _text_to_lines(text: str) -> list[str]:
    if not text:
        return []
    # IPython expects lines with trailing newlines (except possibly last)
    if text.endswith("\n"):
        parts = text.splitlines(keepends=True)
    else:
        parts = [ln + "\n" for ln in text.splitlines()]
        if not parts:
            parts = [text]
    return parts


def tidy3_input_transformer(lines: list[str]) -> list[str]:
    """IPython cleanup transformer: rewrite multi-line tidy3 ``>>`` pipes."""
    if not lines:
        return lines
    from tidy3.partial_run import maybe_rewrite_cell

    text = _lines_to_text(lines)
    rewritten = maybe_rewrite_cell(text)
    if rewritten is None:
        return lines
    return _text_to_lines(rewritten)


def enable_pipe_transform(ipython: Any | None = None) -> bool:
    """Register the pipe input transformer on the active IPython shell."""
    global _TRANSFORMER
    if ipython is None:
        try:
            from IPython import get_ipython

            ipython = get_ipython()
        except Exception:
            ipython = None
    if ipython is None:
        return False

    # Avoid double-registration
    transformers = getattr(ipython, "input_transformers_cleanup", None)
    if transformers is None:
        return False
    if tidy3_input_transformer in transformers:
        _TRANSFORMER = tidy3_input_transformer
        return True
    transformers.append(tidy3_input_transformer)
    _TRANSFORMER = tidy3_input_transformer
    return True


def disable_pipe_transform(ipython: Any | None = None) -> None:
    """Remove the pipe input transformer."""
    global _TRANSFORMER
    if ipython is None:
        try:
            from IPython import get_ipython

            ipython = get_ipython()
        except Exception:
            ipython = None
    if ipython is None:
        return
    transformers = getattr(ipython, "input_transformers_cleanup", None)
    if transformers and tidy3_input_transformer in transformers:
        transformers.remove(tidy3_input_transformer)
    _TRANSFORMER = None


def inject_api(ipython: Any | None = None) -> None:
    """Put tidy3 public names into ``user_ns`` (for bare ``filter`` / ``col``)."""
    if ipython is None:
        try:
            from IPython import get_ipython

            ipython = get_ipython()
        except Exception:
            return
    if ipython is None or not getattr(ipython, "user_ns", None):
        return
    import tidy3 as t3

    for name in t3.__all__:
        if name.startswith("_"):
            continue
        try:
            ipython.user_ns.setdefault(name, getattr(t3, name))
        except AttributeError:
            pass
    ipython.user_ns.setdefault("tidy3", t3)


def load_ipython_extension(ipython: Any) -> None:
    """Register magics + pipe transformer (any Jupyter, including SolveIt)."""
    from IPython.core.magic import Magics, cell_magic, line_magic, magics_class

    from tidy3.partial_run import partial_run

    enable_pipe_transform(ipython)
    inject_api(ipython)

    @magics_class
    class Tidy3Magics(Magics):
        @line_magic("tidy3_run")
        def tidy3_run_line(self, line: str = ""):
            """Run a one-line pipe: ``%tidy3_run tidy(df) >> filter(...)``."""
            line = (line or "").strip()
            if not line:
                print(
                    "Usage: %tidy3_run tidy(df) >> filter(col('x') > 0)\n"
                    "Or put a multi-line prefix in a cell — the pipe transformer\n"
                    "rewrites it automatically (no VS Code extension needed)."
                )
                return None
            return partial_run(line, namespace=self.shell.user_ns)

        @cell_magic("tidy3_run")
        def tidy3_run_cell(self, line: str = "", cell: str = ""):
            """Run a multi-line pipe prefix (explicit partial run).

            Example::

                %%tidy3_run
                tidy(cars)
                >> filter(col("mpg") > 20)
            """
            source = cell if cell.strip() else line
            return partial_run(source, namespace=self.shell.user_ns)

        @line_magic("tidy3_pipes")
        def tidy3_pipes(self, line: str = ""):
            """Enable/disable auto rewrite: ``%tidy3_pipes on|off|status``."""
            arg = (line or "status").strip().lower()
            if arg in ("on", "1", "true", "enable"):
                enable_pipe_transform(self.shell)
                print("tidy3: pipe input transformer ON (multi-line >> auto-wrapped)")
            elif arg in ("off", "0", "false", "disable"):
                disable_pipe_transform(self.shell)
                print("tidy3: pipe input transformer OFF")
            else:
                on = tidy3_input_transformer in getattr(
                    self.shell, "input_transformers_cleanup", []
                )
                print(f"tidy3: pipe input transformer {'ON' if on else 'OFF'}")

    ipython.register_magics(Tidy3Magics)


def unload_ipython_extension(ipython: Any) -> None:
    disable_pipe_transform(ipython)
