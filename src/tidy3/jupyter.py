"""IPython/Jupyter/SolveIt integration — kernel-side.

Highlight + run
---------------
If the frontend sends *selected text* to the kernel (JupyterLab "Run Selected
Text", some SolveIt builds, etc.), the input transformer rewrites multi-line
``>>`` prefixes so they parse. That is true partial-run of a highlight.

If the UI has **no** run-selection command, use one of:

* ``%%tidy3_run`` with the highlighted lines pasted into the cell body
* put the prefix in its own cell and run the cell
* ``%tidy3_run`` for a one-liner

Remote large data (CRAFT)
-------------------------
``%%tidy3_remote`` / ``remote(...)`` run the pipe on the GPU host via
``remote_run_``; only a small Polars preview returns to SolveIt.
"""

from __future__ import annotations

from typing import Any, Callable

_TRANSFORMER: Callable[[list[str]], list[str]] | None = None


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
    if tidy3_input_transformer not in transformers:
        transformers.append(tidy3_input_transformer)
    # Also post — some frontends feed selection after cleanup
    post = getattr(ipython, "input_transformers_post", None)
    if isinstance(post, list) and tidy3_input_transformer not in post:
        post.append(tidy3_input_transformer)
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
        if transformers and tidy3_input_transformer in transformers:
            transformers.remove(tidy3_input_transformer)
    _TRANSFORMER = None


def inject_api(ipython: Any | None = None) -> None:
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
    # Remote helpers
    try:
        from tidy3.remote import remote, remote_bind, remote_collect, remote_status

        ipython.user_ns.setdefault("remote", remote)
        ipython.user_ns.setdefault("remote_bind", remote_bind)
        ipython.user_ns.setdefault("remote_collect", remote_collect)
        ipython.user_ns.setdefault("remote_status", remote_status)
    except Exception:
        pass


def _register_local_if_craft(names: list[str]) -> None:
    """Keep tidy3 magics on the host under %gpu (they call remote_run_)."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return
        ns = ip.user_ns or {}
        reg = ns.get("register_local_magic")
        if not callable(reg):
            # try gpudev_craft
            try:
                from gpudev_craft.core import register_local_magic as reg
            except Exception:
                return
        for name in names:
            try:
                reg(name if name.startswith("%") else f"%{name}")
            except Exception:
                pass
    except Exception:
        pass


def load_ipython_extension(ipython: Any) -> None:
    from IPython.core.magic import Magics, cell_magic, line_magic, magics_class

    from tidy3.partial_run import partial_run

    enable_pipe_transform(ipython)
    inject_api(ipython)

    @magics_class
    class Tidy3Magics(Magics):
        @line_magic("tidy3_run")
        def tidy3_run_line(self, line: str = ""):
            """Run a one-line pipe prefix (local)."""
            line = (line or "").strip()
            if not line:
                print(
                    "Partial run (highlight):\n"
                    "  If SolveIt/Jupyter can 'Run Selected Text', select a pipe\n"
                    "  prefix and run it — the kernel rewrites multi-line >>.\n"
                    "  Else: paste the highlight into %%tidy3_run  or use its own cell.\n"
                    "Remote large data:  %%tidy3_remote   or  remote(\"\"\"...\"\"\")"
                )
                return None
            return partial_run(line, namespace=self.shell.user_ns)

        @cell_magic("tidy3_run")
        def tidy3_run_cell(self, line: str = "", cell: str = ""):
            """Run highlighted/pasted pipe prefix locally; show Polars-style preview."""
            source = cell if cell.strip() else line
            return partial_run(source, namespace=self.shell.user_ns)

        @cell_magic("tidy3_remote")
        def tidy3_remote_cell(self, line: str = "", cell: str = ""):
            """Run pipe on CRAFT remote; return small Polars preview only.

            Paths in the pipe must exist **on the GPU host**. Example::

                %%tidy3_remote
                scan_parquet("/home/gpudev/data/big.parquet")
                >> filter(col("x") > 0)
                >> group_by("g")
                >> summarise(n=n())
            """
            from tidy3.remote import remote

            source = cell if cell.strip() else line
            # Optional: %%tidy3_remote n=20
            n = None
            bind = None
            for part in (line or "").split():
                if part.startswith("n="):
                    n = int(part.split("=", 1)[1])
                elif part.startswith("bind="):
                    bind = part.split("=", 1)[1]
            return remote(source, n=n, bind=bind)

        @line_magic("tidy3_remote")
        def tidy3_remote_line(self, line: str = ""):
            from tidy3.remote import remote

            line = (line or "").strip()
            if not line:
                print("Usage: %tidy3_remote scan_parquet('/data/x.parquet') >> filter(...)")
                return None
            return remote(line)

        @line_magic("tidy3_pipes")
        def tidy3_pipes(self, line: str = ""):
            arg = (line or "status").strip().lower()
            if arg in ("on", "1", "true", "enable"):
                enable_pipe_transform(self.shell)
                print("tidy3: pipe input transformer ON")
            elif arg in ("off", "0", "false", "disable"):
                disable_pipe_transform(self.shell)
                print("tidy3: pipe input transformer OFF")
            else:
                on = tidy3_input_transformer in getattr(
                    self.shell, "input_transformers_cleanup", []
                )
                print(f"tidy3: pipe input transformer {'ON' if on else 'OFF'}")

    ipython.register_magics(Tidy3Magics)
    _register_local_if_craft(
        ["%tidy3_run", "%tidy3_remote", "%tidy3_pipes", "%%tidy3_run", "%%tidy3_remote"]
    )


def unload_ipython_extension(ipython: Any) -> None:
    disable_pipe_transform(ipython)
