"""Every ```python block in README.md runs, in order, like notebook cells.

The blocks run in one IPython shell with ``%load_ext tidy3.jupyter``, so
multi-line ``>>`` pipes, bare column names, and ``%%tidy3_run`` work exactly
as a reader would type them. Blocks tagged ```python notest are skipped:
they need a GPU, CRAFT ``%gpu``, a long benchmark, or files the reader
supplies. Files the examples write land in a temporary directory.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("IPython")
pytest.importorskip("plot3")

README = Path(__file__).resolve().parents[1] / "README.md"


def _blocks() -> list[tuple[int, str]]:
    text = README.read_text(encoding="utf-8")
    out = []
    for match in re.finditer(r"^```(python[^\n]*)\n(.*?)^```", text, re.M | re.S):
        info, body = match.group(1), match.group(2)
        if "notest" in info:
            continue
        line = text[: match.start()].count("\n") + 1
        out.append((line, body))
    return out


def test_readme_has_examples():
    assert len(_blocks()) >= 30


def _materialize(value) -> None:
    """Compute what a notebook would show, so lazy plan errors surface."""
    from tidy3 import TidyFrame

    if isinstance(value, TidyFrame):
        value.collect()
    elif type(value).__module__.startswith("plot3"):
        value.html()


def test_readme_examples_run(tmp_path, monkeypatch):
    from IPython.core.interactiveshell import InteractiveShell

    monkeypatch.chdir(tmp_path)
    shell = InteractiveShell.instance()
    # Show every expression statement, not just the last one in a cell.
    shell.ast_node_interactivity = "all"
    shell.display_trap.hook = lambda value: _materialize(value)
    try:
        shell.run_line_magic("load_ext", "tidy3.jupyter")
        # plot3 sets up its notebook support when first imported inside
        # IPython. Other tests may have imported it already, so do that here.
        import plot3

        plot3.register_plot3(quiet=True)
        for line, body in _blocks():
            before = {name: id(value) for name, value in shell.user_ns.items()}
            result = shell.run_cell(body, store_history=False)
            error = result.error_before_exec or result.error_in_exec
            if error is None:
                # Tables and plots the cell assigned (wide = long >> ...).
                try:
                    for name, value in list(shell.user_ns.items()):
                        if before.get(name) != id(value):
                            _materialize(value)
                except Exception as exc:  # noqa: BLE001 - report any failure
                    error = exc
            if error is not None:
                pytest.fail(
                    f"README.md block at line {line} failed: "
                    f"{type(error).__name__}: {error}\n\n{body}"
                )
    finally:
        shell.run_line_magic("unload_ext", "tidy3.jupyter")
        InteractiveShell.clear_instance()
