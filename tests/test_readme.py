"""Every ```python block in README.md and the docs pages runs, in order,
like notebook cells, one fresh shell per file.

The blocks run in one IPython shell, which loads ``tidy3.jupyter`` only
when a block does (as a reader's notebook must), so
multi-line ``>>`` pipes, bare column names, and ``%%tidy3_run`` work exactly
as a reader would type them. Blocks tagged ```python notest are skipped:
they need a GPU, CRAFT ``%gpu``, a long benchmark, or files the reader
supplies. Files the examples write land in a temporary directory.

The examples read their datasets (penguins, mtcars, gapminder) from the
web. Without a network connection the run is skipped, not failed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("IPython")
pytest.importorskip("plot3")

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "notebooks.md",
    ROOT / "docs" / "reference.md",
    ROOT / "docs" / "craft.md",
    ROOT / "docs" / "benchmarks.md",
]
DATA_URL = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"


def _online() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(DATA_URL, timeout=10):
            return True
    except OSError:
        return False


def _blocks(doc: Path = DOCS[0]) -> list[tuple[int, str]]:
    text = doc.read_text(encoding="utf-8")
    out = []
    for match in re.finditer(r"^```(python[^\n]*)\n(.*?)^```", text, re.M | re.S):
        info, body = match.group(1), match.group(2)
        if "notest" in info:
            continue
        line = text[: match.start()].count("\n") + 1
        out.append((line, body))
    return out


def test_readme_has_examples():
    assert len(_blocks()) >= 6
    assert len(_blocks(ROOT / "docs" / "reference.md")) >= 20


def _materialize(value) -> None:
    """Compute what a notebook would show, so lazy plan errors surface."""
    from tidy3 import TidyFrame

    if isinstance(value, TidyFrame):
        value.collect()
    elif type(value).__module__.startswith("plot3"):
        value.html()


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.name)
def test_readme_examples_run(doc, tmp_path, monkeypatch):
    from IPython.core.interactiveshell import InteractiveShell

    if not _online():
        pytest.skip("examples read their data from the web; no network")

    monkeypatch.chdir(tmp_path)
    shell = InteractiveShell.instance()
    # Show every expression statement, not just the last one in a cell.
    shell.ast_node_interactivity = "all"
    shell.display_trap.hook = lambda value: _materialize(value)
    try:
        # No %load_ext here: each document loads the extension in its own
        # cell, as a reader's fresh notebook must. A cell that loads it and
        # uses multi-line pipes in the same cell fails, as it would there.
        # plot3 sets up its notebook support when first imported inside
        # IPython. Other tests may have imported it already, so do that here.
        import plot3

        plot3.register_plot3(quiet=True)
        for line, body in _blocks(doc):
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
                    f"{doc.name} block at line {line} failed: "
                    f"{type(error).__name__}: {error}\n\n{body}"
                )
    finally:
        shell.run_line_magic("unload_ext", "tidy3.jupyter")
        InteractiveShell.clear_instance()
