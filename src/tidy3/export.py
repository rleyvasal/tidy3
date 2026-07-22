"""Export notebooks / R-style source to plain CPython scripts.

Author in Jupyter with bare names and backticks; ship automation as explicit
Python. Mirrors the spirit of nbdev's ``nb_export``: the notebook is the source
of truth, the ``.py`` file is a build artifact.

Example::

    from tidy3 import nb_export

    nb_export("analysis.ipynb", "analysis_pipeline.py")
    nb_export("analysis.ipynb", "lib.py", only_export=True)  # #| export cells

CLI::

    python -m tidy3 export analysis.ipynb -o analysis_pipeline.py
"""

from __future__ import annotations

import ast
import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "ExportResult",
    "collect_known_names",
    "nb_export",
    "transform_source",
    "transform_cell",
]

# nbdev-style directives: #| export   #|export   #| skip   #|skip
_DIRECTIVE_RE = re.compile(
    r"^\s*#\s*\|\s*(?P<name>[A-Za-z_][\w-]*)\s*(?::\s*(?P<arg>.*))?\s*$"
)
_MAGIC_LINE_RE = re.compile(r"^\s*%{1,2}")
_SHELL_LINE_RE = re.compile(r"^\s*!")


@dataclass
class ExportResult:
    """Outcome of :func:`nb_export`."""

    path: Path
    cells_seen: int = 0
    cells_exported: int = 0
    cells_skipped: int = 0
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        w = f", {len(self.warnings)} warning(s)" if self.warnings else ""
        return (
            f"nb_export → {self.path} "
            f"({self.cells_exported}/{self.cells_seen} cells{w})"
        )


def _parse_directives(source: str) -> tuple[set[str], dict[str, str], str]:
    """Split leading ``#| …`` directives from cell source.

    Returns ``(directive_names, args_by_name, remaining_source)``.
    """
    names: set[str] = set()
    args: dict[str, str] = {}
    lines = source.splitlines(keepends=True)
    i = 0
    # Allow blank lines before directives
    while i < len(lines) and not lines[i].strip():
        i += 1
    body_start = i
    while i < len(lines):
        raw = lines[i]
        # Directives must appear before real code (blank lines ok between them)
        if not raw.strip():
            i += 1
            continue
        m = _DIRECTIVE_RE.match(raw.rstrip("\n"))
        if not m:
            break
        name = m.group("name").lower().replace("-", "_")
        names.add(name)
        if m.group("arg") is not None:
            args[name] = m.group("arg").strip()
        i += 1
        body_start = i
    return names, args, "".join(lines[body_start:])


def collect_known_names(sources: Iterable[str]) -> set[str]:
    """Names that must not become column refs (imports, assigns, defs)."""
    from tidy3.masking import default_known_names

    known = default_known_names()
    for src in sources:
        text = src
        if "`" in text:
            from tidy3.masking import rewrite_backticks

            try:
                text = rewrite_backticks(text)
            except Exception:
                pass
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                known.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    _add_target_names(t, known)
            elif isinstance(node, ast.AnnAssign) and node.target is not None:
                _add_target_names(node.target, known)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    known.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    known.add(alias.asname or alias.name)
    return known


def _add_target_names(target: ast.AST, known: set[str]) -> None:
    if isinstance(target, ast.Name):
        known.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _add_target_names(elt, known)


def _try_rewrite_plot3_magic(line: str) -> str | None:
    """Best-effort ``%plot3 df x=a y=b`` → ggplot expression."""
    s = line.strip()
    if not s.startswith("%plot3"):
        return None
    rest = s[len("%plot3") :].strip()
    if not rest:
        return None
    try:
        parts = shlex.split(rest)
    except ValueError:
        return None
    if not parts:
        return None
    expr = parts[0]
    m: dict[str, str] = {}
    kind = "point"
    for tok in parts[1:]:
        k, _, v = tok.partition("=")
        if not v:
            continue
        if k in ("x", "y", "z", "color", "colour", "group"):
            m["color" if k == "colour" else k] = v
        elif k == "kind":
            kind = v
    if "x" not in m or "y" not in m:
        return None
    aes_parts = [f'{k}="{v}"' for k, v in m.items()]
    geoms: list[str] = []
    for part in kind.split("+"):
        part = part.strip()
        if part == "point":
            geoms.append("geom_point()")
        elif part == "line":
            geoms.append("geom_line()")
        elif part == "path":
            geoms.append("geom_path()")
    if not geoms:
        geoms = ["geom_point()"]
    layers = " + ".join(geoms)
    return f"(ggplot({expr}, aes({', '.join(aes_parts)})) + {layers})"


def _handle_magic_cell(source: str, warnings: list[str], cell_no: int) -> str | None:
    """Return rewritten source, empty string to skip, or None if not a magic cell."""
    lines = [ln for ln in source.splitlines() if ln.strip()]
    if not lines:
        return ""
    # Entire cell is magics / shell?
    if not all(
        _MAGIC_LINE_RE.match(ln) or _SHELL_LINE_RE.match(ln) for ln in lines
    ):
        # Mixed: comment magic lines, keep code lines
        out: list[str] = []
        has_code = False
        for ln in source.splitlines():
            if _MAGIC_LINE_RE.match(ln) or _SHELL_LINE_RE.match(ln):
                rewritten = _try_rewrite_plot3_magic(ln)
                if rewritten is not None:
                    out.append(rewritten)
                    has_code = True
                else:
                    out.append(f"# notebook-only: {ln.lstrip()}")
                    warnings.append(
                        f"cell {cell_no}: commented magic/shell line (not portable)"
                    )
            else:
                out.append(ln)
                if ln.strip() and not ln.strip().startswith("#"):
                    has_code = True
        if not has_code:
            return ""
        return "\n".join(out)

    # Pure magic cell
    out_lines: list[str] = []
    for ln in lines:
        rewritten = _try_rewrite_plot3_magic(ln)
        if rewritten is not None:
            out_lines.append(rewritten)
        else:
            out_lines.append(f"# notebook-only: {ln.lstrip()}")
            warnings.append(
                f"cell {cell_no}: skipped/commented magic `{ln.strip()[:60]}`"
            )
    if all(x.lstrip().startswith("#") for x in out_lines):
        return ""
    return "\n".join(out_lines)


def _publicize_sentinels(source: str) -> tuple[str, bool]:
    """Rewrite internal masking sentinels to public API names.

    ``__tidy3_col__("mpg")`` → ``col("mpg")`` so exported scripts run with a
    normal ``from tidy3 import *`` (no Jupyter injectors).

    Returns ``(source, needs_assign_helper)`` when ``__tidy3_assign__`` remains
    (spaced *new* column names in mutate/select/rename).
    """
    from tidy3.masking import ASSIGN_NAME, BT_NAME, COL_NAME

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, ASSIGN_NAME in source

    needs_assign = False

    class _Fix(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if isinstance(node.func, ast.Name):
                if node.func.id in {COL_NAME, BT_NAME}:
                    node.func.id = "col"
                elif node.func.id == ASSIGN_NAME:
                    nonlocal needs_assign
                    needs_assign = True
            return node

    tree = _Fix().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), needs_assign


def transform_source(
    source: str,
    *,
    known: set[str] | None = None,
    with_plot3: bool = True,
    publicize: bool = True,
) -> str:
    """Rewrite R-style bare names / backticks / multi-line pipes to plain Python.

    This is the same logical pipeline the Jupyter transformers apply at runtime,
    intended for export and offline automation.

    When ``publicize`` is True (default for export), internal sentinels become
    public names (``col(...)`` instead of ``__tidy3_col__(...)``).
    """
    if not source or not source.strip():
        return source

    text = source
    # Multi-line >> pipe wrapping + backtick preparser
    from tidy3.partial_run import maybe_rewrite_cell

    rewritten = maybe_rewrite_cell(text)
    if rewritten is not None:
        text = rewritten
        if text.endswith("\n"):
            text = text[:-1]

    from tidy3.masking import apply_masking, default_known_names

    kn = set(known) if known is not None else default_known_names()
    try:
        text = apply_masking(text, known=kn, backticks=True)
    except SyntaxError:
        # Leave as-is; caller may warn
        raise

    if with_plot3:
        try:
            from plot3.masking import (
                apply_masking as plot3_apply_masking,
                default_known_names as plot3_known,
            )

            kn2 = kn | plot3_known()
            text = plot3_apply_masking(text, known=kn2, backticks=False)
        except ImportError:
            pass
        except SyntaxError:
            raise

    if publicize:
        text, _needs_assign = _publicize_sentinels(text)
    return text


def transform_cell(
    source: str,
    *,
    known: set[str] | None = None,
    with_plot3: bool = True,
    cell_no: int = 0,
    warnings: list[str] | None = None,
    needs_assign_flag: list[bool] | None = None,
) -> str | None:
    """Transform one notebook cell.

    Returns rewritten source, or ``None`` to omit the cell from the export.
    """
    warn = warnings if warnings is not None else []
    directives, _args, body = _parse_directives(source)
    if "skip" in directives:
        return None

    body = body.strip("\n")
    if not body.strip():
        return None

    magic_out = _handle_magic_cell(body, warn, cell_no)
    if magic_out is not None:
        if not magic_out.strip():
            return None
        body = magic_out

    try:
        text = transform_source(
            body, known=known, with_plot3=with_plot3, publicize=False
        )
        text, needs_assign = _publicize_sentinels(text)
        if needs_assign and needs_assign_flag is not None:
            needs_assign_flag.append(True)
        return text
    except SyntaxError as e:
        warn.append(
            f"cell {cell_no}: could not transform ({e.msg}); exported raw "
            "(may need manual fix)"
        )
        return body


def _load_notebook(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "cells" not in data:
        raise ValueError(f"not a Jupyter notebook: {path}")
    return list(data["cells"])


def _cell_source(cell: dict[str, Any]) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def _header(
    nb_path: Path | None,
    *,
    only_export: bool,
    needs_assign: bool = False,
) -> str:
    src = nb_path.as_posix() if nb_path is not None else "<string>"
    mode = "only #| export cells" if only_export else "all code cells (except #| skip)"
    lines = [
        f"# %% Auto-exported by tidy3.nb_export from {src}",
        f"# Mode: {mode}",
        "# R-style bare names / backticks rewritten to plain CPython.",
        "# Re-export from the notebook rather than hand-editing this file.",
        "",
    ]
    if needs_assign:
        lines.extend(
            [
                "# Helper for exported `new col` = expr renames / mutates",
                "from tidy3.masking import make_named_assign as __tidy3_assign__  # noqa: F401,E402",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def nb_export(
    nb_path: str | Path,
    dest: str | Path | None = None,
    *,
    only_export: bool = False,
    with_plot3: bool = True,
    known_extra: Sequence[str] | None = None,
    write: bool = True,
) -> ExportResult | str:
    """Export a Jupyter notebook to a plain Python script.

    Parameters
    ----------
    nb_path:
        Path to ``.ipynb`` notebook.
    dest:
        Output ``.py`` path. Default: same stem next to the notebook
        (``analysis.ipynb`` → ``analysis.py``).
    only_export:
        If True, only cells marked ``#| export`` (nbdev-style) are included.
        If False (default), all code cells are included except ``#| skip``.
    with_plot3:
        Also apply plot3 ``aes`` / ``facet_wrap`` masking when plot3 is installed.
    known_extra:
        Extra names treated as non-columns (frame variables, helpers).
    write:
        If False, return the script text instead of writing a file.

    Returns
    -------
    ExportResult
        When ``write=True`` (default).
    str
        Script source when ``write=False``.
    """
    path = Path(nb_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in {".ipynb", ".json"}:
        raise ValueError(f"nb_export expects a .ipynb file (got {path.suffix!r})")

    cells = _load_notebook(path)
    code_cells = [c for c in cells if c.get("cell_type") == "code"]

    # Pre-scan sources for known names (imports / assigns), using raw cells
    raw_bodies: list[str] = []
    for c in code_cells:
        _dirs, _, body = _parse_directives(_cell_source(c))
        if "skip" in _dirs:
            continue
        raw_bodies.append(body)
    known = collect_known_names(raw_bodies)
    if known_extra:
        known.update(known_extra)

    warnings: list[str] = []
    exported_parts: list[str] = []
    n_exported = 0
    n_skipped = 0
    needs_assign_flag: list[bool] = []

    for i, cell in enumerate(code_cells, start=1):
        src = _cell_source(cell)
        directives, _, _ = _parse_directives(src)
        if "skip" in directives:
            n_skipped += 1
            continue
        if only_export and "export" not in directives:
            n_skipped += 1
            continue

        out = transform_cell(
            src,
            known=known,
            with_plot3=with_plot3,
            cell_no=i,
            warnings=warnings,
            needs_assign_flag=needs_assign_flag,
        )
        if out is None or not out.strip():
            n_skipped += 1
            continue

        # Newly assigned names in this cell become non-columns for later cells.
        known = collect_known_names(raw_bodies + exported_parts + [out])
        if known_extra:
            known.update(known_extra)

        exported_parts.append(out.rstrip() + "\n")
        n_exported += 1

    script = _header(
        path, only_export=only_export, needs_assign=bool(needs_assign_flag)
    )
    if exported_parts:
        script += "\n\n".join(p.rstrip() for p in exported_parts) + "\n"
    else:
        script += "# (no cells exported)\n"
        warnings.append("no cells exported — check #| export / #| skip filters")

    if not write:
        return script

    out_path = (
        Path(dest).expanduser().resolve()
        if dest is not None
        else path.with_suffix(".py")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script, encoding="utf-8")
    return ExportResult(
        path=out_path,
        cells_seen=len(code_cells),
        cells_exported=n_exported,
        cells_skipped=n_skipped,
        warnings=warnings,
    )
