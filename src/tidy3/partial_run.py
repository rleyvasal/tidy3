"""Partial pipeline run — normalize pipe source for any Jupyter kernel.

Enables R-style workflows without a VS Code extension. When the frontend
sends selected or cell text to the kernel (SolveIt, JupyterLab, classic
notebook, …), multi-line::

    tidy(cars)
    >> filter(col("mpg") > 20)

is rewritten into a valid expression so the **filtered** intermediate displays.

Also used by ``partial_run()`` and ``%%tidy3_run``.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Mapping, MutableMapping


_TRAILING_PIPE = re.compile(r">>\s*$")
_SOURCE_START = re.compile(
    r"^\s*("
    r"tidy\s*\("
    r"|scan_(?:parquet|csv|ipc)\s*\("
    r"|_\b"
    r"|[A-Za-z_][A-Za-z0-9_]*"
    r")"
)
# Multi-line tidy pipe: has >> and looks like dplyr chaining
_HAS_PIPE = re.compile(r"^\s*>>", re.MULTILINE)


def looks_like_tidy_pipe(source: str) -> bool:
    """True if *source* looks like a tidy3 ``>>`` pipe (possibly multi-line)."""
    if not source or ">>" not in source:
        return False
    text = source.strip()
    if not text or text.startswith("%") or text.startswith("!"):
        return False
    # Ignore pure comparisons / bit shifts that are not pipes (heuristic)
    if not _HAS_PIPE.search(text) and "\n" not in text:
        # one-liner: require tidy/scan/_/name then >>
        if not re.search(r"(tidy\s*\(|scan_\w+\s*\(|\b_\b|[A-Za-z_]\w*)\s*>>", text):
            return False
    return True


def _validate_pipe_start(text: str) -> None:
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    if first.lstrip().startswith(">>"):
        raise ValueError(
            "selection starts with '>>' — select from the start of the pipe "
            "(include tidy(...), a frame name, or _)."
        )
    if first.lstrip().startswith("."):
        raise ValueError(
            "selection starts with a leading '.' method — partial multi-line "
            "method chains are not supported; use >> pipes or select from tidy(...)."
        )
    if not _SOURCE_START.match(first):
        raise ValueError(
            "selection does not look like a tidy3 pipe start "
            f"(got {first.strip()!r}). Include tidy(...), scan_*, a name, or _."
        )


def normalize_pipe_source(source: str, *, validate_start: bool = True) -> str:
    """Turn multi-line ``tidy … >> verb`` text into one evaluable expression.

    Rules
    -----
    1. Strip whitespace; drop a trailing incomplete ``>>``.
    2. If not already a single expression, wrap in parentheses.
    """
    if source is None:
        raise TypeError("source must be a string")
    text = source.strip()
    if not text:
        raise ValueError("empty selection — nothing to run")

    text = _TRAILING_PIPE.sub("", text).rstrip()
    if not text:
        raise ValueError("selection is only a trailing >>")

    if validate_start:
        _validate_pipe_start(text)

    try:
        ast.parse(text, mode="eval")
        return text
    except SyntaxError:
        pass

    wrapped = f"(\n{text}\n)"
    try:
        ast.parse(wrapped, mode="eval")
    except SyntaxError as e:
        raise SyntaxError(
            "could not parse pipe selection as an expression after wrapping "
            f"in parentheses: {e.msg}"
        ) from e
    return wrapped


def _split_assignment(text: str) -> tuple[str, str] | None:
    """If first line is ``name = …`` and body is a pipe, return (target, rhs)."""
    lines = text.splitlines()
    if not lines:
        return None
    first = lines[0]
    if first.lstrip().startswith(("#", "%", "!")):
        return None
    # Only simple targets: name or tuple of names
    m = re.match(
        r"^(\s*)([A-Za-z_][A-Za-z0-9_]*|\([^)]+\))\s*=\s*(.*)$",
        first,
    )
    if not m:
        return None
    indent, target, rhs_first = m.group(1), m.group(2), m.group(3)
    rest = "\n".join(lines[1:])
    rhs = rhs_first if not rest.strip() else (rhs_first + "\n" + rest).strip()
    if ">>" not in rhs:
        return None
    return f"{indent}{target}", rhs


def maybe_rewrite_cell(source: str) -> str | None:
    """If *source* is invalid Python but a tidy3 pipe, return rewritten source.

    Returns ``None`` when no rewrite is needed (already valid, or not a pipe).
    Safe for IPython input transformers: never rewrites ordinary Python.
    """
    if not source or not source.strip():
        return None
    text = source.strip("\n")
    # Never touch magic cells
    stripped = text.lstrip()
    if stripped.startswith("%") or stripped.startswith("!") or stripped.startswith("?"):
        return None
    if ">>" not in text:
        return None

    # Already valid module code — leave alone
    try:
        ast.parse(text)
        return None
    except SyntaxError:
        pass

    if not looks_like_tidy_pipe(text):
        return None

    # Assignment form: out = tidy(df)\n>> filter(...)
    split = _split_assignment(text)
    if split is not None:
        target, rhs = split
        try:
            norm = normalize_pipe_source(rhs, validate_start=True)
        except (ValueError, SyntaxError):
            return None
        return f"{target} = {norm}\n"

    # Bare expression pipe / partial selection
    try:
        norm = normalize_pipe_source(text, validate_start=True)
    except (ValueError, SyntaxError):
        return None
    return norm + "\n"


def _default_namespace() -> dict[str, Any]:
    """Namespace with tidy3 public API so snippets need fewer imports."""
    import tidy3 as t3

    ns: dict[str, Any] = {"__builtins__": __builtins__}
    for name in t3.__all__:
        if name.startswith("_"):
            continue
        try:
            ns[name] = getattr(t3, name)
        except AttributeError:
            pass
    ns["tidy3"] = t3
    return ns


def partial_run(
    source: str,
    namespace: Mapping[str, Any] | MutableMapping[str, Any] | None = None,
    *,
    inject_api: bool = True,
) -> Any:
    """Evaluate a pipe **prefix** and return the intermediate value.

    Parameters
    ----------
    source:
        Selected or cell code, e.g. multi-line ``tidy(cars)`` + ``>> filter(...)``.
        Outer parentheses optional.
    namespace:
        Eval namespace (IPython ``user_ns`` / ``globals()``). Frame names like
        ``cars`` must already be bound.
    inject_api:
        Inject tidy3 public names when missing.

    Returns
    -------
    Typically a :class:`~tidy3.frame.TidyFrame` whose display shows a limited
    preview of the intermediate (e.g. filtered rows only).
    """
    code = normalize_pipe_source(source)

    if namespace is None:
        ns: dict[str, Any] = _default_namespace()
    else:
        ns = dict(namespace)
        if inject_api:
            base = _default_namespace()
            for k, v in base.items():
                ns.setdefault(k, v)

    return eval(compile(code, "<tidy3.partial_run>", "eval"), ns, ns)
