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
import io
import re
import tokenize
from dataclasses import dataclass, field
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


_OPENERS = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = set(_OPENERS.values())
_SKIP = {
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
}


@dataclass
class _Statement:
    """One logical line: first and last physical rows, and its tokens."""

    first: int
    last: int
    tokens: list[tokenize.TokenInfo] = field(default_factory=list)


def _statements(text: str) -> list[_Statement] | None:
    """Split *text* into logical lines; ``None`` if it cannot be tokenized.

    Lines that start with ``>>`` or ``+`` are dedented first, so indented
    pipe steps (``    >> filter(...)``) never trip indentation checks.
    """
    analysed = "\n".join(
        line.lstrip() if line.lstrip().startswith((">>", "+")) else line
        for line in text.split("\n")
    )
    statements: list[_Statement] = []
    current: _Statement | None = None
    try:
        for token in tokenize.generate_tokens(io.StringIO(analysed).readline):
            if token.type in _SKIP:
                if token.type == tokenize.NEWLINE and current is not None:
                    statements.append(current)
                    current = None
                continue
            if current is None:
                current = _Statement(token.start[0], token.end[0])
            current.last = token.end[0]
            current.tokens.append(token)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    if current is not None:
        statements.append(current)
    return statements


def _has_top_level_pipe(tokens: list[tokenize.TokenInfo]) -> bool:
    depth = 0
    for token in tokens:
        if token.string in _OPENERS:
            depth += 1
        elif token.string in _CLOSERS:
            depth -= 1
        elif depth == 0 and token.string == ">>":
            return True
    return False


def _assigned_value(tokens: list[tokenize.TokenInfo]) -> tuple[int, int] | None:
    """Start of the value after a top-level ``=`` in the statement, if any."""
    depth = 0
    for index, token in enumerate(tokens):
        if token.string in _OPENERS:
            depth += 1
        elif token.string in _CLOSERS:
            depth -= 1
        elif depth == 0 and token.type == tokenize.OP and token.string == "=":
            following = tokens[index + 1 : index + 2]
            return following[0].start if following else None
    return None


def _wrap_pipe_statements(text: str) -> str | None:
    """Parenthesise each multi-line pipe in *text*; ``None`` if that fails.

    A logical line that starts with ``>>`` continues the statement above it,
    across blank and comment lines. A line that starts with ``+`` continues
    it too when the statement already holds a pipe (plot3 layers). Only
    statements at the top level of the cell are joined. Brackets are added
    on existing lines, so line numbers in errors match the cell as typed.
    """
    statements = _statements(text)
    if not statements:
        return None
    groups: list[list[_Statement]] = []
    for statement in statements:
        lead = statement.tokens[0]
        previous = groups[-1] if groups else None
        joins = (
            previous is not None
            and previous[0].tokens[0].start[1] == 0
            and lead.type == tokenize.OP
            and (
                lead.string == ">>"
                or (
                    lead.string == "+"
                    and _has_top_level_pipe(
                        [t for s in previous for t in s.tokens]
                    )
                )
            )
        )
        if joins:
            previous.append(statement)
        else:
            groups.append([statement])

    lines = text.split("\n")
    # Columns in tokens are for the dedented text; map them back.
    shift = [0] + [len(line) - len(line.lstrip()) for line in lines]

    def column(row: int, col: int) -> int:
        stripped = lines[row - 1].lstrip()
        return col + shift[row] if stripped.startswith((">>", "+")) else col

    edits: list[tuple[int, int, str]] = []  # (row, column, text), 1-based rows
    for group in groups:
        if len(group) == 1:
            continue
        head = group[0].tokens
        if head[0].string in {">>", "+"}:
            return None
        opening = _assigned_value(head) or head[0].start
        closing = group[-1].tokens[-1].end
        edits.append((opening[0], column(*opening), "("))
        edits.append((closing[0], column(*closing), ")"))
    if not edits:
        return None
    # Apply right to left so earlier columns stay valid.
    for row, col, piece in sorted(edits, reverse=True):
        line = lines[row - 1]
        lines[row - 1] = line[:col] + piece + line[col:]
    rewritten = "\n".join(lines)
    try:
        ast.parse(rewritten)
    except SyntaxError:
        return None
    return rewritten


def maybe_rewrite_cell(source: str) -> str | None:
    """If *source* is invalid Python but a tidy3 pipe, return rewritten source.

    Returns ``None`` when no rewrite is needed (already valid, or not a pipe).
    Safe for IPython input transformers: never rewrites ordinary Python.

    Backticks are normalized first (`` `hp new` `` → ``__tidy3_bt__("hp new")``)
    so multi-line pipes with spaced column names can still wrap in parentheses.
    """
    if not source or not source.strip():
        return None
    text = source.strip("\n")
    # Never touch magic cells
    stripped = text.lstrip()
    if stripped.startswith("%") or stripped.startswith("!") or stripped.startswith("?"):
        return None

    # Backtick preparser (harmless if no backticks)
    if "`" in text:
        from tidy3.masking import rewrite_backticks

        text = rewrite_backticks(text)

    if ">>" not in text:
        # Only backticks changed — still return rewrite so transformers apply.
        if text != source.strip("\n"):
            return text + ("\n" if source.endswith("\n") else "")
        return None

    # Already valid module code — leave alone (after backtick rewrite check)
    try:
        ast.parse(text)
        if "`" in source:
            return text + ("\n" if not text.endswith("\n") else "")
        return None
    except SyntaxError:
        pass

    if not looks_like_tidy_pipe(text):
        return None

    # A cell may hold comments, other statements, and several pipes: wrap
    # each pipe on its own and leave every other line as written.
    statements = _wrap_pipe_statements(text)
    if statements is not None:
        leading = source[: len(source) - len(source.lstrip("\n"))]
        return leading + statements + "\n"

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
    # Backticks + pipe parens (same as IPython source transformers).
    text = source
    if "`" in text:
        from tidy3.masking import rewrite_backticks

        text = rewrite_backticks(text)
    code = normalize_pipe_source(text)

    if namespace is None:
        ns: dict[str, Any] = _default_namespace()
    else:
        ns = dict(namespace)
        if inject_api:
            base = _default_namespace()
            for k, v in base.items():
                ns.setdefault(k, v)
    # Sentinel + col for R-style AST masking when eval'd outside full IPython.
    try:
        from tidy3.expr import col
        from tidy3.masking import BT_NAME, COL_NAME, apply_masking, default_known_names

        ns.setdefault(COL_NAME, col)
        ns.setdefault(BT_NAME, col)
        from tidy3.expr import name_ref
        from tidy3.masking import NAME_REF

        ns.setdefault(NAME_REF, name_ref)
        import tidy3
        from tidy3.masking import API_NAME

        ns.setdefault(API_NAME, tidy3)
        code = apply_masking(code, known=default_known_names(set(ns)))
    except SyntaxError:
        pass

    return eval(compile(code, "<tidy3.partial_run>", "eval"), ns, ns)
