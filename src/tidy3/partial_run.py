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


def _line_starts(lines: list[str]) -> list[tuple[int, bool]]:
    """Bracket depth, and whether a string is open, where each line starts.

    A light scan (comments, quotes, triple quotes, escapes), enough to tell
    which lines start a statement before the cell can be tokenized.
    """
    starts: list[tuple[int, bool]] = []
    depth, quote = 0, ""
    for line in lines:
        starts.append((depth, bool(quote)))
        i = 0
        while i < len(line):
            if quote:
                if line[i] == "\\":
                    i += 2
                elif line.startswith(quote, i):
                    i += len(quote)
                    quote = ""
                else:
                    i += 1
                continue
            ch = line[i]
            if ch == "#":
                break
            if ch in "\"'":
                quote = line[i : i + 3] if line[i : i + 3] in ('"""', "'''") else ch
                i += len(quote)
                continue
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(depth - 1, 0)
            i += 1
        if quote in ('"', "'"):  # an unclosed one-line string ends here
            quote = ""
    return starts


def _continuation_indents(text: str) -> list[int | None]:
    """For each line that continues a pipe (it starts with ``>>`` or ``+``
    outside brackets), the indentation of the statement it continues;
    ``None`` for other lines.

    At the top level of a cell that is 0. Inside a function or another
    block it is the block's indentation, so re-indented steps keep the
    block's structure when the cell is tokenized.
    """
    lines = text.split("\n")
    out: list[int | None] = []
    anchor = 0
    for line, (depth, in_string) in zip(lines, _line_starts(lines)):
        stripped = line.lstrip()
        if depth or in_string or not stripped or stripped.startswith("#"):
            out.append(None)
            continue
        if stripped.startswith((">>", "+")):
            out.append(anchor)
            continue
        anchor = len(line) - len(stripped)
        out.append(None)
    return out


def _statements(text: str) -> list[_Statement] | None:
    """Split *text* into logical lines; ``None`` if it cannot be tokenized.

    Lines that start with ``>>`` or ``+`` are re-indented first: to the
    indentation of the statement they continue (column 0 at the top of a
    cell, the block's indentation inside a function), or to column 0 inside
    brackets, so indented pipe steps (``    >> filter(...)``) never trip
    indentation checks.
    """
    indents = _continuation_indents(text)
    analysed = "\n".join(
        (" " * indent + line.lstrip()) if indent is not None
        else (line.lstrip() if line.lstrip().startswith((">>", "+")) else line)
        for line, indent in zip(text.split("\n"), indents)
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
    it too when the statement already holds a pipe (plot3 layers). Pipes
    are joined at the top level of the cell and inside functions, loops and
    other blocks. Brackets are added on existing lines, so line numbers in
    errors match the cell as typed.
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
            # Steps were re-indented to the statement they continue.
            and previous[0].tokens[0].start[1] == lead.start[1]
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
    # Columns in tokens are for the re-indented text; map them back.
    indents = _continuation_indents(text)

    def column(row: int, col: int) -> int:
        line = lines[row - 1]
        stripped = line.lstrip()
        if not stripped.startswith((">>", "+")):
            return col
        given = len(line) - len(stripped)
        return col - (indents[row - 1] or 0) + given

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


def join_broken_pipe(source: str) -> str | None:
    """Join a broken pipe's lines with ``\\`` so Python reports the real error.

    :func:`maybe_rewrite_cell` will not wrap a pipe that holds a mistake (an
    unclosed bracket, a missing argument), so Python would stop at the first
    indented ``>>`` with ``IndentationError: unexpected indent``. Joined
    with backslashes, the pipe reaches Python as one statement and the error
    names the real line and column. Comments on the joined lines are dropped,
    since nothing may follow a backslash. Line numbers are unchanged.

    Returns ``None`` when *source* parses, is not a pipe, or has nothing to
    join. Like :func:`_wrap_pipe_statements`, pipes are joined at the top
    level and inside blocks.
    """
    if not source or ">>" not in source:
        return None
    text = source.strip("\n")
    stripped = text.lstrip()
    if stripped.startswith(("%", "!", "?")) or not looks_like_tidy_pipe(text):
        return None
    try:
        ast.parse(text)
        return None
    except SyntaxError:
        pass

    lines = text.split("\n")
    shift = [
        len(line) - len(line.lstrip())
        if line.lstrip().startswith((">>", "+"))
        else 0
        for line in lines
    ]
    analysed = "\n".join(line[cut:] for line, cut in zip(lines, shift))
    # Per 0-based row: bracket depth after the row, and where its comment
    # starts. Rows the tokenizer never reached stay unknown.
    end_depth: dict[int, int] = {}
    comment: dict[int, int] = {}
    in_string: set[int] = set()  # rows a multi-line string runs past
    depth = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO(analysed).readline):
            row = token.start[0] - 1
            if token.string in _OPENERS:
                depth += 1
            elif token.string in _CLOSERS:
                depth = max(depth - 1, 0)  # Python reports the stray closer
            elif token.type == tokenize.COMMENT:
                comment[row] = token.start[1] + shift[row]
            in_string.update(range(row, token.end[0] - 1))
            end_depth[token.end[0] - 1] = depth
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass

    joins: list[int] = []  # rows that get a trailing backslash
    previous: int | None = None  # last code row
    # ``top``: the current statement starts outside brackets, at any
    # indentation (top level of the cell, or inside a block).
    top = piped = False
    start_depth = 0
    for row, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        continues = start_depth == 0 and (
            stripped.startswith(">>") or (stripped.startswith("+") and piped)
        )
        if continues and top and previous is not None:
            joins.extend(range(previous, row))
        elif start_depth == 0 and not continues:
            top, piped = True, False
        piped = piped or ">>" in line
        if row not in end_depth or row in in_string:
            break
        previous, start_depth = row, end_depth[row]

    # IPython runs transformers twice (cleanup and post), so skip rows a
    # first pass already joined.
    joins = [
        row
        for row in joins
        if not lines[row][: comment.get(row, len(lines[row]))].rstrip().endswith("\\")
    ]
    if not joins:
        return None
    for row in joins:
        line = lines[row]
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            lines[row] = "\\"
        else:
            lines[row] = line[: comment.get(row, len(line))].rstrip() + " \\"
    leading = source[: len(source) - len(source.lstrip("\n"))]
    return leading + "\n".join(lines) + "\n"


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
        from tidy3.masking import ASSIGN_NAME, make_named_assign

        ns.setdefault(ASSIGN_NAME, make_named_assign)
        import tidy3
        from tidy3.masking import API_NAME

        ns.setdefault(API_NAME, tidy3)
        code = apply_masking(code, known=default_known_names(set(ns)))
    except SyntaxError:
        pass

    return eval(compile(code, "<tidy3.partial_run>", "eval"), ns, ns)
