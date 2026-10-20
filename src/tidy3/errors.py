"""Readable errors: a missing column names its verb, and a failed preview is brief.

A verb that uses a column the frame lacks raises :class:`ColumnNotFoundError`
when the pipe is built, even on a lazy Polars frame: the plan's schema is
resolved without reading any rows. Any other error keeps its own type and
message. A preview that still fails at run time (a bad cast, say) shows the
error's type and first line, with the full text one step away.
"""

from __future__ import annotations

import difflib
import html as _html
import re
from typing import Any

import polars as pl

_MAX_COLUMNS = 8
_POLARS_MISSING = (
    re.compile(r'unable to find column "([^"]+)"'),
    re.compile(r'^"([^"\n]+)" not found'),
)
_TIDY3_MISSING = re.compile(r"^columns? not found: \[?'([^']+)'")


class ColumnNotFoundError(KeyError, pl.exceptions.ColumnNotFoundError):
    """A verb used a column the frame does not have.

    A ``KeyError`` (as pandas raises) and a Polars ``ColumnNotFoundError``,
    so existing ``except`` clauses for either still catch it.
    """

    def __init__(self, verb: str, column: str, columns: list[str]):
        super().__init__(column)
        self.verb = verb
        self.column = column
        self.columns = list(columns)

    def __reduce__(self) -> Any:
        return (type(self), (self.verb, self.column, self.columns))

    def __str__(self) -> str:
        lines = [f"{self.verb}(): column {self.column!r} does not exist."]
        close = suggest(self.column, self.columns)
        if close:
            lines.append("Did you mean " + " or ".join(repr(c) for c in close) + "?")
        shown = ", ".join(repr(c) for c in self.columns[:_MAX_COLUMNS])
        hidden = len(self.columns) - _MAX_COLUMNS
        if hidden > 0:
            shown += f", … ({hidden} more)"
        lines.append(f"Columns: {shown}" if self.columns else "The frame has no columns.")
        return "\n".join(lines)


def suggest(name: str, columns: list[str], n: int = 3) -> list[str]:
    """Columns *name* was probably meant to be: same letters, contained, or close."""
    low = name.lower()
    same = [c for c in columns if c.lower() == low]
    contains = [
        c for c in columns
        if len(low) >= 3 and (low in c.lower() or (len(c) >= 3 and c.lower() in low))
    ]
    close = difflib.get_close_matches(name, columns, n=n, cutoff=0.6)
    out: list[str] = []
    for column in same + contains + close:
        if column not in out and column != name:
            out.append(column)
    return out[:n]


def missing_column(error: BaseException, columns: list[str], verb: str = "") -> str | None:
    """The column *error* says is missing, or None when it is about something else."""
    if isinstance(error, ColumnNotFoundError):
        return error.column
    if isinstance(error, pl.exceptions.ColumnNotFoundError):
        text = str(error)
        for pattern in _POLARS_MISSING:
            found = pattern.search(text)
            if found:
                return found.group(1)
        return None
    if not isinstance(error, KeyError) or "join" in verb:
        # Join errors may be about the other frame's columns.
        return None
    if len(error.args) != 1 or not isinstance(error.args[0], str):
        return None
    key = error.args[0]
    found = _TIDY3_MISSING.search(key)
    if found:
        key = found.group(1)
    elif "\n" in key or "not found" in key or "():" in key:
        return None
    return None if key in columns else key


def column_error(verb: str, error: BaseException, columns: list[str]) -> ColumnNotFoundError | None:
    """*error* as a :class:`ColumnNotFoundError` for *verb*, when it is one."""
    column = missing_column(error, columns, verb)
    if column is None:
        return None
    return ColumnNotFoundError(verb, column, columns)


# ── previews ────────────────────────────────────────────────────────────

_MAX_SUMMARY = 6
_MAX_LINE = 300


def _summary_lines(error: BaseException) -> list[str]:
    """The error's opening paragraph: what went wrong, without the query plan."""
    lines: list[str] = []
    for line in str(error).strip().splitlines():
        if not line.strip():
            break
        lines.append(line if len(line) <= _MAX_LINE else line[:_MAX_LINE] + " …")
    if len(lines) > _MAX_SUMMARY:
        lines = lines[:_MAX_SUMMARY] + ["…"]
    return lines


def _headline(error: BaseException) -> list[str]:
    lines = _summary_lines(error) or [""]
    return [f"{type(error).__name__}: {lines[0]}".rstrip(": "), *lines[1:]]


def preview_error_text(error: BaseException) -> str:
    """Plain-text preview of a frame whose rows could not be computed."""
    out = ["TidyFrame (preview failed)", *_headline(error)]
    if str(error).strip().splitlines() != _summary_lines(error):
        out.append("collect() raises the full error.")
    return "\n".join(out)


_MONO = "12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def preview_error_html(error: BaseException, plan: str | None = None) -> str:
    """HTML preview of a failed frame; the full error and plan are folded away."""
    esc = _html.escape
    headline = "<br>".join(esc(line) for line in _headline(error))
    details = esc(str(error).strip())
    if plan:
        details += "\n\nQuery plan:\n" + esc(plan)
    return (
        f"<div style='font:{_MONO}'>"
        "<div style='opacity:.85;margin:0 0 2px 0'>TidyFrame (preview failed)</div>"
        f"<div style='color:#c0392b;white-space:pre-wrap'>{headline}</div>"
        "<details style='margin-top:4px'>"
        "<summary style='cursor:pointer;opacity:.7'>Technical details</summary>"
        f"<pre style='font:{_MONO};white-space:pre-wrap;margin:4px 0'>{details}</pre>"
        "</details></div>"
    )
