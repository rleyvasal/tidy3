"""stringr for tidy3: ``str_*`` helpers that run inside verbs on either backend.

Each helper builds a backend-neutral :class:`~tidy3.expr.Expr`; the Polars
compiler and the pandas engine call :func:`compile_polars` and
:func:`eval_pandas` here for any ``str_*`` node. A plain string as the
first argument is a column name, as in the other helpers (``mean("x")``);
in a notebook, bare column names work as in R. A list, tuple, or Series
instead runs at once and returns a list, like stringr on a character
vector. Patterns are regular expressions unless wrapped in :func:`fixed`;
:func:`regex` adds flags such as ``ignore_case``. Missing values stay
missing, as in stringr.
"""

from __future__ import annotations

import builtins
import math
import re
import string as _string_module
import sys
import textwrap
import unicodedata
from typing import Any

import pandas as pd
import polars as pl

from tidy3.expr import Expr, _func, _node, col, to_polars

__all__ = [
    "fixed", "regex", "coll", "boundary",
    "str_detect", "str_starts", "str_ends", "str_count", "str_like",
    "str_ilike", "str_equal", "str_subset", "str_which",
    "str_locate", "str_locate_all", "invert_match",
    "str_length", "str_width", "str_sub", "str_sub_all", "str_trunc", "word",
    "str_replace", "str_replace_all", "str_remove", "str_remove_all",
    "str_replace_na",
    "str_to_upper", "str_to_lower", "str_to_title", "str_to_sentence",
    "str_trim", "str_squish", "str_pad", "str_dup", "str_wrap",
    "str_extract", "str_extract_all", "str_match", "str_match_all",
    "str_split", "str_split_1", "str_split_fixed", "str_split_i",
    "str_c", "str_flatten", "str_flatten_comma", "str_glue", "str_glue_data",
    "str_sort", "str_order", "str_unique", "str_escape", "str_conv", "str_view",
]


# ── patterns ────────────────────────────────────────────────────────────


class StringPattern:
    """A pattern with its matching rules, from :func:`fixed`, :func:`regex`,
    :func:`coll`, or :func:`boundary`."""

    __slots__ = ("pattern", "literal", "flags", "boundary")

    def __init__(
        self, pattern: str, literal: bool, flags: str = "", boundary: bool = False
    ):
        if not isinstance(pattern, str):
            raise TypeError("a pattern must be a string")
        self.pattern = pattern
        self.literal = literal
        self.flags = flags
        self.boundary = boundary

    def __repr__(self) -> str:
        kind = "fixed" if self.literal else "boundary" if self.boundary else "regex"
        flag = f", flags={self.flags!r}" if self.flags else ""
        return f"{kind}({self.pattern!r}{flag})"


def fixed(pattern: str, ignore_case: bool = False) -> StringPattern:
    """Match *pattern* as literal text, not a regular expression (stringr's ``fixed()``)."""
    return StringPattern(pattern, True, "i" if ignore_case else "")


def coll(pattern: str, ignore_case: bool = False, locale: str = "en") -> StringPattern:
    """Literal text, like :func:`fixed`; there is no locale-aware collation."""
    return fixed(pattern, ignore_case)


def regex(
    pattern: str,
    ignore_case: bool = False,
    multiline: bool = False,
    comments: bool = False,
    dotall: bool = False,
) -> StringPattern:
    """A regular expression with stringr's flags (stringr's ``regex()``)."""
    flags = "".join(
        flag
        for flag, on in (
            ("i", ignore_case), ("m", multiline), ("x", comments), ("s", dotall)
        )
        if on
    )
    return StringPattern(pattern, False, flags)


_BOUNDARIES = {"character": r"(?s:.)", "word": r"\w+(?:['’]\w+)*"}


def boundary(type: str = "character") -> StringPattern:  # noqa: A002
    """Characters or words as the pattern (stringr's ``boundary()``).

    Splitting on a boundary returns its pieces: ``str_split(x, boundary("word"))``
    gives the words. Only ``"character"`` and ``"word"`` are supported.
    """
    if type not in _BOUNDARIES:
        raise NotImplementedError(
            'boundary() supports type="character" and type="word"'
        )
    return StringPattern(_BOUNDARIES[type], False, "", True)


def _pattern_kwargs(name: str, pattern: Any) -> dict[str, Any]:
    if isinstance(pattern, str):
        pattern = regex(pattern)
    if not isinstance(pattern, StringPattern):
        raise TypeError(
            f"{name}() pattern must be a string, fixed(...) or regex(...)"
        )
    return {
        "pattern": pattern.pattern,
        "literal": pattern.literal,
        "flags": pattern.flags,
        "boundary": pattern.boundary,
    }


def _split_kwargs(name: str, pattern: Any) -> dict[str, Any]:
    # stringr splits on "" into characters.
    if isinstance(pattern, str) and pattern == "":
        pattern = boundary("character")
    return _pattern_kwargs(name, pattern)


# Characters with a meaning in both Python's ``re`` and Rust's ``regex``.
_SPECIAL = re.compile(r"([\\.^$|?*+()\[\]{}])")


def _regex(kw: dict, wrap: str = "{}") -> str:
    """The pattern as one regex string that Polars and Python's re both read."""
    body = _SPECIAL.sub(r"\\\1", kw["pattern"]) if kw["literal"] else kw["pattern"]
    flags = f"(?{kw['flags']})" if kw["flags"] else ""
    return flags + wrap.format(body)


def _plain(kw: dict) -> bool:
    """True when the pattern is plain text matched case-sensitively."""
    return (
        not kw["flags"]
        and not kw["boundary"]
        and kw["pattern"] != ""
        and (bool(kw["literal"]) or not _SPECIAL.search(kw["pattern"]))
    )


def _compiled(kw: dict) -> re.Pattern:
    return re.compile(_regex(kw))


def _group_names(name: str, kw: dict) -> list[str]:
    try:
        compiled = _compiled(kw)
    except re.error as error:
        raise ValueError(f"{name}() could not read the pattern: {error}") from None
    labels = {index: label for label, index in compiled.groupindex.items()}
    # A list, not a tuple: tuples inside expression nodes are nodes.
    return [labels.get(i, str(i)) for i in range(1, compiled.groups + 1)]


# ── building expressions ────────────────────────────────────────────────

_X = "__tidy3_string__"
_AGGREGATES = {"str_flatten"}


def _is_vector(x: Any) -> bool:
    if isinstance(x, (str, Expr, pl.Expr)):
        return False
    return isinstance(x, (list, tuple, pd.Series, pl.Series)) or (
        type(x).__name__ == "ndarray"
    )


def _values(x: Any) -> list:
    if isinstance(x, pl.Series):
        return x.to_list()
    return [None if pd.api.types.is_scalar(v) and pd.isna(v) else v for v in x]


def _eager(expr: Expr, columns: dict[str, Any], aggregate: bool = False) -> Any:
    """Run *expr* now on the vectors in *columns* and return a list (or a value)."""
    frame = pl.DataFrame({name: pl.Series(name, _values(v)) for name, v in columns.items()})
    out = frame.select(to_polars(expr).alias(_X))[_X].to_list()
    return out[0] if aggregate else out


def _string(x: Any) -> Expr:
    return col(x) if isinstance(x, str) else Expr(_node(x))


def _str(name: str, string: Any, *args: Any, **kwargs: Any) -> Any:
    if _is_vector(string):
        expr = _func(name, col(_X), *args, **kwargs)
        return _eager(expr, {_X: string}, aggregate=name in _AGGREGATES)
    return _func(name, _string(string), *args, **kwargs)


def _check_int(name: str, arg: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name}() {arg} must be an integer")


def _check_bool(name: str, arg: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name}() {arg} must be a boolean")


def _check_side(name: str, side: str, sides: tuple[str, ...]) -> None:
    if side not in sides:
        choices = ", ".join(repr(s) for s in sides)
        raise ValueError(f"{name}() side must be one of {choices}")


def _require_vector(name: str, string: Any, hint: str) -> list:
    if not _is_vector(string):
        raise TypeError(f"{name}() takes a list or Series of strings; {hint}")
    return _values(string)


# ── detect ──────────────────────────────────────────────────────────────


def str_detect(string: Any, pattern: Any, negate: bool = False) -> Any:
    """True where *pattern* matches anywhere in *string*."""
    return _str("str_detect", string, negate=negate, **_pattern_kwargs("str_detect", pattern))


def str_starts(string: Any, pattern: Any, negate: bool = False) -> Any:
    """True where *string* starts with *pattern*."""
    return _str("str_starts", string, negate=negate, **_pattern_kwargs("str_starts", pattern))


def str_ends(string: Any, pattern: Any, negate: bool = False) -> Any:
    """True where *string* ends with *pattern*."""
    return _str("str_ends", string, negate=negate, **_pattern_kwargs("str_ends", pattern))


def str_count(string: Any, pattern: Any) -> Any:
    """Number of non-overlapping matches of *pattern* in *string*."""
    return _str("str_count", string, **_pattern_kwargs("str_count", pattern))


def _like_regex(pattern: str) -> str:
    out, chars = [], iter(pattern)
    for char in chars:
        if char == "\\":
            out.append(re.escape(next(chars, "\\")))
        elif char == "%":
            out.append(".*")
        elif char == "_":
            out.append(".")
        else:
            out.append(_SPECIAL.sub(r"\\\1", char))
    return "^" + "".join(out) + "$"


def str_like(string: Any, pattern: str, ignore_case: bool = False) -> Any:
    """SQL ``LIKE``: ``%`` matches any run of characters, ``_`` any one; ``\\`` escapes."""
    if not isinstance(pattern, str):
        raise TypeError("str_like() pattern must be a string")
    flags = "si" if ignore_case else "s"
    return _str(
        "str_detect", string, negate=False, pattern=_like_regex(pattern),
        literal=False, flags=flags, boundary=False,
    )


def str_ilike(string: Any, pattern: str) -> Any:
    """Case-insensitive :func:`str_like`."""
    return str_like(string, pattern, ignore_case=True)


def str_equal(x: Any, y: Any, ignore_case: bool = False) -> Any:
    """True where *x* equals *y*; a plain string *y* is text, not a column."""
    _check_bool("str_equal", "ignore_case", ignore_case)
    if _is_vector(x) or _is_vector(y):
        columns = {_X: x}
        other: Any = y
        if _is_vector(y):
            columns["__tidy3_other__"] = y
            other = col("__tidy3_other__")
        return _eager(_func("str_equal", col(_X), other, ignore_case=ignore_case), columns)
    return _func("str_equal", _string(x), y, ignore_case=ignore_case)


def str_subset(string: Any, pattern: Any, negate: bool = False) -> list:
    """The strings that match *pattern* (in a pipe, use ``filter(str_detect(...))``)."""
    values = _require_vector("str_subset", string, "in a pipe use filter(str_detect(...))")
    keep = str_detect(values, pattern, negate=negate)
    return [value for value, found in zip(values, keep) if found is True]


def str_which(string: Any, pattern: Any, negate: bool = False) -> list:
    """1-based positions of the strings that match *pattern*, as in R."""
    values = _require_vector("str_which", string, "in a pipe use str_detect()")
    keep = str_detect(values, pattern, negate=negate)
    return [i + 1 for i, found in enumerate(keep) if found is True]


# ── locate ──────────────────────────────────────────────────────────────


def str_locate(string: Any, pattern: Any) -> Any:
    """1-based ``{"start", "end"}`` of the first match, as a struct column."""
    return _str("str_locate", string, **_pattern_kwargs("str_locate", pattern))


def str_locate_all(string: Any, pattern: Any) -> Any:
    """Every match's ``{"start", "end"}``, as a list column."""
    return _str("str_locate_all", string, **_pattern_kwargs("str_locate_all", pattern))


def invert_match(loc: Any) -> list[dict]:
    """The stretches between matches, from one row of :func:`str_locate_all`."""
    if not isinstance(loc, (list, tuple)):
        raise TypeError("invert_match() takes one row of str_locate_all()")
    starts = [0] + [m["end"] + 1 for m in loc]
    ends = [m["start"] - 1 for m in loc] + [-1]
    return [{"start": s, "end": e} for s, e in zip(starts, ends)]


# ── length and substrings ───────────────────────────────────────────────


def str_length(string: Any) -> Any:
    """Number of characters in *string*."""
    return _str("str_length", string)


def str_width(string: Any) -> Any:
    """Display width: wide East Asian characters count 2, combining marks 0."""
    return _str("str_width", string)


def str_sub(string: Any, start: int = 1, end: int = -1) -> Any:
    """Characters *start* to *end*, 1-based and inclusive; negatives count from the end."""
    _check_int("str_sub", "start", start)
    _check_int("str_sub", "end", end)
    return _str("str_sub", string, start=start, end=end)


def str_sub_all(string: Any, start: Any = 1, end: Any = -1) -> Any:
    """Several substrings of each string, as a list column.

    *start* and *end* are integers or equal-length sequences of them.
    """
    starts = list(start) if isinstance(start, (list, tuple)) else [start]
    ends = list(end) if isinstance(end, (list, tuple)) else [end]
    if len(starts) == 1:
        starts = starts * len(ends)
    if len(ends) == 1:
        ends = ends * len(starts)
    if len(starts) != len(ends):
        raise ValueError("str_sub_all() start and end must have the same length")
    for value in (*starts, *ends):
        _check_int("str_sub_all", "start and end", value)
    return _str("str_sub_all", string, starts=starts, ends=ends)


def str_trunc(
    string: Any, width: int, side: str = "right", ellipsis: str = "..."
) -> Any:
    """Shorten strings longer than *width*, marking the cut with *ellipsis*."""
    _check_int("str_trunc", "width", width)
    _check_side("str_trunc", side, ("right", "left", "center"))
    if width < len(ellipsis):
        raise ValueError("str_trunc() width must be at least the ellipsis length")
    return _str("str_trunc", string, width=width, side=side, ellipsis=ellipsis)


def word(string: Any, start: int = 1, end: int | None = None, sep: Any = fixed(" ")) -> Any:
    """Words *start* to *end* (1-based, negatives from the end) of *string*."""
    end = start if end is None else end
    _check_int("word", "start", start)
    _check_int("word", "end", end)
    return _str("str_word", string, start=start, end=end, **_pattern_kwargs("word", sep))


# ── replace ─────────────────────────────────────────────────────────────


def _replace(name: str, string: Any, pattern: Any, replacement: str) -> Any:
    if not isinstance(replacement, str):
        raise TypeError(f"{name}() replacement must be a string")
    return _str(
        name, string, replacement=replacement, **_pattern_kwargs(name, pattern)
    )


def str_replace(string: Any, pattern: Any, replacement: str) -> Any:
    """Replace the first match of *pattern*; ``\\1`` refers to a capture group."""
    return _replace("str_replace", string, pattern, replacement)


def str_replace_all(string: Any, pattern: Any, replacement: str | None = None) -> Any:
    """Replace every match of *pattern*; ``\\1`` refers to a capture group.

    A dict of ``{pattern: replacement}`` applies each in turn, like R's
    named vector: ``str_replace_all(x, {"one": "1", "two": "2"})``.
    """
    if isinstance(pattern, dict):
        if replacement is not None:
            raise TypeError("str_replace_all() takes a dict or a replacement, not both")
        if _is_vector(string):
            out: Any = _values(string)
            for key, value in pattern.items():
                out = str_replace_all(out, key, value)
            return out
        out = _string(string)
        for key, value in pattern.items():
            out = _replace("str_replace_all", out, key, value)
        return out
    if replacement is None:
        raise TypeError("str_replace_all() needs a replacement")
    return _replace("str_replace_all", string, pattern, replacement)


def str_remove(string: Any, pattern: Any) -> Any:
    """Remove the first match of *pattern*."""
    return _replace("str_replace", string, pattern, "")


def str_remove_all(string: Any, pattern: Any) -> Any:
    """Remove every match of *pattern*."""
    return _replace("str_replace_all", string, pattern, "")


def str_replace_na(string: Any, replacement: str = "NA") -> Any:
    """Turn missing values into *replacement* (``"NA"`` by default, as in R)."""
    if not isinstance(replacement, str):
        raise TypeError("str_replace_na() replacement must be a string")
    return _str("str_replace_na", string, replacement=replacement)


# ── case and whitespace ─────────────────────────────────────────────────


def str_to_upper(string: Any) -> Any:
    """Upper case."""
    return _str("str_to_upper", string)


def str_to_lower(string: Any) -> Any:
    """Lower case."""
    return _str("str_to_lower", string)


def str_to_title(string: Any) -> Any:
    """Title Case: each word capitalised."""
    return _str("str_to_title", string)


def str_to_sentence(string: Any) -> Any:
    """Sentence case: first character upper, the rest lower."""
    return _str("str_to_sentence", string)


def str_trim(string: Any, side: str = "both") -> Any:
    """Remove whitespace from the start, end, or both sides."""
    _check_side("str_trim", side, ("both", "left", "right"))
    return _str("str_trim", string, side=side)


def str_squish(string: Any) -> Any:
    """Trim both ends and collapse inner runs of whitespace to one space."""
    return _str("str_squish", string)


def str_pad(string: Any, width: int, side: str = "left", pad: str = " ") -> Any:
    """Pad *string* to *width* characters with *pad* on *side*."""
    _check_int("str_pad", "width", width)
    _check_side("str_pad", side, ("left", "right", "both"))
    if not isinstance(pad, str) or len(pad) != 1:
        raise ValueError("str_pad() pad must be a single character")
    return _str("str_pad", string, width=width, side=side, pad=pad)


def str_dup(string: Any, times: int) -> Any:
    """Repeat *string* *times* times."""
    _check_int("str_dup", "times", times)
    if times < 0:
        raise ValueError("str_dup() times must be zero or more")
    return _str("str_dup", string, times=times)


def str_wrap(
    string: Any,
    width: int = 80,
    indent: int = 0,
    exdent: int = 0,
    whitespace_only: bool = True,
) -> Any:
    """Wrap text into lines of at most *width* characters joined by ``\\n``."""
    for arg, value in (("width", width), ("indent", indent), ("exdent", exdent)):
        _check_int("str_wrap", arg, value)
    _check_bool("str_wrap", "whitespace_only", whitespace_only)
    return _str(
        "str_wrap", string, width=width, indent=indent, exdent=exdent,
        whitespace_only=whitespace_only,
    )


# ── extract, match, and split ───────────────────────────────────────────


def str_extract(string: Any, pattern: Any, group: int | None = None) -> Any:
    """The first match of *pattern*, or its capture *group*; missing when none."""
    if group is not None:
        _check_int("str_extract", "group", group)
        if group < 1:
            raise ValueError("str_extract() group must be 1 or more")
    kw = _pattern_kwargs("str_extract", pattern)
    return _str("str_extract", string, group=group or 0, **kw)


def str_extract_all(string: Any, pattern: Any) -> Any:
    """Every match of *pattern*, as a list column."""
    return _str("str_extract_all", string, **_pattern_kwargs("str_extract_all", pattern))


def str_match(string: Any, pattern: Any) -> Any:
    """The first match and its capture groups, as a struct column.

    Fields are ``match`` and then each group, by name for named groups
    ``(?P<name>...)`` and by number otherwise; ``unnest_wider()`` spreads them.
    """
    kw = _pattern_kwargs("str_match", pattern)
    return _str("str_match", string, names=_group_names("str_match", kw), **kw)


def str_match_all(string: Any, pattern: Any) -> Any:
    """Every match with its capture groups, as a list of structs."""
    kw = _pattern_kwargs("str_match_all", pattern)
    return _str("str_match_all", string, names=_group_names("str_match_all", kw), **kw)


def str_split(string: Any, pattern: Any, n: int | None = None) -> Any:
    """Split *string* on *pattern* into a list column of at most *n* pieces."""
    if n is not None:
        _check_int("str_split", "n", n)
        if n < 1:
            raise ValueError("str_split() n must be 1 or more")
    return _str("str_split", string, n=n, **_split_kwargs("str_split", pattern))


def str_split_1(string: str, pattern: Any) -> list:
    """Split one string into a list of pieces."""
    if not isinstance(string, str):
        raise TypeError("str_split_1() takes a single string")
    return str_split([string], pattern)[0]


def str_split_fixed(string: Any, pattern: Any, n: int) -> Any:
    """Exactly *n* pieces per string, padded with ``""``, as a list column."""
    _check_int("str_split_fixed", "n", n)
    if n < 1:
        raise ValueError("str_split_fixed() n must be 1 or more")
    return _str("str_split_fixed", string, n=n, **_split_kwargs("str_split_fixed", pattern))


def str_split_i(string: Any, pattern: Any, i: int) -> Any:
    """Piece *i* (1-based, negative from the end) of *string* split on *pattern*."""
    _check_int("str_split_i", "i", i)
    if i == 0:
        raise ValueError("str_split_i() i must not be 0")
    return _str("str_split_i", string, i=i, **_split_kwargs("str_split_i", pattern))


# ── combine ─────────────────────────────────────────────────────────────


def str_c(*strings: Any, sep: str = "") -> Any:
    """Join values row by row with *sep*; a missing value gives a missing result.

    Unlike the other helpers, a plain string here is text, not a column:
    ``str_c(first, " ", last)``. Use ``col("x")`` (or a bare name in a
    notebook) for columns.
    """
    if not strings:
        raise ValueError("str_c() needs at least one value")
    if not isinstance(sep, str):
        raise TypeError("str_c() sep must be a string")
    if builtins.any(_is_vector(s) for s in strings):
        columns, parts = {}, []
        for i, value in enumerate(strings):
            if _is_vector(value):
                columns[f"{_X}{i}"] = value
                parts.append(col(f"{_X}{i}"))
            else:
                parts.append(value)
        return _eager(_func("str_c", *parts, sep=sep), columns)
    return _func("str_c", *strings, sep=sep)


def str_flatten(
    string: Any, collapse: str = "", last: str | None = None, na_rm: bool = False
) -> Any:
    """Aggregate: join a column's values into one string with *collapse*.

    *last* goes between the final two values instead of *collapse*.
    """
    if not isinstance(collapse, str) or (last is not None and not isinstance(last, str)):
        raise TypeError("str_flatten() collapse and last must be strings")
    _check_bool("str_flatten", "na_rm", na_rm)
    return _str(
        "str_flatten", string, collapse=collapse, last=last, na_rm=na_rm, comma=False
    )


def str_flatten_comma(string: Any, last: str | None = None, na_rm: bool = False) -> Any:
    """Aggregate: join with ``", "``; ``last=", and "`` gives an Oxford comma."""
    if last is not None and not isinstance(last, str):
        raise TypeError("str_flatten_comma() last must be a string")
    _check_bool("str_flatten_comma", "na_rm", na_rm)
    return _str(
        "str_flatten", string, collapse=", ", last=last, na_rm=na_rm, comma=True
    )


def _glue(template: str, variables: dict[str, Any]) -> Expr:
    parts: list[Any] = []
    for text, field, spec, conversion in _string_module.Formatter().parse(template):
        if text:
            parts.append(text)
        if field is None:
            continue
        field = field.strip()
        if not field.isidentifier() or spec or conversion:
            raise ValueError(
                f"str_glue() can only fill in names, like {{price}}; got {{{field}}}"
            )
        if field in variables and not callable(variables[field]):
            parts.append(Expr(("name", field, _node(variables[field]))))
        else:
            parts.append(col(field))
    return _func("str_glue", *parts)


def _caller_variables(depth: int) -> dict[str, Any]:
    frame = sys._getframe(depth + 1)
    return {**frame.f_globals, **frame.f_locals}


def str_glue(*template: str, sep: str = "") -> Expr:
    """Fill ``{name}`` with each row's column value (or a variable of that name).

    A column wins over a variable of the same name, as in dplyr. Missing
    values print as ``NA``; ``{{`` and ``}}`` are literal braces.
    """
    if not template or not builtins.all(isinstance(t, str) for t in template):
        raise TypeError("str_glue() takes one or more template strings")
    return _glue(sep.join(template), _caller_variables(1))


def str_glue_data(data: Any, *template: str, sep: str = "") -> list:
    """:func:`str_glue` on *data* (a frame or a dict of columns), returning a list."""
    from tidy3.frame import TidyFrame, tidy

    frame = data if isinstance(data, TidyFrame) else tidy(data)
    expr = _glue(sep.join(template), _caller_variables(1))
    from tidy3.verbs import mutate

    out = (frame >> mutate(**{_X: expr})).collect(as_="polars")
    return out[_X].to_list()


# ── vectors ─────────────────────────────────────────────────────────────


def _natural_key(value: str) -> list:
    return [(0, int(p), "") if p.isdigit() else (1, 0, p) for p in re.split(r"(\d+)", value) if p]


def _order(values: list, decreasing: bool, na_last: bool | None, numeric: bool) -> list[int]:
    present = [i for i, v in enumerate(values) if v is not None]
    missing = [i for i, v in enumerate(values) if v is None]
    key = (lambda i: _natural_key(values[i])) if numeric else (lambda i: values[i])
    present.sort(key=key, reverse=decreasing)
    if na_last is None:
        return present
    return present + missing if na_last else missing + present


def str_order(
    string: Any,
    decreasing: bool = False,
    na_last: bool | None = True,
    numeric: bool = False,
) -> list[int]:
    """1-based positions that sort *string*, as in R; ``na_last=None`` drops missing."""
    values = _require_vector("str_order", string, "in a pipe use arrange()")
    return [i + 1 for i in _order(values, decreasing, na_last, numeric)]


def str_sort(
    string: Any,
    decreasing: bool = False,
    na_last: bool | None = True,
    numeric: bool = False,
) -> list:
    """*string* sorted; ``numeric=True`` sorts ``"x10"`` after ``"x9"``."""
    values = _require_vector("str_sort", string, "in a pipe use arrange()")
    return [values[i] for i in _order(values, decreasing, na_last, numeric)]


def str_unique(string: Any, ignore_case: bool = False) -> list:
    """Distinct strings, in first-seen order."""
    values = _require_vector("str_unique", string, "in a pipe use distinct()")
    seen, out = set(), []
    for value in values:
        key = value.lower() if ignore_case and value is not None else value
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


class StrView(str):
    """Printed form of :func:`str_view`."""

    def __repr__(self) -> str:
        return str(self)


def str_view(string: Any, pattern: Any = None, match: bool | None = True) -> StrView:
    """Show each string with matches of *pattern* in ``<>``.

    With a pattern, only matching strings show (``match=False``: only the
    others; ``match=None``: all).
    """
    values = _require_vector("str_view", string, "pull() a column first")
    rx = _compiled(_pattern_kwargs("str_view", pattern)) if pattern is not None else None
    lines = []
    for i, value in enumerate(values, start=1):
        shown = "NA" if value is None else value
        if rx is not None and value is not None:
            found = rx.search(value) is not None
            if match is not None and found != match:
                continue
            shown = rx.sub(lambda m: f"<{m.group(0)}>", value)
        elif rx is not None and match is not None:
            continue
        lines.append(f"[{i}] │ {shown}")
    return StrView("\n".join(lines))


def str_escape(string: Any) -> Any:
    """Escape regular-expression characters so *string* matches literally."""
    return _str("str_escape", string)


def str_conv(string: Any, encoding: str) -> Any:
    """Decode a column of bytes from *encoding* into text."""
    if not isinstance(encoding, str):
        raise TypeError("str_conv() encoding must be a string")
    return _str("str_conv", string, encoding=encoding)


# ── row-by-row Python, shared by both engines ───────────────────────────


def _locate_all(value: str, rx: re.Pattern) -> list[dict]:
    return [{"start": m.start() + 1, "end": m.end()} for m in rx.finditer(value)]


def _match_row(m: re.Match | None, names: list[str]) -> dict:
    if m is None:
        return {"match": None, **{name: None for name in names}}
    return {"match": m.group(0), **{name: m.group(i + 1) for i, name in enumerate(names)}}


def _re_split(rx: re.Pattern, value: str, n: int | None) -> list[str]:
    pieces, position = [], 0
    for m in rx.finditer(value):
        if n is not None and len(pieces) == n - 1:
            break
        if m.end() == m.start():
            continue
        pieces.append(value[position:m.start()])
        position = m.end()
    pieces.append(value[position:])
    return pieces


def _split_row(value: str, kw: dict, n: int | None) -> list[str]:
    if kw["boundary"]:
        return [m.group(0) for m in _compiled(kw).finditer(value)]
    if _plain(kw):
        return value.split(kw["pattern"], -1 if n is None else n - 1)
    return _re_split(_compiled(kw), value, n)


def _width(value: str) -> int:
    width = 0
    for char in value:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def _wrap(value: str, kw: dict) -> str:
    return "\n".join(
        textwrap.wrap(
            value,
            width=builtins.max(kw["width"], 1),
            initial_indent=" " * kw["indent"],
            subsequent_indent=" " * kw["exdent"],
            break_long_words=not kw["whitespace_only"],
            break_on_hyphens=False,
        )
    )


def _word(value: str, kw: dict) -> Any:
    if _plain(kw):
        pieces = value.split(kw["pattern"])
        seps = [kw["pattern"]] * (len(pieces) - 1)
    else:
        pieces, seps, position = [], [], 0
        for m in _compiled(kw).finditer(value):
            pieces.append(value[position:m.start()])
            seps.append(m.group(0))
            position = m.end()
        pieces.append(value[position:])
    count = len(pieces)
    start = kw["start"] + count + 1 if kw["start"] < 0 else kw["start"]
    end = kw["end"] + count + 1 if kw["end"] < 0 else kw["end"]
    if start > count or end > count:
        return None
    start = builtins.max(start, 1)
    out = ""
    for i in range(start - 1, end):
        out += pieces[i] + (seps[i] if i < end - 1 else "")
    return out


def _flatten(values: list, kw: dict) -> Any:
    if kw["na_rm"]:
        values = [v for v in values if v is not None and v is not pd.NA]
    elif builtins.any(v is None or v is pd.NA for v in values):
        return None
    last = kw["last"]
    if last is None or len(values) < 2:
        return kw["collapse"].join(values)
    if kw["comma"] and len(values) == 2 and last.startswith(","):
        last = last[1:]
    return kw["collapse"].join(values[:-1]) + last + values[-1]


def _sub(value: str, start: int, end: int) -> str:
    length = len(value)
    first = start if start > 0 else length + start + 1
    last = end if end >= 0 else length + end + 1
    first, last = builtins.max(first, 1), builtins.min(last, length)
    return value[first - 1:last] if last >= first else ""


def _trunc(value: str, width: int, side: str, ellipsis: str) -> str:
    if len(value) <= width:
        return value
    keep = width - len(ellipsis)
    if side == "right":
        return value[:keep] + ellipsis
    if side == "left":
        return ellipsis + value[len(value) - keep:]
    return value[:math.ceil(keep / 2)] + ellipsis + value[len(value) - keep // 2:]


def _pad_both(value: str, width: int, pad: str) -> str:
    missing = builtins.max(width - len(value), 0)
    return pad * (missing // 2) + value + pad * (missing - missing // 2)


def _split_i(pieces: list[str], i: int) -> Any:
    index = i - 1 if i > 0 else i
    return pieces[index] if -len(pieces) <= index < len(pieces) else None


def _escape(value: str) -> str:
    return _SPECIAL.sub(r"\\\1", value)


def _decode(value: Any, encoding: str) -> str:
    return value.decode(encoding) if isinstance(value, (bytes, bytearray)) else value


# ── Polars ──────────────────────────────────────────────────────────────

_LOCATION = pl.Struct({"start": pl.Int32, "end": pl.Int32})


def _pl_replacement(replacement: str) -> str:
    """stringr's ``\\1`` back-references in Polars' ``${1}`` form; ``$`` stays text."""
    replacement = replacement.replace("$", "$$")
    return re.sub(r"\\(\d)", r"${\1}", replacement)


def _pl_sub(x: pl.Expr, start: int, end: int) -> pl.Expr:
    length = x.str.len_chars().cast(pl.Int64)
    first = pl.lit(start) if start > 0 else length + start + 1
    last = pl.lit(end) if end >= 0 else length + end + 1
    first = pl.max_horizontal(first, pl.lit(1))
    last = pl.min_horizontal(last, length)
    size = pl.max_horizontal(last - first + 1, pl.lit(0))
    return x.str.slice(first - 1, size)


def _pl_split(x: pl.Expr, kw: dict, n: int | None = None) -> pl.Expr:
    if kw["boundary"]:
        return x.str.extract_all(_regex(kw))
    if n is None and _plain(kw):
        return x.str.split(kw["pattern"])
    if n is None:
        # Polars splits on text only: mark each regex match, then split.
        return x.str.replace_all(_regex(kw), "\x1f").str.split("\x1f")
    if _plain(kw):
        fields = x.str.splitn(kw["pattern"], n)
        pieces = pl.concat_list(
            [fields.struct.field(f"field_{i}") for i in range(n)]
        ).list.drop_nulls()
        return pl.when(x.is_null()).then(None).otherwise(pieces)
    return x.map_elements(
        lambda v: _split_row(v, kw, n), return_dtype=pl.List(pl.String)
    )


def _pl_flatten(x: pl.Expr, kw: dict) -> pl.Expr:
    values = x.drop_nulls() if kw["na_rm"] else x
    joined = values.str.join(kw["collapse"])
    last = kw["last"]
    if last is not None:
        count = values.len()
        head = values.slice(0, count - 1).str.join(kw["collapse"])
        with_last = pl.concat_str([head, pl.lit(last), values.last()])
        if kw["comma"] and last.startswith(","):
            two = pl.concat_str([head, pl.lit(last[1:]), values.last()])
            with_last = pl.when(count == 2).then(two).otherwise(with_last)
        joined = pl.when(count >= 2).then(with_last).otherwise(joined)
    if not kw["na_rm"]:
        joined = pl.when(x.null_count() > 0).then(None).otherwise(joined)
    return joined


def compile_polars(name: str, args: tuple, kw: dict) -> pl.Expr:
    """Polars expression for the ``str_*`` node *name*.

    *args* are compiled Polars expressions; *kw* holds the literal keyword values.
    """
    if name == "str_c":
        return pl.concat_str(
            [arg.cast(pl.String) for arg in args], separator=kw["sep"]
        )
    if name == "str_glue":
        return pl.concat_str([arg.cast(pl.String).fill_null("NA") for arg in args])
    if name == "str_conv":
        return args[0].map_elements(
            lambda v: _decode(v, kw["encoding"]), return_dtype=pl.String
        )
    x = args[0].cast(pl.String)
    if name in {"str_detect", "str_starts", "str_ends"}:
        if _plain(kw) and name != "str_detect":
            fn = x.str.starts_with if name == "str_starts" else x.str.ends_with
            found = fn(kw["pattern"])
        elif _plain(kw):
            found = x.str.contains(kw["pattern"], literal=True)
        else:
            wrap = {"str_detect": "{}", "str_starts": "^(?:{})", "str_ends": "(?:{})$"}
            found = x.str.contains(_regex(kw, wrap[name]))
        return ~found if kw["negate"] else found
    if name == "str_equal":
        y = args[1].cast(pl.String)
        if kw["ignore_case"]:
            return x.str.to_lowercase() == y.str.to_lowercase()
        return x == y
    if name == "str_count":
        if _plain(kw):
            return x.str.count_matches(kw["pattern"], literal=True)
        return x.str.count_matches(_regex(kw))
    if name == "str_locate":
        before = x.str.extract(_regex(kw, "^((?s:.*?))(?:{})"), 1)
        found = x.str.extract(_regex(kw), 0)
        start = (before.str.len_chars() + 1).cast(pl.Int32)
        end = start + found.str.len_chars().cast(pl.Int32) - 1
        return pl.struct(start.alias("start"), end.alias("end"))
    if name == "str_locate_all":
        rx = _compiled(kw)
        return x.map_elements(
            lambda v: _locate_all(v, rx), return_dtype=pl.List(_LOCATION)
        )
    if name == "str_length":
        return x.str.len_chars()
    if name == "str_width":
        return x.map_elements(_width, return_dtype=pl.Int64)
    if name == "str_sub":
        return _pl_sub(x, kw["start"], kw["end"])
    if name == "str_sub_all":
        return pl.concat_list(
            [_pl_sub(x, s, e) for s, e in zip(kw["starts"], kw["ends"])]
        )
    if name == "str_trunc":
        width, ellipsis = kw["width"], kw["ellipsis"]
        keep = width - len(ellipsis)
        length = x.str.len_chars().cast(pl.Int64)
        head = x.str.slice(0, math.ceil(keep / 2) if kw["side"] == "center" else keep)
        tail_size = keep // 2 if kw["side"] == "center" else keep
        tail = x.str.slice(length - tail_size, tail_size)
        cut = {
            "right": pl.concat_str([head, pl.lit(ellipsis)]),
            "left": pl.concat_str([pl.lit(ellipsis), tail]),
            "center": pl.concat_str([head, pl.lit(ellipsis), tail]),
        }[kw["side"]]
        return pl.when(length > width).then(cut).otherwise(x)
    if name == "str_word":
        if not _plain(kw):
            return x.map_elements(lambda v: _word(v, kw), return_dtype=pl.String)
        pieces = x.str.split(kw["pattern"])
        count = pieces.list.len().cast(pl.Int64)
        start, end = kw["start"], kw["end"]
        first = count + start + 1 if start < 0 else pl.lit(start, pl.Int64)
        last = count + end + 1 if end < 0 else pl.lit(end, pl.Int64)
        out_of_range = (first > count) | (last > count)
        first = pl.max_horizontal(first, pl.lit(1))
        size = pl.max_horizontal(last - first + 1, pl.lit(0))
        words = pieces.list.slice(first - 1, size).list.join(kw["pattern"])
        return pl.when(out_of_range).then(None).otherwise(words)
    if name in {"str_replace", "str_replace_all"}:
        fn = x.str.replace_all if name == "str_replace_all" else x.str.replace
        # Polars matches a pattern without special characters as text and
        # then reads the replacement as text too, so take that path here.
        if _plain(kw):
            return fn(kw["pattern"], kw["replacement"], literal=True)
        return fn(_regex(kw), _pl_replacement(kw["replacement"]))
    if name == "str_replace_na":
        return x.fill_null(kw["replacement"])
    if name == "str_to_upper":
        return x.str.to_uppercase()
    if name == "str_to_lower":
        return x.str.to_lowercase()
    if name == "str_to_title":
        return x.str.to_titlecase()
    if name == "str_to_sentence":
        return pl.concat_str(
            [x.str.slice(0, 1).str.to_uppercase(), x.str.slice(1).str.to_lowercase()]
        )
    if name == "str_trim":
        return {
            "both": x.str.strip_chars(),
            "left": x.str.strip_chars_start(),
            "right": x.str.strip_chars_end(),
        }[kw["side"]]
    if name == "str_squish":
        return x.str.strip_chars().str.replace_all(r"\s+", " ")
    if name == "str_pad":
        width, pad = kw["width"], kw["pad"]
        if kw["side"] == "left":
            return x.str.pad_start(width, pad)
        if kw["side"] == "right":
            return x.str.pad_end(width, pad)
        length = x.str.len_chars().cast(pl.Int64)
        left = pl.max_horizontal(pl.lit(width) - length, pl.lit(0)) // 2
        return x.str.pad_start(length + left, pad).str.pad_end(width, pad)
    if name == "str_dup":
        if kw["times"] == 0:
            return pl.when(x.is_null()).then(None).otherwise(pl.lit(""))
        return pl.concat_str([x] * kw["times"])
    if name == "str_wrap":
        return x.map_elements(lambda v: _wrap(v, kw), return_dtype=pl.String)
    if name == "str_extract":
        return x.str.extract(_regex(kw), kw["group"])
    if name == "str_extract_all":
        return x.str.extract_all(_regex(kw))
    if name == "str_match":
        rx = _regex(kw)
        fields = [x.str.extract(rx, 0).alias("match")] + [
            x.str.extract(rx, i + 1).alias(label) for i, label in enumerate(kw["names"])
        ]
        return pl.struct(fields)
    if name == "str_match_all":
        rx = _compiled(kw)
        names = kw["names"]
        dtype = pl.List(pl.Struct({label: pl.String for label in ("match", *names)}))
        return x.map_elements(
            lambda v: [_match_row(m, names) for m in rx.finditer(v)], return_dtype=dtype
        )
    if name in {"str_split", "str_split_i"}:
        pieces = _pl_split(x, kw, kw.get("n"))
        if name == "str_split":
            return pieces
        i = kw["i"]
        return pieces.list.get(i - 1 if i > 0 else i, null_on_oob=True)
    if name == "str_split_fixed":
        n = kw["n"]
        padding = pl.lit(pl.Series([[""] * n], dtype=pl.List(pl.String)))
        padded = pl.concat_list([_pl_split(x, kw, n), padding]).list.head(n)
        return pl.when(x.is_null()).then(None).otherwise(padded)
    if name == "str_flatten":
        return _pl_flatten(x, kw)
    if name == "str_escape":
        return x.str.replace_all(r"([.^$|?*+()\[\]{}\\])", r"\${1}")
    raise ValueError(f"unknown string function: {name}")


# ── pandas ──────────────────────────────────────────────────────────────

_NA = pd.NA


def _missing(value: Any) -> bool:
    return value is None or value is _NA or (isinstance(value, float) and math.isnan(value))


def _map(series: Any, fn: Any) -> Any:
    """Apply *fn* to each present value; missing values stay missing (strings)."""
    return series.map(lambda v: _NA if _missing(v) else fn(v)).astype("string")


def _map_object(series: Any, fn: Any) -> Any:
    """Like :func:`_map` for lists and dicts."""
    return series.astype(object).map(lambda v: None if _missing(v) else fn(v))


def _to_str(series: Any) -> Any:
    """A string result in pandas' default string dtype, missing as missing."""
    return series.astype("str")


def _text(value: Any, index: Any) -> Any:
    """A value as a nullable string Series, or a scalar string."""
    if isinstance(value, pd.Series):
        return value.reindex(index).astype("string")
    return _NA if _missing(value) else str(value)


def eval_pandas(
    name: str,
    args: tuple,
    kw: dict,
    df: Any,
    groups: list[str] | None,
    mode: str,
) -> Any:
    """pandas result for the ``str_*`` node *name*; *args* are evaluated values."""
    from tidy3.pandas_engine import _as_series, _grouped

    if name in {"str_c", "str_glue"}:
        sep = kw.get("sep", "")
        out: Any = None
        for arg in args:
            part = _text(arg, df.index)
            if name == "str_glue":
                part = part.fillna("NA") if isinstance(part, pd.Series) else (
                    "NA" if part is _NA else part
                )
            out = part if out is None else out + sep + part
        if not isinstance(out, pd.Series):
            out = pd.Series(out, index=df.index, dtype="string")
        return _to_str(out)
    if name == "str_conv":
        series = _as_series(args[0], df.index)
        return _to_str(_map(series.astype(object), lambda v: _decode(v, kw["encoding"])))

    x = _as_series(args[0], df.index).astype("string")
    if name in {"str_detect", "str_starts", "str_ends"}:
        if _plain(kw) and name == "str_starts":
            found = x.str.startswith(kw["pattern"])
        elif _plain(kw) and name == "str_ends":
            found = x.str.endswith(kw["pattern"])
        elif _plain(kw):
            found = x.str.contains(kw["pattern"], regex=False)
        else:
            wrap = {"str_detect": "{}", "str_starts": "^(?:{})", "str_ends": r"(?:{})\Z"}
            found = x.str.contains(_regex(kw, wrap[name]), regex=True)
        found = found.astype("boolean")
        return ~found if kw["negate"] else found
    if name == "str_equal":
        y = _text(args[1], df.index)
        if kw["ignore_case"]:
            x = x.str.lower()
            y = y.str.lower() if isinstance(y, pd.Series) else (
                y if y is _NA else y.lower()
            )
        return (x == y).astype("boolean")
    if name == "str_count":
        return x.str.count(_regex(kw)).astype("Int64")
    if name == "str_locate":
        rx = _compiled(kw)

        def locate(value: Any) -> dict:
            m = None if _missing(value) else rx.search(value)
            if m is None:
                return {"start": None, "end": None}
            return {"start": m.start() + 1, "end": m.end()}

        return x.astype(object).map(locate)
    if name == "str_locate_all":
        rx = _compiled(kw)
        return _map_object(x, lambda v: _locate_all(v, rx))
    if name == "str_length":
        return x.str.len().astype("Int64")
    if name == "str_width":
        return x.astype(object).map(lambda v: _NA if _missing(v) else _width(v)).astype("Int64")
    if name == "str_sub":
        return _to_str(_map(x, lambda v: _sub(v, kw["start"], kw["end"])))
    if name == "str_sub_all":
        pairs = list(zip(kw["starts"], kw["ends"]))
        return x.astype(object).map(
            lambda v: [None if _missing(v) else _sub(v, s, e) for s, e in pairs]
        )
    if name == "str_trunc":
        return _to_str(
            _map(x, lambda v: _trunc(v, kw["width"], kw["side"], kw["ellipsis"]))
        )
    if name == "str_word":
        return _to_str(_map(x, lambda v: _word(v, kw)).fillna(_NA))
    if name in {"str_replace", "str_replace_all"}:
        n = -1 if name == "str_replace_all" else 1
        if _plain(kw):
            out = x.str.replace(kw["pattern"], kw["replacement"], n=n, regex=False)
        else:
            out = x.str.replace(_regex(kw), kw["replacement"], n=n, regex=True)
        return _to_str(out)
    if name == "str_replace_na":
        return _to_str(x.fillna(kw["replacement"]))
    if name == "str_to_upper":
        return _to_str(x.str.upper())
    if name == "str_to_lower":
        return _to_str(x.str.lower())
    if name == "str_to_title":
        return _to_str(x.str.title())
    if name == "str_to_sentence":
        return _to_str(_map(x, lambda v: v[:1].upper() + v[1:].lower()))
    if name == "str_trim":
        side = {"both": "strip", "left": "lstrip", "right": "rstrip"}[kw["side"]]
        return _to_str(getattr(x.str, side)())
    if name == "str_squish":
        return _to_str(x.str.strip().str.replace(r"\s+", " ", regex=True))
    if name == "str_pad":
        if kw["side"] == "both":
            out = _map(x, lambda v: _pad_both(v, kw["width"], kw["pad"]))
        else:
            out = x.str.pad(kw["width"], side=kw["side"], fillchar=kw["pad"])
        return _to_str(out)
    if name == "str_dup":
        return _to_str(x.str.repeat(kw["times"]))
    if name == "str_wrap":
        return _to_str(_map(x, lambda v: _wrap(v, kw)))
    if name == "str_extract":
        rx = _compiled(kw)

        def extract(value: str) -> Any:
            found = rx.search(value)
            if found is None or found.group(kw["group"]) is None:
                return _NA
            return found.group(kw["group"])

        return _to_str(_map(x, extract))
    if name == "str_extract_all":
        rx = _compiled(kw)
        return _map_object(x, lambda v: [m.group(0) for m in rx.finditer(v)])
    if name == "str_match":
        rx = _compiled(kw)
        return x.astype(object).map(
            lambda v: _match_row(None if _missing(v) else rx.search(v), kw["names"])
        )
    if name == "str_match_all":
        rx = _compiled(kw)
        return _map_object(
            x, lambda v: [_match_row(m, kw["names"]) for m in rx.finditer(v)]
        )
    if name == "str_split":
        return _map_object(x, lambda v: _split_row(v, kw, kw["n"]))
    if name == "str_split_i":
        out = _map(x, lambda v: _split_i(_split_row(v, kw, None), kw["i"]))
        return _to_str(out.fillna(_NA))
    if name == "str_split_fixed":
        n = kw["n"]
        return _map_object(
            x, lambda v: (_split_row(v, kw, n) + [""] * n)[:n]
        )
    if name == "str_flatten":
        def join(values: Any) -> Any:
            out = _flatten([None if _missing(v) else v for v in values], kw)
            return _NA if out is None else out

        if groups:
            grouped = _grouped(x, df, groups)
            if mode == "agg":
                return _to_str(grouped.agg(join).astype("string"))
            return _to_str(grouped.transform(join).astype("string"))
        result = join(x)
        if mode == "agg":
            return None if result is _NA else result
        return _to_str(pd.Series(result, index=df.index, dtype="string"))
    if name == "str_escape":
        return _to_str(_map(x, _escape))
    raise ValueError(f"unknown string function: {name}")
