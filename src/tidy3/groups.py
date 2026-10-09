"""Group metadata: dplyr's group_data(), group_keys(), group_rows(), ….

Each function takes a frame or pipes: ``group_size(df)`` or
``df >> group_size()``. Groups come in dplyr's order: sorted by their keys,
missing values last. With ``group_by(..., drop=False)``, unused levels of a
categorical (factor) key form empty groups, following dplyr's rule.

Row positions (``group_rows``, ``group_data``'s ``.rows``) and group numbers
(``group_indices``) are 1-based, as in dplyr and the rest of tidy3
(``slice(1)`` is the first row), so ``slice(*group_rows(df)[0])`` picks the
first group's rows. Subtract 1 to index a pandas frame or NumPy array.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from tidy3.eda import _MISSING, _pipeable

__all__ = [
    "group_data",
    "group_indices",
    "group_keys",
    "group_rows",
    "group_size",
    "group_trim",
    "group_walk",
    "groups",
    "nest_by",
]


def _frame(data: Any) -> Any:
    from tidy3.frame import TidyFrame, tidy

    return data if isinstance(data, TidyFrame) else tidy(data)


def _levels(tf: Any, pdf: pd.DataFrame, name: str) -> list[Any] | None:
    levels = tf._category_levels.get(name)
    if levels is not None:
        return list(levels)
    if isinstance(pdf[name].dtype, pd.CategoricalDtype):
        return list(pdf[name].cat.categories)
    return None


def _positions_by_value(
    values: pd.Series, positions: np.ndarray
) -> tuple[dict[Any, np.ndarray], np.ndarray]:
    """Row positions for each distinct value, and for missing values.

    One factorize and one sort, so many groups stay O(n log n).
    """
    codes, uniques = pd.factorize(values, use_na_sentinel=True)
    uniques = list(uniques)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    chunks = np.split(order, np.flatnonzero(np.diff(sorted_codes)) + 1)
    found: dict[Any, np.ndarray] = {}
    missing = positions[:0]
    for chunk in chunks:
        if not len(chunk):
            continue
        code = codes[chunk[0]]
        if code < 0:
            missing = positions[chunk]
        else:
            found[uniques[code]] = positions[chunk]
    return found, missing


def _group_table(
    tf: Any, names: list[str] | None = None
) -> tuple[list[str], pd.DataFrame, list[tuple[tuple, np.ndarray]]]:
    """Keys, the materialized frame, and (key, row positions) per group.

    ``names`` groups by other columns than the frame's own groups.
    """
    pdf = tf.collect(as_="pandas").reset_index(drop=True)
    own = names is None
    names = list(tf._groups or []) if own else list(names)
    every_row = np.arange(len(pdf))
    if own and tf._rowwise:
        rows = [
            (tuple(pdf.iloc[i][name] for name in names), np.array([i]))
            for i in every_row
        ]
        return names, pdf, rows
    if not names:
        return names, pdf, [((), every_row)]

    expand = own and not tf._group_drop
    table: list[tuple[tuple, np.ndarray]] = []

    def split(positions: np.ndarray, depth: int, prefix: tuple) -> None:
        if depth == len(names):
            table.append((prefix, positions))
            return
        name = names[depth]
        values = pdf[name].iloc[positions]
        found, missing = _positions_by_value(values, positions)
        levels = _levels(tf, pdf, name)
        if levels is not None:
            # Factor keys sort by level. dplyr's drop=False keeps every
            # level, plus missing if present; an unused level is one empty
            # group with the later keys missing.
            keys = list(levels)
        else:
            keys = sorted(found)
        if len(missing):
            keys.append(None)
        for key in keys:
            subset = missing if key is None else found.get(key, positions[:0])
            if len(subset) == 0 and expand and levels is not None:
                rest = (None,) * (len(names) - depth - 1)
                table.append(((*prefix, key, *rest), subset))
            elif len(subset):
                split(subset, depth + 1, (*prefix, key))

    split(every_row, 0, ())
    return names, pdf, table


def _keys_frame(tf: Any, names: list[str], pdf: pd.DataFrame, table: list) -> pd.DataFrame:
    keys = pd.DataFrame(
        {name: [key[index] for key, _ in table] for index, name in enumerate(names)}
    )
    for name in names:
        levels = _levels(tf, pdf, name)
        if levels is not None:
            # Keep unused levels (empty groups) that the data never shows.
            keys[name] = pd.Categorical(keys[name], categories=levels)
            continue
        try:
            keys[name] = keys[name].astype(pdf[name].dtype)
        except (TypeError, ValueError):
            pass
    return keys


def _result(tf: Any, pdf: pd.DataFrame) -> Any:
    from tidy3.frame import tidy

    return tidy(pdf, backend=tf._backend)


def group_data(data: Any = _MISSING):
    """Group keys plus a ``.rows`` column of each group's 1-based row positions."""
    if data is _MISSING:
        return _pipeable("group_data", group_data)
    tf = _frame(data)
    names, pdf, table = _group_table(tf)
    out = _keys_frame(tf, names, pdf, table)
    out[".rows"] = [(positions + 1).tolist() for _, positions in table]
    return _result(tf, out)


def group_keys(data: Any = _MISSING):
    """One row per group with the grouping columns."""
    if data is _MISSING:
        return _pipeable("group_keys", group_keys)
    tf = _frame(data)
    names, pdf, table = _group_table(tf)
    return _result(tf, _keys_frame(tf, names, pdf, table))


def group_rows(data: Any = _MISSING):
    """List of each group's 1-based row positions (as in dplyr)."""
    if data is _MISSING:
        return _pipeable("group_rows", group_rows)
    _, _, table = _group_table(_frame(data))
    return [(positions + 1).tolist() for _, positions in table]


def group_size(data: Any = _MISSING):
    """Number of rows in each group."""
    if data is _MISSING:
        return _pipeable("group_size", group_size)
    _, _, table = _group_table(_frame(data))
    return [len(positions) for _, positions in table]


def group_indices(data: Any = _MISSING):
    """The 1-based group number of every row (dplyr ``group_indices()``)."""
    if data is _MISSING:
        return _pipeable("group_indices", group_indices)
    _, pdf, table = _group_table(_frame(data))
    numbers = np.zeros(len(pdf), dtype=int)
    for number, (_, positions) in enumerate(table, start=1):
        numbers[positions] = number
    return numbers.tolist()


def groups(data: Any = _MISSING):
    """Names of the grouping columns (dplyr ``groups()``)."""
    if data is _MISSING:
        return _pipeable("groups", groups)
    return list(_frame(data)._groups or [])


def group_trim(data: Any = _MISSING):
    """Drop unused levels of categorical grouping columns, then regroup."""
    if data is _MISSING:
        return _pipeable("group_trim", group_trim)
    tf = _frame(data)
    names = list(tf._groups or [])
    if not names:
        return tf
    pdf = tf.collect(as_="pandas")
    levels = dict(tf._category_levels)
    for name in names:
        current = _levels(tf, pdf, name)
        if current is None:
            continue
        used = set(pdf[name].dropna().tolist())
        levels[name] = [level for level in current if level in used]
        if isinstance(pdf[name].dtype, pd.CategoricalDtype) and tf._backend == "pandas":
            pdf = pdf.assign(**{name: pdf[name].cat.remove_unused_categories()})
    if tf._backend == "pandas":
        return tf._with_pdf(pdf, groups=names, category_levels=levels)
    return tf._with_lf(tf._lf, groups=names, category_levels=levels)


def group_walk(fn: Callable[[Any, dict[str, Any]], Any]):
    """Call ``fn(group, keys)`` for each group, then pass the frame on.

    For side effects such as saving one file per group::

        df >> group_by("region") >> group_walk(
            lambda part, key: part.write_csv(f"{key['region']}.csv")
        )
    """
    if not callable(fn):
        raise TypeError("group_walk() fn must be callable")
    from tidy3.verbs import Verb, group_map

    def _apply(tf):
        tf >> group_map(fn)
        return tf

    return Verb(_apply, "group_walk")


def nest_by(*cols: Any, name: str = "data"):
    """One row per group with the rest of the columns nested in ``name``.

    The result is rowwise by the grouping columns, like dplyr ``nest_by()``.
    """
    from tidy3.verbs import Verb, group_nest

    def _apply(tf):
        from tidy3.tidyselect import resolve_selection

        keys = resolve_selection(tf, cols) if cols else list(tf._groups or [])
        nested = tf >> group_nest(*keys, name=name)
        return nested._with_lf(nested._lf, groups=keys, rowwise=True) if (
            nested._backend == "polars"
        ) else nested._with_pdf(nested._pdf, groups=keys, rowwise=True)

    return Verb(_apply, "nest_by")
