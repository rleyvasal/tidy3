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
import polars as pl

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


def _levels(tf: Any, name: str) -> list[Any] | None:
    """Factor levels of a key column, or None when it is not a factor."""
    levels = tf._category_levels.get(name)
    if levels is not None:
        return list(levels)
    dtype = tf._lf.collect_schema()[name]
    return dtype.categories.to_list() if isinstance(dtype, pl.Enum) else None


def _missing(value: Any) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and value != value)


def _key(values: Any) -> tuple:
    """A group key as a tuple of plain values, missing as None."""
    values = values if isinstance(values, tuple) else (values,)
    return tuple(
        None if _missing(v) else (v.item() if isinstance(v, np.generic) else v)
        for v in values
    )


def _row_count(tf: Any) -> int:
    return int(tf._lf.select(pl.len()).collect().item())


def _observed(tf: Any, names: list[str], positions: bool = True) -> dict[tuple, Any]:
    """Row positions of each key combination that occurs (backend-native).

    With ``positions=False`` each key maps to its row count instead, which
    never materializes the positions.
    """
    index = "__tidy3_position"
    while index in names:
        index += "_"
    # A float key's NaN and null are separate Polars groups but one missing
    # key here, as in pandas; merge them rather than let one replace the other.
    if not positions:
        counts = tf._lf.group_by(names).agg(pl.len().alias(index)).collect()
        tallies: dict[tuple, int] = {}
        for row in counts.iter_rows():
            key = _key(tuple(row[:-1]))
            tallies[key] = tallies.get(key, 0) + int(row[-1])
        return tallies
    found = (
        tf._lf.with_row_index(index)
        .group_by(names)
        .agg(pl.col(index))
        .collect()
    )
    # One flat array split at the group boundaries, not a Python list per group.
    from tidy3.tidyr import _EXPLODE_OPTIONS

    flat = (
        found.lazy().select(index).explode(index, **_EXPLODE_OPTIONS).collect()[index]
        .to_numpy().astype(np.int64, copy=False)
    )
    bounds = np.cumsum(found[index].list.len().to_numpy())[:-1]
    keys = found.select(names).iter_rows()
    observed: dict[tuple, np.ndarray] = {}
    for key, part in zip(keys, np.split(flat, bounds)):
        key = _key(key)
        observed[key] = np.sort(np.concatenate([observed[key], part])) if key in observed else part
    return observed


def _ordered_keys(tf: Any, names: list[str], keys: list[tuple], expand: bool) -> list[tuple]:
    """*keys* in dplyr's order: sorted (factor keys by level), missing last.

    With ``expand`` (dplyr's ``drop = FALSE``), an unused level of a factor
    key adds one empty group with the later keys missing. Only the key
    combinations are looked at, never the rows.
    """
    out: list[tuple] = []

    def split(subset: list[tuple], depth: int, prefix: tuple) -> None:
        if depth == len(names):
            out.append(prefix)
            return
        levels = _levels(tf, names[depth])
        present = {k[depth] for k in subset}
        order = list(levels) if levels is not None else sorted(v for v in present if v is not None)
        if None in present:
            order.append(None)
        for value in order:
            inner = [k for k in subset if k[depth] == value]
            if inner:
                split(inner, depth + 1, (*prefix, value))
            elif expand and levels is not None:
                out.append((*prefix, value, *([None] * (len(names) - depth - 1))))

    split(keys, 0, ())
    return out


def _group_table(
    tf: Any, names: list[str] | None = None, positions: bool = True
) -> tuple[list[str], list[tuple[tuple, Any]]]:
    """Each group's key and row positions, in dplyr's order.

    ``names`` groups by other columns than the frame's own groups. With
    ``positions=False`` each group carries its row count instead.
    """
    own = names is None
    names = list(tf._groups or []) if own else list(names)
    if own and tf._rowwise:
        if not names:
            # rowwise() with no keys: still one group per row, as in dplyr.
            # Iterating a frame with no columns would yield no rows at all.
            count = _row_count(tf)
            return names, [((), np.array([i]) if positions else 1) for i in range(count)]
        rows = tf._lf.select(names).collect().iter_rows()
        return names, [(_key(tuple(r)), np.array([i]) if positions else 1) for i, r in enumerate(rows)]
    if not names:
        count = _row_count(tf)
        return names, [((), np.arange(count) if positions else count)]
    observed = _observed(tf, names, positions)
    empty = np.array([], dtype=np.int64) if positions else 0
    ordered = _ordered_keys(tf, names, list(observed), expand=own and not tf._group_drop)
    return names, [(key, observed.get(key, empty)) for key in ordered]


def _keys_frame(tf: Any, names: list[str], keys: list[tuple]) -> Any:
    """The key columns for *keys*, in the frame's backend and column types."""
    schema = tf._lf.collect_schema()
    return pl.DataFrame(
        {name: [k[i] for k in keys] for i, name in enumerate(names)}, strict=False
    ).with_columns(pl.col(name).cast(schema[name], strict=False) for name in names)


def _result(tf: Any, frame: Any) -> Any:
    from tidy3.frame import TidyFrame

    if isinstance(frame, pl.DataFrame):
        frame = frame.lazy()
    return TidyFrame(frame, category_levels=tf._category_levels)


def group_data(data: Any = _MISSING):
    """Group keys plus a ``.rows`` column of each group's 1-based row positions."""
    if data is _MISSING:
        return _pipeable("group_data", group_data)
    tf = _frame(data)
    names, table = _group_table(tf)
    out = _keys_frame(tf, names, [key for key, _ in table])
    if not names:
        # Ungrouped: one group, no key columns to attach to.
        out = pl.DataFrame({".rows": _rows_column(table)})
    else:
        out = out.with_columns(_rows_column(table))
    return _result(tf, out)


def _rows_column(table: list[tuple[tuple, np.ndarray]]) -> pl.Series:
    """Each group's 1-based rows as one list column, without Python lists per group."""
    import pyarrow as pa

    parts = [positions for _, positions in table]
    values = (np.concatenate(parts) if parts else np.array([], dtype=np.int64)).astype(np.int64) + 1
    offsets = np.concatenate([[0], np.cumsum([len(part) for part in parts], dtype=np.int64)])
    rows = pa.LargeListArray.from_arrays(pa.array(offsets, pa.int64()), pa.array(values, pa.int64()))
    return pl.Series(".rows", rows).cast(pl.List(pl.Int64))


def group_keys(data: Any = _MISSING):
    """One row per group with the grouping columns."""
    if data is _MISSING:
        return _pipeable("group_keys", group_keys)
    tf = _frame(data)
    names, table = _group_table(tf, positions=False)
    return _result(tf, _keys_frame(tf, names, [key for key, _ in table]))


def group_rows(data: Any = _MISSING):
    """List of each group's 1-based row positions (as in dplyr)."""
    if data is _MISSING:
        return _pipeable("group_rows", group_rows)
    _, table = _group_table(_frame(data))
    return [(positions + 1).tolist() for _, positions in table]


def group_size(data: Any = _MISSING):
    """Number of rows in each group."""
    if data is _MISSING:
        return _pipeable("group_size", group_size)
    _, table = _group_table(_frame(data), positions=False)
    return [count for _, count in table]


def group_indices(data: Any = _MISSING):
    """The 1-based group number of every row (dplyr ``group_indices()``)."""
    if data is _MISSING:
        return _pipeable("group_indices", group_indices)
    tf = _frame(data)
    _, table = _group_table(tf)
    numbers = np.zeros(_row_count(tf), dtype=int)
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
    from tidy3.frame import FactorLevels

    levels = dict(tf._category_levels)
    casts = []
    for name in names:
        current = _levels(tf, name)
        if current is None:
            continue
        used = set(tf._lf.select(pl.col(name).drop_nulls().unique()).collect().to_series().to_list())
        recorded = tf._category_levels.get(name)
        kept = [level for level in current if level in used]
        levels[name] = FactorLevels(kept, ordered=bool(getattr(recorded, "ordered", False)))
        if all(isinstance(level, str) for level in kept):
            # The Enum's own categories are the levels Polars shows and sorts by.
            casts.append(pl.col(name).cast(pl.String).cast(pl.Enum(kept)))
    lf = tf._lf.with_columns(casts) if casts else tf._lf
    trimmed = ("group_trim() reads which levels the data uses",)
    return tf._with_lf(lf, groups=names, category_levels=levels, data_reads=trimmed)


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
        return nested._with_lf(nested._lf, groups=keys, rowwise=True)

    return Verb(_apply, "nest_by")
