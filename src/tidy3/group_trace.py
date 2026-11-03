"""Running a group callback's pipeline in one pass over the grouped frame.

``group_modify(fn)`` hands each group to a Python callback, which costs one
copy of the frame's columns per group. When the callback is a pipeline of
tidy3 verbs that each mean the same on a grouped frame as on every group
alone, the pipeline runs once over the whole frame instead, grouped by
hidden copies of the keys so that the key columns stay ordinary columns, as
they are in each group's part.

The callback runs first on a probe: an empty frame with the group's schema
that records the verbs piped into it and refuses to compute anything. The
trace is used only when the callback read nothing (not the key, not the
rows, not the data), piped only verbs from :data:`ONE_PASS_VERBS`, and
returned the probe. Otherwise, and for groupings the one pass cannot
reproduce (float keys, where NaN and null are one missing group per group
but two in a grouped plan; factor keys; kept empty groups; rowwise frames),
``group_modify`` falls back to one callback per group.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from tidy3.frame import TidyFrame

# Verbs whose grouped meaning is their per-group meaning: rows are filtered,
# added to, summarised, sliced, or ordered within each group, and group
# columns are kept. arrange sorts the whole frame, which a later stable
# sort by key turns into the within-group order.
ONE_PASS_VERBS = frozenset({
    "filter", "filter_out", "mutate", "transmute", "select", "drop", "rename",
    "relocate", "head", "slice", "slice_head", "slice_tail", "slice_min", "slice_max",
    "summarise", "count", "tally", "add_count", "add_tally", "distinct",
    "drop_na", "arrange",
})


class TraceAbort(Exception):
    """The callback did something a trace cannot stand for."""


class _NoRows:
    """What ``cur_group_rows()`` finds during a trace: nothing it can list."""

    def tolist(self) -> Any:
        raise TraceAbort("cur_group_rows() was read")


class _Key(dict):
    """The callback's key: reading any of it ends the trace."""

    def _read(self, *args: Any, **kwargs: Any) -> Any:
        raise TraceAbort("the group key was read")

    __getitem__ = __contains__ = __iter__ = __len__ = get = keys = values = items = _read  # type: ignore[assignment]


class _ProbeLazy(pl.LazyFrame):
    """A plan that resolves its schema but never computes."""

    def collect(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        raise TraceAbort("data was computed")


def _probe_plan(lf: pl.LazyFrame) -> _ProbeLazy:
    return _ProbeLazy._from_pyldf(lf.clear()._ldf)


class _Probe(TidyFrame):
    """An empty frame with the group's schema that records the verbs piped in."""

    _trace: list[Any]

    @classmethod
    def _start(cls, tf: TidyFrame) -> "_Probe":
        probe = cls(_probe_plan(tf._lf), category_levels=tf._category_levels)
        probe._trace = []
        return probe

    def __rshift__(self, other: Any) -> Any:
        from tidy3.verbs import Verb

        if not isinstance(other, Verb) or other.name not in ONE_PASS_VERBS:
            raise TraceAbort(f"{getattr(other, 'name', other)!r} is not a one-pass verb")
        result = other.__rrshift__(self)
        if not isinstance(result, TidyFrame):
            raise TraceAbort("the verb did not return a frame")
        probe = _Probe(
            _probe_plan(result._lf),
            groups=result._groups,
            rowwise=result._rowwise,
            group_drop=result._group_drop,
            category_levels=result._category_levels,
        )
        probe._trace = [*self._trace, other]
        return probe

    def _computes(self, *args: Any, **kwargs: Any) -> Any:
        raise TraceAbort("data was computed")

    collect = compute = to_polars = to_pandas = to_numpy = pull = glimpse = show = _computes  # type: ignore[assignment]
    __len__ = __iter__ = __getitem__ = __repr__ = _repr_html_ = _computes  # type: ignore[assignment]

    @property
    def height(self) -> int:  # type: ignore[override]
        raise TraceAbort("data was computed")

    @property
    def shape(self) -> tuple[int, int]:  # type: ignore[override]
        raise TraceAbort("data was computed")


def one_pass(tf: TidyFrame, fn: Any, names: list[str]) -> TidyFrame | None:
    """``group_modify(fn)`` as one grouped pipeline, or None for the per-group path."""
    from tidy3.groups import _levels
    from tidy3.tidyselect import _group_rows_token, _reset_group_rows
    from tidy3.verbs import _dplyr_sort_expressions, _temp_column

    if tf._rowwise or not tf._group_drop:
        return None
    schema = tf._schema()
    if any(schema[name].is_float() or _levels(tf, name) is not None for name in names):
        return None
    token = _group_rows_token(_NoRows())
    try:
        result = fn(_Probe._start(tf), _Key())
    except Exception:
        return None
    finally:
        _reset_group_rows(token)
    if not isinstance(result, _Probe):
        return None
    columns = result._lf.collect_schema().names()
    # Hidden copies of the keys carry the grouping, so the key columns are
    # ordinary columns the pipeline may select away or change, as in a part.
    occupied = list(schema)
    hidden = []
    for name in names:
        hidden.append(_temp_column([*occupied, *hidden], f"__tidy3_group_{name}"))
    grouped = tf._with_lf(
        tf._lf.with_columns(pl.col(name).alias(key) for name, key in zip(names, hidden)),
        groups=hidden,
        rowwise=False,
    )
    try:
        out = grouped
        for verb in result._trace:
            out = out >> verb
        present = out._lf.collect_schema().names()
    except Exception:
        return None
    if any(key not in present for key in hidden) or any(column not in present for column in columns):
        return None
    # Groups in dplyr's order, missing last; rows within a group as the
    # pipeline left them.
    lf = out._lf.sort(
        _dplyr_sort_expressions(out._lf, hidden), nulls_last=True, maintain_order=True
    ).select(columns)
    from tidy3.frame import tidy

    return tidy(lf)
