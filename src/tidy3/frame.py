"""TidyFrame: lazy dplyr-like frame over Polars LazyFrame."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, overload

import polars as pl

from tidy3.options import get_options, options  # re-export

if TYPE_CHECKING:
    from tidy3.verbs import Verb

__all__ = ["TidyFrame", "tidy", "options"]

_POLARS_EXECUTION_ENGINES = frozenset({"auto", "streaming", "gpu"})


def _validate_execution_engine(engine: Any, *, backend: str) -> None:
    if isinstance(engine, str) and engine not in _POLARS_EXECUTION_ENGINES:
        choices = ", ".join(sorted(_POLARS_EXECUTION_ENGINES))
        raise ValueError(f"engine must be one of: {choices}")
    if backend == "pandas" and engine != "auto":
        raise ValueError(
            "engine is a Polars execution option; use engine='auto' with "
            "the pandas backend"
        )


def _uses_gpu_engine(engine: Any) -> bool:
    return engine == "gpu" or isinstance(engine, pl.GPUEngine)


class TidyFrame:
    """dplyr-style frame with ``>>`` pipes: lazy Polars (default) or eager pandas.

    The pandas backend exists for 1:1 engine comparisons (same pipeline,
    different engine) — see ``tidy(df, backend="pandas")`` and ``tidy3.bench``.
    """

    __slots__ = (
        "_data",
        "_groups",
        "_rowwise",
        "_group_drop",
        "_category_levels",
        # After select(): (pre-projection data, re-apply projection). Lets a
        # later filter/filter_out reference columns dropped by select —
        # equivalent to rewriting ``select >> filter`` as ``filter >> select``.
        "_select_base",
    )

    def __init__(
        self,
        data: Any,
        groups: list[str] | None = None,
        *,
        rowwise: bool = False,
        group_drop: bool = True,
        category_levels: dict[str, list[Any]] | None = None,
        select_base: Any | None = None,
    ):
        if not isinstance(data, pl.LazyFrame):
            import pandas as pd

            if not isinstance(data, pd.DataFrame):
                raise TypeError(
                    "TidyFrame requires a polars LazyFrame or pandas DataFrame"
                )
        self._data = data
        self._groups = list(groups) if groups else None
        self._rowwise = bool(rowwise)
        self._group_drop = bool(group_drop) if self._groups else True
        if category_levels is None and self.backend == "pandas":
            import pandas as pd

            category_levels = {
                str(name): series.cat.categories.tolist()
                for name, series in data.items()
                if isinstance(series.dtype, pd.CategoricalDtype)
            }
        self._category_levels = dict(category_levels or {})
        self._select_base = select_base

    @property
    def backend(self) -> str:
        return "polars" if isinstance(self._data, pl.LazyFrame) else "pandas"

    @property
    def _backend(self) -> str:
        return self.backend

    @property
    def _lf(self) -> pl.LazyFrame:
        if not isinstance(self._data, pl.LazyFrame):
            raise TypeError(
                "this TidyFrame runs on the pandas backend (no LazyFrame); "
                "polars-only APIs need tidy(..., backend='polars')"
            )
        return self._data

    @property
    def _pdf(self):
        return self._data  # pandas DataFrame when backend == "pandas"

    # ── EDA / inspection (schema-first; row counts materialize lazily) ──

    @property
    def columns(self) -> list[str]:
        """Column names as a Python ``list[str]`` (Polars-style).

        Schema-only — does not scan row data.
        """
        if self.backend == "pandas":
            return [str(c) for c in self._pdf.columns]
        return list(self._lf.collect_schema().names())

    @property
    def names(self) -> list[str]:
        """Alias of :attr:`columns` (base R ``names()``)."""
        return self.columns

    @property
    def width(self) -> int:
        """Number of columns (Polars ``width``)."""
        return len(self.columns)

    @property
    def height(self) -> int:
        """Number of rows (materializes a count for lazy frames)."""
        return len(self)

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_rows, n_cols)`` like NumPy/pandas/Polars."""
        return (self.height, self.width)

    @property
    def dtypes(self) -> dict[str, Any]:
        """``{column: dtype}`` from the frame schema (no full collect)."""
        if self.backend == "pandas":
            return {str(k): v for k, v in self._pdf.dtypes.items()}
        schema = self._lf.collect_schema()
        return {name: dtype for name, dtype in schema.items()}

    @property
    def schema(self) -> Any:
        """Backend schema object (Polars ``Schema`` or pandas dtypes map)."""
        if self.backend == "pandas":
            return self.dtypes
        return self._lf.collect_schema()

    def summary(
        self,
        *,
        percentiles: tuple[float, ...] | list[float] | None = None,
    ) -> TidyFrame:
        """Column-wise summary statistics; see :func:`tidy3.eda.summary`."""
        from tidy3.eda import summary as summary_fn

        return summary_fn(self, percentiles=percentiles)

    def describe(
        self,
        *,
        percentiles: tuple[float, ...] | list[float] | None = None,
    ) -> TidyFrame:
        """Alias of :meth:`summary` (pandas/Python naming)."""
        return self.summary(percentiles=percentiles)

    def colnames(self):
        """Paste-ready column selectors; see :func:`tidy3.eda.colnames`."""
        from tidy3.eda import colnames as colnames_fn

        return colnames_fn(self)

    def names_list(self) -> list[str]:
        """Plain column-name list (same as :attr:`columns` / :func:`names`)."""
        return list(self.columns)

    def _with_lf(
        self,
        lf: pl.LazyFrame,
        groups: list[str] | None = None,
        *,
        rowwise: bool | None = None,
        group_drop: bool | None = None,
        category_levels: dict[str, list[Any]] | None = None,
        select_base: Any | None = None,
        keep_select_base: bool = False,
    ) -> TidyFrame:
        # Default: clear select_base (most verbs invalidate the rewrite window).
        # select/filter set select_base=...; pass keep_select_base to preserve.
        if keep_select_base and select_base is None:
            select_base = self._select_base
        return TidyFrame(
            lf,
            groups=groups,
            rowwise=self._rowwise if rowwise is None else rowwise,
            group_drop=(
                self._group_drop if group_drop is None else group_drop
            ),
            category_levels=(
                self._category_levels
                if category_levels is None
                else category_levels
            ),
            select_base=select_base,
        )

    def _with_pdf(
        self,
        pdf: Any,
        groups: list[str] | None = None,
        *,
        rowwise: bool | None = None,
        group_drop: bool | None = None,
        category_levels: dict[str, list[Any]] | None = None,
        select_base: Any | None = None,
        keep_select_base: bool = False,
    ) -> TidyFrame:
        if keep_select_base and select_base is None:
            select_base = self._select_base
        return TidyFrame(
            pdf,
            groups=groups,
            rowwise=self._rowwise if rowwise is None else rowwise,
            group_drop=(
                self._group_drop if group_drop is None else group_drop
            ),
            category_levels=(
                self._category_levels
                if category_levels is None
                else category_levels
            ),
            select_base=select_base,
        )

    # ── pipe ────────────────────────────────────────────────────────────
    @overload
    def __rshift__(self, other: Verb) -> TidyFrame: ...

    @overload
    def __rshift__(self, other: Callable[[TidyFrame], Any]) -> Any: ...

    def __rshift__(self, other: Any) -> Any:
        """``tf >> verb`` — prefer Verb.__rrshift__, but support callables.

        Annotated so Pylance treats ``tidy(df) >> select(...)`` as ``TidyFrame``.
        """
        from tidy3.verbs import Verb

        other_cls = type(other)
        is_tidy3_verb = (
            other_cls.__module__ == "tidy3.verbs" and other_cls.__name__ == "Verb"
        )
        if isinstance(other, Verb) or is_tidy3_verb:
            return other.__rrshift__(self)
        if callable(other):
            return other(self)
        return NotImplemented

    # ── method-chain aliases ────────────────────────────────────────────
    def filter(self, *predicates: Any, by: Any = None) -> TidyFrame:  # noqa: A001
        from tidy3.verbs import filter as filter_verb

        return self >> filter_verb(*predicates, by=by)

    def filter_out(self, *predicates: Any, by: Any = None) -> TidyFrame:
        from tidy3.verbs import filter_out as filter_out_verb

        return self >> filter_out_verb(*predicates, by=by)

    def mutate(
        self,
        *specs: Any,
        by: Any = None,
        keep: str = "all",
        before: Any = None,
        after: Any = None,
        **kwargs: Any,
    ) -> TidyFrame:
        from tidy3.verbs import mutate as mutate_verb

        return self >> mutate_verb(
            *specs,
            by=by,
            keep=keep,
            before=before,
            after=after,
            **kwargs,
        )

    def transmute(self, *specs: Any, by: Any = None, **kwargs: Any) -> TidyFrame:
        from tidy3.verbs import transmute as transmute_verb

        return self >> transmute_verb(*specs, by=by, **kwargs)

    def select(self, *cols: Any, **renames: Any) -> TidyFrame:
        from tidy3.verbs import select as select_verb

        return self >> select_verb(*cols, **renames)

    def drop(self, *cols: Any) -> TidyFrame:
        from tidy3.verbs import drop as drop_verb

        return self >> drop_verb(*cols)

    def rename(self, **kwargs: str) -> TidyFrame:
        from tidy3.verbs import rename as rename_verb

        return self >> rename_verb(**kwargs)

    def rename_with(self, fn, *cols: Any) -> TidyFrame:
        from tidy3.verbs import rename_with as rename_with_verb

        return self >> rename_with_verb(fn, *cols)

    def relocate(
        self,
        *cols: Any,
        before: Any = None,
        after: Any = None,
    ) -> TidyFrame:
        from tidy3.verbs import relocate as relocate_verb

        return self >> relocate_verb(*cols, before=before, after=after)

    def pull(self, var: str | int = -1, *, name: str | int | None = None) -> Any:
        from tidy3.verbs import pull as pull_verb

        return self >> pull_verb(var, name=name)

    def glimpse(self, n: int = 10) -> TidyFrame:
        from tidy3.verbs import glimpse as glimpse_verb

        return self >> glimpse_verb(n)

    def arrange(self, *keys: Any, by_group: bool = False) -> TidyFrame:
        from tidy3.verbs import arrange as arrange_verb

        return self >> arrange_verb(*keys, by_group=by_group)

    def distinct(
        self,
        *cols: Any,
        keep_all: bool = False,
        maintain_order: bool = True,
        **computed: Any,
    ) -> TidyFrame:
        from tidy3.verbs import distinct as distinct_verb

        return self >> distinct_verb(
            *cols,
            keep_all=keep_all,
            maintain_order=maintain_order,
            **computed,
        )

    def group_by(
        self,
        *cols: Any,
        add: bool = False,
        drop: bool | None = None,
        **computed: Any,
    ) -> TidyFrame:
        from tidy3.verbs import group_by as group_by_verb

        return self >> group_by_verb(
            *cols, add=add, drop=drop, **computed
        )

    def rowwise(self, *cols: Any) -> TidyFrame:
        from tidy3.verbs import rowwise as rowwise_verb

        return self >> rowwise_verb(*cols)

    def ungroup(self, *cols: Any) -> TidyFrame:
        from tidy3.verbs import ungroup as ungroup_verb

        return self >> ungroup_verb(*cols)

    def summarise(
        self,
        *specs: Any,
        by: Any = None,
        groups: str | None = None,
        **kwargs: Any,
    ) -> TidyFrame:
        from tidy3.verbs import summarise as summarise_verb

        return self >> summarise_verb(
            *specs, by=by, groups=groups, **kwargs
        )

    summarize = summarise

    def reframe(self, *specs: Any, by: Any = None, **kwargs: Any) -> TidyFrame:
        from tidy3.verbs import reframe as reframe_verb

        return self >> reframe_verb(*specs, by=by, **kwargs)

    def count(
        self,
        *cols: str,
        wt: Any = None,
        sort: bool = False,
        name: str | None = None,
        drop: bool | None = None,
    ) -> TidyFrame:
        from tidy3.verbs import count as count_verb

        return self >> count_verb(
            *cols, wt=wt, sort=sort, name=name, drop=drop
        )

    def tally(
        self,
        *,
        wt: Any = None,
        sort: bool = False,
        name: str | None = None,
    ) -> TidyFrame:
        from tidy3.verbs import tally as tally_verb

        return self >> tally_verb(wt=wt, sort=sort, name=name)

    def add_count(
        self,
        *cols: str,
        wt: Any = None,
        sort: bool = False,
        name: str | None = None,
    ) -> TidyFrame:
        from tidy3.verbs import add_count as add_count_verb

        return self >> add_count_verb(*cols, wt=wt, sort=sort, name=name)

    def add_tally(
        self,
        *,
        wt: Any = None,
        sort: bool = False,
        name: str | None = None,
    ) -> TidyFrame:
        from tidy3.verbs import add_tally as add_tally_verb

        return self >> add_tally_verb(wt=wt, sort=sort, name=name)

    def head(self, n: int = 10) -> TidyFrame:
        from tidy3.verbs import head as head_verb

        return self >> head_verb(n)

    def slice(self, *rows: Any, by: Any = None) -> TidyFrame:
        from tidy3.verbs import slice as slice_verb

        return self >> slice_verb(*rows, by=by)

    def slice_head(
        self, *, n: int | None = None, prop: float | None = None, by: Any = None
    ) -> TidyFrame:
        from tidy3.verbs import slice_head as slice_head_verb

        return self >> slice_head_verb(n=n, prop=prop, by=by)

    def slice_tail(
        self, *, n: int | None = None, prop: float | None = None, by: Any = None
    ) -> TidyFrame:
        from tidy3.verbs import slice_tail as slice_tail_verb

        return self >> slice_tail_verb(n=n, prop=prop, by=by)

    def slice_min(
        self,
        order_by: Any,
        *,
        n: int | None = None,
        prop: float | None = None,
        with_ties: bool = True,
        na_rm: bool = False,
        by: Any = None,
    ) -> TidyFrame:
        from tidy3.verbs import slice_min as slice_min_verb

        return self >> slice_min_verb(
            order_by, n=n, prop=prop, with_ties=with_ties, na_rm=na_rm, by=by
        )

    def slice_max(
        self,
        order_by: Any,
        *,
        n: int | None = None,
        prop: float | None = None,
        with_ties: bool = True,
        na_rm: bool = False,
        by: Any = None,
    ) -> TidyFrame:
        from tidy3.verbs import slice_max as slice_max_verb

        return self >> slice_max_verb(
            order_by, n=n, prop=prop, with_ties=with_ties, na_rm=na_rm, by=by
        )

    def slice_sample(
        self,
        *,
        n: int | None = None,
        prop: float | None = None,
        weight_by: Any = None,
        replace: bool = False,
        seed: int | None = None,
        by: Any = None,
    ) -> TidyFrame:
        from tidy3.verbs import slice_sample as slice_sample_verb

        return self >> slice_sample_verb(
            n=n,
            prop=prop,
            weight_by=weight_by,
            replace=replace,
            seed=seed,
            by=by,
        )

    def sample_n(self, n: int, *, seed: int | None = None) -> TidyFrame:
        from tidy3.verbs import sample_n as sample_n_verb

        return self >> sample_n_verb(n, seed=seed)

    def sample_frac(self, frac: float, *, seed: int | None = None) -> TidyFrame:
        from tidy3.verbs import sample_frac as sample_frac_verb

        return self >> sample_frac_verb(frac, seed=seed)

    # ── reshape / missing data ─────────────────────────────────────────
    def drop_na(self, *cols: Any) -> TidyFrame:
        from tidy3.tidyr import drop_na as drop_na_verb

        return self >> drop_na_verb(*cols)

    def replace_na(self, replace: dict[str, Any]) -> TidyFrame:
        from tidy3.tidyr import replace_na as replace_na_verb

        return self >> replace_na_verb(replace)

    def fill(
        self, *cols: Any, direction: str = "down", by: Any = None
    ) -> TidyFrame:
        from tidy3.tidyr import fill as fill_verb

        return self >> fill_verb(*cols, direction=direction, by=by)

    def expand(self, *cols: Any) -> TidyFrame:
        from tidy3.tidyr import expand as expand_verb

        return self >> expand_verb(*cols)

    def complete(
        self,
        *cols: Any,
        fill: dict[str, Any] | None = None,
        explicit: bool = True,
    ) -> TidyFrame:
        from tidy3.tidyr import complete as complete_verb

        return self >> complete_verb(*cols, fill=fill, explicit=explicit)

    def pivot_longer(
        self,
        cols: Any,
        *,
        names_to: Any = "name",
        values_to: str = "value",
        names_prefix: str | None = None,
        names_sep: str | None = None,
        names_pattern: str | None = None,
        values_drop_na: bool = False,
        cols_vary: str = "fastest",
    ) -> TidyFrame:
        from tidy3.tidyr import pivot_longer as pivot_longer_verb

        return self >> pivot_longer_verb(
            cols,
            names_to=names_to,
            values_to=values_to,
            names_prefix=names_prefix,
            names_sep=names_sep,
            names_pattern=names_pattern,
            values_drop_na=values_drop_na,
            cols_vary=cols_vary,
        )

    def pivot_wider(
        self,
        *,
        names_from: Any = "name",
        values_from: Any = "value",
        id_cols: Any = None,
        names_prefix: str = "",
        names_sort: bool = False,
        values_fill: Any = None,
        values_fn: str | None = None,
        names: Any = None,
    ) -> TidyFrame:
        from tidy3.tidyr import pivot_wider as pivot_wider_verb

        return self >> pivot_wider_verb(
            names_from=names_from,
            values_from=values_from,
            id_cols=id_cols,
            names_prefix=names_prefix,
            names_sort=names_sort,
            values_fill=values_fill,
            values_fn=values_fn,
            names=names,
        )

    def separate(
        self,
        column: str,
        into: Any,
        *,
        sep: str = r"[^A-Za-z0-9]+",
        remove: bool = True,
        convert: bool = False,
        extra: str = "warn",
        fill: str = "warn",
    ) -> TidyFrame:
        from tidy3.tidyr import separate as separate_verb

        return self >> separate_verb(
            column,
            into,
            sep=sep,
            remove=remove,
            convert=convert,
            extra=extra,
            fill=fill,
        )

    def unite(
        self,
        column: str,
        *cols: Any,
        sep: str = "_",
        remove: bool = True,
        na_rm: bool = False,
    ) -> TidyFrame:
        from tidy3.tidyr import unite as unite_verb

        return self >> unite_verb(
            column, *cols, sep=sep, remove=remove, na_rm=na_rm
        )

    def nest(
        self,
        column: str = "data",
        *,
        cols: Any = None,
        by: Any = None,
    ) -> TidyFrame:
        from tidy3.tidyr import nest as nest_verb

        return self >> nest_verb(column, cols=cols, by=by)

    def unnest_longer(
        self,
        column: str,
        *,
        values_to: str | None = None,
        indices_to: str | None = None,
        keep_empty: bool = False,
    ) -> TidyFrame:
        from tidy3.tidyr import unnest_longer as unnest_longer_verb

        return self >> unnest_longer_verb(
            column,
            values_to=values_to,
            indices_to=indices_to,
            keep_empty=keep_empty,
        )

    def unnest(
        self,
        column: str,
        *,
        keep_empty: bool = False,
        names_sep: str | None = None,
    ) -> TidyFrame:
        from tidy3.tidyr import unnest as unnest_verb

        return self >> unnest_verb(
            column, keep_empty=keep_empty, names_sep=names_sep
        )

    def unnest_wider(
        self, column: str, *, names_sep: str | None = None
    ) -> TidyFrame:
        from tidy3.tidyr import unnest_wider as unnest_wider_verb

        return self >> unnest_wider_verb(column, names_sep=names_sep)

    def left_join(
        self, right: Any, *, on: Any = None, by: Any = None, **kwargs: Any
    ) -> TidyFrame:
        from tidy3.verbs import left_join as left_join_verb

        return self >> left_join_verb(right, on=on, by=by, **kwargs)

    def inner_join(
        self, right: Any, *, on: Any = None, by: Any = None, **kwargs: Any
    ) -> TidyFrame:
        from tidy3.verbs import inner_join as inner_join_verb

        return self >> inner_join_verb(right, on=on, by=by, **kwargs)

    def right_join(
        self, right: Any, *, on: Any = None, by: Any = None, **kwargs: Any
    ) -> TidyFrame:
        from tidy3.verbs import right_join as right_join_verb

        return self >> right_join_verb(right, on=on, by=by, **kwargs)

    def full_join(
        self, right: Any, *, on: Any = None, by: Any = None, **kwargs: Any
    ) -> TidyFrame:
        from tidy3.verbs import full_join as full_join_verb

        return self >> full_join_verb(right, on=on, by=by, **kwargs)

    def nest_join(
        self,
        right: Any,
        *,
        on: Any = None,
        by: Any = None,
        name: str = "data",
        keep: bool = False,
        na_matches: str = "na",
    ) -> TidyFrame:
        from tidy3.verbs import nest_join as nest_join_verb

        return self >> nest_join_verb(
            right,
            on=on,
            by=by,
            name=name,
            keep=keep,
            na_matches=na_matches,
        )

    def semi_join(
        self,
        right: Any,
        *,
        on: Any = None,
        by: Any = None,
        na_matches: str = "na",
    ) -> TidyFrame:
        from tidy3.verbs import semi_join as semi_join_verb

        return self >> semi_join_verb(
            right, on=on, by=by, na_matches=na_matches
        )

    def anti_join(
        self,
        right: Any,
        *,
        on: Any = None,
        by: Any = None,
        na_matches: str = "na",
    ) -> TidyFrame:
        from tidy3.verbs import anti_join as anti_join_verb

        return self >> anti_join_verb(
            right, on=on, by=by, na_matches=na_matches
        )

    def cross_join(self, right: Any, **kwargs: Any) -> TidyFrame:
        from tidy3.verbs import cross_join as cross_join_verb

        return self >> cross_join_verb(right, **kwargs)

    def bind_rows(self, *others: Any, id: str | None = None) -> TidyFrame:  # noqa: A002
        from tidy3.verbs import bind_rows as bind_rows_verb

        return self >> bind_rows_verb(*others, id=id)

    def bind_cols(self, *others: Any) -> TidyFrame:
        from tidy3.verbs import bind_cols as bind_cols_verb

        return self >> bind_cols_verb(*others)

    def union(self, right: Any) -> TidyFrame:
        from tidy3.verbs import union as union_verb

        return self >> union_verb(right)

    def union_all(self, right: Any) -> TidyFrame:
        from tidy3.verbs import union_all as union_all_verb

        return self >> union_all_verb(right)

    def intersect(self, right: Any) -> TidyFrame:
        from tidy3.verbs import intersect as intersect_verb

        return self >> intersect_verb(right)

    def setdiff(self, right: Any) -> TidyFrame:
        from tidy3.verbs import setdiff as setdiff_verb

        return self >> setdiff_verb(right)

    def symdiff(self, right: Any) -> TidyFrame:
        from tidy3.verbs import symdiff as symdiff_verb

        return self >> symdiff_verb(right)

    def setequal(self, right: Any) -> bool:
        from tidy3.verbs import setequal as setequal_verb

        return self >> setequal_verb(right)

    def rows_insert(
        self,
        right: Any,
        *,
        by: str | list[str] | None = None,
        conflict: str = "error",
    ) -> TidyFrame:
        from tidy3.verbs import rows_insert as rows_insert_verb

        return self >> rows_insert_verb(right, by=by, conflict=conflict)

    def rows_append(self, right: Any) -> TidyFrame:
        from tidy3.verbs import rows_append as rows_append_verb

        return self >> rows_append_verb(right)

    def rows_update(
        self,
        right: Any,
        *,
        by: str | list[str] | None = None,
        unmatched: str = "error",
    ) -> TidyFrame:
        from tidy3.verbs import rows_update as rows_update_verb

        return self >> rows_update_verb(right, by=by, unmatched=unmatched)

    def rows_patch(
        self,
        right: Any,
        *,
        by: str | list[str] | None = None,
        unmatched: str = "error",
    ) -> TidyFrame:
        from tidy3.verbs import rows_patch as rows_patch_verb

        return self >> rows_patch_verb(right, by=by, unmatched=unmatched)

    def rows_upsert(
        self, right: Any, *, by: str | list[str] | None = None
    ) -> TidyFrame:
        from tidy3.verbs import rows_upsert as rows_upsert_verb

        return self >> rows_upsert_verb(right, by=by)

    def rows_delete(
        self,
        right: Any,
        *,
        by: str | list[str] | None = None,
        unmatched: str = "error",
    ) -> TidyFrame:
        from tidy3.verbs import rows_delete as rows_delete_verb

        return self >> rows_delete_verb(right, by=by, unmatched=unmatched)

    def with_polars(self, fn) -> TidyFrame:
        """Escape hatch: ``fn(LazyFrame) -> LazyFrame`` applied to the plan."""
        out = fn(self._lf)
        if not isinstance(out, pl.LazyFrame):
            if isinstance(out, pl.DataFrame):
                out = out.lazy()
            else:
                raise TypeError("with_polars(fn) must return a Polars LazyFrame or DataFrame")
        return self._with_lf(out, groups=self._groups)

    # ── materialize ─────────────────────────────────────────────────────
    def collect(
        self,
        as_: Literal["polars", "pandas", "arrow", "numpy"] = "polars",
        *,
        columns: Any = None,
        arrow_backed: bool = False,
        engine: Any = "auto",
        dtype: Any = None,
        order: Literal["c", "fortran"] = "fortran",
        writable: bool = False,
        allow_copy: bool = True,
    ) -> Any:
        """Materialize the frame, optionally projecting columns first.

        ``arrow_backed=True`` returns a pandas DataFrame whose columns use
        Arrow extension dtypes.  For the Polars backend this avoids copying
        every column into NumPy buffers at the handoff boundary.

        ``engine`` controls Polars execution: ``"auto"`` (the default),
        ``"streaming"``, or ``"gpu"``. A configured ``polars.GPUEngine``
        object is also accepted. Non-default engines require the Polars
        backend.
        """
        _validate_execution_engine(engine, backend=self.backend)
        if arrow_backed and as_ != "pandas":
            raise ValueError("arrow_backed=True requires as_='pandas'")
        numpy_options = (
            dtype is not None
            or order != "fortran"
            or writable
            or not allow_copy
        )
        if numpy_options and as_ != "numpy":
            raise ValueError(
                "dtype, order, writable, and allow_copy require as_='numpy'"
            )
        if as_ == "numpy":
            return self.to_numpy(
                columns=columns,
                dtype=dtype,
                order=order,
                writable=writable,
                allow_copy=allow_copy,
                engine=engine,
            )
        selected = None
        if columns is not None:
            from tidy3.tidyselect import resolve_selection

            selected = resolve_selection(self, [columns])
        if self.backend == "pandas":
            pdf = self._data if selected is None else self._data.loc[:, selected]
            if as_ == "pandas":
                if arrow_backed:
                    return pdf.convert_dtypes(dtype_backend="pyarrow")
                return pdf
            if as_ == "polars":
                return pl.from_pandas(pdf)
            if as_ == "arrow":
                return pl.from_pandas(pdf).to_arrow()
            raise ValueError(
                "as_ must be 'polars', 'pandas', 'arrow', or 'numpy'"
            )
        lf = self._lf if selected is None else self._lf.select(selected)
        df = lf.collect(engine=engine)
        if as_ == "polars":
            return df
        if as_ == "pandas":
            return df.to_pandas(use_pyarrow_extension_array=arrow_backed)
        if as_ == "arrow":
            return df.to_arrow()
        raise ValueError(
            "as_ must be 'polars', 'pandas', 'arrow', or 'numpy'"
        )

    def to_polars(
        self, *, columns: Any = None, engine: Any = "auto"
    ) -> pl.DataFrame:
        return self.collect(as_="polars", columns=columns, engine=engine)

    def to_pandas(
        self,
        *,
        columns: Any = None,
        arrow_backed: bool = False,
        engine: Any = "auto",
    ):
        return self.collect(
            as_="pandas",
            columns=columns,
            arrow_backed=arrow_backed,
            engine=engine,
        )

    def to_arrow(self, *, columns: Any = None, engine: Any = "auto"):
        return self.collect(as_="arrow", columns=columns, engine=engine)

    def to_numpy(
        self,
        *,
        columns: Any = None,
        dtype: Any = None,
        order: Literal["c", "fortran"] = "fortran",
        writable: bool = False,
        allow_copy: bool = True,
        engine: Any = "auto",
    ):
        """Materialize selected columns as a host NumPy matrix.

        The default Fortran order matches the columnar dataframe layout and
        offers the best chance of avoiding a copy. Use ``order="c"`` and
        ``writable=True`` for consumers that require a mutable C-contiguous
        matrix. ``allow_copy=False`` turns an otherwise implicit copy into an
        error. Polars execution is controlled independently by ``engine``.
        """
        import numpy as np

        _validate_execution_engine(engine, backend=self.backend)
        if order not in {"c", "fortran"}:
            raise ValueError("order must be 'c' or 'fortran'")

        selected = None
        if columns is not None:
            from tidy3.tidyselect import resolve_selection

            selected = resolve_selection(self, [columns])

        if self.backend == "polars":
            lf = self._lf if selected is None else self._lf.select(selected)
            frame = lf.collect(engine=engine)
            array = frame.to_numpy(
                order=order,
                writable=writable,
                allow_copy=allow_copy,
            )
            if dtype is not None and array.dtype != np.dtype(dtype):
                if not allow_copy:
                    raise RuntimeError(
                        "requested NumPy dtype conversion requires a copy"
                    )
                array = array.astype(
                    dtype,
                    order="C" if order == "c" else "F",
                    copy=False,
                )
            return array

        frame = self._data if selected is None else self._data.loc[:, selected]
        array = frame.to_numpy(dtype=dtype, copy=False)
        if not allow_copy and array.size:
            shares_data = any(
                np.shares_memory(array, frame[name].to_numpy(copy=False))
                for name in frame.columns
            )
            if not shares_data:
                raise RuntimeError(
                    "pandas cannot provide this NumPy matrix without copying"
                )
        contiguous = (
            array.flags.c_contiguous
            if order == "c"
            else array.flags.f_contiguous
        )
        if not contiguous:
            if not allow_copy:
                raise RuntimeError(
                    f"NumPy order={order!r} requires a copy for this frame"
                )
            array = np.array(
                array, dtype=dtype, order="C" if order == "c" else "F", copy=True
            )
        if writable and not array.flags.writeable:
            if not allow_copy:
                raise RuntimeError("a writable NumPy matrix requires a copy")
            array = array.copy(order="C" if order == "c" else "F")
        return array

    def __array__(self, dtype: Any = None, copy: bool | None = None):
        """NumPy array protocol; materializes the complete lazy plan."""
        array = self.to_numpy(dtype=dtype, allow_copy=copy is not False)
        if copy is True:
            return array.copy(order="K")
        return array

    # ── write results ──────────────────────────────────────────────────
    def write_csv(
        self,
        path: Any,
        *,
        include_header: bool = True,
        separator: str = ",",
        engine: Any = "auto",
        **kwargs: Any,
    ) -> None:
        """Execute and write CSV; Polars plans stream directly to disk."""
        _validate_execution_engine(engine, backend=self.backend)
        if self.backend == "polars":
            if _uses_gpu_engine(engine):
                self.collect(as_="polars", engine=engine).write_csv(
                    path,
                    include_header=include_header,
                    separator=separator,
                    **kwargs,
                )
                return
            self._lf.sink_csv(
                path,
                include_header=include_header,
                separator=separator,
                engine=engine,
                **kwargs,
            )
            return
        self._data.to_csv(
            path,
            index=False,
            header=include_header,
            sep=separator,
            **kwargs,
        )

    def write_parquet(
        self,
        path: Any,
        *,
        compression: str = "zstd",
        engine: Any = "auto",
        **kwargs: Any,
    ) -> None:
        """Execute and write Parquet; Polars plans stream directly to disk."""
        _validate_execution_engine(engine, backend=self.backend)
        if self.backend == "polars":
            if _uses_gpu_engine(engine):
                self.collect(as_="polars", engine=engine).write_parquet(
                    path, compression=compression, **kwargs
                )
                return
            self._lf.sink_parquet(
                path, compression=compression, engine=engine, **kwargs
            )
            return
        self._data.to_parquet(
            path, index=False, compression=compression, **kwargs
        )

    def write_ipc(
        self,
        path: Any,
        *,
        compression: str | None = "uncompressed",
        engine: Any = "auto",
        **kwargs: Any,
    ) -> None:
        """Execute and write Arrow IPC/Feather output."""
        _validate_execution_engine(engine, backend=self.backend)
        if self.backend == "polars":
            if _uses_gpu_engine(engine):
                self.collect(as_="polars", engine=engine).write_ipc(
                    path, compression=compression, **kwargs
                )
                return
            self._lf.sink_ipc(
                path, compression=compression, engine=engine, **kwargs
            )
            return
        self._data.reset_index(drop=True).to_feather(
            path, compression=compression, **kwargs
        )

    def write_excel(
        self,
        path: Any,
        *,
        worksheet: str | None = None,
        engine: Any = "auto",
        **kwargs: Any,
    ) -> None:
        """Materialize and write XLSX output using Polars/XlsxWriter."""
        try:
            self.collect(as_="polars", engine=engine).write_excel(
                workbook=path, worksheet=worksheet, **kwargs
            )
        except ModuleNotFoundError as error:
            if error.name == "xlsxwriter" or "xlsxwriter" in str(error).lower():
                raise ImportError(
                    "Excel output requires XlsxWriter; install tidy3[excel]"
                ) from error
            raise

    def lazy(self) -> pl.LazyFrame:
        return self._lf

    def explain(self, **kwargs: Any) -> str:
        if self.backend == "pandas":
            return "<pandas backend: eager, no query plan>"
        return self._lf.explain(**kwargs)

    def preview(self, n: int | None = None) -> Any:
        """Materialize only the first *n* rows (for inspection)."""
        n = get_options().preview_rows if n is None else n
        if self.backend == "pandas":
            return self._data.head(n)
        return self._lf.head(n).collect()

    def show(self, n: int | None = None) -> None:
        print(self.preview(n))

    # ── plot3 bridge ────────────────────────────────────────────────────
    def ggplot(self, mapping=None, **kwargs: Any):
        """Hand off to plot3.ggplot (pandas). Optional dependency."""
        try:
            from plot3 import ggplot
        except ImportError as e:
            raise ImportError(
                "plot3 is not installed. Clone https://github.com/rleyvasal/plot3 "
                "or ensure plot3 is on PYTHONPATH."
            ) from e
        return ggplot(self.to_pandas(), mapping, **kwargs)

    # ── display (match Polars table formatting) ─────────────────────────
    def _preview_df(self) -> Any:
        opts = get_options()
        n = opts.preview_rows if opts.preview else 10
        if self.backend == "pandas":
            return self._data.head(n)
        return self._lf.head(n).collect()

    def _caption(self) -> str:
        be = ", pandas" if self.backend == "pandas" else ""
        groups = f" groups={self._groups}" if self._groups else ""
        rowwise = " rowwise" if self._rowwise else ""
        return f"TidyFrame (preview{be}{groups}{rowwise})"

    def __repr__(self) -> str:
        try:
            pdf = self._preview_df()
        except Exception as e:  # pragma: no cover
            return f"<TidyFrame groups={self._groups!r} (preview failed: {e})>"
        # Same text table style as a bare polars/pandas DataFrame
        return f"{self._caption()}\n{pdf!r}"

    def _repr_html_(self) -> str:
        """Inline-styled table — survives CRAFT's remote republish (no <style>)."""
        try:
            pdf = self._preview_df()
        except Exception as e:  # pragma: no cover
            return f"<pre>TidyFrame preview failed: {e}</pre>"
        from tidy3.display import df_to_html

        return df_to_html(pdf, caption=self._caption())

    # NOTE: no _repr_mimebundle_ — SolveIt renders text/plain when a
    # mimebundle is present; separate __repr__/_repr_html_ (like polars
    # itself) makes it pick the HTML table.

    def __len__(self) -> int:
        if self.backend == "pandas":
            return len(self._data)
        return int(self._lf.select(pl.len()).collect().item())


def tidy(data: Any = None, backend: str | None = None, **kwargs: Any) -> TidyFrame:
    """Wrap data as a TidyFrame.

    Accepts pandas DataFrame, polars DataFrame/LazyFrame, dict of columns,
    or keyword columns like ``tidy(x=[1,2], y=[3,4])``.

    ``backend`` selects the engine: ``"polars"`` (lazy, default) or
    ``"pandas"`` (eager — for 1:1 engine comparisons). The default comes
    from ``options(backend=...)``.
    """
    # Best-effort: multi-line >> rewrite in IPython/SolveIt/CRAFT.
    try:
        from tidy3.jupyter import ensure_ipython_integration

        ensure_ipython_integration(quiet=True)
    except Exception:
        pass

    if data is None and kwargs:
        data = kwargs
        kwargs = {}
    if data is None:
        raise TypeError("tidy() requires data or column kwargs")

    backend = backend or getattr(get_options(), "backend", "polars")
    if backend not in ("polars", "pandas"):
        raise ValueError(f"backend must be 'polars' or 'pandas', got {backend!r}")

    if backend == "pandas":
        import pandas as pd

        if isinstance(data, TidyFrame):
            if data.backend == "pandas":
                return data
            return TidyFrame(
                data.collect(as_="pandas"),
                groups=data._groups,
                rowwise=data._rowwise,
                group_drop=data._group_drop,
                category_levels=data._category_levels,
            )
        if isinstance(data, pd.DataFrame):
            return TidyFrame(data)
        if isinstance(data, pl.LazyFrame):
            return TidyFrame(data.collect().to_pandas())
        if isinstance(data, pl.DataFrame):
            return TidyFrame(data.to_pandas())
        return TidyFrame(pd.DataFrame(data))

    if isinstance(data, TidyFrame):
        if data.backend == "polars":
            return data
        return TidyFrame(
            pl.from_pandas(data._pdf).lazy(),
            groups=data._groups,
            rowwise=data._rowwise,
            group_drop=data._group_drop,
            category_levels=data._category_levels,
        )
    if isinstance(data, pl.LazyFrame):
        return TidyFrame(data)
    if isinstance(data, pl.DataFrame):
        return TidyFrame(data.lazy())
    # pandas
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            category_levels = {
                str(name): series.cat.categories.tolist()
                for name, series in data.items()
                if isinstance(series.dtype, pd.CategoricalDtype)
            }
            return TidyFrame(
                pl.from_pandas(data).lazy(),
                category_levels=category_levels,
            )
    except ImportError:
        pass
    if isinstance(data, dict):
        return TidyFrame(pl.DataFrame(data).lazy())
    # fallback: try polars constructor
    return TidyFrame(pl.DataFrame(data).lazy())
