"""TidyFrame: lazy dplyr-like frame over Polars LazyFrame."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Callable, Literal, overload

import polars as pl

from tidy3.options import get_options, options  # re-export

if TYPE_CHECKING:
    from tidy3.verbs import Verb

__all__ = ["TidyFrame", "tidy", "options"]

_POLARS_EXECUTION_ENGINES = frozenset({"auto", "streaming", "gpu"})
_KEEP = object()
# Reasons the current verb read row data to build its plan; see note_data_read().
_DATA_READS: ContextVar[list[str] | None] = ContextVar("tidy3_data_reads", default=None)


def note_data_read(reason: str) -> None:
    """Record that the running verb built its plan from values in the data.

    Such a plan holds those values as constants (factor levels, pivot
    names, completed values), so it is right only for this data and
    :meth:`TidyFrame.to_polars_code` refuses to export it. Checks that only
    raise errors do not call this.
    """
    reads = _DATA_READS.get()
    if reads is not None and reason not in reads:
        reads.append(reason)


class FactorLevels(list):
    """Factor level names plus whether the factor is ordered.

    A list subclass so ``levels == ["a", "b"]``, ``enumerate(levels)``, and
    ``pl.Enum(levels)`` keep working. ``list(levels)`` drops the flag.
    """

    def __init__(self, levels: Any, ordered: bool = False) -> None:
        super().__init__(levels)
        self.ordered = bool(ordered)


def _as_levels(levels: Any, ordered: bool | None = None) -> FactorLevels:
    """Wrap *levels* without dropping an ordered flag already stored on it."""
    if isinstance(levels, FactorLevels) and ordered is None:
        return levels
    if ordered is None:
        ordered = bool(getattr(levels, "ordered", False))
    return FactorLevels(list(levels), ordered=bool(ordered))


def _recorded_levels(levels: dict[str, Any] | None) -> dict[str, FactorLevels]:
    return {str(name): _as_levels(values) for name, values in (levels or {}).items()}


def _is_categorical_dtype(dtype: Any) -> bool:
    import pandas as pd

    if isinstance(dtype, pd.CategoricalDtype):
        return True
    if isinstance(dtype, pd.ArrowDtype):
        return "dictionary" in str(dtype)
    return False


def _numeric_level(level: Any) -> bool:
    return isinstance(level, (int, float)) and not isinstance(level, bool)


def _restore_factors(pdf: Any, levels: dict[str, Any]) -> Any:
    """Put recorded factor levels back onto a pandas frame.

    Polars drops unused levels and the ordered flag when a pandas factor is
    imported. Restoration runs only when the collected column is already
    categorical, numeric levels survived as numbers, or the column is empty
    or entirely missing. A string column is left alone, so a later overwrite
    is not turned back into a factor or into missing values.
    """
    if not levels:
        return pdf
    import pandas as pd

    for name, recorded in levels.items():
        if name not in pdf.columns:
            continue
        categories = list(recorded)
        ordered = bool(getattr(recorded, "ordered", False))
        series = pdf[name]
        if (
            isinstance(series.dtype, pd.CategoricalDtype)
            and list(series.cat.categories) == categories
            and bool(series.cat.ordered) is ordered
        ):
            continue
        try:
            present = [value for value in series.tolist() if not pd.isna(value)]
            if any(value not in set(categories) for value in present):
                continue
            numeric = bool(categories) and all(_numeric_level(level) for level in categories)
            if not (
                _is_categorical_dtype(series.dtype)
                or not present
                or (numeric and pd.api.types.is_numeric_dtype(series))
            ):
                continue
            pdf[name] = pd.Categorical(series.tolist(), categories=categories, ordered=ordered)
        except (TypeError, ValueError):
            continue
    return pdf


def _validate_execution_engine(engine: Any) -> None:
    if isinstance(engine, str) and engine not in _POLARS_EXECUTION_ENGINES:
        choices = ", ".join(sorted(_POLARS_EXECUTION_ENGINES))
        raise ValueError(f"engine must be one of: {choices}")


def _uses_gpu_engine(engine: Any) -> bool:
    return engine == "gpu" or isinstance(engine, pl.GPUEngine)


def _from_pandas(pdf: Any) -> pl.DataFrame:
    """``pl.from_pandas`` that keeps a text factor's levels as a ``pl.Enum``.

    Plain ``from_pandas`` gives ``pl.Categorical``, which drops level order,
    while Polars-side ``fct_*`` results are ``pl.Enum``.
    """
    import pandas as pd

    df = pl.from_pandas(pdf)
    casts = []
    for name, series in pdf.items():
        if not isinstance(series.dtype, pd.CategoricalDtype):
            continue
        levels = series.cat.categories.tolist()
        if all(isinstance(level, str) for level in levels):
            casts.append(pl.col(str(name)).cast(pl.String).cast(pl.Enum(levels)))
    return df.with_columns(casts) if casts else df


def _merge_reads(*groups: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(reason for group in groups for reason in group))


def _pandas_levels(pdf: Any) -> dict[str, FactorLevels]:
    """Factor levels and ordered flags of a pandas frame's categorical columns."""
    import pandas as pd

    return {
        str(name): FactorLevels(series.cat.categories.tolist(), ordered=bool(series.cat.ordered))
        for name, series in pdf.items()
        if isinstance(series.dtype, pd.CategoricalDtype)
    }


def _excel_frame(frame: Any, engine: Any) -> pl.DataFrame:
    if isinstance(frame, TidyFrame):
        return frame.collect(as_="polars", engine=engine)
    if isinstance(frame, pl.DataFrame):
        return frame
    if isinstance(frame, pl.LazyFrame):
        return frame.collect()
    try:
        import pandas as pd

        if isinstance(frame, pd.DataFrame):
            return pl.from_pandas(frame)
    except ImportError:
        pass
    raise TypeError("write_excel() sheets must be tidy, Polars, or pandas frames")


class TidyFrame:
    """dplyr-style frame with ``>>`` pipes over a lazy Polars query.

    pandas DataFrames are accepted and converted; ``collect(as_="pandas")``
    hands a result back to pandas.
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
        # Parse failures captured at mutate time: a lazy plan that still
        # reads the pre-mutate source.
        "_problems",
        # Steps whose plan holds values read from the data (note_data_read).
        "_data_reads",
        # The plan's schema, resolved once: a frame's plan never changes.
        "_schema_cache",
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
        problems: Any | None = None,
        data_reads: tuple[str, ...] = (),
    ):
        if not isinstance(data, pl.LazyFrame):
            import pandas as pd

            if not isinstance(data, pd.DataFrame):
                raise TypeError("TidyFrame requires a polars LazyFrame or pandas DataFrame")
            if category_levels is None:
                category_levels = _pandas_levels(data)
            data = _from_pandas(data).lazy()
        self._data = data
        self._groups = list(groups) if groups else None
        self._rowwise = bool(rowwise)
        self._group_drop = bool(group_drop) if self._groups else True
        self._category_levels = _recorded_levels(category_levels)
        self._select_base = select_base
        self._problems = problems
        self._data_reads = tuple(data_reads)
        self._schema_cache = None

    @property
    def _lf(self) -> pl.LazyFrame:
        return self._data

    def _schema(self) -> pl.Schema:
        """The plan's schema, resolved on first use (Polars re-resolves each call)."""
        if self._schema_cache is None:
            self._schema_cache = self._data.collect_schema()
        return self._schema_cache

    # ── EDA / inspection (schema-first; row counts materialize lazily) ──

    @property
    def columns(self) -> list[str]:
        """Column names as a Python ``list[str]`` (Polars-style).

        Schema-only — does not scan row data.
        """
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
        schema = self._lf.collect_schema()
        return {name: dtype for name, dtype in schema.items()}

    @property
    def schema(self) -> Any:
        """The Polars ``Schema``, without collecting rows."""
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
        problems: Any = _KEEP,
        data_reads: tuple[str, ...] = (),
    ) -> TidyFrame:
        # Default: clear select_base (most verbs invalidate the rewrite window).
        # select/filter set select_base=...; pass keep_select_base to preserve.
        if keep_select_base and select_base is None:
            select_base = self._select_base
        if problems is _KEEP:
            problems = self._problems
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
            problems=problems,
            data_reads=_merge_reads(self._data_reads, data_reads),
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

    def set_names(self, *names: Any) -> TidyFrame:
        from tidy3.verbs import set_names as set_names_verb

        return self >> set_names_verb(*names)

    def clean_names(
        self, *, case: str = "snake", unique: bool = True
    ) -> TidyFrame:
        from tidy3.verbs import clean_names as clean_names_verb

        return self >> clean_names_verb(case=case, unique=unique)

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

    def get_dupes(self, *cols: Any) -> TidyFrame:
        from tidy3.verbs import get_dupes as get_dupes_verb

        return self >> get_dupes_verb(*cols)

    def remove_empty(self, which: Any = "both", *, cutoff: float = 1) -> TidyFrame:
        from tidy3.janitor import remove_empty as remove_empty_verb

        return self >> remove_empty_verb(which, cutoff=cutoff)

    def remove_constant(self, *, na_rm: bool = False) -> TidyFrame:
        from tidy3.janitor import remove_constant as remove_constant_verb

        return self >> remove_constant_verb(na_rm=na_rm)

    def row_to_names(
        self,
        row_number: Any,
        *,
        remove_row: bool = True,
        remove_rows_above: bool = True,
        value: Any = None,
        column: Any = None,
    ) -> TidyFrame:
        from tidy3.janitor import row_to_names as row_to_names_verb

        return self >> row_to_names_verb(
            row_number,
            remove_row=remove_row,
            remove_rows_above=remove_rows_above,
            value=value,
            column=column,
        )

    def find_header(self, *, value: Any = None, column: Any = 1) -> int:
        from tidy3.janitor import find_header as find_header_verb

        return self >> find_header_verb(value=value, column=column)

    def compare_df_cols(self, *frames: Any, **kwargs: Any) -> TidyFrame:
        from tidy3.janitor import compare_df_cols as compare_df_cols_fn

        return compare_df_cols_fn(self, *frames, **kwargs)

    def compare_df_cols_same(self, *frames: Any, **kwargs: Any) -> bool:
        from tidy3.janitor import compare_df_cols_same as compare_same

        return compare_same(self, *frames, **kwargs)

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

    def head(self, n: int = 6) -> TidyFrame:
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
    def compute(self, *, engine: Any = "auto") -> TidyFrame:
        """Run the plan now and keep piping from the result (dplyr's ``compute()``).

        A lazy frame re-runs its whole plan each time it is materialized,
        so a pipe that starts with ``scan_csv(url)`` downloads the file again
        for every ``collect()``, plot, or ``to_numpy()``. Computing once keeps
        the result in memory; later verbs build on it. Groups and rowwise
        state carry over. Parse failures are
        computed in the same run, so ``problems()`` afterwards reads the
        stored failures and does not read the source again.
        """
        return self._compute(engine, ("compute() replaces the plan with its result",))

    def _compute(self, engine: Any = "auto", reads: tuple[str, ...] = ()) -> TidyFrame:
        """compute() for tidy3's own steps, which record why they read data."""
        _validate_execution_engine(engine)
        if self._problems is None:
            return self._with_lf(
                self._lf.collect(engine=engine).lazy(), groups=self._groups, data_reads=reads
            )
        data, found = pl.collect_all([self._lf, self._problems], engine=engine)
        return self._with_lf(
            data.lazy(), groups=self._groups, problems=found.lazy(), data_reads=reads
        )

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
        every column into NumPy buffers at the handoff boundary. Factor
        columns recorded on a Polars frame are written back as pandas
        categoricals, including unused levels and ordered status.

        ``engine`` controls Polars execution: ``"auto"`` (the default),
        ``"streaming"``, or ``"gpu"``. A configured ``polars.GPUEngine``
        object is also accepted. Non-default engines require the Polars
        backend.
        """
        _validate_execution_engine(engine)
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
        lf = self._lf if selected is None else self._lf.select(selected)
        df = lf.collect(engine=engine)
        if as_ == "polars":
            return df
        if as_ == "pandas":
            return _restore_factors(
                df.to_pandas(use_pyarrow_extension_array=arrow_backed),
                self._category_levels,
            )
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

        _validate_execution_engine(engine)
        if order not in {"c", "fortran"}:
            raise ValueError("order must be 'c' or 'fortran'")

        selected = None
        if columns is not None:
            from tidy3.tidyselect import resolve_selection

            selected = resolve_selection(self, [columns])

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
        _validate_execution_engine(engine)
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

    def write_parquet(
        self,
        path: Any,
        *,
        compression: str = "zstd",
        engine: Any = "auto",
        **kwargs: Any,
    ) -> None:
        """Execute and write Parquet; Polars plans stream directly to disk."""
        _validate_execution_engine(engine)
        if _uses_gpu_engine(engine):
            self.collect(as_="polars", engine=engine).write_parquet(
                path, compression=compression, **kwargs
            )
            return
        self._lf.sink_parquet(
            path, compression=compression, engine=engine, **kwargs
        )
        return

    def write_ipc(
        self,
        path: Any,
        *,
        compression: str | None = "uncompressed",
        engine: Any = "auto",
        **kwargs: Any,
    ) -> None:
        """Execute and write Arrow IPC/Feather output."""
        _validate_execution_engine(engine)
        if _uses_gpu_engine(engine):
            self.collect(as_="polars", engine=engine).write_ipc(
                path, compression=compression, **kwargs
            )
            return
        self._lf.sink_ipc(
            path, compression=compression, engine=engine, **kwargs
        )
        return

    def problems(self) -> TidyFrame:
        """Parse failures for this frame. See :func:`tidy3.convert.problems`."""
        from tidy3.convert import problems_frame

        return problems_frame(self)

    def stop_for_problems(self) -> TidyFrame:
        """Raise when this frame has parse failures."""
        from tidy3.convert import stop_for_problems

        return stop_for_problems(self)

    def write_excel(
        self,
        path: Any,
        *,
        worksheet: str | None = None,
        sheets: dict[str, Any] | None = None,
        engine: Any = "auto",
        **kwargs: Any,
    ) -> None:
        """Write an xlsx workbook.

        ``worksheet`` names the single sheet for this frame. ``sheets`` is
        a dict of sheet name to frame and writes exactly those sheets, not
        this frame as well. Passing both is an error. Text that starts with
        ``=`` is stored as text, not run as a formula.
        """
        if sheets is not None and worksheet is not None:
            raise TypeError(
                "write_excel() takes worksheet= for this frame, or sheets= "
                "for several frames, not both"
            )
        if sheets is not None and (not isinstance(sheets, dict) or not sheets):
            raise TypeError(
                "write_excel() sheets= must be a non-empty dict of sheet name to frame"
            )
        _validate_execution_engine(engine)
        try:
            import xlsxwriter
        except ImportError as error:
            raise ImportError(
                "Excel output requires XlsxWriter. "
                "Install it with pip install 'tidy3[excel]'."
            ) from error
        book = xlsxwriter.Workbook(path, {"strings_to_formulas": False})
        try:
            targets = list(sheets.items()) if sheets is not None else [(worksheet, self)]
            for name, frame in targets:
                data = _excel_frame(frame, engine)
                options = dict(kwargs)
                if name is not None:
                    options["worksheet"] = name
                data.write_excel(workbook=book, **options)
        finally:
            book.close()

    def lazy(self) -> pl.LazyFrame:
        return self._lf

    def explain(self, **kwargs: Any) -> str:
        return self._lf.explain(**kwargs)

    def to_polars_code(self, *, name: str = "pipeline") -> str:
        """Python source of a function that runs this pipeline with Polars only.

        Each input frame becomes an argument (``df``, then ``df2``, ...). Raises
        :class:`tidy3.ExportError` for a step with no plain-Polars form: a
        column computed by a Python function, or a step that read values from
        the data to build its query (factor levels, ``pivot_wider`` names).
        """
        from tidy3.polars_export import to_polars_code

        return to_polars_code(self, name=name)

    def preview(self, n: int | None = None) -> Any:
        """Materialize only the first *n* rows (for inspection)."""
        n = get_options().preview_rows if n is None else n
        return self._lf.head(n).collect()

    def show(self, n: int | None = None) -> None:
        """Print the first *n* rows, all of them, like tibble's ``print(n = )``."""
        print(self._preview_text(n))

    # ── plot3 bridge ────────────────────────────────────────────────────
    def ggplot(self, mapping=None, **kwargs: Any):
        """Hand off to plot3.ggplot. Optional dependency."""
        try:
            from plot3 import ggplot
        except ImportError as e:
            raise ImportError(
                "plot3 is not installed: pip install plot3"
            ) from e
        # plot3 reads tidy3 tables directly (no pandas copy).
        return ggplot(self, mapping, **kwargs)

    # ── display (match Polars table formatting) ─────────────────────────
    def _preview_df(self) -> Any:
        return self._preview_parts()[0]

    def _preview_parts(self, n: int | None = None) -> tuple[Any, int | None]:
        """The first *n* rows and the total row count (None when not counted).

        Like a tibble, the shape is the whole frame's, not the preview's.
        With ``options(preview_count=False)`` a lazy frame is not counted.
        """
        opts = get_options()
        if n is None:
            n = opts.preview_rows if opts.preview else 10
        if opts.preview_count:
            head, count = pl.collect_all([self._lf.head(n), self._lf.select(pl.len())])
            return head, int(count.item())
        # One extra row tells whether more rows follow.
        head = self._lf.head(n + 1).collect()
        return head.head(n), (head.height if head.height <= n else None)

    def _preview_text(self, n: int | None = None) -> str:
        pdf, total = self._preview_parts(n)
        shown = len(pdf)
        # The preview is already cut to *n* rows; stop Polars' own printer
        # (10 rows) from cutting it again.
        with pl.Config(tbl_rows=max(shown, 1)):
            text = repr(pdf)
        lines = text.splitlines()
        if lines and lines[0].startswith("shape:"):
            lines = lines[1:]
        rows = "?" if total is None else f"{total}"
        out = [self._caption(), f"shape: ({rows}, {pdf.shape[1]})", *lines]
        if total is None:
            out.append("… more rows")
        elif total > shown:
            more = total - shown
            out.append(f"… {more:,} more row{'s' if more != 1 else ''}")
        return "\n".join(out)

    def _caption(self) -> str:
        be = ""
        groups = f" groups={self._groups}" if self._groups else ""
        rowwise = " rowwise" if self._rowwise else ""
        return f"TidyFrame (preview{be}{groups}{rowwise})"

    def __repr__(self) -> str:
        try:
            return self._preview_text()
        except Exception as e:
            from tidy3.errors import preview_error_text

            return preview_error_text(e)

    def _repr_html_(self) -> str:
        """Inline-styled table — survives CRAFT's remote republish (no <style>)."""
        try:
            pdf, total = self._preview_parts()
        except Exception as e:
            from tidy3.errors import preview_error_html

            plan = None
            try:
                plan = self._lf.explain(optimized=False)
            except Exception:
                plan = None
            return preview_error_html(e, plan)
        from tidy3.display import df_to_html

        return df_to_html(
            pdf,
            caption=self._caption(),
            max_rows=len(pdf),
            total_rows=total,
            rows_unknown=total is None,
        )

    # NOTE: no _repr_mimebundle_ — SolveIt renders text/plain when a
    # mimebundle is present; separate __repr__/_repr_html_ (like polars
    # itself) makes it pick the HTML table.

    def __len__(self) -> int:
        return int(self._lf.select(pl.len()).collect().item())


def tidy(data: Any = None, **kwargs: Any) -> TidyFrame:
    """Wrap data as a TidyFrame.

    Accepts a pandas DataFrame, a Polars DataFrame or LazyFrame, a dict of
    columns, or keyword columns like ``tidy(x=[1,2], y=[3,4])``. pandas input
    becomes a lazy Polars frame that keeps its factor levels; use
    ``collect(as_="pandas")`` to hand a result back to pandas.
    """
    if "backend" in kwargs:
        raise TypeError(_NO_PANDAS_BACKEND)
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
    if isinstance(data, TidyFrame):
        return data
    if isinstance(data, pl.LazyFrame):
        return TidyFrame(data)
    if isinstance(data, pl.DataFrame):
        return TidyFrame(data.lazy())
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return TidyFrame(data)
    except ImportError:
        pass
    return TidyFrame(pl.DataFrame(data).lazy())


_NO_PANDAS_BACKEND = (
    "tidy3 runs on Polars only; the pandas backend was removed in 0.8. "
    "pandas DataFrames are still accepted as input, and collect(as_=\"pandas\") "
    "returns a pandas result."
)
