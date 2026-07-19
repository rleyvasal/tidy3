"""TidyFrame: lazy dplyr-like frame over Polars LazyFrame."""

from __future__ import annotations

from typing import Any, Literal

import polars as pl

from tidy3.options import get_options, options  # re-export

__all__ = ["TidyFrame", "tidy", "options"]


class TidyFrame:
    """dplyr-style frame with ``>>`` pipes: lazy Polars (default) or eager pandas.

    The pandas backend exists for 1:1 engine comparisons (same pipeline,
    different engine) — see ``tidy(df, backend="pandas")`` and ``tidy3.bench``.
    """

    __slots__ = ("_data", "_groups", "_rowwise")

    def __init__(
        self,
        data: Any,
        groups: list[str] | None = None,
        *,
        rowwise: bool = False,
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

    def _with_lf(
        self,
        lf: pl.LazyFrame,
        groups: list[str] | None = None,
        *,
        rowwise: bool | None = None,
    ) -> TidyFrame:
        return TidyFrame(
            lf,
            groups=groups,
            rowwise=self._rowwise if rowwise is None else rowwise,
        )

    def _with_pdf(
        self,
        pdf: Any,
        groups: list[str] | None = None,
        *,
        rowwise: bool | None = None,
    ) -> TidyFrame:
        return TidyFrame(
            pdf,
            groups=groups,
            rowwise=self._rowwise if rowwise is None else rowwise,
        )

    # ── pipe ────────────────────────────────────────────────────────────
    def __rshift__(self, other: Any) -> Any:
        """``tf >> verb`` — prefer Verb.__rrshift__, but support callables."""
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

    def mutate(self, *specs: Any, by: Any = None, **kwargs: Any) -> TidyFrame:
        from tidy3.verbs import mutate as mutate_verb

        return self >> mutate_verb(*specs, by=by, **kwargs)

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

    def arrange(self, *keys: Any) -> TidyFrame:
        from tidy3.verbs import arrange as arrange_verb

        return self >> arrange_verb(*keys)

    def distinct(self, *cols: str) -> TidyFrame:
        from tidy3.verbs import distinct as distinct_verb

        return self >> distinct_verb(*cols)

    def group_by(self, *cols: str) -> TidyFrame:
        from tidy3.verbs import group_by as group_by_verb

        return self >> group_by_verb(*cols)

    def rowwise(self, *cols: Any) -> TidyFrame:
        from tidy3.verbs import rowwise as rowwise_verb

        return self >> rowwise_verb(*cols)

    def ungroup(self) -> TidyFrame:
        from tidy3.verbs import ungroup as ungroup_verb

        return self >> ungroup_verb()

    def summarise(self, *specs: Any, by: Any = None, **kwargs: Any) -> TidyFrame:
        from tidy3.verbs import summarise as summarise_verb

        return self >> summarise_verb(*specs, by=by, **kwargs)

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
    ) -> TidyFrame:
        from tidy3.verbs import count as count_verb

        return self >> count_verb(*cols, wt=wt, sort=sort, name=name)

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
    def collect(self, as_: Literal["polars", "pandas", "arrow"] = "polars") -> Any:
        if self.backend == "pandas":
            pdf = self._data
            if as_ == "pandas":
                return pdf
            if as_ == "polars":
                return pl.from_pandas(pdf)
            if as_ == "arrow":
                return pl.from_pandas(pdf).to_arrow()
            raise ValueError("as_ must be 'polars', 'pandas', or 'arrow'")
        df = self._lf.collect()
        if as_ == "polars":
            return df
        if as_ == "pandas":
            return df.to_pandas()
        if as_ == "arrow":
            return df.to_arrow()
        raise ValueError("as_ must be 'polars', 'pandas', or 'arrow'")

    def to_polars(self) -> pl.DataFrame:
        return self.collect(as_="polars")

    def to_pandas(self):
        return self.collect(as_="pandas")

    def to_arrow(self):
        return self.collect(as_="arrow")

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
        )
    if isinstance(data, pl.LazyFrame):
        return TidyFrame(data)
    if isinstance(data, pl.DataFrame):
        return TidyFrame(data.lazy())
    # pandas
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return TidyFrame(pl.from_pandas(data).lazy())
    except ImportError:
        pass
    if isinstance(data, dict):
        return TidyFrame(pl.DataFrame(data).lazy())
    # fallback: try polars constructor
    return TidyFrame(pl.DataFrame(data).lazy())
