"""TidyFrame: lazy dplyr-like frame over Polars LazyFrame."""

from __future__ import annotations

from typing import Any, Literal

import polars as pl

from tidy3.options import get_options, options  # re-export

__all__ = ["TidyFrame", "tidy", "options"]


class TidyFrame:
    """Lazy frame with dplyr-style methods and ``>>`` pipe support."""

    __slots__ = ("_lf", "_groups")

    def __init__(self, lf: pl.LazyFrame, groups: list[str] | None = None):
        if not isinstance(lf, pl.LazyFrame):
            raise TypeError("TidyFrame requires a polars LazyFrame")
        self._lf = lf
        self._groups = list(groups) if groups else None

    def _with_lf(self, lf: pl.LazyFrame, groups: list[str] | None = None) -> TidyFrame:
        return TidyFrame(lf, groups=groups)

    # ── pipe ────────────────────────────────────────────────────────────
    def __rshift__(self, other: Any) -> Any:
        """``tf >> verb`` — prefer Verb.__rrshift__, but support callables."""
        from tidy3.verbs import Verb

        if isinstance(other, Verb):
            return other.__rrshift__(self)
        if callable(other):
            return other(self)
        return NotImplemented

    # ── method-chain aliases ────────────────────────────────────────────
    def filter(self, *predicates: Any) -> TidyFrame:  # noqa: A001
        from tidy3.verbs import filter as filter_verb

        return self >> filter_verb(*predicates)

    def mutate(self, **kwargs: Any) -> TidyFrame:
        from tidy3.verbs import mutate as mutate_verb

        return self >> mutate_verb(**kwargs)

    def transmute(self, **kwargs: Any) -> TidyFrame:
        from tidy3.verbs import transmute as transmute_verb

        return self >> transmute_verb(**kwargs)

    def select(self, *cols: Any) -> TidyFrame:
        from tidy3.verbs import select as select_verb

        return self >> select_verb(*cols)

    def drop(self, *cols: str) -> TidyFrame:
        from tidy3.verbs import drop as drop_verb

        return self >> drop_verb(*cols)

    def rename(self, **kwargs: str) -> TidyFrame:
        from tidy3.verbs import rename as rename_verb

        return self >> rename_verb(**kwargs)

    def arrange(self, *keys: Any) -> TidyFrame:
        from tidy3.verbs import arrange as arrange_verb

        return self >> arrange_verb(*keys)

    def distinct(self, *cols: str) -> TidyFrame:
        from tidy3.verbs import distinct as distinct_verb

        return self >> distinct_verb(*cols)

    def group_by(self, *cols: str) -> TidyFrame:
        from tidy3.verbs import group_by as group_by_verb

        return self >> group_by_verb(*cols)

    def ungroup(self) -> TidyFrame:
        from tidy3.verbs import ungroup as ungroup_verb

        return self >> ungroup_verb()

    def summarise(self, **kwargs: Any) -> TidyFrame:
        from tidy3.verbs import summarise as summarise_verb

        return self >> summarise_verb(**kwargs)

    summarize = summarise

    def count(self, *cols: str, name: str = "n") -> TidyFrame:
        from tidy3.verbs import count as count_verb

        return self >> count_verb(*cols, name=name)

    def head(self, n: int = 10) -> TidyFrame:
        from tidy3.verbs import head as head_verb

        return self >> head_verb(n)

    slice_head = head

    def sample_n(self, n: int, *, seed: int | None = None) -> TidyFrame:
        from tidy3.verbs import sample_n as sample_n_verb

        return self >> sample_n_verb(n, seed=seed)

    def sample_frac(self, frac: float, *, seed: int | None = None) -> TidyFrame:
        from tidy3.verbs import sample_frac as sample_frac_verb

        return self >> sample_frac_verb(frac, seed=seed)

    def left_join(self, right: Any, *, on: str | list[str] | None = None, **kwargs: Any) -> TidyFrame:
        from tidy3.verbs import left_join as left_join_verb

        return self >> left_join_verb(right, on=on, **kwargs)

    def inner_join(self, right: Any, *, on: str | list[str] | None = None, **kwargs: Any) -> TidyFrame:
        from tidy3.verbs import inner_join as inner_join_verb

        return self >> inner_join_verb(right, on=on, **kwargs)

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
        return self._lf.explain(**kwargs)

    def preview(self, n: int | None = None) -> pl.DataFrame:
        """Materialize only the first *n* rows (for inspection)."""
        n = get_options().preview_rows if n is None else n
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
    def _preview_df(self) -> pl.DataFrame:
        opts = get_options()
        n = opts.preview_rows if opts.preview else 10
        return self._lf.head(n).collect()

    def __repr__(self) -> str:
        try:
            pdf = self._preview_df()
        except Exception as e:  # pragma: no cover
            return f"<TidyFrame groups={self._groups!r} (preview failed: {e})>"
        groups = f" groups={self._groups}" if self._groups else ""
        # Same text table style as a bare polars DataFrame
        return f"TidyFrame (preview){groups}\n{pdf!r}"

    def _repr_html_(self) -> str:
        """Prefer Polars' own HTML table (not pandas)."""
        try:
            pdf = self._preview_df()
        except Exception as e:  # pragma: no cover
            return f"<pre>TidyFrame preview failed: {e}</pre>"
        groups = f" groups={self._groups}" if self._groups else ""
        header = (
            f"<div style='font:12px ui-monospace,SFMono-Regular,Menlo,monospace;"
            f"margin:0 0 4px 0;opacity:.85'>"
            f"TidyFrame (preview{groups})</div>"
        )
        # Polars DataFrame HTML (shape + styled table) — same as printing `cars`
        h = getattr(pdf, "_repr_html_", None)
        if callable(h):
            try:
                body = h()
                if body:
                    return header + body
            except Exception:
                pass
        # Fallback: monospaced polars text table (matches notebook plain output)
        return header + f"<pre style='font:12px ui-monospace,Menlo,monospace'>{pdf!r}</pre>"

    def _repr_mimebundle_(self, include=None, exclude=None):
        """Let Jupyter pick text/html or text/plain like Polars does."""
        plain = repr(self)
        try:
            html = self._repr_html_()
        except Exception:
            html = f"<pre>{plain}</pre>"
        return {"text/plain": plain, "text/html": html}

    def __len__(self) -> int:
        return int(self._lf.select(pl.len()).collect().item())


def tidy(data: Any = None, **kwargs: Any) -> TidyFrame:
    """Wrap data as a lazy TidyFrame.

    Accepts pandas DataFrame, polars DataFrame/LazyFrame, dict of columns,
    or keyword columns like ``tidy(x=[1,2], y=[3,4])``.
    """
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
