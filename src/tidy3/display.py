"""Self-contained HTML tables — inline styles only.

Polars' ``_repr_html_`` emits a ``<style>`` block plus ``class="dataframe"``
and relies on the notebook to keep both. CRAFT republishes remote output
through ``display_pub.publish``, where that styling is lost and tables
collapse. Every element here carries its style inline, so the table renders
identically local, remote, and inside sslive exports.
"""

from __future__ import annotations

import html as _html

import polars as pl

_MONO = "12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
_CELL = f"font:{_MONO};padding:3px 10px;text-align:right;white-space:pre;border:none"
_RULE = "border-bottom:1px solid rgba(127,127,127,.45)"


def _fmt(v: object) -> str:
    if v is None:
        return "null"
    return _html.escape(str(v))


def _dtype_short(t: pl.DataType) -> str:
    """Polars-style short dtype name ("i64"); falls back to str(t)."""
    try:
        from polars._plr import dtype_str_repr

        return dtype_str_repr(t)
    except Exception:
        return str(t)


def _adapt(df):
    """(height, width, names, dtypes, rows_fn) for polars or pandas frames."""
    if isinstance(df, pl.DataFrame):
        return (
            df.height,
            df.width,
            list(df.columns),
            [_dtype_short(t) for t in df.dtypes],
            lambda n: df.head(n).rows(),
        )
    # pandas (duck-typed; avoids a hard import)
    return (
        len(df),
        df.shape[1],
        [str(c) for c in df.columns],
        [str(t) for t in df.dtypes],
        lambda n: list(df.head(n).itertuples(index=False, name=None)),
    )


def df_to_html(
    df,
    *,
    caption: str | None = None,
    max_rows: int = 25,
    max_cols: int = 20,
) -> str:
    """Render a polars or pandas frame as an HTML table, all styling inline."""
    height, width, names, dtypes, rows_fn = _adapt(df)
    cols = list(range(width))
    col_gap = None
    if width > max_cols:
        left = max_cols // 2
        right = max_cols - left
        cols = list(range(left)) + list(range(width - right, width))
        col_gap = left  # position of the "…" column within `cols`

    n_rows = min(height, max_rows)
    rows = rows_fn(n_rows)

    def _tr(cells: list[str], style: str) -> str:
        tds = []
        for j, c in enumerate(cells):
            tds.append(f"<td style='{style}'>{c}</td>")
            if col_gap is not None and j + 1 == col_gap:
                tds.append(f"<td style='{style}'>…</td>")
        return "<tr>" + "".join(tds) + "</tr>"

    head_tr = _tr([f"<b>{_fmt(names[i])}</b>" for i in cols], f"{_CELL};{_RULE}")
    dtype_tr = _tr([_fmt(dtypes[i]) for i in cols], f"{_CELL};opacity:.55;{_RULE}")
    body = [_tr([_fmt(r[i]) for i in cols], _CELL) for r in rows]
    if height > n_rows:
        body.append(_tr(["…"] * len(cols), f"{_CELL};opacity:.55"))

    cap = ""
    if caption:
        cap = (
            f"<div style='font:{_MONO};opacity:.85;margin:0 0 2px 0'>"
            f"{_html.escape(caption)}</div>"
        )
    shape = (
        f"<small style='font:{_MONO};opacity:.65'>shape: ({height}, {width})</small>"
    )
    return (
        "<div style='overflow-x:auto'>"
        + cap
        + shape
        + "<table style='border-collapse:collapse;margin:2px 0'>"
        + head_tr
        + dtype_tr
        + "".join(body)
        + "</table></div>"
    )


def register_polars_formatter(ip=None) -> bool:
    """Make bare ``pl.DataFrame`` display via :func:`df_to_html` in IPython.

    Registered type formatters take precedence over ``_repr_html_``, so this
    swaps polars' class-based HTML for the inline-styled table. Used on the
    CRAFT remote kernel, where republished output loses ``<style>`` blocks.
    """
    if ip is None:
        try:
            from IPython import get_ipython

            ip = get_ipython()
        except Exception:
            return False
    if ip is None:
        return False
    try:
        fmt = ip.display_formatter.formatters["text/html"]
    except Exception:
        return False
    fmt.for_type(pl.DataFrame, lambda df: df_to_html(df))
    return True


def unregister_polars_formatter(ip=None) -> None:
    if ip is None:
        try:
            from IPython import get_ipython

            ip = get_ipython()
        except Exception:
            return
    if ip is None:
        return
    try:
        ip.display_formatter.formatters["text/html"].pop(pl.DataFrame)
    except Exception:
        pass
