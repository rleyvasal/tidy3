"""Remaining public-API benchmark cases.

Native callables use Polars only. tidy3 is used on the tidy side.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl


def _raw_full_seq(df, extra=None):
    """full_seq(x, 1): check the values lie on the grid, then fill the range."""
    values = extra.cast(pl.Float64)
    low, high = values.min(), values.max()
    offset = (values - low) % 1
    if ((offset > 1e-6) & ((1 - offset) > 1e-6)).any():
        raise ValueError("full_seq() x is not a regular sequence")
    steps = round(high - low)
    return (pl.int_range(0, steps + 1, eager=True).cast(pl.Float64) + low).to_list()


def _raw_setequal(df, extra=None):
    """Whether two frames hold the same distinct rows."""
    left = df.select("species", "island").unique()
    right = df.select("species", "island").unique(maintain_order=True).unique()
    if left.height != right.height:
        return False
    return left.join(right, on=["species", "island"], how="anti", nulls_equal=True).height == 0


def _raw_remove_empty(df, extra=None):
    """Drop the columns whose every value is missing (null, or NaN in floats)."""
    data = df.with_columns(empty=pl.lit(None, dtype=pl.String))
    present = data.select(
        (pl.col(name).is_not_null() & ~pl.col(name).is_nan() if dtype.is_float() else pl.col(name).is_not_null())
        .sum()
        .alias(name)
        for name, dtype in data.schema.items()
    ).row(0, named=True)
    return data.select(name for name, count in present.items() if count)


def attach(add, Case) -> None:
    from tidy3 import bench_api as api
    import tidy3 as t

    def verb(name, family, functions, tidy, native, methods=(), prepare=None):
        add(Case(name, family, functions, tidy, native, methods, prepare))

    def pipe(df):
        return t.tidy(df)

    # ── single-table verbs ─────────────────────────────────────────────
    verb(
        "filter", "verbs", ["filter"],
        lambda df, extra=None: (pipe(df) >> t.filter(t.col("flag")) >> t.select("id")).collect(),
        lambda df, extra=None: df.filter(pl.col("flag")).select("id"),
    )
    verb(
        "filter_out", "verbs", ["filter_out"],
        lambda df, extra=None: (pipe(df) >> t.filter_out(t.col("flag")) >> t.select("id")).collect(),
        lambda df, extra=None: df.filter(~pl.col("flag")).select("id"),
    )
    verb(
        "mutate", "verbs", ["mutate"],
        lambda df, extra=None: (pipe(df) >> t.mutate(mass=t.col("mass") * 2) >> t.select("id", "mass")).collect(),
        lambda df, extra=None: df.select("id", (pl.col("mass") * 2).alias("mass")),
    )
    verb(
        "select", "verbs", ["select"],
        lambda df, extra=None: (pipe(df) >> t.select("species", "mass")).collect(),
        lambda df, extra=None: df.select("species", "mass"),
    )
    verb(
        "arrange", "verbs", ["arrange"],
        lambda df, extra=None: (pipe(df) >> t.arrange("species", "id") >> t.select("id")).collect(),
        lambda df, extra=None: df.sort(["species", "id"], nulls_last=True, maintain_order=True).select("id"),
    )
    verb(
        "transmute", "verbs", ["transmute"],
        lambda df, extra=None: (pipe(df) >> t.transmute(out=t.col("mass") * 2)).collect(),
        lambda df, extra=None: df.select((pl.col("mass") * 2).alias("out")),
    )
    verb(
        "drop", "verbs", ["drop"],
        lambda df, extra=None: (pipe(df) >> t.drop("words")).collect(),
        lambda df, extra=None: df.drop("words"),
    )
    verb(
        "drop_na", "verbs", ["drop_na"],
        lambda df, extra=None: (pipe(df) >> t.drop_na("gap") >> t.select("id")).collect(),
        lambda df, extra=None: df.filter(pl.col("gap").is_not_null()).select("id"),
    )
    verb(
        "replace_na", "verbs", ["replace_na"],
        lambda df, extra=None: (pipe(df) >> t.replace_na({"gap": 0.0}) >> t.select("id", "gap")).collect(),
        lambda df, extra=None: df.select("id", pl.col("gap").fill_null(0.0)),
    )
    verb(
        "fill", "verbs", ["fill"],
        lambda df, extra=None: (pipe(df) >> t.fill("gap") >> t.select("id", "gap")).collect(),
        lambda df, extra=None: df.select("id", pl.col("gap").forward_fill()),
    )
    verb(
        "relocate", "verbs", ["relocate"],
        lambda df, extra=None: (pipe(df) >> t.relocate("mass")).collect(),
        lambda df, extra=None: df.select(["mass", *[name for name in df.columns if name != "mass"]]),
    )
    verb(
        "rename", "verbs", ["rename"],
        lambda df, extra=None: (pipe(df) >> t.rename(weight="mass") >> t.select("id", "weight")).collect(),
        lambda df, extra=None: df.select("id", pl.col("mass").alias("weight")),
    )
    verb(
        "rename_with", "verbs", ["rename_with"],
        lambda df, extra=None: (pipe(df) >> t.rename_with(str.upper, "mass") >> t.select("id", "MASS")).collect(),
        lambda df, extra=None: df.select("id", pl.col("mass").alias("MASS")),
    )
    verb(
        "set_names", "verbs", ["set_names"],
        lambda df, extra=None: (pipe(df.select("id", "species", "mass")) >> t.set_names("a", "b", "c")).collect(),
        lambda df, extra=None: df.select("id", "species", "mass").rename({"id": "a", "species": "b", "mass": "c"}),
    )
    verb(
        "pull", "verbs", ["pull"],
        lambda df, extra=None: pipe(df) >> t.pull("mass"),
        lambda df, extra=None: df.get_column("mass"),
    )
    verb(
        "count", "verbs", ["count"],
        lambda df, extra=None: (pipe(df) >> t.count("species")).collect(),
        lambda df, extra=None: df.group_by("species", maintain_order=True).agg(pl.len().alias("n")).sort("species"),
    )
    verb(
        "tally", "verbs", ["tally", "group_by"],
        lambda df, extra=None: (pipe(df) >> t.group_by("g") >> t.tally()).collect(),
        lambda df, extra=None: df.group_by("g", maintain_order=True).agg(pl.len().alias("n")).sort("g"),
    )
    verb(
        "add_count", "verbs", ["add_count"],
        lambda df, extra=None: (pipe(df) >> t.add_count("species") >> t.select("id", "n")).collect(),
        lambda df, extra=None: df.select("id", pl.len().over("species").alias("n")),
    )
    verb(
        "add_tally", "verbs", ["add_tally"],
        lambda df, extra=None: (pipe(df) >> t.group_by("g") >> t.add_tally() >> t.ungroup() >> t.select("id", "n")).collect(),
        lambda df, extra=None: df.select("id", pl.len().over("g").alias("n")),
    )
    verb(
        "summarise", "verbs", ["summarise"],
        lambda df, extra=None: (pipe(df) >> t.group_by("g") >> t.summarise(out=t.mean("mass"))).collect(),
        lambda df, extra=None: df.group_by("g", maintain_order=True).agg(pl.col("mass").mean().alias("out")).sort("g"),
    )
    verb(
        "summarize", "verbs", ["summarize"],
        lambda df, extra=None: (pipe(df) >> t.group_by("g") >> t.summarize(out=t.mean("mass"))).collect(),
        lambda df, extra=None: df.group_by("g", maintain_order=True).agg(pl.col("mass").mean().alias("out")).sort("g"),
    )
    verb(
        "reframe", "verbs", ["reframe"],
        lambda df, extra=None: (pipe(df) >> t.group_by("g") >> t.reframe(out=t.mean("mass"))).collect(),
        lambda df, extra=None: df.group_by("g", maintain_order=True).agg(pl.col("mass").mean().alias("out")).sort("g"),
    )
    verb(
        "head", "verbs", ["head"],
        lambda df, extra=None: (pipe(df) >> t.head(6) >> t.select("id")).collect(),
        lambda df, extra=None: df.head(6).select("id"),
    )
    verb(
        "slice", "verbs", ["slice"],
        lambda df, extra=None: (pipe(df) >> t.slice(1, 2) >> t.select("id")).collect(),
        lambda df, extra=None: df.head(2).select("id"),
    )
    verb(
        "slice_head", "verbs", ["slice_head"],
        lambda df, extra=None: (pipe(df) >> t.slice_head(n=3) >> t.select("id")).collect(),
        lambda df, extra=None: df.head(3).select("id"),
    )
    verb(
        "slice_tail", "verbs", ["slice_tail"],
        lambda df, extra=None: (pipe(df) >> t.slice_tail(n=3) >> t.select("id")).collect(),
        lambda df, extra=None: df.tail(3).select("id"),
    )
    verb(
        "slice_min", "verbs", ["slice_min"],
        lambda df, extra=None: (pipe(df) >> t.slice_min("mass", n=2) >> t.select("id", "mass")).collect(),
        # with_ties keeps every row tied with the nth extreme, so a repeated
        # minimum is taller than n.
        lambda df, extra=None: (
            df.filter(pl.col("mass").rank(method="min") <= 2)
            .sort("mass", nulls_last=True, maintain_order=True)
            .select("id", "mass")
        ),
    )
    verb(
        "slice_max", "verbs", ["slice_max"],
        lambda df, extra=None: (pipe(df) >> t.slice_max("mass", n=2) >> t.select("id", "mass")).collect(),
        lambda df, extra=None: (
            df.filter(pl.col("mass").rank(method="min", descending=True) <= 2)
            .sort("mass", descending=True, nulls_last=True, maintain_order=True)
            .select("id", "mass")
        ),
    )
    verb(
        "distinct", "verbs", ["distinct"],
        lambda df, extra=None: (pipe(df) >> t.distinct("species")).collect(),
        lambda df, extra=None: df.select("species").unique(maintain_order=True),
    )
    verb(
        "compute", "verbs", ["compute"],
        lambda df, extra=None: pipe(df) >> t.compute(),
        lambda df, extra=None: df,
    )
    verb(
        "collect", "verbs", ["collect"],
        lambda df, extra=None: pipe(df) >> t.collect(),
        lambda df, extra=None: df,
    )
    verb(
        "ungroup", "verbs", ["ungroup"],
        lambda df, extra=None: (pipe(df) >> t.group_by("g") >> t.ungroup()).collect(),
        lambda df, extra=None: df,
    )
    verb(
        "tidy", "verbs", ["tidy"],
        lambda df, extra=None: t.tidy(df).collect(),
        lambda df, extra=None: df,
    )
    verb(
        "bind_rows", "verbs", ["bind_rows"],
        lambda df, extra=None: (pipe(df.select("id", "species").head(2)) >> t.bind_rows(df.select("id", "species").tail(2))).collect(),
        lambda df, extra=None: pl.concat([df.select("id", "species").head(2), df.select("id", "species").tail(2)]),
    )
    verb(
        "bind_cols", "verbs", ["bind_cols"],
        lambda df, extra=None: (pipe(df.select("id")) >> t.bind_cols(df.select("species"))).collect(),
        lambda df, extra=None: df.select("id", "species"),
    )

    # ── joins and set operations ───────────────────────────────────────
    def _even(df):
        return df.filter(pl.col("id") % 2 == 0).select("id", pl.lit("even").alias("tag"))

    verb(
        "left_join", "joins", ["left_join", "eq", "join_by"],
        lambda df, extra=None: (pipe(df.select("id", "species")) >> t.left_join(_even(df), on=t.join_by(t.eq("id")))).collect(),
        lambda df, extra=None: df.select("id", "species").join(_even(df), on="id", how="left", maintain_order="left", nulls_equal=True),
    )
    verb(
        "inner_join", "joins", ["inner_join"],
        lambda df, extra=None: (pipe(df.select("id")) >> t.inner_join(_even(df), on="id")).collect(),
        lambda df, extra=None: df.select("id").join(_even(df), on="id", how="inner", maintain_order="left", nulls_equal=True),
    )
    verb(
        "semi_join", "joins", ["semi_join"],
        lambda df, extra=None: (pipe(df.select("id")) >> t.semi_join(_even(df).select("id"), on="id")).collect(),
        lambda df, extra=None: df.select("id").join(_even(df).select("id"), on="id", how="semi", maintain_order="left", nulls_equal=True),
    )
    verb(
        "anti_join", "joins", ["anti_join"],
        lambda df, extra=None: (pipe(df.select("id")) >> t.anti_join(_even(df).select("id"), on="id")).collect(),
        lambda df, extra=None: df.select("id").join(_even(df).select("id"), on="id", how="anti", maintain_order="left", nulls_equal=True),
    )

    def _extra_right():
        return pl.DataFrame({"id": [0, 10**9], "tag": ["keep", "new"]})

    verb(
        "right_join", "joins", ["right_join"],
        lambda df, extra=None: (pipe(df.select("id", "species")) >> t.right_join(_extra_right(), on="id")).collect(),
        lambda df, extra=None: df.select("id", "species").join(_extra_right(), on="id", how="right", maintain_order="left_right", nulls_equal=True, coalesce=True).select("id", "species", "tag"),
    )
    verb(
        "full_join", "joins", ["full_join"],
        lambda df, extra=None: (pipe(df.select("id", "species")) >> t.full_join(_extra_right(), on="id")).collect(),
        lambda df, extra=None: df.select("id", "species").join(_extra_right(), on="id", how="full", maintain_order="left_right", nulls_equal=True, coalesce=True),
    )
    verb(
        "cross_join", "joins", ["cross_join"],
        lambda df, extra=None: (pipe(df.select("id")) >> t.cross_join(pl.DataFrame({"tag": ["x"]}))).collect(),
        lambda df, extra=None: df.select("id").join(pl.DataFrame({"tag": ["x"]}), how="cross", maintain_order="left"),
    )

    def _nested(df, extra=None):
        right = df.filter(pl.col("id") < 4).select("g", "species")
        left = df.select("id", "g")
        nested = right.group_by("g", maintain_order=True).agg(pl.struct("species").alias("data"))
        output = left.join(nested, on="g", how="left", nulls_equal=True, maintain_order="left")
        dtype = output.schema["data"]
        return output.with_columns(pl.col("data").fill_null(pl.lit([], dtype=dtype)))

    verb(
        "nest_join", "joins", ["nest_join"],
        lambda df, extra=None: (pipe(df.select("id", "g")) >> t.nest_join(df.filter(pl.col("id") < 4).select("g", "species"), on="g")).collect(),
        _nested,
    )

    def _asof(df, extra=None):
        right = pl.DataFrame({"y": [0, 8, 16]})
        return (
            df.select("id").with_row_index("__row").sort("id")
            .join_asof(right.sort("y"), left_on="id", right_on="y", strategy="backward", coalesce=False)
            .sort("__row").drop("__row")
        )

    verb(
        "closest", "joins", ["closest", "ge"],
        lambda df, extra=None: (pipe(df.select("id")) >> t.left_join(pl.DataFrame({"y": [0, 8, 16]}), on=t.join_by(t.closest(t.ge("id", "y"))))).collect(),
        _asof,
    )

    def _matched(df, right, predicate):
        crossed = df.join(right, how="cross")
        return crossed.with_columns(
            pl.when(predicate).then(pl.col(name)).otherwise(None).alias(name)
            for name in right.columns
        )

    verb(
        "gt", "joins", ["gt"],
        lambda df, extra=None: (pipe(df.select("id")) >> t.left_join(pl.DataFrame({"y": [10], "tag": ["hit"]}), on=t.join_by(t.gt("id", "y")))).collect(),
        lambda df, extra=None: _matched(df.select("id"), pl.DataFrame({"y": [10], "tag": ["hit"]}), pl.col("id") > pl.col("y")),
    )
    verb(
        "lt", "joins", ["lt"],
        lambda df, extra=None: (pipe(df.select("id")) >> t.left_join(pl.DataFrame({"y": [10], "tag": ["hit"]}), on=t.join_by(t.lt("id", "y")))).collect(),
        lambda df, extra=None: _matched(df.select("id"), pl.DataFrame({"y": [10], "tag": ["hit"]}), pl.col("id") < pl.col("y")),
    )
    verb(
        "le", "joins", ["le"],
        lambda df, extra=None: (pipe(df.select("id")) >> t.left_join(pl.DataFrame({"y": [10], "tag": ["hit"]}), on=t.join_by(t.le("id", "y")))).collect(),
        lambda df, extra=None: _matched(df.select("id"), pl.DataFrame({"y": [10], "tag": ["hit"]}), pl.col("id") <= pl.col("y")),
    )
    verb(
        "between", "joins", ["between"],
        lambda df, extra=None: (pipe(df.select("id")) >> t.left_join(pl.DataFrame({"lo": [5], "hi": [10], "tag": ["in"]}), on=t.join_by(t.between("id", "lo", "hi")))).collect(),
        lambda df, extra=None: _matched(df.select("id"), pl.DataFrame({"lo": [5], "hi": [10], "tag": ["in"]}), (pl.col("id") >= pl.col("lo")) & (pl.col("id") <= pl.col("hi"))),
    )
    verb(
        "within", "joins", ["within"],
        lambda df, extra=None: (
            pipe(df.select("id", lo=pl.col("id"), hi=pl.col("id") + 5))
            >> t.left_join(pl.DataFrame({"rlo": [2], "rhi": [10], "tag": ["w"]}), on=t.join_by(t.within("lo", "hi", "rlo", "rhi")))
        ).collect(),
        lambda df, extra=None: _matched(
            df.select("id", lo=pl.col("id"), hi=pl.col("id") + 5),
            pl.DataFrame({"rlo": [2], "rhi": [10], "tag": ["w"]}),
            (pl.col("lo") >= pl.col("rlo")) & (pl.col("hi") <= pl.col("rhi")),
        ),
    )
    verb(
        "overlaps", "joins", ["overlaps"],
        lambda df, extra=None: (
            pipe(df.select("id", lo=pl.col("id"), hi=pl.col("id") + 5))
            >> t.left_join(pl.DataFrame({"rlo": [2], "rhi": [10], "tag": ["o"]}), on=t.join_by(t.overlaps("lo", "hi", "rlo", "rhi")))
        ).collect(),
        lambda df, extra=None: _matched(
            df.select("id", lo=pl.col("id"), hi=pl.col("id") + 5),
            pl.DataFrame({"rlo": [2], "rhi": [10], "tag": ["o"]}),
            (pl.col("lo") <= pl.col("rhi")) & (pl.col("hi") >= pl.col("rlo")),
        ),
    )

    pairs = lambda df: df.select("species", "island")  # noqa: E731
    subset = lambda df: df.filter(pl.col("id") % 5 == 0).select("species", "island")  # noqa: E731

    def _unique(df):
        return pairs(df).unique(maintain_order=True)

    verb(
        "union", "joins", ["union"],
        lambda df, extra=None: (pipe(pairs(df)) >> t.union(subset(df))).collect(),
        lambda df, extra=None: pl.concat([pairs(df), subset(df)]).unique(keep="first", maintain_order=True),
    )
    verb(
        "union_all", "joins", ["union_all"],
        lambda df, extra=None: (pipe(pairs(df).head(2)) >> t.union_all(pairs(df).head(1))).collect(),
        lambda df, extra=None: pl.concat([pairs(df).head(2), pairs(df).head(1)], how="vertical"),
    )
    verb(
        "intersect", "joins", ["intersect"],
        lambda df, extra=None: (pipe(pairs(df)) >> t.intersect(subset(df))).collect(),
        lambda df, extra=None: _unique(df).join(subset(df).unique(maintain_order=True), on=["species", "island"], how="semi", nulls_equal=True, maintain_order="left"),
    )
    verb(
        "setdiff", "joins", ["setdiff"],
        lambda df, extra=None: (pipe(pairs(df)) >> t.setdiff(df.filter(pl.col("species") == "Adelie").select("species", "island"))).collect(),
        lambda df, extra=None: _unique(df).join(
            df.filter(pl.col("species") == "Adelie").select("species", "island").unique(maintain_order=True),
            on=["species", "island"], how="anti", nulls_equal=True, maintain_order="left",
        ),
    )
    verb(
        "symdiff", "joins", ["symdiff"],
        lambda df, extra=None: (pipe(pairs(df).head(2)) >> t.symdiff(pairs(df).slice(1, 2))).collect(),
        lambda df, extra=None: pl.concat([
            pairs(df).head(2).unique(maintain_order=True).join(pairs(df).slice(1, 2).unique(maintain_order=True), on=["species", "island"], how="anti", nulls_equal=True, maintain_order="left"),
            pairs(df).slice(1, 2).unique(maintain_order=True).join(pairs(df).head(2).unique(maintain_order=True), on=["species", "island"], how="anti", nulls_equal=True, maintain_order="left"),
        ], how="vertical"),
    )
    verb(
        "setequal", "joins", ["setequal"],
        lambda df, extra=None: pipe(pairs(df)) >> t.setequal(_unique(df)),
        _raw_setequal,
    )

    # ── rows_* ─────────────────────────────────────────────────────────
    def _base(df):
        return df.select("id", "mass", "species").head(6)

    def _update(df, replace):
        right = pl.DataFrame(replace)
        hit = pl.col("id").is_in(right["id"].implode())
        joined = _base(df).join(right, on="id", how="left", suffix="_y", maintain_order="left")
        return joined.with_columns(
            pl.when(hit).then(pl.col("mass_y")).otherwise(pl.col("mass")).alias("mass"),
            pl.when(hit).then(pl.col("species_y")).otherwise(pl.col("species")).alias("species"),
        ).select("id", "mass", "species")

    update_rows = {"id": [1, 3], "mass": [9.0, 9.0], "species": ["Z", "Z"]}
    verb(
        "rows_update", "rows", ["rows_update"],
        lambda df, extra=None: (pipe(_base(df)) >> t.rows_update(pl.DataFrame(update_rows), by="id")).collect(),
        lambda df, extra=None: _update(df, update_rows),
    )

    def _patch(df):
        base = _base(df).with_columns(pl.when(pl.col("id") == 0).then(None).otherwise(pl.col("species")).alias("species"))
        right = pl.DataFrame({"id": [0], "mass": [None], "species": ["Z"]})
        return base, right

    verb(
        "rows_patch", "rows", ["rows_patch"],
        lambda df, extra=None: (pipe(_patch(df)[0]) >> t.rows_patch(_patch(df)[1], by="id")).collect(),
        lambda df, extra=None: _patch(df)[0].with_columns(pl.when(pl.col("id") == 0).then(pl.lit("Z")).otherwise(pl.col("species")).alias("species")),
    )
    verb(
        "rows_delete", "rows", ["rows_delete"],
        lambda df, extra=None: (pipe(_base(df)) >> t.rows_delete(pl.DataFrame({"id": [1]}), by="id")).collect(),
        lambda df, extra=None: _base(df).filter(pl.col("id") != 1),
    )
    verb(
        "rows_append", "rows", ["rows_append"],
        lambda df, extra=None: (pipe(_base(df)) >> t.rows_append(pl.DataFrame({"id": [99], "mass": [1.0], "species": ["Q"]}))).collect(),
        lambda df, extra=None: pl.concat([_base(df), pl.DataFrame({"id": [99], "mass": [1.0], "species": ["Q"]})], how="vertical"),
    )
    inserted = pl.DataFrame({"id": [1, 99], "mass": [7.0, 1.0], "species": ["NO", "Q"]})
    verb(
        "rows_insert", "rows", ["rows_insert"],
        lambda df, extra=None: (pipe(_base(df)) >> t.rows_insert(inserted, by="id", conflict="ignore")).collect(),
        lambda df, extra=None: pl.concat([_base(df), inserted.filter(pl.col("id") == 99)], how="vertical"),
    )
    upserted = pl.DataFrame({"id": [1, 99], "mass": [8.0, 1.0], "species": ["Z", "Q"]})
    verb(
        "rows_upsert", "rows", ["rows_upsert"],
        lambda df, extra=None: (pipe(_base(df)) >> t.rows_upsert(upserted, by="id")).collect(),
        lambda df, extra=None: pl.concat([_update(df, {"id": [1], "mass": [8.0], "species": ["Z"]}), upserted.filter(pl.col("id") == 99)], how="vertical"),
    )

    # ── groups ─────────────────────────────────────────────────────────
    def _grouped(df):
        return pipe(df) >> t.group_by("g")

    verb(
        "group_data", "groups", ["group_data"],
        lambda df, extra=None: t.group_data(_grouped(df)),
        lambda df, extra=None: (
            df.select("g").with_row_index("row", offset=1)
            .group_by("g", maintain_order=True).agg(pl.col("row").cast(pl.Int64).alias(".rows"))
            .sort("g")
        ),
    )
    verb(
        "group_keys", "groups", ["group_keys"],
        lambda df, extra=None: t.group_keys(_grouped(df)),
        lambda df, extra=None: df.select("g").unique(maintain_order=True).sort("g"),
    )
    verb(
        "group_rows", "groups", ["group_rows"],
        lambda df, extra=None: t.group_rows(_grouped(df)),
        lambda df, extra=None: [
            [int(value) + 1 for value in positions]
            for positions in (
                df.with_row_index("row").group_by("g", maintain_order=True).agg(pl.col("row")).sort("g")["row"].to_list()
            )
        ],
    )
    verb(
        "group_size", "groups", ["group_size"],
        lambda df, extra=None: t.group_size(_grouped(df)),
        lambda df, extra=None: [df.filter(pl.col("g") == key).height for key in range(8) if df.filter(pl.col("g") == key).height],
    )
    verb(
        "group_indices", "groups", ["group_indices"],
        lambda df, extra=None: t.group_indices(_grouped(df)),
        lambda df, extra=None: api._group_number(df),
    )
    verb(
        "groups", "groups", ["groups", "group_vars"],
        lambda df, extra=None: (t.groups(_grouped(df)), t.group_vars(_grouped(df))),
        lambda df, extra=None: (["g"], ["g"]),
    )
    verb(
        "n_groups", "groups", ["n_groups"],
        lambda df, extra=None: t.n_groups(_grouped(df)),
        lambda df, extra=None: int(df["g"].n_unique()),
    )
    verb(
        "group_split", "groups", ["group_split"],
        lambda df, extra=None: _grouped(df.select("g", "id")) >> t.group_split(),
        lambda df, extra=None: [df.filter(pl.col("g") == key).select("g", "id") for key in range(8)],
    )
    verb(
        "group_map", "groups", ["group_map", "cur_group_rows"],
        lambda df, extra=None: _grouped(df) >> t.group_map(lambda part, key: len(t.cur_group_rows())),
        lambda df, extra=None: [df.filter(pl.col("g") == key).height for key in range(8)],
    )
    verb(
        "group_modify", "groups", ["group_modify"],
        lambda df, extra=None: (_grouped(df) >> t.group_modify(lambda part, key: part >> t.slice_head(n=1))).collect(),
        lambda df, extra=None: api._first_per_group(df),
    )
    verb(
        "group_nest", "groups", ["group_nest"],
        lambda df, extra=None: (pipe(df.select("g", "species", "mass")) >> t.group_by("g") >> t.group_nest()).collect(),
        lambda df, extra=None: (
            df.select("g", "species", "mass").group_by("g", maintain_order=True)
            .agg(pl.struct("species", "mass").alias("data")).sort("g")
        ),
    )
    verb(
        "nest_by", "groups", ["nest_by"],
        lambda df, extra=None: (pipe(df.select("g", "species", "mass")) >> t.nest_by("g")).collect(),
        lambda df, extra=None: (
            df.select("g", "species", "mass").group_by("g", maintain_order=True)
            .agg(pl.struct("species", "mass").alias("data")).sort("g")
        ),
    )
    verb(
        "with_groups", "groups", ["with_groups"],
        lambda df, extra=None: (pipe(df) >> t.with_groups("g", lambda grouped: grouped >> t.summarise(out=t.mean("mass")))).collect(),
        lambda df, extra=None: df.group_by("g", maintain_order=True).agg(pl.col("mass").mean().alias("out")).sort("g"),
    )

    def _trimmed(df):
        return df.select("id", pl.col("species").cast(pl.Enum(["Adelie", "Chinstrap", "Gentoo", "King"])))

    verb(
        "group_trim", "groups", ["group_trim"],
        lambda df, extra=None: (pipe(_trimmed(df)) >> t.group_by("species") >> t.group_trim() >> t.ungroup()).collect(),
        # dplyr drops unused levels, so King leaves the Enum.
        lambda df, extra=None: _trimmed(df).with_columns(
            pl.col("species").cast(pl.String).cast(pl.Enum(
                [level for level in ["Adelie", "Chinstrap", "Gentoo", "King"] if level in df["species"].unique().to_list()]
            ))
        ),
    )

    # ── selectors and across ───────────────────────────────────────────
    add(api._select_where("where", ["where", "is_float"], t.is_float))
    add(api._select_where("is_integer", ["is_integer"], t.is_integer))
    add(api._select_where("is_numeric", ["is_numeric"], t.is_numeric))
    add(api._select_where("is_string", ["is_string"], t.is_string))
    add(api._select_where("is_character", ["is_character"], t.is_character))
    add(api._select_where("is_bool", ["is_bool"], t.is_bool))
    add(api._select_where("is_boolean", ["is_boolean"], t.is_boolean))
    add(api._select_where("is_datetime", ["is_datetime"], t.is_datetime))
    add(api._select_where("is_temporal", ["is_temporal"], t.is_temporal))
    verb(
        "is_categorical", "select", ["is_categorical"],
        lambda df, extra=None: (
            pipe(df.with_columns(kind=pl.col("species").cast(pl.Enum(["Adelie", "Chinstrap", "Gentoo", "King"]))))
            >> t.select(t.where(t.is_categorical))
        ).collect(),
        lambda df, extra=None: df.select(pl.col("species").cast(pl.Enum(["Adelie", "Chinstrap", "Gentoo", "King"])).alias("kind")),
    )
    verb(
        "starts_with", "select", ["starts_with"],
        lambda df, extra=None: (pipe(df) >> t.select(t.starts_with("bill"))).collect(),
        lambda df, extra=None: df.select("bill", "bill1", "bill2"),
    )
    verb(
        "ends_with", "select", ["ends_with"],
        lambda df, extra=None: (pipe(df) >> t.select(t.ends_with("d"))).collect(),
        lambda df, extra=None: df.select([name for name in df.columns if name.endswith("d") or name.endswith("D")]),
    )
    verb(
        "contains", "select", ["contains"],
        lambda df, extra=None: (pipe(df) >> t.select(t.contains("ill"))).collect(),
        lambda df, extra=None: df.select([name for name in df.columns if "ill" in name.casefold()]),
    )
    verb(
        "matches", "select", ["matches"],
        lambda df, extra=None: (pipe(df) >> t.select(t.matches("^bill"))).collect(),
        lambda df, extra=None: df.select([name for name in df.columns if name.lower().startswith("bill")]),
    )
    verb(
        "everything", "select", ["everything"],
        lambda df, extra=None: (pipe(df) >> t.select(t.everything())).collect(),
        lambda df, extra=None: df,
    )
    verb(
        "all_of", "select", ["all_of"],
        lambda df, extra=None: (pipe(df) >> t.select(t.all_of(["species", "mass"]))).collect(),
        lambda df, extra=None: df.select("species", "mass"),
    )
    verb(
        "any_of", "select", ["any_of"],
        lambda df, extra=None: (pipe(df) >> t.select(t.any_of(["species", "missing", "mass"]))).collect(),
        lambda df, extra=None: df.select("species", "mass"),
    )
    verb(
        "num_range", "select", ["num_range"],
        lambda df, extra=None: (pipe(df) >> t.select(t.num_range("bill", range(1, 3)))).collect(),
        lambda df, extra=None: df.select("bill1", "bill2"),
    )
    verb(
        "last_col", "select", ["last_col"],
        lambda df, extra=None: (pipe(df) >> t.select(t.last_col())).collect(),
        lambda df, extra=None: df.select(df.columns[-1]),
    )
    verb(
        "col_range", "select", ["col_range"],
        lambda df, extra=None: (pipe(df) >> t.select(t.col_range("mass", "flag"))).collect(),
        lambda df, extra=None: df.select("mass", "bill", "flag"),
    )
    verb(
        "cols_between", "select", ["cols_between"],
        lambda df, extra=None: (pipe(df) >> t.select(t.cols_between("mass", "flag"))).collect(),
        lambda df, extra=None: df.select("mass", "bill", "flag"),
    )
    verb(
        "group_cols", "select", ["group_cols"],
        lambda df, extra=None: (_grouped(df) >> t.select(t.group_cols())).collect(),
        lambda df, extra=None: df.select("g"),
    )

    def _across(value):
        t.cur_column()
        t.cur_group()
        t.cur_group_id()
        t.n_groups()
        return value

    verb(
        "across", "select", ["across", "cur_column", "cur_group", "cur_group_id"],
        lambda df, extra=None: (
            _grouped(df) >> t.mutate(t.across(t.where(t.is_float), _across, names="{col}_z"))
            >> t.ungroup() >> t.select(t.ends_with("_z"))
        ).collect(),
        lambda df, extra=None: df.select(
            pl.col(name).alias(f"{name}_z")
            for name, dtype in df.schema.items()
            if dtype.is_float()
        ),
    )
    verb(
        "if_all", "select", ["if_all"],
        lambda df, extra=None: (pipe(df) >> t.filter(t.if_all(("flag", "logic"), lambda value: value)) >> t.select("id")).collect(),
        lambda df, extra=None: df.filter(pl.col("flag") & pl.col("logic")).select("id"),
    )
    verb(
        "if_any", "select", ["if_any"],
        lambda df, extra=None: (pipe(df) >> t.filter(t.if_any(("flag", "logic"), lambda value: value)) >> t.select("id")).collect(),
        lambda df, extra=None: df.filter(pl.col("flag") | pl.col("logic")).select("id"),
    )
    verb(
        "pick", "select", ["pick"],
        lambda df, extra=None: (pipe(df) >> t.mutate(out=t.pick("mass", "bill").sum()) >> t.select("out")).collect(),
        lambda df, extra=None: df.select((pl.col("mass") + pl.col("bill")).alias("out")),
    )
    verb(
        "c_across", "select", ["c_across", "rowwise"],
        lambda df, extra=None: (pipe(df) >> t.rowwise() >> t.mutate(out=t.sum(t.c_across(["mass", "bill"]))) >> t.select("out")).collect(),
        lambda df, extra=None: df.select((pl.col("mass") + pl.col("bill")).alias("out")),
    )

    # ── tidyr ──────────────────────────────────────────────────────────
    def _longer(df, extra=None):
        indexed = df.select("id", "mass", "bill").with_row_index("__i")
        parts = []
        for order, name in enumerate(("mass", "bill")):
            parts.append(indexed.select(
                "__i", "id",
                pl.lit(name).alias("name"),
                pl.col(name).alias("value"),
                pl.lit(order).alias("__o"),
            ))
        return pl.concat(parts).sort(["__i", "__o"]).drop("__i", "__o")

    verb(
        "pivot_longer", "tidyr", ["pivot_longer"],
        lambda df, extra=None: (pipe(df.select("id", "mass", "bill")) >> t.pivot_longer(["mass", "bill"])).collect(),
        _longer,
    )
    verb(
        "build_longer_spec", "tidyr", ["build_longer_spec"],
        lambda df, extra=None: t.build_longer_spec(pipe(df), ["mass", "bill"]),
        lambda df, extra=None: pl.DataFrame({
            ".name": ["mass", "bill"],
            ".value": ["value", "value"],
            "name": ["mass", "bill"],
        }),
    )
    verb(
        "check_pivot_spec", "tidyr", ["check_pivot_spec"],
        lambda df, extra=None: t.check_pivot_spec(t.build_longer_spec(pipe(df), ["mass", "bill"])),
        lambda df, extra=None: pl.DataFrame({
            ".name": ["mass", "bill"],
            ".value": ["value", "value"],
            "name": ["mass", "bill"],
        }),
    )
    verb(
        "pivot_longer_spec", "tidyr", ["pivot_longer_spec"],
        lambda df, extra=None: (
            pipe(df.select("id", "mass", "bill"))
            >> t.pivot_longer_spec(t.build_longer_spec(pipe(df.select("id", "mass", "bill")), ["mass", "bill"]))
        ).collect(),
        _longer,
    )

    def _wide_source(df):
        return df.select("id", "species", "mass").head(3)

    def _wider(df, extra=None):
        source = _wide_source(df)
        names = source["species"].unique(maintain_order=True).to_list()
        wide = source.pivot(on="species", index="id", values="mass")
        return wide.select(["id", *names])

    verb(
        "pivot_wider", "tidyr", ["pivot_wider"],
        lambda df, extra=None: (pipe(_wide_source(df)) >> t.pivot_wider(names_from="species", values_from="mass")).collect(),
        _wider,
    )
    verb(
        "build_wider_spec", "tidyr", ["build_wider_spec"],
        lambda df, extra=None: t.build_wider_spec(pipe(_wide_source(df)), names_from="species", values_from="mass"),
        lambda df, extra=None: pl.DataFrame({
            ".name": ["Adelie", "Chinstrap", "Gentoo"],
            ".value": ["mass", "mass", "mass"],
            "species": ["Adelie", "Chinstrap", "Gentoo"],
        }),
    )
    verb(
        "pivot_wider_spec", "tidyr", ["pivot_wider_spec"],
        lambda df, extra=None: (
            pipe(_wide_source(df))
            >> t.pivot_wider_spec(t.build_wider_spec(pipe(_wide_source(df)), names_from="species", values_from="mass"))
        ).collect(),
        _wider,
    )
    verb(
        "pivot_longer_ptypes", "tidyr", [],
        lambda df, extra=None: (
            pipe(df.select("id", "mass", "bill"))
            >> t.pivot_longer(["mass", "bill"], names_ptypes={"name": pl.Enum(["mass", "bill"])}, values_ptypes=float)
        ).collect(),
        lambda df, extra=None: _longer(df).with_columns(pl.col("name").cast(pl.Enum(["mass", "bill"]))),
    )
    verb(
        "pivot_wider_values_fn", "tidyr", [],
        lambda df, extra=None: (
            pipe(df.select("g", "species", "mass", "bill"))
            >> t.pivot_wider(names_from="species", values_from=["mass", "bill"], values_fn={"mass": "mean", "bill": "max"})
        ).collect(),
        lambda df, extra=None: (
            df.group_by("g", "species", maintain_order=True)
            .agg(pl.col("mass").mean(), pl.col("bill").max())
            .pivot(on="species", index="g", values=["mass", "bill"])
        ),
    )
    verb(
        "separate", "tidyr", ["separate"],
        lambda df, extra=None: (pipe(df) >> t.separate("csv", ["a", "b"], sep=",", extra="drop", fill="right") >> t.select("id", "a", "b")).collect(),
        lambda df, extra=None: df.select("id", pl.col("csv").str.split_exact(",", 1).struct.rename_fields(["a", "b"])).unnest("csv"),
    )
    verb(
        "separate_wider_delim", "tidyr", ["separate_wider_delim"],
        lambda df, extra=None: (pipe(df.select("id", "csv")) >> t.separate_wider_delim("csv", ["a", "b"], delim=",")).collect(),
        lambda df, extra=None: df.select("id", pl.col("csv").str.split_exact(",", 1).struct.rename_fields(["a", "b"])).unnest("csv"),
    )
    verb(
        "separate_longer_delim", "tidyr", ["separate_longer_delim"],
        lambda df, extra=None: (pipe(df.select("id", "csv")) >> t.separate_longer_delim("csv", ",")).collect(),
        lambda df, extra=None: df.select("id", pl.col("csv").str.split(",")).explode("csv", empty_as_null=True),
    )
    verb(
        "separate_wider_position", "tidyr", ["separate_wider_position"],
        lambda df, extra=None: (pipe(df.select("code")) >> t.separate_wider_position("code", [("ab", 2), ("num", 2)])).collect(),
        lambda df, extra=None: df.select(pl.col("code").str.slice(0, 2).alias("ab"), pl.col("code").str.slice(2, 2).alias("num")),
    )
    verb(
        "separate_wider_regex", "tidyr", ["separate_wider_regex"],
        lambda df, extra=None: (pipe(df.select("code")) >> t.separate_wider_regex("code", [("ab", "AB"), ("num", r"\d{2}")])).collect(),
        lambda df, extra=None: df.select(pl.col("code").str.extract(r"(AB)(\d{2})", 1).alias("ab"), pl.col("code").str.extract(r"(AB)(\d{2})", 2).alias("num")),
    )
    verb(
        "separate_longer_position", "tidyr", ["separate_longer_position"],
        lambda df, extra=None: (pipe(df.select("id", "code")) >> t.separate_longer_position("code", 2)).collect(),
        lambda df, extra=None: df.select("id", pl.col("code").str.extract_all(r".{2}")).explode("code", empty_as_null=True),
    )
    verb(
        "unite", "tidyr", ["unite"],
        lambda df, extra=None: (pipe(df.select("species", "island", "id")) >> t.unite("both", "species", "island", sep=" ") >> t.select("both", "id")).collect(),
        lambda df, extra=None: df.select(pl.concat_str(["species", "island"], separator=" ").alias("both"), "id"),
    )
    verb(
        "nest", "tidyr", ["nest"],
        lambda df, extra=None: (pipe(df.select("g", "species", "mass")) >> t.nest(by="g")).collect(),
        lambda df, extra=None: (
            df.select("g", "species", "mass").group_by("g", maintain_order=True)
            .agg(pl.struct("species", "mass").alias("data"))
        ),
    )
    verb(
        "unnest", "tidyr", ["unnest"],
        lambda df, extra=None: (
            pipe(df.select("g", "species", "mass")) >> t.nest(by="g") >> t.unnest("data")
        ).collect(),
        lambda df, extra=None: (
            df.select("g", "species", "mass").group_by("g", maintain_order=True)
            .agg(pl.struct("species", "mass").alias("data"))
            .explode("data", empty_as_null=True).unnest("data")
        ),
    )
    verb(
        "unnest_longer", "tidyr", ["unnest_longer"],
        lambda df, extra=None: (pipe(df.select("id", "items")) >> t.unnest_longer("items")).collect(),
        lambda df, extra=None: df.select("id", "items").explode("items", empty_as_null=True),
    )
    verb(
        "unnest_wider", "tidyr", ["unnest_wider"],
        lambda df, extra=None: (pipe(df.select("id", "rec")) >> t.unnest_wider("rec")).collect(),
        lambda df, extra=None: df.select("id", "rec").unnest("rec"),
    )
    verb(
        "unnest_auto", "tidyr", ["unnest_auto"],
        lambda df, extra=None: (pipe(df.select("id", "rec")) >> t.unnest_auto("rec")).collect(),
        lambda df, extra=None: df.select("id", "rec").unnest("rec"),
    )
    verb(
        "pack", "tidyr", ["pack"],
        lambda df, extra=None: (pipe(df.select("id", "mass", "species")) >> t.pack("box", "mass", "species")).collect(),
        lambda df, extra=None: df.select("id", pl.struct("mass", "species").alias("box")),
    )
    verb(
        "unpack", "tidyr", ["unpack"],
        lambda df, extra=None: (pipe(df.select("id", "rec")) >> t.unpack("rec")).collect(),
        lambda df, extra=None: df.select("id", "rec").unnest("rec"),
    )
    verb(
        "hoist", "tidyr", ["hoist"],
        lambda df, extra=None: (pipe(df.select("id", "rec")) >> t.hoist("rec", a="a")).collect(),
        lambda df, extra=None: df.select(
            "id",
            pl.col("rec").struct.field("a").alias("a"),
            pl.struct(b=pl.col("rec").struct.field("b")).alias("rec"),
        ),
    )
    verb(
        "chop", "tidyr", ["chop"],
        lambda df, extra=None: (pipe(df.select("g", "species")) >> t.chop("species")).collect(),
        lambda df, extra=None: df.select("g", "species").group_by("g", maintain_order=True).agg(pl.col("species")).select("g", "species"),
    )
    verb(
        "unchop", "tidyr", ["unchop"],
        lambda df, extra=None: (pipe(df.select("g", "species")) >> t.chop("species") >> t.unchop("species")).collect(),
        lambda df, extra=None: (
            df.select("g", "species").group_by("g", maintain_order=True).agg(pl.col("species")).explode("species", empty_as_null=True)
        ),
    )
    verb(
        "expand", "tidyr", ["expand"],
        lambda df, extra=None: (pipe(df.select("species", "island")) >> t.expand("species", "island")).collect(),
        lambda df, extra=None: (
            df.select("species").unique().sort("species")
            .join(df.select("island").unique().sort("island"), how="cross")
        ),
    )
    verb(
        "nesting", "tidyr", ["nesting"],
        lambda df, extra=None: (pipe(df.select("species", "island")) >> t.expand(t.nesting("species", "island"))).collect(),
        lambda df, extra=None: df.select("species", "island").unique(maintain_order=True),
    )
    verb(
        "full_seq", "tidyr", ["full_seq"],
        lambda df, extra=None: t.full_seq(extra, 1),
        _raw_full_seq,
        prepare=lambda df: df["whole"],
    )
    verb(
        "complete", "tidyr", ["complete"],
        lambda df, extra=None: (pipe(df.select("whole")) >> t.complete(whole=t.full_seq("whole", 1))).collect(),
        # complete() keeps every original row and inserts the missing steps,
        # so a repeated key stays repeated.
        lambda df, extra=None: pl.DataFrame({
            "whole": pl.int_range(
                int(df["whole"].min()), int(df["whole"].max()) + 1, eager=True, dtype=pl.Int64
            )
        }).join(df.select("whole"), on="whole", how="left", maintain_order="left_right"),
    )
    verb(
        "crossing", "tidyr", ["crossing"],
        lambda df, extra=None: t.crossing(species=["Gentoo", "Adelie"], year=[2021, 2020]),
        lambda df, extra=None: pl.DataFrame({"species": ["Adelie", "Gentoo"]}).join(pl.DataFrame({"year": [2020, 2021]}), how="cross"),
    )
    verb(
        "expand_grid", "tidyr", ["expand_grid"],
        lambda df, extra=None: t.expand_grid(df.select("id", "species").head(3), year=[2020, 2021]),
        lambda df, extra=None: df.select("id", "species").head(3).join(pl.DataFrame({"year": [2020, 2021]}), how="cross"),
    )
    verb(
        "uncount", "tidyr", ["uncount"],
        lambda df, extra=None: (pipe(df.select("id", weights=pl.col("id") % 3 + 1)) >> t.uncount("weights")).collect(),
        lambda df, extra=None: df.select(pl.col("id").repeat_by(pl.col("id") % 3 + 1)).explode("id", empty_as_null=True),
    )

    # ── janitor and eda ────────────────────────────────────────────────
    verb(
        "adorn_totals", "janitor", ["adorn_totals"],
        lambda df, extra=None: (pipe(df.select("species", "mass", "bill")) >> t.adorn_totals("both")).collect(),
        lambda df, extra=None: pl.concat([
            df.select("species", "mass", "bill"),
            df.select(pl.lit("Total").alias("species"), pl.col("mass").sum(), pl.col("bill").sum()),
        ]).with_columns(pl.sum_horizontal("mass", "bill").alias("Total")),
    )
    verb(
        "adorn_percentages", "janitor", ["adorn_percentages"],
        lambda df, extra=None: (pipe(df.select("species", "mass", "bill")) >> t.adorn_percentages()).collect(),
        lambda df, extra=None: df.select(
            "species",
            (pl.col("mass") / (pl.col("mass") + pl.col("bill"))).alias("mass"),
            (pl.col("bill") / (pl.col("mass") + pl.col("bill"))).alias("bill"),
        ),
    )
    verb(
        "adorn_percentages_col", "janitor", [],
        lambda df, extra=None: (pipe(df.select("species", "mass", "bill")) >> t.adorn_percentages("col")).collect(),
        lambda df, extra=None: df.select(
            "species", pl.col("mass") / pl.col("mass").sum(), pl.col("bill") / pl.col("bill").sum(),
        ),
    )
    verb(
        "tabyl", "janitor", ["tabyl"],
        lambda df, extra=None: (pipe(df) >> t.tabyl("species")).collect(),
        lambda df, extra=None: df.group_by("species").len(name="n").sort("species")
        .with_columns((pl.col("n") / pl.col("n").sum()).alias("percent")),
    )
    verb(
        "tabyl_two_way", "janitor", [],
        lambda df, extra=None: (pipe(df) >> t.tabyl("species", "island")).collect(),
        lambda df, extra=None: df.group_by("species", "island").len(name="n")
        .pivot(on="island", index="species", values="n", sort_columns=True)
        .fill_null(0).with_columns(pl.exclude("species").cast(pl.Float64)).sort("species"),
    )
    verb(
        "get_dupes", "janitor", ["get_dupes"],
        lambda df, extra=None: (pipe(df.select("species", "id")) >> t.get_dupes("species")).collect(),
        lambda df, extra=None: (
            df.select("species", "id")
            .with_columns(pl.len().over("species").alias("dupe_count"))
            .filter(pl.col("dupe_count") > 1)
            .select("species", "dupe_count", "id")
            .sort(["dupe_count", "species"], descending=[True, False], maintain_order=True)
        ),
    )
    verb(
        "remove_constant", "janitor", ["remove_constant"],
        lambda df, extra=None: (pipe(df) >> t.remove_constant()).collect(),
        lambda df, extra=None: df.select(api._constant_columns(df)),
    )
    verb(
        "remove_empty", "janitor", ["remove_empty"],
        lambda df, extra=None: (pipe(df.with_columns(empty=pl.lit(None, dtype=pl.String))) >> t.remove_empty(which="cols")).collect(),
        _raw_remove_empty,
    )
    verb(
        "clean_names", "janitor", ["clean_names"],
        lambda df, extra=None: (
            pipe(df.select(pl.col("species").alias("Species Name"), pl.col("mass").alias("Age (%)")))
            >> t.clean_names()
        ).collect(),
        lambda df, extra=None: df.select(pl.col("species").alias("species_name"), pl.col("mass").alias("age_percent")),
    )
    verb(
        "make_clean_names", "janitor", ["make_clean_names"],
        lambda df, extra=None: t.make_clean_names(["Species Name", "Age (%)"]),
        lambda df, extra=None: ["species_name", "age_percent"],
    )
    verb(
        "describe_class", "janitor", ["describe_class"],
        lambda df, extra=None: t.describe_class(df["mass"]),
        lambda df, extra=None: "numeric",
    )
    verb(
        "compare_df_cols", "janitor", ["compare_df_cols"],
        lambda df, extra=None: t.compare_df_cols(pipe(df.select("id", "mass")), pipe(df.select("mass", "species"))),
        lambda df, extra=None: pl.DataFrame({
            "column_name": ["id", "mass", "species"],
            "..1": ["integer", "numeric", None],
            "..2": [None, "numeric", "character"],
        }),
    )
    verb(
        "compare_df_cols_same", "janitor", ["compare_df_cols_same"],
        lambda df, extra=None: t.compare_df_cols_same(pipe(df.select("id")), pipe(df.select("id"))),
        lambda df, extra=None: True,
    )
    verb(
        "find_header", "janitor", ["find_header"],
        lambda df, extra=None: t.find_header(df),
        lambda df, extra=None: 2,
    )
    verb(
        "row_to_names", "janitor", ["row_to_names"],
        lambda df, extra=None: (pipe(df.select("species", "island").head(3)) >> t.row_to_names(1)).collect(),
        lambda df, extra=None: df.select("species", "island").head(3).slice(1).rename({"species": "Adelie", "island": "Biscoe"}),
    )

    small = lambda df: df.select("id", "mass", "flag")  # noqa: E731
    verb("names", "eda", ["names"], lambda df, extra=None: t.names(pipe(df)), lambda df, extra=None: list(df.columns))
    verb("colnames", "eda", ["colnames"], lambda df, extra=None: list(t.colnames(pipe(df))), lambda df, extra=None: list(df.columns))
    verb("nrow", "eda", ["nrow"], lambda df, extra=None: t.nrow(pipe(df)), lambda df, extra=None: df.height)
    verb("ncol", "eda", ["ncol"], lambda df, extra=None: t.ncol(pipe(df)), lambda df, extra=None: df.width)
    verb("dim", "eda", ["dim"], lambda df, extra=None: t.dim(pipe(df)), lambda df, extra=None: (df.height, df.width))
    verb(
        "dtypes", "eda", ["dtypes"],
        lambda df, extra=None: {name: str(dtype) for name, dtype in t.dtypes(pipe(df)).items()},
        lambda df, extra=None: {name: str(dtype) for name, dtype in df.schema.items()},
    )
    verb(
        "summary", "eda", ["summary"],
        lambda df, extra=None: t.summary(pipe(small(df))),
        lambda df, extra=None: api._describe(small(df)),
    )
    verb(
        "describe", "eda", ["describe"],
        lambda df, extra=None: t.describe(pipe(small(df))),
        lambda df, extra=None: api._describe(small(df)),
    )

    # ── methods and io ─────────────────────────────────────────────────
    verb("height", "frame", [], lambda df, extra=None: pipe(df).height, lambda df, extra=None: df.height, methods=["height"])
    verb("width", "frame", [], lambda df, extra=None: pipe(df).width, lambda df, extra=None: df.width, methods=["width"])
    verb("shape", "frame", [], lambda df, extra=None: pipe(df).shape, lambda df, extra=None: df.shape, methods=["shape"])
    verb("columns", "frame", [], lambda df, extra=None: list(pipe(df).columns), lambda df, extra=None: list(df.columns), methods=["columns"])
    verb(
        "schema", "frame", [],
        lambda df, extra=None: {name: str(dtype) for name, dtype in pipe(df).schema.items()},
        lambda df, extra=None: {name: str(dtype) for name, dtype in df.schema.items()},
        methods=["schema"],
    )
    verb("names_list", "frame", [], lambda df, extra=None: pipe(df).names_list(), lambda df, extra=None: list(df.columns), methods=["names_list"])
    verb("lazy", "frame", [], lambda df, extra=None: pipe(df).lazy().collect(), lambda df, extra=None: df.lazy().collect(), methods=["lazy"])
    verb("to_polars", "frame", [], lambda df, extra=None: pipe(df).to_polars(), lambda df, extra=None: df, methods=["to_polars"])
    verb(
        "to_pandas", "frame", [],
        lambda df, extra=None: pl.from_pandas(pipe(small(df)).to_pandas()),
        lambda df, extra=None: pl.from_pandas(small(df).to_pandas()),
        methods=["to_pandas"],
    )
    verb(
        "to_arrow", "frame", [],
        lambda df, extra=None: pl.from_arrow(pipe(small(df)).to_arrow()),
        lambda df, extra=None: pl.from_arrow(small(df).to_arrow()),
        methods=["to_arrow"],
    )
    verb(
        "to_numpy", "frame", ["to_numpy"],
        lambda df, extra=None: pipe(df.select("mass")) >> t.to_numpy(),
        lambda df, extra=None: df.select("mass").to_numpy(order="fortran"),
    )
    verb(
        "with_polars", "frame", [],
        lambda df, extra=None: pipe(df).with_polars(lambda frame: frame.filter(pl.col("id") < 4)),
        lambda df, extra=None: df.filter(pl.col("id") < 4),
        methods=["with_polars"],
    )

    def _reader(suffix, separator=None):
        def prepare(df):
            method = "parquet" if suffix == ".parquet" else "ipc" if suffix == ".ipc" else "csv"
            return api._write(df, suffix, method, separator)
        return prepare

    def _scan(reader, native):
        def tidy(df, extra):
            return reader(extra).collect()

        def raw(df, extra):
            return native(extra).collect()

        return tidy, raw

    tidy_scan, raw_scan = _scan(t.scan_csv, pl.scan_csv)
    verb("scan_csv", "io", ["scan_csv"], tidy_scan, raw_scan, prepare=_reader(".csv"))
    tidy_scan, raw_scan = _scan(t.scan_parquet, pl.scan_parquet)
    verb("scan_parquet", "io", ["scan_parquet"], tidy_scan, raw_scan, prepare=_reader(".parquet"))
    tidy_scan, raw_scan = _scan(t.scan_ipc, pl.scan_ipc)
    verb("scan_ipc", "io", ["scan_ipc"], tidy_scan, raw_scan, prepare=_reader(".ipc"))
    verb(
        "read_csv", "io", ["read_csv"],
        lambda df, extra: t.read_csv(extra).collect(),
        lambda df, extra: pl.read_csv(extra),
        prepare=_reader(".csv"),
    )
    verb(
        "read_tsv", "io", ["read_tsv"],
        lambda df, extra: t.read_tsv(extra).collect(),
        lambda df, extra: pl.read_csv(extra, separator="\t"),
        prepare=_reader(".tsv", "\t"),
    )
    verb(
        "read_delim", "io", ["read_delim"],
        lambda df, extra: t.read_delim(extra, "|").collect(),
        lambda df, extra: pl.read_csv(extra, separator="|"),
        prepare=_reader(".txt", "|"),
    )

    def _write_paths(suffix):
        def prepare(df):
            paths = []
            for _ in range(2):
                handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                handle.close()
                paths.append(Path(handle.name))
            return paths
        return prepare

    def _write_case(name, method, reader):
        def tidy(df, extra):
            # The pipe form; it calls the TidyFrame method of the same name.
            pipe(df.select("id", "species", "whole")) >> getattr(t, method)(extra[0])
            return reader(extra[0])

        def raw(df, extra):
            getattr(df.select("id", "species", "whole"), method)(extra[1])
            return reader(extra[1])

        verb(name, "io", [method], tidy, raw, methods=[method], prepare=_write_paths(name.split("_", 1)[1] if False else {"write_csv": ".csv", "write_parquet": ".parquet", "write_ipc": ".ipc"}[method]))

    _write_case("write_csv", "write_csv", pl.read_csv)
    _write_case("write_parquet", "write_parquet", pl.read_parquet)
    _write_case("write_ipc", "write_ipc", pl.read_ipc)

    # ── round_date, make_date, make_datetime, isoyear, leap_year ─────────
    def _col(name, functions, build, native):
        add(api._column(name, "dates", functions, build, native))

    def _round_month(df):
        below = pl.col("day_d").dt.truncate("1mo")
        above = below.dt.offset_by("1mo")
        return pl.when((above - pl.col("day_d")) <= (pl.col("day_d") - below)).then(above).otherwise(below)

    _col("round_date", ["round_date"], lambda t, df: t.round_date("day_d", "month"), _round_month)
    _col(
        "make_date", ["make_date"],
        lambda t, df: t.make_date(t.col("id") % 400 + 1800, t.col("id") % 12 + 1, t.col("id") % 28 + 1),
        lambda df: pl.date(pl.col("id") % 400 + 1800, pl.col("id") % 12 + 1, pl.col("id") % 28 + 1),
    )
    _col(
        "make_datetime", ["make_datetime"],
        lambda t, df: t.make_datetime(2024, 1, t.col("id") % 28 + 1, t.col("id") % 24, 30, 15.5),
        lambda df: pl.datetime(2024, 1, pl.col("id") % 28 + 1, pl.col("id") % 24, 30, 15, 500_000, time_zone="UTC"),
    )
    _col("isoyear", ["isoyear"], lambda t, df: t.isoyear("day_d"), lambda df: pl.col("day_d").dt.iso_year().cast(pl.Int64))
    _col("leap_year", ["leap_year"], lambda t, df: t.leap_year("day_d"), lambda df: pl.col("day_d").dt.is_leap_year())

    # ── durations and periods ────────────────────────────────────────────
    def _span(name, functions, build, native):
        add(api._column(name, "dates", functions, build, native))

    def _us(seconds):
        return pl.duration(microseconds=seconds * 1_000_000)

    _span("ddays", ["ddays"], lambda t, df: t.col("day_d") + t.ddays(1), lambda df: pl.col("day_d") + pl.duration(days=1))
    _span("dseconds", ["dseconds"], lambda t, df: t.dseconds(t.col("bill")), lambda df: (pl.col("bill") * 1e6).round().cast(pl.Int64).cast(pl.Duration("us")))
    _span("dminutes", ["dminutes"], lambda t, df: t.col("at") + t.dminutes(90), lambda df: pl.col("at") + _us(5400))
    _span("dhours", ["dhours"], lambda t, df: t.col("at") - t.dhours(1.5), lambda df: pl.col("at") - _us(5400))
    _span("dweeks", ["dweeks"], lambda t, df: t.col("day_d") + t.dweeks(2), lambda df: pl.col("day_d") + pl.duration(weeks=2))
    _span("dmonths", ["dmonths"], lambda t, df: t.col("at") + t.dmonths(1), lambda df: pl.col("at") + _us(2629800))
    _span("dyears", ["dyears"], lambda t, df: t.col("day_d") + t.dyears(1), lambda df: pl.col("day_d").cast(pl.Datetime("us")) + _us(31557600))
    _span("duration", ["duration"], lambda t, df: t.col("at") + t.duration(hours=1, minutes=30), lambda df: pl.col("at") + pl.duration(hours=1, minutes=30))
    _span("as_duration", ["as_duration"], lambda t, df: t.as_duration("whole"), lambda df: (pl.col("whole") * 1_000_000).cast(pl.Duration("us")))
    _span(
        "date_difference", [],
        lambda t, df: t.col("day_d") - t.col("day_d").min(),
        lambda df: pl.col("day_d") - pl.col("day_d").min(),
    )

    def _length(df):
        micros = (pl.col("day_d") - pl.col("day_d").min()).dt.total_microseconds().cast(pl.Float64)
        return micros / (micros * 0.0 + 86_400_000_000.0)

    _span("time_length", ["time_length"], lambda t, df: t.time_length(t.col("day_d") - t.min("day_d"), "day"), _length)
    add(api._summary(
        "duration_sum", [],
        lambda t, df: t.sum(t.col("day_d") - t.col("day_d").min()),
        lambda df: (pl.col("day_d") - pl.col("day_d").min()).sum(),
        family="dates",
    ))
    _span("days", ["days"], lambda t, df: t.col("day_d") + t.days(1), lambda df: pl.col("day_d").dt.offset_by("1d"))
    _span("weeks", ["weeks"], lambda t, df: t.col("day_d") - t.weeks(2), lambda df: pl.col("day_d").dt.offset_by("-14d"))
    _span("months_plus", [], lambda t, df: t.col("day_d") + t.months(1), lambda df: pl.col("day_d").dt.offset_by("1mo"))
    _span("minutes", ["minutes"], lambda t, df: t.col("at") + t.minutes(90), lambda df: pl.col("at").dt.offset_by("5400000000us"))
    _span("seconds", ["seconds"], lambda t, df: t.col("at") + t.seconds(1.5), lambda df: pl.col("at").dt.offset_by("1500000us"))
    _span(
        "period", ["period"],
        lambda t, df: t.col("day_d") + t.period(months=1, days=2, hours=3),
        lambda df: pl.col("day_d").cast(pl.Datetime("us")).dt.offset_by("1mo2d10800000000us"),
    )

    zone = "America/New_York"

    def _zoned(df):
        return df.select("id", pl.col("at").dt.replace_time_zone(zone, ambiguous="earliest", non_existent="null").alias("z"))

    verb(
        "hours", "dates", ["hours"],
        lambda df, extra: (t.tidy(extra) >> t.mutate(out=t.col("z") + t.hours(1)) >> t.select("out")).collect(),
        lambda df, extra: extra.select(
            pl.col("z").dt.replace_time_zone(None).dt.offset_by("1h")
            .dt.replace_time_zone(zone, ambiguous="earliest", non_existent="null").alias("out")
        ),
        prepare=_zoned,
    )
