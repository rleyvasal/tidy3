# tidy3

dplyr-style **lazy** data manipulation for Python, powered by [Polars](https://pola.rs/).

- Readable multi-line pipes (`>>` ≈ R’s `|>`)
- **Partial pipeline run in any Jupyter** (SolveIt, JupyterLab, classic, …) — no VS Code extension
- Lazy until `collect()` / plot / preview (`LIMIT n` only)
- Optional bridge to [plot3](https://github.com/rleyvasal/plot3) and [gpudev](https://github.com/rleyvasal/gpudev)

## Install

```bash
cd /path/to/tidy3
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```python
from tidy3 import tidy, filter, mutate, group_by, summarise, col, n, mean, collect

out = (
    tidy(cars)                          # pandas or polars
    >> filter(col("mpg") > 20)
    >> mutate(km=col("mpg") * 1.609)
    >> group_by("cyl")
    >> summarise(n=n(), avg=mean("mpg"))
    >> collect()                        # polars; collect(as_="pandas") for pandas
)
```

With the Jupyter extension loaded you can omit the outer parentheses — the **kernel** rewrites multi-line `>>` pipes before parse.

## Jupyter / SolveIt

### Load (same as other gpudev addons)

```text
%local
%run /path/to/gpudev/CRAFT.py
%run /path/to/gpudev/addons/tidy3.py
%gpu
```

Or without CRAFT: `%load_ext tidy3.jupyter` / `pip install -e .`

Under **`%gpu`**, cells run on the remote kernel — a separate namespace and
filesystem. The addon handles this automatically: on the first `%gpu` cell it
**pushes the local tidy3 source to the remote kernel** (`tidy3.craft`, ~10 KB
over the existing channel), installs polars there if missing, and loads the
Jupyter extension remotely (API names, `>>` rewriting, `%%tidy3_run`). It
re-seeds by itself after `%restart_kernel` and after local source edits
(content stamp). If something goes sideways: `seed_tidy3_remote(force=True)`.

Then just use normal tidy3 with **paths on the GPU host**:

```python
# after %gpu — file path is on the GPU box
scan_parquet("/home/gpudev/data/huge.parquet")
>> filter(col("year") >= 2020)
>> group_by("region")
>> summarise(n=n(), avg=mean("value"))
```

### Partial run

- **Run Selected Text** (if SolveIt has it): highlight a pipe prefix → run selection  
- Own cell with only the prefix  
- `%%tidy3_run` with the prefix pasted in  

```python
%%tidy3_run
tidy(cars)
>> filter(col("mpg") > 20)
```

```text
%tidy3_pipes on|off|status
```

## plot3

```python
from plot3 import aes, geom_point, ggplot

tidy(big)
>> filter(col("intensity") > 10)
>> select("x", "y", "z", "intensity")
>> ggplot(aes(x="x", y="y", z="z", colour="intensity"))
+ geom_point()
```

The existing method handoff remains available as
``tidy(df).ggplot(aes(...))``.

## gpudev / CRAFT / SolveIt

```text
%local
%run /path/to/gpudev/CRAFT.py
%run /path/to/gpudev/addons/tidy3.py
%run /path/to/gpudev/addons/plot3.py
```

```bash
ln -s ../../tidy3 /path/to/gpudev/addons/tidy3   # if needed
```

## API (v0.2)

| Area | Symbols |
|------|---------|
| Frame | `tidy`, `scan_parquet`, `scan_csv`, `scan_ipc`, `TidyFrame` |
| Rows | `filter`, `filter_out`, `arrange`, `distinct`, `slice`, `slice_head`, `slice_tail`, `slice_min`, `slice_max`, `slice_sample`, `head`, `sample_n`, `sample_frac` |
| Columns | `mutate`, `transmute`, `select`, `drop`, `rename`, `rename_with`, `relocate`, `pull`, `glimpse` |
| Groups | `group_by`, `rowwise`, `ungroup`, `summarise`, `reframe`, `count`, `tally`, `add_count`, `add_tally` |
| Joins | `left_join`, `right_join`, `inner_join`, `full_join`, `semi_join`, `anti_join`, `cross_join` |
| Join specs | `join_by`, `eq`, `ge`, `gt`, `le`, `lt`, `closest`, `between`, `within`, `overlaps` |
| Bind/set | `bind_rows`, `bind_cols`, `union`, `union_all`, `intersect`, `setdiff`, `symdiff`, `setequal` |
| Row mutation | `rows_insert`, `rows_append`, `rows_update`, `rows_patch`, `rows_upsert`, `rows_delete` |
| Selectors | `everything`, `col_range`, `last_col`, `group_cols`, `starts_with`, `ends_with`, `contains`, `matches`, `num_range`, `all_of`, `any_of`, `where` |
| Column-wise | `across`, `if_any`, `if_all`, `pick`, `c_across` |
| Materialize | `collect`, `pull`, `glimpse`, `peek` |
| Expr | `col`, `n`, `mean`, `sum`, `min`, `max`, `median`, `std`, `first`, `last`, `desc`, ranking/window helpers, `n_distinct`, `coalesce`, `if_else`, `case_when` |
| Jupyter | `%load_ext tidy3.jupyter`, `%tidy3_run`, `%%tidy3_run`, `%tidy3_pipes` |
| Partial | `partial_run`, `maybe_rewrite_cell`, `normalize_pipe_source` |
| Escape | `TidyFrame.with_polars(fn)` |

## Backends

The default engine is **Polars lazy**. For 1:1 engine comparisons (e.g.
against datar, which is pandas-only) the same pipeline also runs on an
**eager pandas backend**:

```python
tidy(df, backend="pandas") >> filter(col("x") > 0) >> ...   # per-frame
options(backend="pandas")                                    # session default
```

Expressions (`col("x") * 2`, `mean("y")`, comparisons, `cum_sum`, …) are
backend-neutral: they compile to `pl.Expr` on polars and evaluate natively on
pandas, with dplyr window semantics after `group_by` on both. The pandas
backend covers the documented verb/expression subset; anything
polars-specific raises a clear error pointing back to `backend="polars"`.

### Tidy-select and column-wise operations

Selectors work anywhere columns are selected by `select`, `drop`, `relocate`,
or `rename_with`. Combine them with `|` (union), `&` (intersection), `-`
(difference), or `~` (complement):

```python
tidy(df)
>> select("id", starts_with("measure_"), last_col())

tidy(df)
>> select(everything() - ends_with("_raw"))

tidy(df)
>> relocate(where(is_numeric), after="label")
```

`where` receives column dtypes; portable predicates include `is_numeric`,
`is_string`, `is_boolean`, and `is_temporal`. `all_of(names)` is strict about
missing columns while `any_of(names)` silently ignores them.

Use `across` as a positional argument to `mutate`, `transmute`, or
`summarise`. Python name templates accept `{col}` and `{fn}`; dplyr-style
`{.col}` and `{.fn}` are accepted too:

```python
tidy(df)
>> mutate(across(starts_with("x"), lambda x: x.round(2)))
>> filter(if_any(ends_with("_score"), lambda x: x > 0))
>> group_by("team")
>> summarise(
    across(
        where(is_numeric),
        {"mean": mean, "sd": std},
        names="{col}_{fn}",
    )
)
```

### Row-wise operations and reframing

`rowwise` evaluates ordinary window/aggregate expressions one row at a time.
Its optional tidy-select arguments are identifier columns preserved by
`summarise`. Use `c_across` for row-wise reductions; grouping identifiers are
automatically excluded:

```python
tidy(df)
>> rowwise("id")
>> mutate(
    total=sum(c_across(starts_with("score_"))),
    average=mean(c_across(starts_with("score_"))),
)
>> ungroup()
```

`pick` represents a selected set as one structured column and supports common
horizontal reductions without requiring `rowwise`:

```python
tidy(df) >> mutate(total=pick(starts_with("score_")).sum())
tidy(df) >> mutate(scores=pick(starts_with("score_")))
```

`reframe` accepts vector-valued expressions, recycles scalar results within
each group, and always returns an ungrouped frame:

```python
tidy(df)
>> group_by("team")
>> reframe(score=col("score"), team_mean=mean("score"))
```

### Row mutation and advanced joins

The SQL-inspired row verbs use `y`'s first column as the key by default, or
accept explicit `by=` keys. `y` may contain any subset of `x`'s columns:

```python
tidy(accounts)
>> rows_insert(new_accounts, by="id", conflict="ignore")
>> rows_patch(corrections, by="id")       # only replaces missing values
>> rows_upsert(latest, by="id")           # update matches, append new keys
>> rows_delete(retired, by="id")
```

`rows_insert` supports `conflict="error"|"ignore"`; update, patch, and delete
support `unmatched="error"|"ignore"`. Polars validates these policies lazily
when the plan is collected.

`join_by` accepts same-name equality keys, `(left, right)` equality pairs, or
`(left, operator, right)` conditions. Named helpers are clearer for advanced
joins:

```python
sales
>> left_join(
    promos,
    by=join_by("id", ge("sale_date", "promo_date")),
)

# One nearest earlier promotion per sale
sales
>> left_join(
    promos,
    by=join_by("id", closest(ge("sale_date", "promo_date"))),
)

points >> inner_join(ranges, by=join_by(between("point", "lower", "upper")))
segments >> inner_join(regions, by=join_by(overlaps("lo", "hi", "start", "end")))
```

Advanced specifications work with left/right/inner/full/semi/anti joins.
Polars uses its native inequality-join plan when no equality partition is
needed; equality-partitioned inequalities first reduce candidates by key.

### Per-operation grouping and portable helpers

Use `by=` when grouping is needed for one operation only. It accepts the same
tidy-select specifications as `select()` and returns an ungrouped result:

```python
tidy(sales)
>> mutate(region_average=mean("amount"), by="region")
>> filter(col("amount") > mean("amount"), by="region")

tidy(sales) >> summarise(total=sum("amount"), by=["region", "year"])
tidy(sales) >> slice_max("amount", n=2, by="region")
```

`by=` is supported by `mutate`, `transmute`, `filter`, `filter_out`,
`summarise`, `reframe`, `slice`, and every `slice_*` variant. Like dplyr,
it cannot be combined with an already grouped or rowwise frame.

Portable helpers run with the same grouped semantics on both backends:

```python
tidy(events) >> mutate(
    row=row_number(),
    rank=dense_rank("score"),
    previous=lag("score"),
    running_average=cummean("score"),
    label=case_when(
        (col("score").is_null(), "missing"),
        (col("score") >= 90, "high"),
        default="other",
    ),
    by="team",
)
```

Ranking helpers are `row_number`, `min_rank`, `dense_rank`, `percent_rank`,
`cume_dist`, and `ntile`. Window and value helpers include `lead`, `lag`,
`cummean`, `cumall`, `cumany`, `n_distinct`, `coalesce`, `if_else`, and
`case_when`.

Mutating joins accept dplyr-style safety controls:

```python
orders >> left_join(
    customers,
    on="customer_id",
    relationship="many-to-one",
    unmatched="error",
    multiple="all",
    na_matches="never",
    keep=False,
)
```

`relationship` accepts `one-to-one`, `one-to-many`, `many-to-one`, or
`many-to-many`; `multiple` accepts `all`, `any`, `first`, or `last`.
Polars relationship and unmatched checks remain lazy and raise on collection.

Every frame-returning Phase 1–5 verb stays lazy on the Polars backend. `pull`,
`setequal`, plotting, previews, and explicit `collect` are intentional
materialization boundaries. `slice_sample(weight_by=...)` is currently
available on the pandas backend; unweighted sampling (including
`replace=True`) is supported on both. Experimental `across(..., unpack=...)`,
`nest_join`, ordered `lead`/`lag`, and additional vector helpers are future
phases.

## Benchmark vs datar

```python
from tidy3 import bench
bench.run(rows=10_000_000)        # full pipeline; pip install datar datar-pandas for the datar row
bench.run_ops(rows=10_000_000)    # each verb in isolation vs raw pandas
```

For the adoption-oriented suite, which covers everyday operations plus ML,
event-history, customer aggregation, and join/filter/aggregate workflows:

```bash
python -m tidy3.bench_suite --rows 1000000 --repeat 5 --output pandas
python -m tidy3.bench_suite --rows 1000000 --repeat 5 --output native
```

The default `pandas` output makes every engine return a pandas DataFrame and
therefore includes the Polars-to-pandas ML handoff. `native` keeps Polars
results in Polars and isolates execution from conversion. Data generation,
warm-up, garbage collection, and correctness validation are outside recorded
times; every measured result is checked against raw pandas. The suite reports
medians and rotates engine order between repetitions. Optional datar coverage
remains in the smaller `tidy3.bench` benchmark as a backup comparison.

Apple-silicon laptop, 10M rows × 100 groups, filter→mutate→group_by→summarise→arrange:

| engine | time | vs fastest |
|---|---|---|
| **tidy3[polars]** | 40.6ms | 1.0x |
| polars (raw lazy) | 41.5ms | 1.0x |
| pandas (raw) | 60.1ms | 1.5x |
| **tidy3[pandas]** | 60.8ms | 1.5x |
| datar[pandas] | 166.3ms | 4.1x |

Per operation (`run_ops`; ratios vs raw pandas; tidy3[polars] includes plan
execution + `to_pandas`):

| op | pandas | tidy3[pandas] | datar | tidy3[polars] |
|---|---|---|---|---|
| filter x>0 | 27.4ms | 22.9ms (0.8x) | 25.6ms (0.9x) | 25.1ms (0.9x) |
| mutate z=x*2+y | 4.8ms | 5.1ms (1.1x) | 13.0ms (2.7x) | 41.9ms (8.7x)¹ |
| group+summarise | 66.3ms | 67.7ms (**1.0x**) | 264.2ms (4.0x) | 54.5ms (**0.8x**) |
| arrange y | 1141.8ms | 1308.9ms (1.1x) | 1363.6ms (1.2x) | 240.6ms (**0.2x**) |

¹ the op itself is ~free in polars; timed in isolation, the cost is
materializing 40M values back to pandas — in a real lazy pipeline that
happens once at the end (see the pipeline table).

The takeaway: **tidy3's wrapper is free on both engines** (simple
aggregations batch into one `groupby().agg()` pass, matching hand-written
pandas), datar pays 2.7–4x everywhere it matters, and the polars backend
wins outright. Under `%gpu`, run the same `bench.run(...)` on the remote
kernel (`!uv pip install datar datar-pandas` there first for the datar row).

## Notes

- **Grouped semantics are dplyr's**: after `group_by`, `mutate`/`filter`/
  `slice_*`/`sample_n`/`head` evaluate **per group** (Polars window `.over`).
  `summarise` aggregates per group.
- **Display is self-contained**: tables carry inline styles (no `<style>`
  block), so they render identically in local cells, republished `%gpu`
  output, and sslive exports. Under `%gpu`, bare polars DataFrames are
  restyled the same way (`seed_tidy3_remote(style_polars=False)` to opt out).
- **Builtins are shadowed** by design: injecting the API puts `filter`, `slice`,
  `sum`, `min`, `max` into the notebook namespace (dplyr ergonomics). Use
  `builtins.filter` etc. when you need the Python originals.
- **Plotting big remote data**: aggregate remotely, then let plot3's own
  remote path pull the small result: `%plot3 res.to_pandas() x=... y=...`.

## Why not datar?

datar’s pandas/Python wrappers struggle on large frames. tidy3 compiles verbs to a **Polars Lazy** plan and only materializes at the edge. Previews use `LIMIT n`.

## License

MIT
