# tidy3 reference

Every verb family in detail: saving results, the API table, backends, tidy-select, reshaping, row-wise work, joins, grouping helpers, performance controls, and the NumPy bridge. Names follow dplyr and tidyr, so their documentation applies too.

Back to the [README](../README.md).

The examples on this page share this setup:

```python
from tidy3 import *

# mtcars, read from the web; `model` holds the car names
cars = (
    scan_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv")
    >> rename(model="rownames")
    >> select("model", "mpg", "cyl", "hp", "wt")
    >> compute()
)
cars.write_parquet("cars.parquet")

result = (
    scan_parquet("cars.parquet")
    >> filter(col("mpg") > 20)
    >> mutate(km=col("mpg") * 1.609)
    >> group_by("cyl")
    >> summarise(n=n(), avg=mean("mpg"))
)
```

## Save results

Write a pipeline directly without first calling `collect()`:

```python
result.write_parquet("summary.parquet")  # recommended for data pipelines
result.write_csv("summary.csv")
result.write_ipc("summary.arrow")        # Arrow IPC / Feather
```

With the default Polars backend, these execute the lazy plan and stream its
result to disk, avoiding a second fully materialized DataFrame in memory.
The same methods also work with `backend="pandas"`. GPU writers execute the
plan on the GPU, then materialize before serialization because current GPU
file sinks are not stable across all formats; `auto` and `streaming` retain
direct lazy sinks.

Choose how Polars executes when the plan is materialized or written:

```python
result.collect(engine="auto")             # default: let Polars choose
result.collect(engine="streaming")        # execute in streaming batches

result.write_parquet("summary.parquet", engine="streaming")
result.write_ipc("summary.arrow", engine="auto")
```

On an NVIDIA GPU with `pip install cudf-polars`, Polars runs what it can on
the GPU and falls back to the CPU for the rest:

```python notest
result.collect(engine="gpu")
result.write_csv("summary.csv", engine="gpu")
```

The same `engine=` argument is available on `to_polars()`, `to_pandas()`,
`to_arrow()`, and `write_excel()`. It applies only to the default Polars
backend. For a GPU run that must not silently fall back to CPU, pass
`engine=polars.GPUEngine(raise_on_fail=True)`; the benchmark suite does this
automatically whenever `--polars-engine gpu` is selected.

Excel output is intended for smaller reporting datasets and must materialize
the result:

```bash
pip install "tidy3[excel]"
```

```python
result.write_excel(
    "summary.xlsx",
    worksheet="Summary",
    autofit=True,
    freeze_panes="A2",
)
```

You can still collect first when another library needs the result:

```python
result.collect().write_csv("summary.csv")          # Polars API
result.collect(as_="pandas").to_csv("summary.csv", index=False)
```

## API

| Area | Symbols |
|------|---------|
| Export | `nb_export`, `transform_source`, `python -m tidy3 export` / `run` |
| Frame | `tidy`, `scan_parquet`, `scan_csv`, `scan_ipc`, `TidyFrame` |
| Output | `collect`, `to_numpy`, `TidyFrame.write_parquet`, `TidyFrame.write_csv`, `TidyFrame.write_ipc`, `TidyFrame.write_excel` |
| Rows | `filter`, `filter_out`, `arrange`, `distinct`, `slice`, `slice_head`, `slice_tail`, `slice_min`, `slice_max`, `slice_sample`, `head`, `sample_n`, `sample_frac` |
| Columns | `mutate`, `transmute`, `select`, `drop`, `rename`, `rename_with`, `relocate`, `pull`, `glimpse` |
| Groups | `group_by`, `rowwise`, `ungroup`, `with_groups`, `group_split`, `group_map`, `group_modify`, `group_walk`, `group_nest`, `nest_by`, `group_data`, `group_keys`, `group_rows`, `group_size`, `group_indices`, `group_vars`, `groups`, `n_groups`, `group_trim`, `summarise`, `reframe`, `count`, `tally`, `add_count`, `add_tally` |
| Missing data | `drop_na`, `replace_na`, `fill`, `complete`, `expand`, `nesting`, `crossing`, `expand_grid`, `full_seq`, `uncount` |
| Reshape | `pivot_longer`, `pivot_wider`, `build_longer_spec`, `pivot_longer_spec`, `build_wider_spec`, `pivot_wider_spec`, `check_pivot_spec`, `separate`, `separate_longer_delim`, `separate_wider_delim`, `separate_wider_regex`, `separate_wider_position`, `separate_longer_position`, `unite`, `nest`, `unnest`, `unnest_longer`, `unnest_wider`, `unnest_auto`, `chop`, `unchop`, `hoist`, `pack`, `unpack` |
| Joins | `left_join`, `right_join`, `inner_join`, `full_join`, `semi_join`, `anti_join`, `cross_join`, `nest_join` |
| Join specs | `join_by`, `eq`, `ge`, `gt`, `le`, `lt`, `closest`, `between`, `within`, `overlaps` |
| Bind/set | `bind_rows`, `bind_cols`, `union`, `union_all`, `intersect`, `setdiff`, `symdiff`, `setequal` |
| Row mutation | `rows_insert`, `rows_append`, `rows_update`, `rows_patch`, `rows_upsert`, `rows_delete` |
| Selectors | `everything`, `col_range`/`cols_between`, `last_col`, `group_cols`, `starts_with`, `ends_with`, `contains`, `matches`, `num_range`, `all_of`, `any_of`, `where`; set ops `\|` `&` `-` `~`/`!`/`-helper`; predicates `is_numeric`, `is_integer`, `is_float`, `is_string`/`is_character`, `is_bool`/`is_boolean`, `is_datetime`, `is_categorical`, `is_temporal` |
| Column-wise | `across`, `if_any`, `if_all`, `pick`, `c_across` |
| Materialize | `collect`, `pull`, `glimpse`, `peek` |
| Expr | `col`, `n`, `mean`, `sum`, `min`, `max`, `median`, `std`/`sd`, `var`, `any`, `all`, `first`, `last`, `nth`, `near`, `na_if`, `between`, `consecutive_id`, `case_match`, `recode`, ranking/window helpers, `n_distinct`, `coalesce`, `if_else`, `case_when`, `recode_values`, `replace_values`, `replace_when`, `when_any`, `when_all`, `order_by`, `with_order` |
| Jupyter | `%load_ext tidy3.jupyter`, `%tidy3_run`, `%%tidy3_run`, `%tidy3_pipes` |
| Partial | `partial_run`, `maybe_rewrite_cell`, `normalize_pipe_source` |
| Escape | `TidyFrame.with_polars(fn)` |

### API maturity

tidy3 is alpha. Symbols work today, but not every verb is equally polished for
production pipelines or R byte-for-byte parity. Prefer the **stable core** when
you need predictable performance and semantics.

| Tier | Intent | Symbols / areas |
|------|--------|-----------------|
| **Stable core** | Daily driver: declarative dplyr on Polars, dual-backend tests, R oracle where installed | `tidy`, `scan_*`, `filter`, `filter_out`, `mutate`, `transmute`, `select`, `drop`, `rename`, `relocate`, `arrange`, `distinct`, `group_by`, `ungroup`, `summarise`/`summarize`, `count`, `tally`, `add_count`, `add_tally`, equality joins (`left`/`right`/`inner`/`full`/`semi`/`anti`/`cross`), `bind_rows`/`bind_cols`, set ops (`union`, `union_all`, `intersect`, `setdiff`, `symdiff`, `setequal`), `head`/`slice`/`slice_head`/`slice_tail`/`slice_min`/`slice_max`, core expr helpers (`col`, `n`, `mean`, `sum`, …), tidyselect basics, `collect` and file writers |
| **Growing** | Feature-complete enough for real work; more edge cases and schema-discovery cost | tidyr missing/reshape (`drop_na`, `replace_na`, `fill`, `complete`, `expand`, `pivot_*`, `separate`, `unite`, `nest`/`unnest*`), `across`/`if_any`/`if_all`/`pick`/`c_across`, `rowwise`, `reframe`, `rename_with`, `join_by` inequality joins, `nest_join`, `rows_*`, advanced ranking/window helpers, NumPy/`to_numpy` handoff |
| **Experimental** | Correctness-first or Python-callback paths; may materialize eagerly or lag R oracle coverage | `group_split`, `group_map`, `group_modify`, `group_nest`, `with_groups`, `hoist`, `pack`, `unpack`, `separate_longer_delim`, `separate_wider_delim`, stochastic sampling (`sample_n`, `sample_frac`, `slice_sample`) |

**Performance note:** the stable core is the path that tracks raw Polars and
beats pandas on realistic sizes. `group_nest` compiles to a Polars
`group_by().agg(struct)` plan and is typically **faster** than building
nested pandas DataFrames by hand; for counts alone use `count()`/`tally()`
instead of nesting. Python-callback verbs (`group_map` / `group_modify` /
`group_split`) still materialize and loop in Python — fine for small groups,
not for hot large-data paths. Prefer declarative `summarise` when
performance matters.

## Backends

The default engine is **Polars lazy**. For 1:1 engine comparisons (e.g.
against datar, which is pandas-only) the same pipeline also runs on an
**eager pandas backend**:

```python
import pandas as pd

df = pd.DataFrame({"x": [-1.5, 0.5, 2.0], "y": [10, 20, 30]})

tidy(df, backend="pandas") >> filter(col("x") > 0)   # per frame
options(backend="pandas")                            # session default
options(backend="polars")                            # back to the default
```

Expressions (`col("x") * 2`, `mean("y")`, comparisons, `cum_sum`, …) are
backend-neutral: they compile to `pl.Expr` on polars and evaluate natively on
pandas, with dplyr window semantics after `group_by` on both. The pandas
backend covers the documented verb/expression subset; anything
polars-specific raises a clear error pointing back to `backend="polars"`.

### Tidy-select and column-wise operations

Selectors work anywhere columns are selected by `select`, `drop`, `relocate`,
or `rename_with`. Combine them with `|` (union), `&` (intersection), `-`
(difference), or `~` (complement / negation):

```python
df = tidy({
    "id": [1, 2, 3],
    "label": ["a", "b", "c"],
    "measure_1": [0.5, 0.7, 0.2],
    "measure_2": [1.5, 1.1, 0.9],
    "mpg": [21.0, 22.8, 18.7],
    "cyl": [6, 4, 8],
    "hp": [110, 93, 175],
    "tmp_flag": [True, False, True],
    "hp_raw": ["110", "93", "175"],
})

df >> select("id", starts_with("measure_"), last_col())
df >> select(~starts_with("tmp_"))             # preferred: Python-native invert
df >> select(where(is_numeric) & ~starts_with("id"))
df >> select(cols_between("mpg", "hp"))        # inclusive column range
df >> select(everything() - ends_with("_raw"))
df >> relocate(where(is_numeric), after="label")
```

**Negation: prefer `~`.** It is always valid Python (`Expr` / `Selector`
implement `__invert__`) and works in scripts, notebooks, and exports with no
preparser.

In Jupyter/SolveIt only, `!` is optional R-like sugar **inside tidy3 verb
calls** (e.g. `select(!starts_with("tmp_"))` → `~`). Outside tidy3 contexts
`!` is left alone so shell cells stay literal (`!pip install …`).

`where` receives column dtypes; portable predicates include `is_numeric`,
`is_string`, `is_boolean`, and `is_temporal`. `all_of(names)` is strict about
missing columns while `any_of(names)` silently ignores them.

### Reshape and missing data

The familiar tidyr operations work as pipe verbs or `TidyFrame` methods on
both backends:

```python
from tidy3 import (
    complete, drop_na, fill, nest, pivot_longer, pivot_wider, replace_na,
    separate, starts_with, tidy, unite, unnest, unnest_longer,
)

# Weekly readings per visit; the patient id is only on each patient's first row
measurements = {
    "patient_id": ["p1", None, "p2", None],
    "visit": ["baseline", "follow-up", "baseline", "follow-up"],
    "week_1": [5.1, 4.8, 6.0, None],
    "week_2": [5.3, None, 6.2, 6.4],
}
long_measures = {
    "store": ["A", "A", "B", "B"],
    "quarter": ["Q1", "Q2", "Q1", "Q2"],
    "sales": [100, 120, 90, 95],
    "cost": [60, 70, 55, 50],
}
visits = {"subject": [1, 2], "mean_1": [5.0, 6.1], "mean_2": [5.4, 6.3],
          "sd_1": [0.4, 0.5], "sd_2": [0.3, 0.6]}
labels = {"code": ["north-1", "north-2", "south-1"]}
events = {
    "team": ["red", "red", "blue", "blue"],
    "time": [1, 2, 1, 2],
    "value": [3.5, 4.0, 2.5, 3.0],
    "score": [95, None, 88, 91],
    "items": [["a", "b"], ["c"], [], ["d", "e"]],
}
values = {"score": [7, None, 9], "required_field": ["x", "y", None]}
observations = {"subject": ["s1", "s1", "s2"], "visit": [1, 2, 1], "score": [3, None, 4]}

long = (
    tidy(measurements)
    >> fill("patient_id", direction="down")
    >> pivot_longer(
        starts_with("week_"),
        names_to="week",
        names_prefix="week_",
        values_to="reading",
        values_drop_na=True,
    )
)

wide = long >> pivot_wider(
    names_from="week",
    values_from="reading",
    names_prefix="week_",
    values_fill=0,
)

# Multiple measures use tidyr's value-first names (sales_Q1, cost_Q1).
wide_measures = tidy(long_measures) >> pivot_wider(
    names_from="quarter", values_from=["sales", "cost"]
)

# .value takes output value-column names from the input column names.
tidy(visits) >> pivot_longer(
    starts_with(("mean_", "sd_")),
    names_to=[".value", "visit"],
    names_sep="_",
)

parts = tidy(labels) >> separate("code", ["region", "id"], sep="-")
parts >> unite("code", "region", "id", sep="-")
tidy(events) >> unnest_longer("items", indices_to="item_index")
tidy(values) >> replace_na({"score": 0}) >> drop_na("required_field")

tidy(observations) >> complete(
    "subject", "visit", fill={"score": 0}, explicit=False
)

nested = tidy(events) >> nest("records", cols=["time", "value"])
restored = nested >> unnest("records")
```

Build grids and repeat rows like tidyr:

```python
expand_grid(store=["A", "B"], week=[1, 2, 3])          # every combination, as given
crossing(store=["B", "A", "B"], week=[2, 1])           # distinct values, sorted
tidy({"item": ["pen", "cup"], "n": [2, 1]}) >> uncount("n", id="copy")

# Fill in missing years: full_seq() spans min to max in steps of 5
tidy({"country": ["A", "A", "B"], "year": [1952, 1962, 1957], "pop": [1.0, 2.0, 3.0]}) >> complete(
    "country", year=full_seq("year", 5)
)
```

Split text by pattern or position, and move between lists and rows:

```python
codes = tidy({"id": [1, 2], "code": ["ab-12", "cd-3"], "date": ["20240115", "20231201"]})
codes >> separate_wider_regex("code", [("letters", "[a-z]+"), "-", ("number", "[0-9]+")])
codes >> separate_wider_position("date", [("year", 4), ("month", 2), ("day", 2)])
codes >> separate_longer_position("code", 2)            # "ab", "-1", "2", ...

chopped = tidy({"g": ["a", "a", "b"], "x": [1, 2, 3]}) >> chop("x")
chopped >> unchop("x")                                   # back to one row per value
```

In `separate_wider_regex()`, a bare string is matched but not kept; in
`separate_wider_position()`, a bare number is a width that is skipped.
`unnest_auto()` picks `unnest_wider()` for named elements and
`unnest_longer()` otherwise.

`expand()` and `complete()` take named values like this too. The values keep
the column's type, so a whole-number `year` stays whole (R's `full_seq()`
would make it a decimal).

`fill(..., by=...)` provides temporary grouping, while an existing
`group_by()` is respected automatically. Most Polars operations add only lazy
plan nodes. `pivot_wider()` must know its output schema, so it discovers the
distinct `names_from` values unless `names=[...]` is supplied; providing
`names` avoids that metadata query for known categories. `unnest_wider()`
similarly discovers the width of unnamed list values.

`pivot_wider()` accepts multiple `names_from` and `values_from` columns, and
tidyr's naming and shaping options:

```python
scores = tidy({"id": [1, 1, 2, 2], "term": ["fall", "spring", "fall", "spring"],
               "math": [80, 85, 70, 75], "art": [90, 88, 60, 65]})
scores >> pivot_wider(names_from="term", values_from=["math", "art"],
                      names_glue="{term}_{.value}", names_vary="slowest")
# id, fall_math, fall_art, spring_math, spring_art

tidy({"id": [1, 2], "week_1": [5, 6], "week_2": [7, 8]}) >> pivot_longer(
    ["week_1", "week_2"], names_to="week", names_prefix="week_",
    names_transform={"week": int},                 # "1" -> 1
)
```

For full control, build the pivot's mapping as a table (a "spec"), edit it,
and pivot with it:

```python
long = tidy({"id": [1, 1, 2, 2], "key": ["a", "b", "a", "b"], "v": [1, 2, 3, 4]})
spec = build_wider_spec(long, names_from="key", values_from="v").collect(as_="pandas")
spec[".name"] = ["first", "second"]          # rename the output columns
long >> pivot_wider_spec(spec, id_cols="id")  # id, first, second
```

`build_longer_spec()` / `pivot_longer_spec()` do the same for lengthening.

`names_sep` joins name parts (default `_`), `names_expand=True` / `id_expand=True`
add a column / row for every possible value (unused factor levels too), and
`unused_fn` summarises columns that are not ids, names, or values, e.g.
`unused_fn={"note": lambda s: "; ".join(s)}`.
As in tidyr, values that the id and name columns do not uniquely identify
become list columns with a warning; pass `values_fn="list"` to ask for lists,
or `values_fn="mean"` (`sum`, `first`, …) to summarise duplicates.
`pivot_longer()` supports the `.value` sentinel, and `separate(convert=True)`
performs R-style logical/numeric inference on both backends. Schema-dependent
reshape and conversion operations issue a metadata-only query on Polars.

Use `across` as a positional argument to `mutate`, `transmute`, or
`summarise`. Python name templates accept `{col}` and `{fn}`; dplyr-style
`{.col}` and `{.fn}` are accepted too. Use `cur_column()` inside a function
when its calculation depends on the selected column name, and
`cur_group()["group_name"]` when it depends on a grouped key:

```python
df = tidy({
    "id": [1, 2, 3, 4],
    "team": ["red", "red", "blue", "blue"],
    "region": ["north", "north", "south", "south"],
    "year": [1999, 2004, 2011, 2016],
    "target": [2.0, 2.0, 1.0, 1.0],
    "x": [1.234, 2.5, -0.75, 3.0],
    "x_adj": [1.1, 2.4, -0.7, 2.9],
    "math_score": [80, -5, 92, 70],
    "art_score": [-1, -3, 88, 75],
})

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

tidy(df) >> mutate(
    across(starts_with("x"), lambda x: x + len(cur_column()))
)

tidy(df) >> group_by("team", "target") >> mutate(
    across(starts_with("x"), lambda x: x - cur_group()["target"])
)

tidy(df) >> group_by("team") >> mutate(
    across(starts_with("x"), lambda x: x + cur_group_id())
)
```

Functions passed to `across` can return a mapping of named expressions. Pass
`unpack=True` to expand those fields into regular columns; use an
`{outer}`/`{inner}` template to control the resulting names:

```python
tidy(df) >> mutate(
    across(
        starts_with("x"),
        lambda x: {"double": x * 2, "plus_one": x + 1},
        unpack="{outer}__{inner}",
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
    total=sum(c_across(ends_with("_score"))),
    average=mean(c_across(ends_with("_score"))),
)
>> ungroup()
```

`pick` represents a selected set as one structured column and supports common
horizontal reductions without requiring `rowwise`:

```python
tidy(df) >> mutate(total=pick(ends_with("_score")).sum())
tidy(df) >> mutate(scores=pick(ends_with("_score")))
```

`reframe` accepts vector-valued expressions, recycles scalar results within
each group, and always returns an ungrouped frame:

```python
tidy(df)
>> group_by("team")
>> reframe(score=col("math_score"), team_mean=mean("math_score"))
```

### dplyr-compatible evaluation controls

Assignments in one `mutate()` are evaluated from left to right, so later
expressions can use columns created earlier. Independent assignments are
still fused into one backend operation:

```python
tidy(df) >> mutate(
    doubled=col("x") * 2,
    squared=col("doubled") ** 2,
    keep="used",                  # all, used, unused, or none
    before="x",
)
```

`distinct("id")` returns only the grouping columns and `id`, matching
dplyr. Use `distinct("id", keep_all=True)` to retain the first complete row.
Multi-level summaries default to dropping only the final group; override this
with `groups="drop"`, `"drop_last"`, `"keep"`, or `"rowwise"`.

Computed and additive groups and partial ungrouping are supported:

```python
tidy(df)
>> group_by("region")
>> group_by(add=True, decade=col("year") // 10)
>> ungroup("decade")
```

Ask about the groups with dplyr's group functions, called or piped:

```python
by_team = tidy(df) >> group_by("team")

group_keys(by_team)        # one row per group, sorted by key
group_size(by_team)        # [2, 2]
group_rows(by_team)        # [[3, 4], [1, 2]]: 1-based row positions
by_team >> group_indices() # each row's 1-based group number
by_team >> group_walk(lambda part, key: print(key["team"], part.collect().height))

tidy(df) >> nest_by("team")  # one row per team, other columns nested in `data`
```

Groups come in dplyr's order, sorted by key with missing last, in these
functions and in `group_split()`, `group_map()`, `group_modify()`, and
`group_nest()`. Row positions are 1-based, as in dplyr and `slice()`;
subtract 1 to index a pandas frame or NumPy array.

Aggregates accept `na_rm=` and default to `False`, matching dplyr: a missing
value propagates unless removal is requested explicitly. Use
`mean("x", na_rm=True)` to ignore missing values.

### Row mutation and advanced joins

The SQL-inspired row verbs use `y`'s first column as the key by default, or
accept explicit `by=` keys. `y` may contain any subset of `x`'s columns:

```python
accounts = {"id": [1, 2, 3], "owner": ["Ana", "Ben", None], "balance": [100.0, None, 50.0]}
new_accounts = {"id": [3, 4], "owner": ["Cy", "Dee"], "balance": [0.0, 75.0]}
corrections = {"id": [2, 3], "owner": ["Bob", "Cy"], "balance": [20.0, 999.0]}
latest = {"id": [1, 5], "owner": ["Ana", "Eve"], "balance": [120.0, 10.0]}
retired = {"id": [4]}

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
from datetime import date

sales = tidy({
    "id": [1, 1, 2, 2, 3],
    "region": ["north", "north", "south", "south", "south"],
    "year": [2024, 2025, 2024, 2025, 2025],
    "sale_date": [date(2025, 1, 10), date(2025, 3, 2), date(2025, 1, 5),
                  date(2025, 2, 20), date(2025, 2, 1)],
    "amount": [120.0, 80.0, 45.0, 60.0, 30.0],
})
promos = tidy({
    "id": [1, 1, 2],
    "promo_date": [date(2025, 1, 1), date(2025, 2, 1), date(2025, 2, 15)],
    "discount": [0.10, 0.15, 0.05],
})
points = tidy({"point": [1, 5, 12]})
ranges = tidy({"lower": [0, 10], "upper": [6, 20], "band": ["low", "high"]})
segments = tidy({"lo": [0, 8], "hi": [4, 12]})
regions = tidy({"start": [3, 10], "end": [9, 15], "name": ["A", "B"]})

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

dplyr 1.2's recoding helpers map values with `(values, replacement)` pairs,
where `None` matches missing values, or with a `from_=`/`to=` lookup table:

```python
tidy(events) >> mutate(
    team_name=recode_values("team", ("red", "Red Rockets"), ("blue", "Blue Jays")),
    score=replace_values("score", (None, 0)),
    capped=replace_when("value", (col("value") > 3.5, 3.5)),
    flagged=when_any(col("score") >= 90, col("value") < 3),
)
```

`recode_values()` builds a new column (unmatched values become `default`, or
an error with `unmatched="error"`); `replace_values()` and `replace_when()`
change some values and keep the rest. `when_any()` and `when_all()` combine
conditions with `|` and `&`; `na_rm=True` ignores missing values. They
supersede `case_match()` and `recode()`, which still work.

Ranking helpers are `row_number`, `min_rank`, `dense_rank`, `percent_rank`,
`cume_dist`, and `ntile`. Window and value helpers include `lead`, `lag`,
`cummean`, `cumall`, `cumany`, `n_distinct`, `coalesce`, `if_else`, and
`case_when`. `lead`, `lag`, `first`, `last`, and `nth` accept `order_by=`.
For any other window expression, `order_by()` computes it as if the rows were
sorted, then returns results in the original row order:

```python
tidy(sales) >> mutate(
    running=order_by("sale_date", col("amount").cum_sum()),
    previous=with_order("sale_date", lag, "amount"),
    by="region",
)
```

Columns found in both tables get dplyr's suffixes in Python spelling:
`_x` and `_y` rather than `.x` and `.y` (as `na.rm` is `na_rm`), so
`v_x` works as a bare name in notebooks. Pass a pair to choose others:

```python
before = tidy({"id": [1, 2], "price": [10.0, 12.0]})
after = tidy({"id": [1, 2], "price": [11.0, 15.0]})

before >> left_join(after, by="id")                             # id, price_x, price_y
before >> left_join(after, by="id", suffix=("_old", "_new"))    # id, price_old, price_new
```

Mutating joins accept dplyr-style safety controls:

```python
orders = tidy({"order_id": [1, 2, 3], "customer_id": [10, 11, 10], "total": [25.0, 40.0, 15.0]})
customers = tidy({"customer_id": [10, 11], "name": ["Ana", "Ben"]})

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

Every frame-returning Phase 1–8 verb stays lazy on the Polars backend. `pull`,
`setequal`, plotting, previews, and explicit `collect` are intentional
materialization boundaries. Schema-dependent reshape operations may run a
small metadata query as described above. `slice_sample(weight_by=...)`,
including `replace=True`, is supported on both backends. Group-context helpers
and additional vector helpers remain future phases.

### Performance controls

Grouped `mutate()` expressions automatically reuse embedded or repeated
statistics. Expressions such as the following compute each group mean and
standard deviation once on both backends; temporary columns never appear in
the result:

```python
features = {"segment": ["a", "a", "a", "b", "b"], "x": [1.0, None, 3.0, 10.0, 14.0]}

tidy(features) >> mutate(
    filled=coalesce(col("x"), mean("x", na_rm=True)),
    z=(
        coalesce(col("x"), mean("x", na_rm=True))
        - mean("x", na_rm=True)
    ) / std("x", na_rm=True),
    by="segment",
)
```

Project before a pandas/Arrow handoff without adding another pipeline verb:

```python
# A model-ready table: one row per customer
result = tidy({
    "customer_id": [101, 102, 103],
    "segment": ["a", "b", "a"],
    "feature_a": [0.1, 0.4, 0.3],
    "feature_b": [1.0, 0.0, 1.0],
    "feature_c": [12.5, 9.0, 11.0],
})

matrix = result.collect(
    as_="pandas",
    columns=["customer_id", starts_with("feature_")],
    arrow_backed=True,
)
```

`arrow_backed=True` still returns a pandas DataFrame, but avoids copying
Polars columns into NumPy buffers. Leave it off when a downstream library
specifically requires NumPy-backed pandas dtypes.

`distinct(..., maintain_order=True)` preserves dplyr row order by default.
Use `maintain_order=False` when output order is irrelevant and throughput is
more important.

### NumPy, Numba, and PyTorch

Build the matrix directly from the lazy pipeline and project away identifiers
or string columns before materializing:

```python
import numpy as np
from tidy3 import starts_with, to_numpy

features = result.to_numpy(
    columns=["feature_a", "feature_b", "feature_c"],
    dtype=np.float32,
    order="c",
    writable=True,
)

# Equivalent output boundaries
features = result.collect(as_="numpy", columns=starts_with("feature_"))
features = result >> to_numpy(columns=starts_with("feature_"))
features = np.asarray(result)  # all columns; materializes the lazy plan
```

Numba functions accept the returned array directly. For PyTorch,
`torch.from_numpy(features)` shares the CPU array's memory, so request
`writable=True`; use `order="c"` when a consumer requires C-contiguous input.
The default is Fortran order because tidy3/Polars are columnar and it offers
the best chance of avoiding a copy. Set `allow_copy=False` when an unexpected
copy should be an error.

This is a host-memory bridge even when `engine="gpu"`: Polars may execute the
query on the GPU, but the NumPy result resides in CPU memory. A future DLPack
bridge would be needed for a direct device-to-PyTorch handoff.

## Notes

- **Grouped semantics are dplyr's**: after `group_by`, `mutate`/`filter`/
  `slice_*`/`sample_n`/`head` evaluate **per group** (Polars window `.over`).
  `summarise` aggregates per group.
- **Display is self-contained**: tables carry inline styles (no `<style>`
  block), so they render identically in local cells, republished `%gpu`
  output, and sslive exports. Under `%gpu`, bare polars DataFrames are
  restyled the same way (`seed_tidy3_remote(style_polars=False)` to opt out).
- **Builtins are shadowed** by design: injecting the API puts `filter`, `slice`,
  `sum`, `min`, `max`, `any`, and `all` into the notebook namespace. Use
  `builtins.filter` etc. when you need the Python originals.
- **Plotting big remote data**: aggregate remotely, then let plot3's own
  remote path pull the small result: `%plot3 res.to_pandas() x=... y=...`.
- **API maturity tiers** (stable / growing / experimental) are listed under
  [API maturity](#api-maturity). Prefer the stable core for adoption and
  performance-sensitive work.
