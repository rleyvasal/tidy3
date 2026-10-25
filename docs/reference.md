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

Write a pipeline directly without first calling `collect()`. Each writer
works as a pipe step, as a function, and as a method:

```python
cars >> filter(col("cyl") == 4) >> write_csv("four_cyl.csv")   # in a pipe
write_parquet(result, "summary.parquet")  # frame first, as in readr
result.write_ipc("summary.arrow")         # Arrow IPC / Feather
```

In a pipe the writer passes the frame on, so the pipeline can keep going:
`cars >> write_csv("all.csv") >> filter(col("cyl") == 4)`. A step after
the writer runs the plan again; call `compute()` before writing to run it
once. `write_csv(df, path)` returns `df`, and it also takes Polars and
pandas frames.

With the default Polars backend, these execute the lazy plan and stream its
result to disk, avoiding a second fully materialized DataFrame in memory.
GPU writers execute the
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

```python
result.write_excel(
    "summary.xlsx",
    worksheet="Summary",
    autofit=True,
    freeze_panes="A2",
)

# several sheets; this frame is not added on top of `sheets`
result.write_excel("report.xlsx", sheets={"Summary": result, "Cars": cars})

# the same as pipe steps and functions
result >> write_excel("summary.xlsx", worksheet="Summary")
write_excel("report.xlsx", sheets={"Summary": result, "Cars": cars})
```

A cell whose text starts with `=` is written as text. `read_excel` and
`excel_sheets` need the optional reader: `pip install 'tidy3[excel]'`.
Writing a workbook uses XlsxWriter, which is already installed with tidy3.

You can still collect first when another library needs the result:

```python
result.collect().write_csv("summary.csv")          # Polars API
result.collect(as_="pandas").to_csv("summary.csv", index=False)
```

## API

| Area | Symbols |
|------|---------|
| Export | `nb_export`, `transform_source`, `python -m tidy3 export` / `run` |
| Frame | `tidy`, `scan_parquet`, `scan_csv`, `scan_ipc`, `read_csv`, `read_tsv`, `read_delim`, `read_excel`, `excel_sheets`, `TidyFrame` |
| Output | `collect`, `to_numpy`, `write_parquet`, `write_csv`, `write_ipc`, `write_excel` (pipe steps, functions, and `TidyFrame` methods) |
| Helpers | `is_na`, `is_nan`, `is_finite`, `is_infinite`, `abs`, `floor`, `ceiling`, `round`, `pmin`, `pmax`, `row_sums`, `row_means` |
| Convert | `as_numeric`/`as_double`, `as_integer`, `as_character`/`as_string`, `as_logical`, `as_date`, `as_datetime`, `ConversionError` |
| Parse | `parse_double`, `parse_integer`, `parse_logical`, `parse_date`, `parse_datetime`, `parse_time`, `parse_number`, `locale`, `problems`, `stop_for_problems` |
| Dates | `ymd`, `mdy`, `dmy`, `ym`, `ymd_hms`, `ymd_hm`, `ymd_h`, `mdy_hms`, `mdy_hm`, `mdy_h`, `dmy_hms`, `dmy_hm`, `dmy_h`, `parse_date_time`, `with_tz`, `force_tz`, `year`, `month`, `day`, `wday`, `yday`, `week`, `isoweek`, `isoyear`, `quarter`, `hour`, `minute`, `second`, `floor_date`, `ceiling_date`, `round_date`, `days_in_month`, `leap_year`, `make_date`, `make_datetime`, `months`, `years`, `add_with_rollback`, `today`, `now`, `excel_numeric_to_date` |
| Factors | `as_factor`, `fct_relevel`, `fct_reorder`, `fct_reorder2` (`last2`, `first2`), `fct_infreq`, `fct_inorder`, `fct_rev`, `fct_recode`, `fct_collapse`, `fct_lump_n`, `fct_lump_prop`, `fct_lump_min`, `fct_drop`, `fct_expand`, `fct_na_value_to_level`, `fct_na_level_to_value` |
| Cleaning | `clean_names`, `make_clean_names`, `set_names`, `get_dupes`, `remove_empty`, `remove_constant`, `row_to_names`, `find_header`, `compare_df_cols`, `compare_df_cols_same`, `describe_class` |
| Reporting | `tabyl` (one-variable frequencies and two-variable count tables), `adorn_totals`, `adorn_percentages` |
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
| **Growing** | Feature-complete enough for real work; more edge cases and schema-discovery cost | tidyr missing/reshape (`drop_na`, `replace_na`, `fill`, `complete`, `expand`, `pivot_*`, `separate`, `unite`, `nest`/`unnest*`), `across`/`if_any`/`if_all`/`pick`/`c_across`, `rowwise`, `reframe`, `rename_with`, `join_by` inequality joins, `nest_join`, `rows_*`, advanced ranking/window helpers, NumPy/`to_numpy` handoff, column helpers (`is_na`, `round`, `pmin`, `row_sums`), `as_*` conversion, lubridate-style dates, readr-style `parse_*` / `problems`, `read_csv` |
| **Experimental** | Correctness-first or Python-callback paths; may materialize eagerly or lag R oracle coverage | `group_split`, `group_map`, `group_modify`, `group_nest`, `with_groups`, `hoist`, `pack`, `unpack`, `separate_longer_delim`, `separate_wider_delim`, stochastic sampling (`sample_n`, `sample_frac`, `slice_sample`) |

**Performance note:** the stable core is the path that tracks raw Polars and
beats pandas on realistic sizes. `group_nest` compiles to a Polars
`group_by().agg(struct)` plan and is typically **faster** than building
nested pandas DataFrames by hand; for counts alone use `count()`/`tally()`
instead of nesting. Python-callback verbs (`group_map` / `group_modify` /
`group_split`) still materialize and loop in Python — fine for small groups,
not for hot large-data paths. Prefer declarative `summarise` when
performance matters.

## pandas input and output

tidy3 runs on Polars. A pandas DataFrame is accepted wherever a frame is, and
becomes a lazy Polars frame that keeps its factor levels and ordered flags.
`collect(as_="pandas")` hands a result back to pandas, restoring those levels:

```python
import pandas as pd

df = pd.DataFrame({"x": [-1.5, 0.5, 2.0], "y": [10, 20, 30]})

(tidy(df) >> filter(col("x") > 0)).collect(as_="pandas")
```

## Export to Polars code

`to_polars_code()` prints a pipeline as a Python function that needs only
Polars. Each input frame becomes an argument (`df`, then `df2`, ...), and the
docstring lists its columns:

```python
frame = (
    tidy({"g": ["a", "b", "a"], "x": [1.0, 2.0, None]})
    >> filter(col("x") > 0)
    >> group_by("g")
    >> summarise(m=mean("x"), n=n())
)
print(frame.to_polars_code(name="summarise_x"))
```

The function runs the query tidy3 compiled, so it returns the same result,
dplyr's rules included: a `mean` with a missing value is missing, and
`arrange` puts missing values last. Two kinds of step raise `ExportError`
instead, naming the step:

- a column computed by a Python function: the text conversions (`as_*`,
  `parse_*`, `ymd` and the other date parsers) with their `problems()`
  tracking, R-style `round`, `force_tz`, `excel_numeric_to_date`, and the
  string functions that run per value (`str_wrap`, `str_width`,
  `str_match_all`). The error names the column and the function;
- a step that built its query from values in the data, such as factor
  levels, `pivot_wider` names, `complete(full_seq(...))`, `remove_empty`, or
  `compute()`. That query is right only for the data it saw. Joining a frame
  with such a step blocks the export too.

Helpers whose rule depends on the column type (`is_na` and the other `is_*`
checks, `pmin`/`pmax`, `row_sums`/`row_means`, `hour`/`minute`/`second`,
`ceiling_date`, `with_tz`) export as plain Polars: tidy3 reads the type from
the lazy schema when it compiles them.

The exported code skips tidy3's input checks (a `separate_wider_delim` with
too many pieces, a weight column with negative values, a `rows_update` key
missing from x), so it returns the same result on valid input but does not
raise on invalid input.

A quoted string assigned in `mutate`, `transmute`, `summarise`, or `reframe`
is text, as in dplyr: `mtcars >> mutate(source="mtcars")`
writes `"mtcars"` in every row. A column is `col("hp")` in plain Python and
the bare name `hp` in a notebook pipe, which `nb_export` writes as
`col("hp")`. Strings that name columns elsewhere (`select("hp")`,
`arrange("hp")`, `mean("hp")`) are unchanged.

### Duplicate records

`get_dupes` follows janitor's duplicate inspection helper. It returns every
record sharing the selected key combination, including the first occurrence,
and adds `dupe_count` with the number of records in that combination.

```python
observations = tidy({"id": [1, 2, 1], "mass": [3500, 3800, 3550]})
observations >> get_dupes("id")  # both id=1 records, each with dupe_count=2
observations >> get_dupes()      # compare all columns
```

Accepts multiple names, lists, and tidyselect helpers such as
`get_dupes(starts_with("key_"))`. `observations.get_dupes("id")` also works.
Notebook pipes accept bare names, for example `get_dupes(id)`.
Selected keys appear first, followed by `dupe_count`, then the remaining
original columns. Results sort by decreasing count and ascending keys;
factor keys use their recorded level order, missing keys sort last, and
rows tied on these keys retain their input order. Missing keys match each
other. No matches returns an empty table with the output columns intact.
An explicit selector that selects no keys treats the whole table as one
key combination; omitting arguments instead checks all columns.

Duplicate detection spans the whole table even after `group_by`. Original
grouping, group-drop settings, factor metadata, and parsing diagnostics are
preserved. Polars remains lazy. The helper does not print janitor's status
messages. Unlike janitor, an existing `dupe_count` column raises an error;
rename it first to keep its values.

### Frequency tables

`tabyl("column")` tabulates **one variable**, returning a `TidyFrame` with
the selected column, `n`, and `percent` (a numeric proportion from 0 to 1).
When missing values are present, it also returns `valid_percent`, calculated
from non-missing rows; the missing row's valid percentage is missing.

```python
tidy({"answer": ["yes", "no", "yes", None]}) >> tabyl("answer")
```

`tabyl("row", "column")` makes a **two-variable count table**: values of
the second variable become column names, the first variable labels rows,
and empty combinations have numeric zero counts. Factor order determines
row and column order; ordinary numeric values sort numerically.

```python
tidy({"group": ["A", "A", "B"], "answer": ["yes", "no", "yes"]}) >> tabyl("group", "answer")
```

Missing column values use the name `NA_`, NaN uses `NaN`, and blank strings
use `emptystring_`. The blank and missing columns come last. Numeric and
date column names follow tidy3's header text rules: for example, `1e6`
becomes `"1000000"` rather than R's `"1e+06"`. If column values produce
duplicate labels, or a label matches the row column's name, rename or
recode those values first. An empty input (including one emptied by
`show_na=False`) returns zero rows and only the row column, as janitor does.

Use `show_na=False` to omit null and NaN rows and recalculate percentages.
For two variables, it omits records missing either selected value.
Unused factor levels have zero counts by default; `show_missing_levels=False`
omits those rows. Factor order and the ordered flag survive. Null and NaN
are separate rows, and both are excluded from valid percentages.
Existing grouping is ignored and the output is ungrouped.

In notebooks, `answers >> tabyl(answer)` and `answers >> tabyl(group, answer)`
accept bare column names.
A vector can be passed directly: `tabyl(["yes", "no", "yes"])`; its column
name is `"x"`, or the name of a supplied Series. Rename an input column named
`n`, `percent`, or `valid_percent` before one-variable tabulation to avoid duplicate
output names. The count table and existing parse diagnostics are computed
once; later steps run from that snapshot. Three-variable tables and
percentage/formatting helpers are not implemented yet.

Both forms are checked against [janitor's `tabyl`](https://sfirke.github.io/janitor/reference/tabyl.html).

### Totals

`adorn_totals()` appends a totals row. Use `"col"` for a totals column or
`"both"` (also `("row", "col")`) for both, including the grand total.
By default it sums numeric columns after the first column, treating the
first as a row label even if it is numeric. Existing groups are ignored.

```python
tidy({"group": ["A", "A", "B"], "answer": ["yes", "no", "yes"]}) >> tabyl("group", "answer") >> adorn_totals("both")
```

`cols=` accepts names, `col(...)`, or tidyselect helpers. It can explicitly
include the first column; in that case its total is numeric, with no text
label. Non-numeric selections are ignored. For one-variable frequency
tables, use `adorn_totals(cols="n", fill=None)` to total counts while
leaving the percentage columns numeric and their total cells missing.

Missing values, including NaN, are omitted by default; `na_rm=False`
propagates them. `name="All"` sets both labels, or `name=("All", "Grand")`
sets the row and column labels separately. Unselected columns are converted
to text and filled with `fill="-"` in the total row. `fill=None` preserves
their types and inserts missing values; the first column still becomes a
text label. A factor row label keeps its level order and ordered flag, with
the new label appended. Use a fresh label if it is already a factor level.
Column totals are floating-point numbers, matching R's row sums.

Totals stay lazy and numeric, and parse diagnostics are retained. Adding
the same totals dimension twice raises an error; adding row and column
totals in separate consecutive steps works. `compute()` preserves that
record. Ordinary data-changing verbs clear it, so keep reporting steps
together at the end of the pipe. An existing column with the requested
totals name raises an error instead of overwriting its values.

Behavior is checked against [janitor's `adorn_totals`](https://sfirke.github.io/janitor/reference/adorn_totals.html).

### Percentages

`adorn_percentages()` divides each count by its row's sum. Use `"col"` for
column sums or `"all"` for the grand total. Results are numeric proportions
(0.25, not "25%"), so they can still be sorted, plotted, or modeled.

```python
tidy({"group": ["A", "A", "B"], "answer": ["yes", "no", "yes"]}) >> tabyl("group", "answer") >> adorn_percentages("col")
```

As with `adorn_totals`, numeric columns after the first are divided by
default, and `cols=` selects others, including the first column; text and
factor columns in a selection are skipped. Missing values are skipped in
row and column sums; `na_rm=False` makes a sum with a missing value
missing. The grand total always skips missing values, as janitor does. A
zero sum gives NaN (0 / 0). With `na_rm=False`, a row holding both a missing
value and NaN sums to missing; R may give either NA or NaN there.

Totals added earlier are not counted twice. A totals row is left out of
column and grand sums and is divided like any other row. A totals column
becomes 1 when dividing by row, and a share of its own sum otherwise.
Selecting columns with `cols=` that leave the totals column out keeps it as
counts. Add totals first and percentages second, as in janitor:
`adorn_totals("both") >> adorn_percentages("all")`. The totals record
survives, so adding the same totals again still raises an error.

Percentages stay lazy, drop existing groups, and keep parse diagnostics.
Divisions are exact, as in R: 3 / 5 is 0.6.
The table needs a numeric column after the first column. Behavior is checked
against [janitor's `adorn_percentages`](https://sfirke.github.io/janitor/reference/adorn_percentages.html).

### Empty columns, headers, and class checks

`remove_empty` drops rows and columns that are entirely missing. `""` is a
value, so convert blanks to missing first if those should go too.
`cutoff=1` (the default) keeps a row or column with any non-missing value.
A smaller cutoff keeps it only when the share of non-missing values is
strictly greater than `cutoff`, and that cutoff cannot be combined with
both rows and columns. Rows are removed before columns. A zero-row frame
has no non-missing values, so `which="cols"` drops every column.
`remove_empty("rows")` on Polars is a lazy filter. Choosing which columns
to drop counts missing values first, so the surviving columns depend on
the data.

`remove_constant` drops a column with fewer than two distinct values.
Missing counts as its own value unless `na_rm=True`. A column that is
entirely missing is removed either way. On Polars, null and NaN are
different, so a column of only those two is kept when `na_rm=False`.
Pandas stores both as NaN in a float column, so that column is removed.
Unused factor levels do not count. The check reads the current values and
is global, not per group.

```python
raw = tidy({"a": ["id", "a", None], "b": [None, None, None], "c": ["n", "", "x"]})
raw >> row_to_names(1) >> remove_empty()
```

`row_to_names(row_number)` uses that 1-based row as the column names.
`row_number="find_header"` calls `find_header`. With no value, the header
is the first row with no missing cells. `find_header(data, value="Total", column="name")`
searches one column; `column` is a name or a 1-based position. The name
row and the rows above it are removed by default. A missing header cell
becomes the string `"NA"`, including NaN. janitor leaves a missing name as
`NA` (and `as.character(NaN)` is `"NaN"`); a string name can be stored.
Each cell is converted on its own: `1.0` becomes `"1"` and
`TRUE` becomes `"TRUE"`. janitor 2.2.1 instead coerces the whole header
row with `unlist` before converting, so a row of `1` and `TRUE` becomes
two `"1"` names there. Duplicate names are made unique (`id`, `id_2`)
because a frame cannot repeat column names; janitor only warns. Reading
the name row runs the plan up to this verb. Grouping and factor levels
follow the new names. A group column that is removed is dropped from the
grouping; if none remain, the frame is ungrouped.

`compare_df_cols(old=old, new=new)` returns `column_name` plus one class
column per frame. Pass frames by keyword, in a mapping, or in a list
(`pair=[old, new]` becomes `pair_1`, `pair_2`). Positional frames are
named `..1`, `..2`. No input may be named `column_name`.
`return_="match"` or `"mismatch"` filters the rows. With
`bind_method="bind_rows"` (the default), a column missing from some frames
still matches when every present class agrees. `"rbind"` requires every
frame to have the column and the same class. Two or more frames sort
column names by code point; one frame keeps its column order. R's `merge`
sorts in the session locale, so this comparison is locked for ASCII names.
`strict_description=True` includes factor levels. `compare_df_cols_same`
returns true when there is no mismatch and does not print the table.
`verbose` is accepted and ignored. One frame is not the same as another
frame, so a single frame with columns returns `False`, as janitor does.
janitor 2.2.1 errors on `compare_df_cols_same` of zero-column frames;
tidy3 returns `True`.

Class names follow R: `integer`, `numeric`, `logical`, `character`,
`Date`, and `POSIXct, POSIXt`. A Python `int` is `integer`; pass a float
for `numeric`. Descriptions use dtypes and recorded factor levels. A
Polars Enum that has no recorded levels is treated as ordered. A pandas
object column may be read far enough to tell a date from text. A frame
with no columns is omitted.

These helpers do not print janitor's removal messages. `cutoff` other
than 1 needs a positive number of rows when dropping columns, and a
positive number of columns when dropping rows; a zero denominator raises
rather than following R's `NA` mask. A Polars frame that loses every
column also loses its rows, because Polars cannot store a row count
without a column. Pandas keeps that row count, as janitor does.

`excel_numeric_to_date(42370)` is 2016-01-01. Serial 60 is missing on
both date systems: Excel treated 1900 as a leap year, and janitor applies
that adjustment before choosing the origin, including for
`date_system="mac pre-2011"`. Mac serial 1 is therefore 1904-01-03, and
Mac serial 0 is 1904-01-02. `include_time=False` ignores the fraction.
`round_seconds=True` rounds seconds the way R's `round` does. The default
zone is UTC, not the machine zone. A quoted string is a column name inside
`mutate`. A number or list is converted immediately. Leap seconds are not
applied, and values below 1 are converted without a warning.

### Column helpers, dates, and readers

These work inside `mutate`, `filter`, `summarise`, and `across` on both
backends. `from tidy3 import *` binds `abs`, `round`, `sum`, `min`, `max`,
`any`, and `all` to tidy3's versions, which shadows the Python built-ins of
those names. Call `builtins.abs` or `builtins.round` when you want the
built-in. The aggregations already did this for `sum`, `min`, `max`, `any`,
and `all`.

```python
cars >> filter(is_na(col("mpg")))
cars >> mutate(hp_capped=pmin(col("hp"), 200))
cars >> mutate(total=row_sums("mpg", "hp", "wt", na_rm=True, min_count=1))
```

`is_na` is true for a Polars null and for NaN, and the result is never
itself missing, so `filter` keeps the row. `is_nan` is true only for NaN.
`is_finite` is true only for a finite number. `is_infinite` is true only
for positive or negative infinity. The string
`"nan"` is text, not a missing number.

`round` follows R's `round()` (R 4.0 and later). An exact tie goes to the
even digit: `round(0.5)` is 0, `round(1.5)` is 2, and `round(0.125, 2)` is
0.12. Otherwise the result is the candidate nearer to the number as
stored, so `round(2.675, 2)` is 2.67 because 2.675 is stored as
2.67499999.... Negative `digits` round to tens (`round(123, digits=-1)` is
120), and integer columns stay integers. `pmin` and `pmax`
are element-wise. With `na_rm=False`, any missing input makes that row
missing. `row_sums(..., na_rm=True)` of an all-missing row is 0;
`min_count=1` makes that row missing. `row_means` of an all-missing row is
missing. `sum(c_across(cols))` with the defaults still skips nulls and
returns 0 for an empty row. Pass `na_rm=True` or `min_count` to `sum` of a
selection to use the `row_sums` rule. `sum(..., min_count=)` on a column
does the same for a group.

`as_double` (also `as_numeric`), `as_integer`, `as_character` (also
`as_string`), `as_logical`, `as_date`, and `as_datetime` share one
converter. `strict=True` raises `ConversionError` at collect time on Polars
and when `mutate` runs on pandas. The message names the function you
called (`as_numeric` or `as_double`, `as_string` or `as_character`), the
column, and up to five bad values.
`strict=False` turns those values into missing and records them for
`problems()`. `as_integer(2.5)` is an error; `2` and `2.0` succeed, and the
strings `"2.0"` and `"2.5"` do not. `as_double` does not strip thousands
separators. `as_logical` accepts exactly `TRUE`, `true`, `True`, and `T`
(and the `FALSE` forms), as R does: `"tRuE"` and `" TRUE"` fail, and so
does the string `"1"`. Numbers work (`0` is false). `as_date` accepts ISO `YYYY-MM-DD`.
`as_datetime` accepts an ISO date or a time after a space or `T`, treats
`Z` and numeric offsets as UTC, and returns a naive datetime of that UTC
clock time.

`as_character` and `as_string` write each value as its column type shows
it: a float stays a float (`21.0` is
`"21.0"`, a `float32` `22.8` is `"22.8"`), an integer is `"21"`, a date is
`"2010-01-31"`, and a datetime is `"2010-01-31 15:30:00"`, with fractional
seconds only when it has them and an offset (`"-08:00"`) when it has a
time zone. Numbers are never rounded, so the text reads back as exactly
the same number. Logical values are `"true"` and `"false"` by default;
`true=` and `false=` set the spelling, such as `true="TRUE",
false="FALSE"` for R or `true="True", false="False"` for Python's `str()`.
For an id stored as a float, convert to an integer first:
`as_string(as_integer(id))` is `"1001"`, not `"1001.0"`.

`parse_double`, `parse_integer`, `parse_logical`, `parse_date`,
`parse_datetime`, and `parse_number` are permissive unless you pass
`strict=True`. They follow readr, checked against it in the R oracle
tests. `parse_number` reads the first number in the text:
`"$1,234.50"` is 1234.5, `"50%"` is 50, and `"1e3"` is 1000. Grouping
marks are skipped anywhere in the whole part (`"1,23"` is 123), and a
second decimal mark ends the number (`"1.2.3"` is 1.2). A minus counts only
right before the number: `"$-5"` is -5, while `"-$5"` and `"- 5"` are
missing. `parse_double` requires the whole string to be one number, with an
optional exponent; grouping marks, `Inf`, and `NaN` fail (`"1,234.5"` is
missing; use `parse_number`). `parse_integer` fails on `"1,234"`, `"2.5"`,
and `"1e3"`, and is 64-bit, so it reads values past R's 32-bit limit.
`locale(decimal_mark=",", grouping_mark=".")` switches the marks; the two
marks are single characters and must differ.
On Polars, `as_double`, `as_integer`, and `parse_double` on text and
number columns run as native casts; they give the same results as the
per-value rules above.
`parse_logical` accepts `true`/`false`/`t`/`f`/`1`/`0`. `parse_date` and
`parse_datetime` take a `strptime` format. `parse_datetime` returns a naive
datetime. A `%z` offset is converted to UTC and the zone is dropped, as
`as_datetime` does and as readr shows the instant in UTC:
`2024-01-01 12:00:00+0200` is 10:00 and `2024-01-01 12:00:00-0500` is 17:00.
Mixed offsets in one column parse. `ymd_hms` is the timezone-aware parser.

`parse_time` reads times of day as readr does and returns a Polars `Time`
column. The default format reads `"13:05"`, `"1:05:09"`, `"1:05 pm"`, and
`"01:05:09.25"`: one or two hour digits, a colon, two minute digits,
optional seconds (readr's default drops a fraction), and an optional AM/PM
in any case. A format uses readr's codes: `%H` (0-23), `%I` (1-12) with
`%p`, `%M` (two digits), `%S` (drops a fraction), `%OS` (keeps it), `%T`
(the default clock without AM/PM), `%R`, `%AT` (the default), and `%.`,
`%+`, and `%*` to skip one, one or more, or any non-digits. Date and zone
codes (`%Y`, `%m`, `%d`, `%b`, `%B`, `%z`, `%Z`) must match but do not change
the time. A space matches any whitespace, including none. readr keeps
`"24:00"`, `"25:00"`, and `"13:00 PM"` as durations of a day or more; a time
of day cannot hold them, so tidy3 records them as problems. Failures use
the same `problems()` recorder and `strict=True` option as the other parsers.

`problems(frame)` returns a frame with columns `row` (1-based), `column`,
`original`, and `expected`. As in readr, `row` is the position in the
column that was parsed: after `filter`, row 1 is the first row that
remains. For `read_csv`, `row` counts the header line, so the first data
row is row 2; skipped and comment lines do not count. On Polars that frame is a lazy plan captured
when the parse was added. Calling `problems()` does not collect the data.
`compute()`, and so `read_csv`, works out the failures in the same run as
the data, so `problems()` afterwards does not read the source again. `stop_for_problems()`
computes the data and its failures in one run, raises `ConversionError` when
any row failed, and otherwise returns the computed frame.
`ymd`, `mdy`, `dmy`, and `ym` use the same recorder. They default to
permissive. Orders accept `-`, `/`, `.`, spaces, compact digits
(`20100131`), month names (`"Jan 31 2010"`, `"31 January 2010"`), and
ordinals (`"1st March 2010"`). As in lubridate, text with a time such as
`"2010-01-31 10:00"` is missing; use `ymd_hm` for it. `ym` returns the first of the month. A two-digit year fails
unless you pass `cutoff`: years up to and including the cutoff are 20xx,
and the rest are 19xx, so `cutoff=68` reads `68` as 2068 and `69` as 1969.
Four-digit years ignore the cutoff.

`ymd_hms`, `ymd_hm`, `mdy_hms`, and `dmy_hms` parse dates with times on both
backends. They return timezone-aware datetimes in `tz="UTC"` by default;
`tz="America/New_York"` interprets unzoned text as that local clock. A `Z`
or numeric offset (`+05:30`, `-0400`, `-05`) preserves the instant and
converts its display to `tz`. They accept English month names, ordinals,
AM/PM, and compact fields with four-digit years (`20240229123456`, also
as a number). Fractions are truncated to microseconds. `ymd_hm` rejects
seconds rather than discarding them. Two-digit years in separated text
require `cutoff`, and failures use the same `problems` recorder and
`strict=True` option as the date parsers.
Signs (`+` and `-`) after the calendar date are reserved for offsets;
use `T`, spaces, or punctuation such as `:` and `.` to separate time fields.

Missing time fields require `truncated`: for `ymd_hms`, 1 permits a missing
second, 2 permits just an hour, and 3 permits a date alone. For `ymd_hm`,
1 permits just an hour and 2 a date alone. Omitted time fields become zero.
The `mdy_hms` and `dmy_hms` options follow `ymd_hms`. A nonexistent clock
during a spring daylight saving change fails; a repeated autumn clock
chooses its later occurrence. Supply an offset to select either occurrence.
Polars expressions stay lazy, including their diagnostics.

These parsers deliberately avoid two R guessing behaviors: a four-digit
year is never split into a two-digit year and an hour to rescue incomplete
input, and `ymd_hm` always honors offsets (R's helper can drop `Z` or reject
numeric offsets). English month names are fixed rather than locale-dependent;
arbitrary words around a timestamp, timezone abbreviations such as `EST`,
and truncating the calendar date itself are unsupported. Existing
`as_datetime` continues to return naive UTC clocks.

`ymd_h`, `mdy_hm`, `mdy_h`, `dmy_hm`, and `dmy_h` complete the family with
the same rules: `ymd_h("2024-02-29 13")` is 13:00, `"2024-02-29 1 PM"` is
13:00 too, and `"2024-02-29 24"` is midnight on 1 March. Minutes or seconds
the order does not name fail. `truncated=1` permits a date alone for the
`_h` orders; for the `_hm` orders, 1 permits just an hour and 2 a date alone.

`parse_date_time(x, orders)` is lubridate's multi-order parser. An order
names fields: `y`/`Y` year, `m` month (number or English name), `b`/`B`
month name, `d` day, `H` hour, `I` hour 1-12 with `p` AM/PM, `M` minute, `S`
second (with any fraction), `z` an offset (`Z`, `+0530`, `+05:30`, `-05`),
`a`/`A` a weekday name, and `T`, `R`, `r` for an `H:M:S`, `H:M`, or `H`
clock with or without AM/PM. lubridate's `O` forms (`OS`, `Om`, `Op`, `Oz`)
read the same as the plain letters. Separators and `%` in an order are
ignored, so `"ymd HMS"`, `"Ymd_HMS"`, and `"%Y-%m-%d %H:%M:%S"` are one
order. As in lubridate, the text may use any non-digits between fields, or
none (`"20240229123456"`), and words before or after the fields are
ignored (`"Thursday 2024-02-29"`), but a letter right before the first
field is not (`"x2024-02-29"` fails). A final AM/PM may be missing when
other values of the column have one.

With several orders, each value takes the first that reads it in
lubridate's ranking, not in the listed order: formats with more fields
first, then the format that reads more of the column, then the listed
order. A format is an order with the text's own separators, so
`"01/02/2024"` and `"13-01-2024"` count apart. lubridate ranks on a sample
of a column of 100 values or more (rows 1, 2, 3, 5, 7, and so on up the
primes), retrains on the values that sample did not read, and stops when a
round reads nothing; tidy3 does the same. The result is a datetime in `tz`
(UTC by default). Text with an offset keeps its instant; other text is a
clock in `tz`, with the daylight saving rules of `ymd_hms`. `truncated=2`
also tries each order without its last one or two fields. `exact=True`
takes the orders as `strptime` formats and uses the first that reads each
value.

```python
eggs = tidy({
    "nest": ["N1A1", "N1A2", "N2A1", "N2A2"],
    "laid": ["2007-11-11", "11/16/2007 2:05 PM", "16 Nov 2007", "bad"],
    "check": ["13:05", "1:05 pm", "07:30:15", "25:00"],
})
logged = eggs >> mutate(
    laid=parse_date_time("laid", ["ymd", "mdy IMp", "dbY"]),
    check=parse_time("check"),
)
logged
```

```python
problems(logged)
```

Differences from lubridate: two-digit years need `cutoff` (lubridate
reads them with a cutoff of 68, so pass `cutoff=68` for its result), and
orders must include a year (lubridate returns year 0 for an order such as
`"HMS"`; use `parse_time`). Seconds from 61 to 69, which lubridate rolls into
the next minute, fail as they do in `ymd_hms`. Negative `truncated`,
`select_formats`, `train=False`, `drop`, `locale`, and the order letters `j`,
`q`, `U`, `u`, `W`, and `w` are not supported; an unsupported letter raises
when the expression is built. `exact=True` formats use Python's `strptime`
codes, as `parse_date` does: write `%S.%f` where lubridate takes `%OS`. lubridate
ranks each guessed format by small bonuses for four-digit years and month
names; tidy3 counts fields only, which changes the choice only between
formats with equal fields and equal counts.

`with_tz(x, tzone="UTC")` displays the same instant in another zone.
`force_tz(x, tzone="UTC")` keeps the local clock fields and interprets them
in a new zone, changing the instant. For example, noon in New York in
January becomes 17:00 UTC with `with_tz`, or 12:00 UTC with `force_tz`.
Both helpers accept date/datetime columns and expressions; parse text first.
Naive datetimes use UTC for `with_tz`; dates begin at midnight UTC.
`force_tz` interprets naive clocks and date midnights directly in `tzone`.
Both preserve missing values and existing datetime precision, including
nanoseconds. Polars plans remain lazy; discovering the target dtype does
not read source rows. Timezone names must be valid; unknown or empty names
raise instead of using a local zone or silently falling back to UTC.

`force_tz(..., roll_dst=("NA", "post"))` matches R's default: skipped spring
clocks become missing and repeated autumn clocks choose the later occurrence.
Supply one policy string for both cases, or a pair for skipped/repeated
clocks separately. `"pre"` and `"post"` choose the instant before/after
the transition; for skipped clocks, they move the clock backward/forward
by the size of the gap. `"boundary"` chooses the transition itself,
and `"NA"` returns missing. When a source clock is already a resolved
repeat and the target clock also repeats, its occurrence is preserved.
These rules also cover half-hour transitions and a skipped calendar day.

Two subsecond details intentionally differ from R: `"boundary"` always
returns the exact whole-second transition, including when paired with a
different repeat policy; preserving an already resolved repeat also
preserves its fractions, even if `roll_dst="boundary"` was requested.
R can retain fractions in the former case and discard them in the latter.

`year`, `month`, and `day` pull those fields as integers. In a notebook
pipe, a bare `year` is still the column when penguins (or any frame) has
one: `filter(year == 2007)` filters the column, and `year(date)` calls the
function.

`wday` numbers Sunday as 1 by default; `week_start=1` numbers Monday as 1.
With `label=True`, it returns an ordered categorical/Enum with all seven
English weekday levels, starting on `week_start`. `abbr=False` uses full
names. Labels are always English; R's labels depend on its locale.
`yday` counts from January 1 as day 1. `week` counts seven-day blocks from
January 1, while `isoweek` follows ISO 8601: 2021-01-01 is week 1 under
`week` and week 53 under `isoweek`.

`quarter(x, fiscal_start=1)` returns quarters 1–4. `type="year.quarter"`
returns the fiscal ending year plus quarter/10. `type="date_first"` and
`"date_last"` return boundary dates; `"year_start/end"` returns text such
as `"2023/24 Q1"`. As in R, missing input under that last type produces
the literal text `"NA/NA QNA"`; the other types keep it missing.

`hour`, `minute`, and `second` extract the input's local clock time,
preserving the timezone already attached to a datetime. Dates give 0;
missing input stays missing. `second` includes fractional seconds, to the
precision of the input. Component helpers stay lazy on Polars, including
weekday labels whose seven levels are known in advance.

`floor_date` and `ceiling_date` round to `second`, `minute`,
`hour`, `day`, `week`, `month`, `quarter`, or `year`. Weeks start on Sunday
(`week_start=7`); `1` is Monday. A week starts at midnight, so a datetime
floors to 00:00 on the first day of its week. Under `ceiling_date`, as in
lubridate, a datetime already on the boundary stays there, but a date moves
to the next boundary: `ceiling_date` of 2010-02-01 to a month is
2010-03-01, and of any date to a day is the next day. `days_in_month` is 29 for 29 February 2012.

Timezone-aware `floor_date` and `ceiling_date` keep the zone and round to
local boundaries, including weeks and daylight saving
transitions. Both occurrences of a repeated hour stay distinct. Flooring
`2024-11-03 01:30:00.125-04:00` to a second keeps that occurrence and drops
the fraction. Ceiling the first 01:30 of an autumn fallback to an hour
lands on the second 01:00, the next hour. lubridate's `ceiling_date` lands
on 02:00 for that one hour; the other units match it.

```python
cars >> mutate(
    built=make_date(2000 + col("cyl"), col("cyl"), 31),
    leap=leap_year(2000 + col("cyl")),
    tested=make_datetime(2024, 3, 10, col("cyl") - 2, 30, tz="America/New_York"),
    month=round_date(make_date(2024, 2, col("cyl") * 2), "month"),
    iso=isoyear(make_date(2016 + col("cyl"), 1, 1)),
) >> select("model", "cyl", "built", "leap", "tested", "month", "iso") >> head(3)
```

`round_date` rounds to the nearer of `floor_date` and the next boundary,
with the same units and `week_start`. Distance is elapsed time, and a tie
goes to the later boundary, as in lubridate: Wednesday noon rounds to the
next Sunday, and 15 February 2010 rounds to 1 March (14 days either way).
On the 23-hour spring-forward day in New York, 12:00 is 11 hours after
midnight and 12 hours before the next one, so it rounds down; 12:30 is a
tie and rounds up. A date on a boundary stays there, unlike `ceiling_date`.
Two choices differ from lubridate. The default unit is `day`, as for
`floor_date`; lubridate's is `second`. The first 01:30 of an autumn
fallback is a tie between 01:00 EDT and the next hour, 01:00 EST, so it
rounds up to 01:00 EST; lubridate takes 02:00 EST as the next hour (see
`ceiling_date` above) and rounds down to 01:00 EDT. Sub-day units leave a
date unchanged, where lubridate returns a datetime.

`make_date(year, month, day)` and `make_datetime(year, month, day, hour,
min, sec, tz="UTC")` build values from columns (a name or `col()`) and
numbers, with lubridate's defaults of 1970-01-01 00:00:00. A missing part
gives a missing result. `make_date` returns missing for a date that does
not exist (30 February, month 13, day 0) and drops fractions toward zero,
so year 2023.7 is 2023. `make_datetime` rolls out-of-range parts over
instead, as lubridate does: 31 February 2023 is 3 March, hour 25 is 01:00
the next day, minute -61 is 22:59 the previous day, and month 13 is January
of the next year. `sec` keeps fractions to the microsecond. In `tz`, a clock
that daylight saving skips lands on the transition (02:30 on 10 March 2024
in New York is 03:00 EDT), and a repeated clock takes the later occurrence
(01:30 on 3 November 2024 is EST). lubridate raises on a fractional year,
month, day, hour, or minute; tidy3 raises for a number and returns missing
for a value in a column. An infinite `sec` is missing, where lubridate
returns an infinite time. Both run in Polars without reading the rows; a
`tz` other than UTC resolves skipped and repeated clocks one batch at a
time.

`isoyear` is the ISO 8601 year that `isoweek` counts in: 1 January 2021 is
in week 53 of 2020. `leap_year` takes dates, datetimes, or numeric years.
As in lubridate, it cuts a fractional year toward zero (2000.5 is a leap
year), and 1900 is not a leap year while 2000 is.

`date + months(1)` and `add_with_rollback` roll an impossible day back to
the last real day of the destination month. 31 January 2010 plus one month
is 28 February 2010. 29 February 2012 plus `years(1)` is 28 February 2013. A period with both
parts moves by `12 * years + months` in one step, as lubridate's
`add_with_rollback` does, so 29 February 2024 plus `years(1) + months(1)`
is 29 March 2025. `months(col("n"))` takes a count per row; a missing
count gives a missing date.
Subtracting a date from `months()` or `years()` is an error. `today()` and
`now()` are Python values taken when the pipe is built. The default
timezone is UTC. `tzone="local"` uses the machine zone, and any other
string is an IANA name such as `"America/Los_Angeles"`. `now()` returns the
wall time in that zone as a naive datetime.

```python
from pathlib import Path

Path("ids.csv").write_text("id,mass\n001,NA\n002,3700\n# skipped\n", encoding="utf-8")
read_csv(
    "ids.csv",
    col_types={"id": "character", "mass": "double"},
    na="NA",
    comment="#",
    id="source",
)
```

`read_csv`, `read_tsv`, and `read_delim` load a `TidyFrame` and share those
options with `scan_csv`. `read_*` use readr's defaults, `trim_ws=True` and
`na=["", "NA"]`; `scan_csv` keeps `trim_ws=False` and Polars' null values.
With `trim_ws=True`, fields are trimmed before column types are guessed
(from the first `infer_schema_length` rows, 100 by default), so `" 6"` is
the integer 6 and `" NA "` is missing. `col_types` values `character`, `integer`,
`double`, `logical`, `date`, and `datetime` are read as text and then
parsed, so an id column keeps leading zeros and a bad integer becomes
missing with a row in `problems()`. Guessed numeric and logical columns
use the same parsers: a bad value after the guessing sample becomes missing
and is recorded in `problems()`. `strict=True` raises `ConversionError`
instead. Date columns are ISO `YYYY-MM-DD`.
`id` adds a column with each row's source path, including when `source` is
a glob. A non-UTF-8 `encoding` is available on `read_csv` for one file;
`scan_csv` asks you to use `read_csv` for that. `read_excel(path, sheet=)`
and `excel_sheets(path)` need `pip install 'tidy3[excel]'`.

#### Where these differ from R on purpose

The R oracle tests (`tests/test_r_oracle_parity.py`) check these helpers
against R, dplyr, readr, and lubridate. These differences are deliberate:

- `as_character` writes values as their column type shows them: `21.0`
  is `"21.0"` where R gives `"21"`, `100000.0` is `"100000.0"` where R
  gives `"1e+05"`, and nothing is rounded to 15 digits. Logical values are
  `"true"` / `"false"` unless you pass `true="TRUE", false="FALSE"`.
- `as_integer` refuses `2.5` and the string `"2.0"`; R truncates them.
- Integers are 64-bit, so `parse_integer("2147483648")` is a number, not NA.
- A two-digit year needs an explicit `cutoff`; readr and lubridate assume 68.
- `read_csv` guesses whole-number columns as integers (as Polars does);
  readr guesses double. The values are the same.

### Factor ordering and recoding

Use `fct_relevel` to move named levels to the front, `fct_reorder` to order
them by another column, `fct_infreq` to put frequent levels first, and
`fct_rev` to reverse them. These change level order while preserving values
and missing values. Existing categorical inputs keep unused levels. Ties
keep the previous level order.

```python
ordered_cars = (
    cars >> mutate(cyl=as_character("cyl"))
    >> mutate(cyl=fct_reorder("cyl", "hp", mean))
)
ordered_cars >> arrange("cyl")
ordered_cars >> mutate(cyl=fct_rev("cyl"))
```

`fct_recode(f, new="old")` renames a level and keeps unmentioned levels.
`NULL="old"` removes that level and makes its rows missing. To merge several
levels, use `fct_collapse(f, new=["old1", "old2"])`. Its optional
`other_level="Other"` combines the remaining levels at the end. Unknown
old levels warn; renamed unused levels and the ordered flag are preserved.

```python
ordered_cars >> mutate(cyl=fct_recode("cyl", four="4"))
ordered_cars >> mutate(cyl=fct_collapse("cyl", small=["4", "6"], other_level="large"))
ordered_cars >> mutate(cyl=fct_lump_n("cyl", 2))
```

`fct_lump_n(f, n)` keeps the most frequent levels; negative `n` keeps the
least frequent. `ties_method="min"` keeps all levels tied at the cutoff;
`"average"`, `"first"`, `"last"`, `"max"`, and `"random"` select other rank
rules. Random ties use NumPy's random state, independently of R's generator.
`fct_lump_min(f, min)` keeps counts at least `min`.
`fct_lump_prop(f, prop)` keeps proportions strictly greater than `prop`;
negative `prop` keeps proportions at most `abs(prop)`. All three accept
row weights through `w=` and a replacement name through `other_level=`
(default `"Other"`). Missing rows remain missing and count toward the
proportion denominator, including their weights. Weights must be finite,
non-negative, and non-missing. A zero proportion denominator with existing
levels raises `ValueError`. Grouped mutations compute frequencies per group.
The replacement level comes last when any levels are lumped.

The contracts are checked against [forcats recoding](https://forcats.tidyverse.org/reference/fct_recode.html),
[collapsing](https://forcats.tidyverse.org/reference/fct_collapse.html), and
[lumping](https://forcats.tidyverse.org/reference/fct_lump.html).

`fct_reorder(f, x, fun=median)` summarizes numeric `x` with the median; `fun` may be a tidy3
aggregation or a callable returning one number. `na_rm=None` removes missing
`x` with a warning; `True` removes silently, and `False` passes missing values
to `fun`. Empty levels use `default=float("inf")`; `desc=True` reverses the
summary order, keeping missing summaries last. `fct_infreq(f, w=)` accepts a
weight column with non-negative, non-missing weights.

`frame.factor_levels()` returns each factor column's levels in order, such as
`{"cyl": ["4", "6", "8"]}`: columns with recorded levels and Polars `Enum`
columns, as plain lists, read from the schema without computing rows.
Packages built on tidy3 frames, such as stats3, read factor order through it.

`fct_relevel(f, "Gentoo", "Adelie", after=0)` keeps unmentioned levels in
their existing order. `after=float("inf")` moves the named levels to the end.
One callable, such as `sorted`, may replace the level names. Unknown levels
warn and are ignored; duplicate names are an error.

`as_factor(x)` makes a factor without sorting text: levels follow first
appearance, as in forcats. Numbers get levels in numeric order and logical
values get `FALSE` and `TRUE`. Level names follow tidy3's header text rules,
so `2.0` becomes `"2"`; unlike R, numbers are never rounded to 15 digits or
written in scientific notation (`1e6` is `"1000000"`, not `"1e+06"`).
Dates and other types raise; convert them with `as_character` first.
`fct_inorder(f)` orders existing levels by first appearance, with unused
levels after them.

```python
cars >> mutate(cyl=as_factor("cyl"), model=fct_inorder("model"))
cars >> mutate(cyl=fct_reorder2(as_factor("cyl"), "wt", "mpg"))
```

`fct_reorder2(f, x, y)` orders levels by `fun(x, y)`, largest first. The
default `last2` is the `y` value at the largest `x`, so a line chart's legend
lists lines in the order they end; `first2` takes the `y` value at the
smallest `x`. Ties in `x` use the first row. `fun` may be any callable taking
the level's `x` and `y` Series and returning one number; extra keyword
arguments pass through. Rows missing `x` or `y` are left out of the
summaries with a warning (`na_rm=None`), silently (`True`), or passed to
`fun` (`False`). Empty levels use `default=-inf`, and `desc=False` puts the
smallest first. Every row stays in the result; forcats also drops the
removed rows from the factor it returns.

`fct_drop(f)` removes unused levels; `only=` limits that to the named
levels. `fct_expand(f, "a", ["b", "c"], after=inf)` adds levels without
changing values, ignoring ones already present. `fct_na_value_to_level(f,
level="(Missing)")` turns missing values into a level, added last if new,
even when nothing is missing. forcats defaults to an `NA` level, which a
tidy3 factor cannot hold, so the default is forcats' earlier `"(Missing)"`.
`fct_na_level_to_value(f, extra_levels=)` turns those levels' rows missing
and removes the levels; without `extra_levels` nothing changes. All of these
keep the ordered flag and run per group in grouped mutations.

These are checked against forcats 1.0.1
([as_factor](https://forcats.tidyverse.org/reference/as_factor.html),
[fct_reorder2](https://forcats.tidyverse.org/reference/fct_reorder.html),
[fct_inorder](https://forcats.tidyverse.org/reference/fct_inorder.html),
[fct_drop](https://forcats.tidyverse.org/reference/fct_drop.html),
[fct_expand](https://forcats.tidyverse.org/reference/fct_expand.html), and
[missing values](https://forcats.tidyverse.org/reference/fct_na_value_to_level.html)).

Use factor expressions in `mutate`, `transmute`, or `across` before sorting,
grouping, or plotting. The result is a Polars Enum, so its level order
survives collection and plot3 handoff. Every factor expression reads the
data once to discover its levels. **A single call on a text column runs in
Polars**: it reads only the columns it uses, and the frame stays lazy. That
covers every helper without groups, as well as `fct_relevel`, `fct_reorder`
(with a tidy3 aggregation), `fct_infreq`, and `fct_rev` in grouped
mutations; `as_factor` also covers numeric and logical columns, and
`fct_reorder2` covers `last2` and `first2`. **Other factor mutations compute a
snapshot**, including the source frame, and later verbs build a lazy plan on
that snapshot: nested calls, factor (Enum) inputs, grouped value changes,
callables, and float columns holding NaN. That path uses Python callbacks.
Other mutations stay lazy.

A list, pandas categorical/Series, or Polars Series runs immediately and
returns a pandas categorical. Character input starts with levels sorted in
Unicode order; R's character-to-factor sorting depends on locale. Existing
categorical input preserves its order. `fct_infreq(ordered=)` controls the
pandas ordered flag; Polars Enums always carry an explicit sort order. The
ordered flag is stored with the level names, so `collect(as_="pandas")`
still reports `ordered=False` after `fct_infreq(ordered=False)`.

An imported pandas factor keeps unused levels, level order, and its ordered
flag on the frame. Collecting a Polars plan with `as_="pandas"` writes those
levels back. `arrange` orders factor columns by that level list.
Missing keys stay last, and ties keep the previous row order.

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

The familiar tidyr operations work as pipe verbs or `TidyFrame` methods:

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
performs R-style logical/numeric inference. Schema-dependent
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

Joins return rows in dplyr's order: x's rows in order, each
followed by its matches in y's order, and then, for `right_join` and
`full_join`, the rows of y that matched nothing, in y's order. So
`islands >> right_join(penguins, by="island")` lists penguins grouped by the
order of `islands`, not in their own order; pandas' `merge(how="right")`
keeps y's order instead. Use `penguins >> left_join(islands, by="island")`
to keep the penguins in their own order.

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

Portable helpers run with dplyr's grouped semantics:

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
`case_when`. `scale(x, center=True, scale=True)` standardises like R's
`scale()` and `rescale(x, to=(0, 1))` min-max scales like `scales::rescale()`;
both skip missing values and return a plain column. `lead`, `lag`, `first`, `last`, and `nth` accept `order_by=`.
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
including `replace=True`, is supported. Group-context helpers
and additional vector helpers remain future phases.

### Strings (stringr)

All of stringr's functions are here, and they run inside any verb on either
backend. Patterns are regular expressions; wrap one in `fixed()` (or
`coll()`) to match text as written, use `regex(..., ignore_case=True,
multiline=, dotall=, comments=)` for flags, and `boundary("word")` or
`boundary("character")` to work in words or characters. Missing values stay
missing, as in R.

```python
cars
>> filter(str_starts(model, "Merc"))
>> mutate(
    make = word(model, 1),
    short = str_trunc(model, 10),
    label = str_glue("{make} gets {mpg} mpg"),
)
```

| Task | Helpers |
|---|---|
| Detect | `str_detect`, `str_starts`, `str_ends` (each takes `negate=`), `str_count`, `str_like`, `str_ilike` (SQL `LIKE`), `str_equal` |
| Locate | `str_locate` (a `{start, end}` struct), `str_locate_all` (a list of them), `invert_match` |
| Length and pieces | `str_length`, `str_width`, `str_sub` (1-based, negatives count from the end), `str_sub_all`, `str_trunc`, `word` |
| Replace | `str_replace`, `str_replace_all` (`\\1` for a capture group; a dict `{pattern: replacement}` applies several), `str_remove`, `str_remove_all`, `str_replace_na` |
| Case | `str_to_upper`, `str_to_lower`, `str_to_title`, `str_to_sentence` |
| Whitespace and layout | `str_trim`, `str_squish`, `str_pad`, `str_dup`, `str_wrap` |
| Extract and match | `str_extract` (`group=`), `str_extract_all`, `str_match` (a struct of `match` and each group), `str_match_all` |
| Split | `str_split` (a list column, `n=`), `str_split_fixed` (exactly `n` pieces), `str_split_i`, `str_split_1` |
| Combine | `str_c` (row by row, `sep=`), `str_glue` (`{column}` templates), `str_glue_data`, and the aggregates `str_flatten` (`collapse=`, `last=`, `na_rm=`) and `str_flatten_comma` |
| Other | `str_escape`, `str_conv` (decode bytes) |

Struct results spread into columns with `unnest_wider()`:
`mutate(loc = str_locate(x, "an")) >> unnest_wider(loc)` gives `start` and
`end`.

Given a list or Series instead of a column, every helper runs at once and
returns a list, as stringr does on a character vector:
`str_detect(["apple", "banana"], "an")` is `[False, True]`. The helpers that
only make sense on a vector work that way only: `str_subset`, `str_which`,
`str_sort`, `str_order` (`numeric=True` puts `"x9"` before `"x10"`),
`str_unique`, and `str_view`, which prints matches in `<>`. In a pipe, use
`filter(str_detect(...))`, `arrange()`, and `distinct()` instead.
`str_which` and `str_order` return 1-based positions, as in R.

How this differs from R:

- In `str_c`, a quoted string is text, not a column: name columns bare in a
  notebook or with `col()` in a script. In `str_equal`, a quoted second
  argument is text. Elsewhere a quoted first argument is a column name, as
  in `mean("x")`.
- `str_glue` fills in names only (`{price}`, not `{price * 2}`); a column
  wins over a notebook variable of the same name, as in dplyr.
- `str_split_fixed` returns a list column of `n` pieces rather than a matrix.
- `str_equal` compares Unicode NFC, after lowercasing when `ignore_case` is
  set, so `é` and `e\u0301` are equal. It is not stringi collation: `ﬁ` is
  not equal to `fi` (stringr's `ignore_case` says they are), and a joined
  family emoji is not equal to the same emoji without ZWJ.
- `coll()` matches literal text without locale rules. `boundary()` supports
  `"character"` and `"word"` only. `boundary("character")` splits code
  points, so a combining accent and the parts of a joined emoji are separate
  pieces; stringr keeps each grapheme together.
- On Polars, regular expressions use Rust's syntax, which has no
  look-around. `str_locate_all`, `str_match_all`, `str_width`, `str_wrap`,
  `str_conv`, `word` with a regex separator, and `str_split` with both a regex
  and `n=` run row by row in Python, so they are slower on large frames.
- `str_to_title("don't")` gives `"Don'T"` (R: `"Don't"`).
- There is no `str_sub<-`; build the new string with `str_c()` and
  `str_sub()`. `str_interp` (deprecated in stringr) is not included.

### Performance controls

Grouped `mutate()` expressions automatically reuse embedded or repeated
statistics. Expressions such as the following compute each group mean and
standard deviation once; temporary columns never appear in
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
- **Missing columns fail at the verb**: a verb that uses a column the frame
  lacks raises `ColumnNotFoundError` as soon as the pipe is built, even on a
  lazy frame (only the schema is resolved; no rows are read). The message
  names the verb, suggests close names, and lists the columns:

  ```text
  mutate(): column 'bill_length' does not exist.
  Did you mean 'bill_length_mm'?
  Columns: 'species', 'island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year'
  ```

  It is also a `KeyError` and a Polars `ColumnNotFoundError`. Other errors
  keep their own type and message; on Python 3.11+ a note names the verb.
  A preview that fails while computing rows shows the error's first line,
  with the full error and query plan under **Technical details**.
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
