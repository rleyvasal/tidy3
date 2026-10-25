# Changelog

All notable changes to tidy3. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may change behaviour, and those changes are listed
under **Changed**.

## Unreleased

### Added

- Date helpers from lubridate: `round_date()` rounds to the nearer calendar boundary (ties go up, distance in elapsed time, local boundaries in a timezone); `make_date()` and `make_datetime()` build values from column and number parts (`make_date` gives missing for impossible dates, `make_datetime` rolls parts over and resolves daylight-saving gaps to the transition); `isoyear()` gives the ISO 8601 year; `leap_year()` takes dates, datetimes, or numeric years. They run as native Polars expressions, except that `make_datetime` in a zone other than UTC resolves skipped and repeated clocks per batch. Checked against lubridate 1.9.5 for ties, week starts, daylight saving, leap days, invalid and missing parts, with raw Polars benchmark cases. Differences from lubridate are documented: `round_date` defaults to `"day"`, keeps a date for sub-day units, and rounds the first 01:30 of an autumn fallback to 01:00 EST; `make_datetime` gives missing, not an error, for a fractional part in a column.

- `adorn_percentages()` divides report counts by row (`"row"`, the default), column (`"col"`), or grand (`"all"`) totals, with `cols=` selection and `na_rm`. Results stay numeric and lazy. Totals from `adorn_totals()` are not counted twice: the totals row is left out of column and grand sums, and the totals column becomes 1 or its own share unless `cols=` leaves it out. Divisions are exact (Polars would otherwise multiply by the reciprocal, giving 0.6000000000000001 for 3 / 5). Checked against R janitor 2.2.1 for every denominator with and without totals, selections, missing values, factor labels, and tabyl reports, with raw Polars benchmark cases.
- `ymd_h`, `mdy_hm`, `mdy_h`, `dmy_hm`, and `dmy_h` complete the datetime order family, with the same separators, month names, compact digits, AM/PM, offsets, `tz`, `truncated`, and diagnostics as `ymd_hms`.
- `parse_date_time(x, orders)` parses with several lubridate orders (`y Y m b B d H I M S p z a A T R r`, separators ignored). Each value takes the first format in lubridate's training ranking: more fields first, then the format that reads more of the column, sampled on prime rows for columns of 100 values or more. Supports `tz`, `truncated`, and `exact=True` formats. Zero-padded text parses natively in Polars.
- `parse_time()` reads times of day with readr's default and explicit formats (`%H %I %M %S %OS %p %T %R %AT`, skips, and ignored date and zone codes) into a Polars `Time` column. Failures, and readr's durations of a day or more, go to `problems()`.
- The new parsers are checked against lubridate 1.9.5 and readr 2.2.0 in the R oracle tests (separators, 12/24-hour clocks, fractions, offsets, time zones, daylight saving, mixed and ambiguous orders, long columns), with raw Polars benchmark cases. Differences from R are documented in the reference.

## 0.9.0 — 2026-10-25

Frequency tables, totals, and the everyday forcats value and level helpers, computed in Polars on text columns.

### Added

- `adorn_totals()` adds numeric row, column, or grand totals, with `cols=` selection, custom labels, optional typed missing fills, and missing-value control. It ignores existing groups, retains parse diagnostics, and extends factor row labels without losing their order or ordered flag. Totals stay lazy. Re-adding a totals dimension raises an error, including after `compute()`; ordinary data-changing verbs clear the reporting markers. R janitor comparisons cover row/column composition, selected columns, empty inputs, missing values, and factor levels. A checked raw Polars benchmark case is included.

- Two-variable frequency tables: `frame >> tabyl("row", "column")` returns numeric counts with zero-filled combinations, factor levels in order, optional unused levels, and `NA_`/`emptystring_` columns for missing and blank values. `show_na=False` drops records missing either variable. Tables preserve row-factor metadata and stored parse failures, ignore existing groups, and compute one count snapshot. Results are checked against R janitor, with a separate raw Polars benchmark case. Numeric headers use tidy3's header text rules, and ambiguous labels raise an error.

- `tabyl()` starts the reporting helpers with one-variable frequency tables: `frame >> tabyl("column")` or `tabyl(vector)`. It returns numeric counts and proportions, adds `valid_percent` when missing rows are shown, and supports `show_na` and `show_missing_levels`. Factor order, ordered flags, and parse diagnostics survive. It computes the count table once and returns an ungrouped snapshot. One-variable results are checked against R janitor, with a raw Polars benchmark case; three-variable tables, `adorn_percentages()`, and the formatting `adorn_*` helpers remain future work.

- Factor value helpers: `fct_recode(new="old", NULL="removed")`, `fct_collapse(new=["old1", "old2"], other_level=...)`, and `fct_lump_n`, `fct_lump_prop`, and `fct_lump_min`. They work on vectors and in mutations, including per-group frequencies, row weights, tie rules, nested calls, and `across`. Missing values, unused levels, level order, and the ordered flag are checked against R forcats. On a text column without groups they run in Polars (see Changed); other uses compute a snapshot. Each has a checked raw Polars benchmark case.

- Factor level helpers: `as_factor`, `fct_inorder`, `fct_reorder2` (with `last2` and `first2`), `fct_drop`, `fct_expand`, `fct_na_value_to_level`, and `fct_na_level_to_value`. They work on vectors and in mutations, including grouped ones, and keep unused levels and the ordered flag. Results are checked against R forcats 1.0.1, and each has a raw Polars benchmark case. `as_factor` names numeric levels with tidy3's header text rules (no 15-digit rounding or scientific notation), `fct_na_value_to_level` defaults to `"(Missing)"` because a tidy3 factor cannot have an `NA` level, and `fct_reorder2(na_rm=True)` keeps every row where forcats drops the missing ones.

### Changed

- Factor helpers on a text column compute their levels in Polars when the mutation is ungrouped: `fct_recode`, `fct_collapse`, the lumping helpers, `fct_drop`, `fct_expand`, `fct_inorder`, `fct_na_value_to_level`, `fct_na_level_to_value`, `as_factor` (also on numeric and logical columns), and `fct_reorder2` with `last2` or `first2`. They read only the columns they use and the frame stays lazy, as the ordering helpers already did. On 100,000 rows each now takes 1.6–3.2 ms instead of 5.9–11.7 ms, within 0.7–2.0 times raw Polars; their total excess over raw Polars fell from 70 ms to about zero. Results, warnings, and errors are unchanged; grouped value changes, nested calls, factor inputs, and other callables keep the previous path.
- Notebook bare names are rewritten only in calls to tidy3's own verbs. A notebook's own `mutate` function, or a verb with the same name from another package (such as tidy3a), is left alone, so `def select(...)` or `from tidy3a import *` no longer has its arguments turned into tidy3 columns. The cell that does the import is handled too, because the cell's own imports decide before the live namespace does.

### Fixed

- Notebook ownership follows statement order: defining or assigning your own `mutate` after importing tidy3 in the same cell leaves its arguments alone; importing tidy3 again restores bare-column rewriting. Imports inside an uncalled function do not change ownership outside that function.
- In notebooks, the first argument of `fct_relevel`, `fct_reorder`, `fct_infreq`, and `fct_rev` is a column even when it has a builtin's name, such as `type`, as it already was for `fct_recode` and the lumping helpers.
- Factor helpers used outside `mutate` on Polars (for example in `filter`) raise "must be assigned with mutate() first" for every `fct_*` function, instead of "unknown expression function" for some.
- Sequential factor mutations use the levels of the intermediate column, so recoding a factor created earlier in the same `mutate` does not replace renamed values with missing ones.

## 0.8.6 — 2026-10-24

Verbs that take column selections accept `col("x")`.

### Fixed

- `group_by()`, `distinct()`, `ungroup()`, `with_groups()`, `relocate()`, `pull()`, `rowwise()`, the `by=` argument of `slice_*()` and other verbs that take column selections accept `col("x")` (and `pl.col("x")`) wherever they accept a column name, as `count()` does since 0.8.4, so scripts can write `group_by(col("species"))`. It used to raise `TypeError: unsupported column selection`. `select(col("x") * 2)` still makes a computed column.

## 0.8.5 — 2026-10-23

`filter()` works with one function nested inside another.

### Fixed

- `filter()` works when one function is nested inside another, such as `filter(str_trim(str_replace_na(col("island"), "")) == "")`. In 0.8.4 any such nesting in `filter()` raised `TypeError: cannot use 'tuple' as a set element`, while the same expression worked in `mutate()`.

## 0.8.4 — 2026-10-23

Multi-line pipes work inside functions, and `count()` takes `col("x")` and computed groups.

### Added

- `count()` and `add_count()` take computed groups as keyword arguments, as dplyr's `count(big = x > 1)` does: `count(heavy=col("body_mass_g") > 4000)` counts by a new `heavy` column, and `add_count()` keeps that column on each row.

### Fixed

- Multi-line `>>` pipes now work inside functions, loops, `if` blocks and other indented code in notebooks, not only at the top level of a cell. Before, a pipe step on its own line inside a `def` raised `IndentationError: unexpected indent`, so pipes there had to be wrapped in parentheses. Each step joins the statement above it at the same indentation; top-level cells rewrite exactly as before, and line numbers in errors still match the cell as typed. A broken pipe inside a block now reports the real mistake rather than an `IndentationError`.
- `count()` and `add_count()` accept `col("x")` (and `pl.col("x")`) wherever they accept a column name, so scripts can write `count(col("species"))`; it used to raise `TypeError: cannot use 'tidy3.expr.Expr' as a dict key`. An unnamed computed expression such as `count(col("x") > 1)` now raises a `TypeError` that shows the keyword form.

## 0.8.3 — 2026-10-23

`tidy()` works on fitted models whichever package was imported last.

### Added

- `tidy()` gives the table of any object with a `__tidy__` method, passing its arguments on: `tidy(fit, conf_int=True)` on a model fitted with stats3 returns its coefficients, as broom's `tidy` does in R, so `tidy()` works on models whichever of tidy3 and stats3 was imported last. With no arguments `tidy()` is a pipe step: `fit >> tidy()`, and `df >> tidy()` wraps data.

## 0.8.2 — 2026-10-23

Factor levels have a public accessor.

### Added

- `TidyFrame.factor_levels()` returns each factor column's levels in order, as `{column: [levels]}`: columns with recorded levels (from `fct_*` and imported pandas categoricals) and Polars `Enum` columns, as plain lists. It reads only the schema, and leaves out levels recorded for columns a later `select` dropped. Packages built on tidy3 frames, such as stats3, use it instead of tidy3's internals.

## 0.8.1 — 2026-10-22

The file writers work as pipe steps.

### Added

- `write_csv`, `write_parquet`, `write_ipc`, and `write_excel` work as pipe steps and as functions, not only as `TidyFrame` methods. `df >> write_csv("out.csv")` writes the file and passes the frame on, so a pipeline can continue after it; `write_csv(df, "out.csv")` writes and returns `df`, as readr does, and also takes Polars and pandas frames. `write_excel("book.xlsx", sheets={...})` writes several sheets without a frame first.

## 0.8.0 — 2026-10-22

tidy3 runs on Polars only: pandas comes in and goes out, but the pandas backend is gone. Pipelines export as plain Polars code, and most operations run within a millisecond of raw Polars.

### Added

- `TidyFrame.to_polars_code()` (and `tidy3.to_polars_code(frame)`) prints a pipeline as a Python function that needs only Polars, with each input frame as an argument and its columns listed in the docstring. It prints the Polars query tidy3 compiled, so the function returns the same result, dplyr's rules included, and reads like hand-written Polars: plain columns by name, no redundant selects or casts, one argument per line in long calls. A column computed by a Python function (the text conversions, R-style `round`, `force_tz`, `excel_numeric_to_date`, per-value string functions) or a step that built its query from values in the data (factor levels, `pivot_wider` names, `compute()`) raises `ExportError` naming the column or step; tidy3 records such steps as the pipeline is built, including through joins. tidy3's lazy input checks (`rows_*` keys, `separate_wider_*` pieces) are skipped in exported code. Of the 279 benchmark cases that build a query, 233 export and match tidy3 exactly when run, and the other 46 are refused for one of those two reasons.
- Works with Polars 2.0: `to_polars_code()` reads Polars 2's join plans as well as 1.x's. `cloudpickle` is a new dependency, because Polars needs it to serialize a plan with Python steps; without it, `to_polars_code()` raises an `ExportError` that says so.
- `is_na`, `is_nan`, `is_finite`, `is_infinite`, `pmin`, `pmax`, `row_sums`, `row_means`, `hour`, `minute`, `second`, `ceiling_date`, and `with_tz` compile to plain Polars when the column type is known from the lazy schema, which is the usual case; a per-batch type check remains for the rest. Results are unchanged (checked against the previous code on 496 type and context combinations).

### Removed

- The pandas backend. tidy3 runs on Polars only. `tidy(df, backend="pandas")` and `options(backend="pandas")` raise a `TypeError` that names the replacement, and `TidyFrame.backend` is gone. pandas stays supported at both ends: a pandas DataFrame passed to `tidy()` (or to a join or bind) becomes a lazy Polars frame that keeps its factor levels and ordered flags, and `collect(as_="pandas")` returns a pandas result. Every verb was implemented twice, and the pandas versions were 2–25 times slower than Polars on 1,000,000 rows, so about 3,400 lines of pandas-only code are removed. Factor calls that Polars cannot yet compute natively (nested calls, callables, NaN values) still evaluate their arguments with pandas internally. `python -m tidy3.bench_suite` drops its `tidy3[pandas]` engine and the `--max-tidy-pandas-geo` budget; it still compares tidy3 with raw pandas.

### Changed

- Ten of the cases slowest against raw Polars got faster (times on 100,000 rows). `group_modify`, `group_map`, and `group_split` give each of up to 32 groups a lazy filter of the frame instead of gathering every column in group order, bind `group_modify` results in one collect, and build `cur_group_rows()` only when called: `group_modify` 10.5 ms → 5.3 ms. `read_csv`/`read_tsv`/`read_delim` trim the text once and parse each guessed column in one pass that also records its problems: 6.0 ms → 4.2 ms. `stop_for_problems()` computes the data and its failures together and returns the computed frame: 2.3 ms → 1.4 ms. `closest()` against an in-memory table whose keys do not repeat searches the keys once and attaches the matched row by its key, with no tie join or pair sort: 1.75 ms → 0.70 ms (raw 0.30 ms), and 7.1 ms → 5.6 ms on a shuffled 200,000-row, 21-column x. An inequality join against a one-row table compares in place instead of joining back: `gt` 0.96 ms → 0.42 ms, `between` 0.72 ms → 0.47 ms. `force_tz` from a naive or UTC time with the default skipped-time rule runs in Polars: 0.49 ms → 0.15 ms. `remove_empty("cols")` reads stored null counts: 1.09 ms → 0.53 ms. Results are unchanged, checked against the previous code (strict reads keep their previous error). The total excess over raw Polars fell from 71 ms to about 58 ms; 11 cases remain at twice raw Polars and 0.3 ms over.
- String functions that apply Python rules per value (`str_wrap`, `str_match_all`, `str_locate_all` with a regex, `word` and `str_split` with a regex separator, `str_conv` on bytes) run them once per distinct value and look the results up, so repeated text costs one call per distinct string. `str_width` counts ASCII text natively and applies the Unicode rules only to other text; `str_locate_all` with a plain-text pattern runs in Polars. On 100,000 rows of repeated text: `str_wrap` 177 ms → 1.3 ms, `str_match_all` 277 ms → 26 ms, `str_locate_all` 273 ms → 23 ms, `str_width` 73 ms → 2 ms; on all-distinct text `str_width` 123 ms → 2 ms and literal `str_locate_all` 271 ms → 63 ms, with the rest unchanged. Results are unchanged (checked against the previous code on 117 cases, including wide and combining Unicode).
- An inequality `join_by()` (`<`, `between`, `within`, `overlaps`) against an in-memory table of up to 64 rows runs as a Polars nested-loop join, which keeps dplyr's row order without sorting the matches; larger or lazy tables keep the IE join. `between` 1.3 ms → 0.7 ms and `gt` 1.6 ms → 1.0 ms on 100,000 rows, and up to 2.3 times faster when most rows match. Results are unchanged (39,340 join cases).
- Less fixed cost per verb: a frame resolves its schema once and the next verb reuses it, `mutate` skips its column-order `select` when new columns already come last, and an ungrouped `slice()` of consecutive rows is a Polars slice (0.41 ms → 0.04 ms). `row_means`/`row_sums` without `min_count` use one horizontal kernel (0.90 ms → 0.48 ms), and `cummean` counts rows directly. A trivial `mutate` pipeline went from 0.19 ms to 0.13 ms. On the 328 benchmark cases at 100,000 rows the total excess over raw Polars fell from 141 ms to 71 ms, and 320 cases are within 1 ms.
- The `join_by`, set-operation, and vector-helper benchmark cases give tidy3 the same choice of eager or lazy input as the raw recipe. They had computed tidy3's inputs eagerly inside the timing, which made equality joins look 1.5 ms slower than raw Polars; with equal inputs they are within 0.15 ms. `second`'s raw recipe now keeps fractional seconds, as `second()` does.
- `python -m tidy3.bench_api` compares each case with the best raw Polars: a recipe that also runs on `df.lazy()` and gives the same result is timed both ways, and the faster counts (`raw_mode` in the JSON). Vector helpers (`str_sort`, `str_order`, `str_unique`, `str_subset`, `str_which`, `full_seq`) take the same Series on both sides, and the `setequal`, `remove_empty`, `problems`, and `full_seq` recipes now do the operation's work instead of returning a constant. Nested results are compared flattened, so `--rows 1000000` finishes (about 2.5 minutes) instead of stalling on `group_nest`. On 100,000 rows the cases where tidy3 appeared faster than raw Polars fell from 39 to 15; the lazy baselines show about 1.5 ms of overhead left in joins.
- `full_seq()` of a Series or array no longer copies the values into a Python list: 3.7 ms → 0.6 ms on 100,000 values. Results are unchanged.
- `python -m tidy3.bench_api` times every public function against a raw Polars recipe on one shared table. It checks values, names, dtypes, and order before timing, and lists functions that are not operations with a reason instead of a timing. On 100,000 rows the excess over those recipes dropped from 383 ms to 168 ms. The largest closes, best of 3 after one warmup, against the previous one-sample times on the same machine: `excel_numeric_to_date` 42.8 ms → 1.5 ms (raw 0.4 ms), `str_which` 24.0 ms → 3.9 ms and `str_subset` 24.1 ms → 3.8 ms (raw about 1.6 ms), `word` 22.0 ms → 3.4 ms (raw 3.1 ms), `str_sort` 20.5 ms → 4.8 ms and `str_order` 21.5 ms → 4.7 ms (raw about 1.9 ms), `str_conv` 13.2 ms → 0.17 ms, `str_unique` 12.6 ms → 3.2 ms, `str_split_fixed` 58 ms → 18.3 ms (raw 18.1 ms), and `complete()` over `full_seq` 9.6 ms → 0.8 ms (raw 0.5 ms). In-range numeric serials use a Polars date offset; serial 60 stays missing, and out-of-range serials still raise. Text `str_conv` is left as text. Results are unchanged.
- Uniform zero-padded dates and grouped string factors stay on Polars. One 100,000-row sample: `ymd` 1.65 ms vs 0.71 ms, `mdy` 2.76 ms vs 0.67 ms, `ymd_hms` 5.0 ms vs 3.9 ms, `fct_infreq` 3.24 ms vs 3.26 ms, `fct_reorder` 3.16 ms vs 4.18 ms, `fct_rev` 2.72 ms vs 3.49 ms, `fct_relevel` 4.47 ms vs 3.22 ms.
- Speed guards compare matching raw Polars operations and record measured timings and ratios separately from CI limits. `python -m tidy3.bench_polars` produces a reproducible `closest()` timing and optional peak-memory report through one million rows, checking complete outputs before measurement.
- Polars rolling `join_by(closest(...))` joins use a sorted lookup for a single rolling inequality with matching numeric, temporal, or text key types and the default `multiple="all"`. All ties, null grouping keys, row order, and join validation are preserved. Matching carries only keys and row IDs; payload columns are attached after choosing matches. Rolling joins with additional conditions also narrow their candidate columns. On the original 200,000-row, 30-column workload: 861 ms → 36 ms. A 1,000,000-row, 21-column join against 100,000 lookup rows takes 62 ms locally.
- `is_na`, `is_nan`, `is_finite`, and `is_infinite` compute only the requested flag in Polars, without a Python list per call. `group_keys`, `group_size`, and `n_groups` count rows per group instead of collecting every row position, and the other group metadata functions split one position array rather than one Python list per group. One 1,000,000-row sample with 8 groups: `is_na` 6.9 ms → 0.36 ms, `is_finite` 6.7 ms → 0.76 ms, `n_groups` 28.3 ms → 0.9 ms, `group_indices` 31.2 ms → 6.1 ms, `group_data` 46.4 ms → 21.6 ms. Results are unchanged.
- Clean text columns convert with one Polars cast or word lookup: `as_integer`, `parse_integer`, `as_double`, `parse_double`, `as_numeric`, `as_logical`, and `parse_logical` skip the whitespace strip and shape check when every present value converts, and the full rules still read any column with padding, `NA` text, overflow, or `inf`/`nan` spellings. `read_csv`, `read_tsv`, and `read_delim` read the file once instead of again to guess types. `separate` with a plain-text `sep` and `separate_wider_delim` take fields straight from the split, and `separate_wider_delim` counts pieces for both checks in one pass. One 100,000-row sample: `read_csv` 9.3 ms → 5.6 ms, `separate` 7.4 ms → 2.9 ms (raw Polars 2.6 ms), `separate_wider_delim` 10.1 ms → 4.2 ms, `parse_logical` 5.2 ms → 2.5 ms, `as_integer` 2.1 ms → 0.5 ms. Results, `problems()`, and strict errors are unchanged (checked against the previous code on 13,400 randomized cases).
- On Polars, `join_by()` with only equality conditions and default options joins in one native step, as `by=` does, renaming a differently named right key first; it used to match row ids and reattach unmatched rows. Left and inner `join_by()` joins with inequality, `between`, `within`, `overlaps`, or `closest` conditions sort only the matched row ids instead of the whole result. Rowwise `mutate` skips the one-row window for element-wise and `c_across()` work; aggregates and window functions keep it. `round` finds R's nearer candidate with fewer array passes, `group_data` builds `.rows` from one array, and `group_split`, `group_map`, and `group_modify` gather each group's rows in one step. One 100,000-row sample: `left_join(by = join_by(id))` 4.1 ms → 2.1 ms (raw Polars 2.0 ms), `join_by(id > y)` left join 3.1 ms → 2.0 ms, rowwise `sum(c_across())` 1.1 ms → 0.3 ms, `round` 1.4 ms → 0.7 ms, `group_data` 2.3 ms → 0.7 ms, `group_modify` 16.1 ms → 11.1 ms. Results are unchanged (checked against the previous code on 39,340 join and 3,440 rowwise and group cases, and on 7.6 million `round` values).

### Fixed

- On Polars, a float grouping key holding both `NaN` and a missing value no longer loses rows in `group_rows`, `group_data`, `group_size`, `group_indices`, or `group_split`. Both are one missing-key group, as on pandas; before, one replaced the other, so its rows disappeared and `group_indices` could return 0.
- `show(n)` and `options(preview_rows=n)` print all *n* rows. The text preview stopped at Polars' own 10-row limit, and the notebook table stopped at 25, so `show(50)` showed 10 of 50 states.
- A Polars list column on the pandas backend arrives as NumPy arrays. `unnest_longer`, `unnest`, `unnest_wider`, and `hoist` now read those cells; `unnest_longer` returned no rows. `where(is_character)` no longer selects list, struct, or date columns, and `remove_constant` no longer raises on them.
- `tidy(polars_frame, backend="pandas")` keeps Date columns as dates. They became datetimes, so `ceiling_date(x, "month")` kept `2024-01-01` instead of moving it to `2024-02-01` as lubridate does, and the result came back as a datetime.
- Text factors leaving the pandas backend (`collect()`, `collect(as_="arrow")`, `tidy(..., backend="polars")`, and pandas input on the Polars backend) are `pl.Enum` in level order, as Polars-side `fct_*` results are. They were `pl.Categorical`, which lost the order.
- On Polars, `group_trim` also drops unused levels from the column's Enum, as dplyr does; it had only updated the recorded levels.
- `excel_numeric_to_date` on a numeric column gives janitor's date for a serial just under a whole day, such as 1899-06-19 for `-194.0000000115741`. The unreleased Polars fast path divided by 86400 through its reciprocal and gave the next day for about 1 in 500 such serials.
- `group_data`, `group_rows`, `group_size`, `n_groups`, and `group_indices` of `rowwise()` with no keys have one group per row, as in dplyr; they had none.
- `build_wider_spec` on no rows types `.name` and `.value` as text and the name columns as in the data, and `pivot_wider_spec` with a spec of no rows returns the distinct id columns instead of raising `IndexError`.

## 0.7.13 — 2026-10-21

A quoted string in `mutate` is text, as in dplyr, on both backends; columns are `col("hp")` in plain Python and bare names in notebook pipes.

### Changed

- A quoted string assigned in `mutate` or `transmute` is text, as in dplyr, on both backends: `mutate(source="mtcars")` writes `"mtcars"` in every row. Polars used to read it as a column name (so `mutate(exists="Y")` raised `ColumnNotFoundError`) while pandas read it as text. Use `col("hp")` for a column in plain Python; in a notebook pipe the bare name `hp` already means the column, and `nb_export` writes it as `col("hp")`.

### Fixed

- `partial_run` accepts backtick column names such as `` mutate(`data set` = "mtcars") ``, as notebook cells already did.

## 0.7.12 — 2026-10-21

Grouped work no longer runs a Python step per group: tens of thousands of groups take milliseconds instead of seconds, with results unchanged.

### Changed

- Grouped work no longer runs a Python step per group, with results unchanged (checked against the previous code on 72 cases with ties, missing values, missing group keys, unused levels, and empty frames, and against R with about 100 groups). With 50,000 groups on 200,000 rows: grouped `fct_reorder` 62 s → 0.13 s on both backends (and `fct_infreq`, `fct_relevel`, `fct_rev` likewise); on pandas, `slice_max`/`slice_min` 13 s → 0.08 s, `slice_head`/`slice_tail`/`slice()` 5 s → 0.01 s, seeded `slice_sample`/`sample_n`/`sample_frac` 6.5 s → 0.01 s (the same rows for the same seed), `reframe` with one value per group 10 s → 0.04 s, `summarise` with `mean`, `sum(min_count=)`, `first`, `last` 0.6 s → 0.005 s, `nth` 0.8 s → 0.03 s, grouped `filter` 1.7 s → 0.01 s, and `group_nest`/`nest(by=)` 3.8 s → 0.7 s. `fct_infreq` and `fct_reorder` also no longer scan the column once per level, so high-cardinality factors are fast without groups too.
- `tests/test_function_speed.py` adds many-groups guards: exact checks that pandas grouped slices and one-value `reframe` skip the per-group loop and that grouped aggregates skip Python `apply`, plus timing checks at 20,000 groups for 13 grouped operations on both backends.

### Fixed

- Grouped `fct_relevel` checks for unknown levels without building a Python set for every group. Warning contents and factor ordering are unchanged.

## 0.7.11 — 2026-10-21

Janitor cleaning helpers, timezone helpers, datetime parsers, and `as_string`; factor joins, `bind_rows`, `complete()`, and `right_join` now follow dplyr; common conversions and parsers run natively.

### Added

- `as_string`, an alias of `as_character` for the Python name, as `is_string` is for `is_character`. Both take `true=` and `false=` for the text of logical values: `true="TRUE", false="FALSE"` for R, or `true="True", false="False"` for Python's `str()`.
- Janitor cleaning helpers on both backends: `remove_empty`, `remove_constant`, `compare_df_cols`, `compare_df_cols_same`, `describe_class`, `row_to_names`, `find_header`, and `excel_numeric_to_date`. Row-only `remove_empty` stays lazy. Dropping columns, constant checks, and header rows read the current values. Blank strings are kept, and no removal message is printed. Duplicate header names are made unique. Excel serial 60 is missing on both date systems, and `excel_numeric_to_date` defaults to UTC. A Polars frame that loses every column also loses its rows. Tests compare the shared cases with janitor 2.2.1.
- Janitor-style `get_dupes` on both backends, also available as a frame method. It keeps every duplicated record, adds `dupe_count`, supports tidyselect, sorts by count and keys, and checks globally while preserving frame grouping. Polars stays lazy. Tests compare directly with R's janitor, including missing and composite keys, factor ordering, empty results, and grouped input. Existing `dupe_count` columns must be renamed rather than silently overwritten. CI installs janitor for these R comparisons.
- Timezone helpers `with_tz` (preserve the instant) and `force_tz` (preserve the local clock) on both backends, with UTC defaults, date/naive-datetime rules, all four `roll_dst` policies, resolved-repeat handling, and millisecond through nanosecond precision. Polars plans remain lazy. R comparisons cover five-zone conversions, ordinary and half-hour DST changes, Samoa's skipped day, and resolved repeats. Deliberate subsecond differences in boundary handling are documented.
- Datetime parsers `ymd_hms`, `ymd_hm`, `mdy_hms`, and `dmy_hms` on both backends, with timezone-aware UTC defaults, named timezones, numeric offsets, English month names, compact timestamps, AM/PM, microsecond fractions, explicit truncation, and conversion diagnostics. Nonexistent daylight saving clocks fail; repeated clocks choose the later occurrence. Polars plans stay lazy. R comparisons cover all four orders, timezone conversions, truncation, and daylight saving transitions; deliberate differences in R's guessing and `ymd_hm` offset handling are documented.

### Changed

- `right_join` returns rows in dplyr's order on both backends: x's rows in order, each with its matches in y's order, then y's unmatched rows. It used to keep y's order, as pandas' `merge(how="right")` and Polars do. The set of rows is unchanged; code that relied on y's order should `arrange()` or use `left_join` with the tables swapped. R comparisons cover plain joins, `keep=True`, `multiple="first"`, `join_by()`, and missing keys.
- Faster conversions and parsers, with results unchanged. Dates, datetimes, numbers, and logical values in their common text shapes (`2024-01-07`, `2024-01-07 13:05:09`, `$1,234.50`, `true`) convert natively in Polars, and every other value still goes through the per-value rules, so problems() and strict errors are the same. On 500,000 rows: `ymd_hms` 2.4 s → 0.08 s, `parse_datetime` 1.4 s → 0.02 s, `as_date` 1.1 s → 0.01 s, `ymd`/`mdy` 0.7 s → 0.05 s, `parse_number` 0.6 s → 0.08 s, `parse_integer` 0.34 s → 0.01 s, `as_logical` 0.24 s → 0.02 s. The pandas backend uses the same path through Polars (`ymd_hms` 2.3 s → 0.12 s, `as_double` 0.43 s → 0.05 s), and its `is_na`, `pmin`, and `row_sums` no longer loop in Python (96 ms → 3 ms, 110 ms → 20 ms, 125 ms → 40 ms).
- Factor functions on Polars (`fct_infreq`, `fct_relevel`, `fct_reorder`, `fct_rev`) check only the distinct values and add the factor as a cast of its source column instead of copying every row through Python lists. The snapshot contract is unchanged: a lazy frame is computed once and then no longer reads its source. Building `scan_csv(...) >> mutate(species=fct_infreq(species))` on 500,000 rows takes 35–45 ms instead of 200 ms.
- `tests/test_function_speed.py` guards these paths: exact checks that common shapes skip the per-value rules on both backends, plus timing checks against the native operation for 15 functions.

- `as_character` writes each value as its column type shows it, the same way on both backends: `21.0` is `"21.0"` (R writes `"21"`), a `float32` `22.8` is `"22.8"` (it was `"22.7999992370605"`), `0.1 + 0.2` is `"0.30000000000000004"` instead of being rounded to 15 digits, logical values are `"true"` / `"false"` instead of `"TRUE"` / `"FALSE"`, and a datetime is `"2010-01-31 15:30:00"`, with an offset when it has a time zone. On Polars this is a native cast. For a float id, use `as_character(as_integer(id))`.
- Errors from `as_numeric` and `as_string` name the function you called, not `as_double` or `as_character`.

### Fixed

- Pandas datetime conversions preserve microsecond precision on pandas 2 as well as pandas 3. Native conversions no longer build an unused Python list before executing.

- Joins with factor keys follow dplyr on both backends. A factor joined to a text key, which raised `SchemaError` on Polars, now joins and returns the key as text; two factors with different levels join on the union of their levels, x's first; `semi_join` and `anti_join` return x's factor unchanged, which pandas had turned into text.
- `bind_rows` of factors with different levels gives one factor with every level, in order of first appearance, as vctrs does. It raised `SchemaError` on Polars and gave text on pandas. A factor bound to text is text, and a frame without the column keeps the factor (pandas had turned it into text).
- `complete()` and `expand()` use every level of a factor named outside `nesting()`, including unused ones, as tidyr does; a grouped `complete()` adds NA only in groups that have it. Missing values sort last, and on Polars a row with a missing key keeps its values instead of losing them in the join.
- Pandas reshaping preserves surviving grouping columns, group-drop settings, recorded factor levels, and existing parsing diagnostics in `pack`, `unpack`, `hoist`, `separate_longer_delim`, `separate_longer_position`, `separate_wider_delim`, `separate_wider_regex`, and `separate_wider_position`. Later summaries stay grouped, and `stop_for_problems` still catches earlier conversion failures. Regression checks cover empty inputs and duplicate indexes; R comparisons cover grouping, unused levels, and subsequent summaries on both backends.
- `parse_datetime` with `%z` reads offsets as naive UTC clocks on both backends. `2024-01-01 12:00:00+0200` is 10:00 and `2024-01-01 12:00:00-0500` is 17:00, including when one column mixes the two. A bad offset is a recorded failure. Formats without `%z` stay on the clock that was written. `ymd_hms` is still the timezone-aware parser.
- Pandas `floor_date` and `ceiling_date` keep the timezone and use local calendar boundaries, including weeks and both occurrences of a repeated hour. Flooring `2024-11-03 01:30:00.125-04:00` to a second keeps that occurrence. Ceiling the first 01:30 of an autumn fallback to an hour lands on the second 01:00; lubridate lands on 02:00 for that hour, and the other units match it.
- Polars `collect(as_="pandas")` restores imported factor levels, including unused levels, level order, and ordered status. Empty and all-missing factors keep their levels. `arrange` sorts those factors by level order, with missing values last and ties left stable. `fct_rev` and `fct_infreq(ordered=)` keep the ordered flag across that collection.
- `str_equal` treats Unicode NFC equivalents as equal on both backends, including `ignore_case` (`é` equals `e\u0301`; `É` equals it only when case is ignored). It does not follow stringi collation, so `ﬁ` stays different from `fi` and a joined emoji stays different from the same emoji without ZWJ. `boundary("character")` still splits code points rather than grapheme clusters.

## 0.7.10 — 2026-10-21

The R oracle tests now cover the 0.7.9 helpers against R, readr, and lubridate, and the differences they found are fixed. Where tidy3 still differs on purpose, the reference lists it.

### Added

- Lubridate-style components: `wday`, `yday`, `week`, `isoweek`, `quarter`, `hour`, `minute`, and `second` on both backends. Weekday labels carry ordered English levels; quarters support fiscal years and boundary dates; clock components preserve local time and fractional seconds. Expressions stay lazy on Polars. R comparisons cover calendar and ISO boundaries, weekday level order, every fiscal starting month, and daylight saving transitions.
- Factor ordering: `fct_relevel`, `fct_reorder`, `fct_infreq`, and `fct_rev` in `mutate` and `across`, on both backends. They preserve values, missing values, and unused levels; grouped mutations reconcile factor levels in group order. Vector inputs return a pandas categorical. Polars factor mutations compute a snapshot to discover Enum levels, then resume lazy piping. Tests compare values and levels against R's forcats.

### Fixed

- Source distributions include the notebook loaders, Markdown documentation, and R oracle fixture needed by the shipped tests.
- Guessed CSV columns use the same parsers as explicit `col_types`: a bad value beyond the type-guessing sample becomes missing with a row in `problems()`, or raises `ConversionError` under `strict=True`.
- String helpers around summaries, such as `str_length(first(model))`, return one value per group on pandas, matching Polars and R.
- `round` preserves Float32 columns and handles extreme `digits` without overflowing, including rounding tiny numbers past 308 decimal places.
- `pmin`, `pmax`, `row_sums`, and `row_means` work with pandas 2.0, the declared minimum version.
- Imports and `python -m tidy3` work from the repository root. Notebook setup now lives in `load.py`; both `%run load.py` and `%run tidy3.py` still enable pipes.
- `parse_number` reads text the way readr does. A minus counts only right before the number, so `"$-5"` is -5 but `"-$1,234"` and `"Price - 12"` are missing. Grouping marks are skipped anywhere in the whole part (`"1,23"` is 123), a second decimal mark ends the number (`"1.2.3"` is 1.2), and exponents work (`"1e3"` is 1000).
- `parse_double` takes an exponent (`"1e3"`) and, as in readr, fails on grouping marks, `Inf`, and `NaN`: `"1,234.5"` is missing, so use `parse_number`. `parse_integer` fails on `"1,234"`.
- `parse_double` and `parse_integer` no longer crash on empty or blank text; it is a recorded failure.
- `as_logical` takes exactly `TRUE`, `true`, `True`, and `T` (and the `FALSE` forms), as R does; `"tRuE"` and `" TRUE"` are missing.
- `round` follows R 4's rule: an exact tie goes to the even digit, and otherwise the nearer candidate to the stored number wins, so `round(2.675, 2)` is 2.67 and `round(0.15, 1)` is 0.1. Integer columns stay integers on both backends. Checked against R on about 17,000 values.
- `pmin` and `pmax` leave text such as `"nan"` alone; only a float NaN counts as missing.
- `ymd`, `mdy`, `dmy`, and `ym` read month names (`"Jan 31 2010"`) and ordinals (`"1st March 2010"`), and text with a time is missing, as in lubridate.
- `ceiling_date` on a date moves to the next boundary even from one, as lubridate does: 2010-02-01 rounded up to a month is 2010-03-01. Datetimes on a boundary stay put.
- `floor_date` and `ceiling_date` with `"week"` start the week at midnight, so a datetime with a time of day floors to 00:00.
- `years()` and `months()` added together move by `12 * years + months` in one step, as lubridate's `add_with_rollback` does: 29 February 2024 plus `years(1) + months(1)` is 29 March 2025, not 28 March.
- `months(col("n"))` and `years(col("n"))` work on the pandas backend; a missing count gives a missing date on both backends.
- `problems()` after `read_csv` (or any `compute()`) no longer reads the file again; the failures are worked out in the same run as the data. For `read_csv`, `row` counts the header line, as readr does.
- `read_csv`, `read_tsv`, and `read_delim` trim fields before guessing column types, so a padded `" 6"` in mtcars' `cyl` is a number, not text. They also read `NA` as missing by default (`na=["", "NA"]`, as readr does). `scan_csv` is unchanged.
- On the pandas backend, `summarise(m = round(mean(hp), 1))` and other helpers around a summary give one value per group instead of an error, and conversions treat NaN as missing, so `as_integer` of a float column with a missing value works.

### Changed

- On Polars, `as_double`, `as_integer`, and `parse_double` on text and number columns run as native casts, about 30 times faster on a million rows. Results, failures, and `problems()` are the same as before; values the casts cannot settle, such as non-ASCII digits, still go through the per-value rules.
- The R oracle job in CI installs lubridate and readr, so their cases run there instead of being skipped.

## 0.7.9 — 2026-10-21

Column helpers, type conversion, calendar dates, readr-style parsers, and easier file reading. Each new expression compiles on the Polars and pandas backends.

### Added

- Missing-value checks `is_na`, `is_nan`, `is_finite`, and `is_infinite`, usable in `mutate`, `filter`, and `across`. `is_na` is true for both null and NaN and is never itself missing, so `filter(is_na(body_mass_g))` keeps those rows. On Polars, `is_nan` is true only for NaN. The pandas backend stores a missing float as NaN, so both checks are true there.
- Math helpers `abs`, `floor`, `ceiling`, `round`, `pmin`, and `pmax`. `round` uses half-to-even, the same rule as R's `round()` and IEEE 754 (`round(0.5)` is 0, `round(1.5)` is 2). `pmin` and `pmax` are element-wise; `na_rm=False` turns a row missing when any input is missing.
- `row_sums` and `row_means`, with `na_rm`. `min_count` on `sum` and `row_sums` makes a row or group with too few values missing instead of 0. `sum(c_across(...))` with the defaults is unchanged.
- Type conversion: `as_numeric` / `as_double`, `as_integer`, `as_character`, `as_logical`, `as_date`, and `as_datetime`. `strict=True` (the default) raises `ConversionError`, naming the column and showing example bad values. `strict=False` turns bad values into missing. `as_integer` refuses `2.5` rather than cutting it to 2. `2` and `2.0` succeed.
- Calendar helpers in the style of lubridate: `ymd`, `mdy`, `dmy`, `ym`, `year`, `month`, `day`, `floor_date`, `ceiling_date`, `days_in_month`, `months`, `years`, `add_with_rollback`, `today`, and `now`. Adding `months()` or `years()` rolls an impossible day back to the last real day (31 January plus one month is 28 February, or 29 in a leap year). `today` and `now` default to UTC. A two-digit year needs an explicit `cutoff` (`cutoff=68` reads 68 as 2068 and 69 as 1969). A bare `year` in `filter(year == 2007)` is still the penguins column; `year(date)` is the function.
- readr-style parsers: `parse_double`, `parse_integer`, `parse_logical`, `parse_date`, `parse_datetime`, and `parse_number`, plus `locale()` for the decimal mark, grouping mark, and encoding. Parsers are permissive by default and record failures. `problems()` returns a frame of row, column, original value, and expected type. On a lazy frame that frame is its own lazy plan and does not load the file. `stop_for_problems()` collects that plan and raises.
- `read_csv`, `read_tsv`, and `read_delim` return a `TidyFrame` and share options with `scan_csv`: `col_types`, `na`, `skip`, `comment`, `trim_ws`, and `id` (the source path of each row). `col_types` can keep an id column as text so leading zeros survive. `read_csv` trims whitespace by default; `scan_csv` does not. `read_excel` and `excel_sheets` need the optional extra `pip install 'tidy3[excel]'`.
- `write_excel(..., sheets={name: frame})` writes several sheets. Text that starts with `=` is stored as text.

### Changed

- `abs`, `round`, `sum`, `min`, `max`, `any`, and `all` shadow the Python built-ins of the same names after `from tidy3 import *`, as they already did for the aggregations. See the reference.

## 0.7.8 — 2026-10-20

Clear errors for missing columns and failed previews. (0.7.7 had the same changes and was withdrawn from PyPI.)

### Changed

- A verb that uses a missing column raises `tidy3.ColumnNotFoundError` when
  the pipe is built, on both backends, naming the verb, suggesting close
  names, and listing the frame's columns. On Polars, `mutate`, `filter`, and
  `summarise` used to accept the name and fail only at display with Polars'
  internal plan text; on pandas the error was a bare `KeyError`. The new
  error is still a `KeyError` and a Polars `ColumnNotFoundError`. Checking
  resolves the lazy plan's schema only; no rows are read.
- Other errors from a verb keep their type and message; on Python 3.11+ a
  note names the verb (`tidy3: raised in mutate()`).

### Fixed

- A preview that fails while computing rows shows the error's type and first
  line instead of the whole message; the HTML table folds the full error and
  query plan under "Technical details". The error text is now escaped in
  HTML, where Polars' `<---` markers and `<` in data could garble it.

## 0.7.6 — 2026-10-20

Previews show the whole frame's shape.

### Fixed

- A preview's shape is the whole frame's, not the preview's: a filter that
  keeps 30 rows shows `shape: (30, 2)` above its first 10 and ends with
  `… 20 more rows`, as a tibble does. It used to say `(10, 2)`. Counting
  runs the query to the end; `options(preview_count=False)` skips it and
  shows `?` for the rows. `peek()` shows the same shape.

## 0.7.5 — 2026-10-20

String functions from R's stringr.

### Added

- stringr: every `str_*` function, plus `word`, `invert_match`, and the
  pattern modifiers `fixed()`, `regex()`, `coll()`, and `boundary()`. They
  work in any verb on both backends, so `cars >> filter(str_starts(name,
  "chev"))` reads as in R. Given a list or Series, a helper runs at once and
  returns a list; `str_subset`, `str_which`, `str_sort`, `str_order`,
  `str_unique`, and `str_view` work on lists only. `str_locate` and
  `str_match` return struct columns for `unnest_wider()`. The reference lists
  where tidy3 differs from R.

## 0.7.4 — 2026-10-14

Clearer errors from broken pipes in notebooks.

### Fixed

- A multi-line `>>` pipe with a mistake in it (a missing argument, an extra
  closing bracket) used to fail with `IndentationError: unexpected indent` at
  its first `>>` line. The notebook transformers now join the pipe's lines so
  Python reports the real error at the right line. A bracket left open at the
  end of a cell gives Python's own `incomplete input`, as in any cell.
  `nb_export` leaves broken cells as written.

## 0.7.3 — 2026-10-14

Scaling helpers for handing features to scikit-learn.

### Added

- `scale(x, center=True, scale=True)` standardises a column like R's
  `scale()`, and `rescale(x, to=(0, 1))` min-max scales it like
  `scales::rescale()`. Both skip missing values, work by group, and return a
  plain column. A constant column gives NaN from `scale()` and the middle of
  `to` from `rescale()`, as in R. The README's k-means example uses `scale`.

## 0.7.2 — 2026-10-10

Plot layers on their own lines in exported notebooks, and a README that
loads the extension in a cell of its own.

### Fixed

- The README loads `%load_ext tidy3.jupyter` in a cell of its own. Its first
  example loaded the extension and used multi-line pipes in the same cell,
  which fails with a SyntaxError in a fresh notebook: IPython reads a whole
  cell before running any of it. The README test no longer loads the
  extension itself, so it catches this.

### Changed

- `nb_export` and `python -m tidy3 run` join a plot's `+ geom_*()` lines to
  the `ggplot(...)` above them, as plot3 0.6.6 does in notebooks. tidy3
  needs plot3 0.6.6 or newer.

## 0.7.1 — 2026-10-09

A plain `pip install tidy3`, plotting and Excel included.

### Changed

- `pip install tidy3` is the whole install: plot3 (for `>> ggplot()`) and
  XlsxWriter (for `write_excel`) are regular dependencies, and the
  `jupyter`, `plot3`, and `excel` extras are gone. Notebooks already have
  IPython.

## 0.7.0 — 2026-10-09

`compute()`, notebook masking fixes, R's `head()`, and a README that shows
notebook pipes, plots, and a scikit-learn handoff.

### Added

- `compute()` runs the plan so far and keeps piping from the in-memory
  result, as dplyr's `compute()` does. `scan_csv(url) >> drop_na() >>
  compute()` downloads and cleans once; later collects, plots, and
  `to_numpy()` calls reuse it instead of re-running the scan. Groups and
  rowwise state carry over. Also a method: `frame.compute()`.
- `examples/cluster_penguins.py`: tidy3 prepares the Palmer penguins,
  `to_numpy()` hands them to scikit-learn's k-means, and the labels come
  back into the pipe for counting and plot3 figures.

### Fixed

- In notebooks, a variable made earlier in the same cell stays a variable
  inside a verb. `kmeans = KMeans(...).fit(X)` followed by
  `mutate(cluster = kmeans.labels_)` in one cell used to read `kmeans` as a
  column. Scripts run with `python -m tidy3 run` already worked this way.
- In notebooks, `all_of(features)` and `any_of(features)` read the variable
  `features`, as in tidyselect, instead of treating its name as a column.
- On the Polars backend, `mutate(c = [0, 1, 0])` gives one value per row,
  as in dplyr and on the pandas backend. A Python list or tuple used to put
  the whole list in every row.

### Changed

- `head()` returns 6 rows by default, as R's `head()` does (it was 10).
- The `plot3` extra needs plot3 0.6.0 or newer, for the README's plots
  (`scales="free_y"`, `aes(colour = factor(cluster))` in notebooks).
- `glimpse(penguins)` works as in dplyr: it prints the frame and returns
  it. `penguins >> glimpse()` and `glimpse(5)` work as before.
- The README leads with a penguins pipeline, its typed table, plot3
  figures, and a scikit-learn handoff, with every dataset read from the
  web. The notebook, CRAFT, reference, and benchmark material moved to
  `docs/`, and the README test runs those pages' examples too.

## 0.6.0 — 2026-10-09

dplyr-style join and column naming in Python spelling, the rest of tidyr's
pivot and list options, pivot specs, and Polars-native verbs throughout.

### Added

- `pivot_wider()` gains tidyr's `names_sep`, `names_glue` (`"{key}_{.value}"`),
  `names_vary`, `names_expand`, `id_expand`, and `unused_fn`;
  `pivot_longer()` gains `names_transform` and `values_transform` (a function
  or a dict by column, e.g. `{"week": int}`). Calls that use none of them run
  exactly as before, and on Polars every option stays in Polars (lazy).
- tidyr's smaller list and splitting options: `separate_wider_delim()`'s
  `names_sep` and `cols_remove`; `unnest_longer()`'s `indices_include` and
  `transform`; `unnest_wider()`'s `transform`; `hoist()`'s `remove` and
  `transform`; and `nest()`'s `names_sep` (tidyr's `.names_sep`).
- tidyr's pivot specs: `build_longer_spec()` / `pivot_longer_spec()`,
  `build_wider_spec()` / `pivot_wider_spec()`, and `check_pivot_spec()`.
  Build the mapping as a table, edit it (rename columns, drop
  combinations), and pivot with it; on Polars the pivots stay lazy.

### Changed

- Joins name columns found in both tables like dplyr, in Python spelling:
  both copies get a suffix, `_x` and `_y` by default (dplyr's `.x`/`.y`),
  and `suffix=` takes a pair, e.g. `suffix=("_old", "_new")`. `keep=True`
  gives `id_x`/`id_y`, and a suffixed name that already exists gets the
  suffix again (`v_x_x`), as in dplyr. This applies to left, right, inner,
  full, and cross joins. Before, only the right copy was renamed
  (`v`, `v_right`); `suffix=("", "_right")` gives those names back.
- `suffix="_r"` (one string) still renames only the right copy but is
  deprecated; it warns and will be removed.
- `hoist()` works like tidyr's: new columns go just before the list column,
  and the fields it pulls out are removed from it (the column goes when
  nothing is left). `remove=False` keeps them. Before, it added columns at
  the end and left the list column unchanged.
- `bind_cols()` names duplicated columns `x_1`, `x_3` (each column's
  position), dplyr's `x...1`, `x...3` in Python spelling. A repaired name
  that would clash with an existing column gets its position again.

### Fixed

- On the Polars backend, these verbs now work in Polars instead of
  converting to pandas and back: the `separate_wider_*` and
  `separate_longer_*` verbs, `hoist()`, `expand_grid()`, `crossing()`,
  `full_seq()` in `expand()`/`complete()`, the group functions
  (`group_data()`, `group_keys()`, `group_rows()`, `group_size()`,
  `group_indices()`, `group_trim()`), the group callbacks
  (`group_split()`, `group_map()`, `group_modify()`, `group_walk()`),
  `nest_by()`, `drop = FALSE` empty groups in `summarise()`/`count()`, and
  `setequal()`. `tf.ggplot()` hands the table to plot3 directly.
- pandas backend: a right-table whole-number column stays whole numbers when
  a join leaves some rows unmatched (it became decimals).

## 0.5.1 — 2026-10-08

dplyr's empty groups for `drop = FALSE`, and 1-based group row positions.

### Changed

- `group_rows()` and `group_data()`'s `.rows` give 1-based row positions, as
  in dplyr and the rest of tidy3 (`slice(1)` is the first row). They were
  0-based in 0.4.0 and 0.5.0, so each position is now one higher.

### Fixed

- `summarise()` and `count()` after `group_by(..., drop=False)` with several
  keys follow dplyr's empty-group rule: an unused factor level adds one empty
  group (later keys missing) instead of a full grid of combinations. On
  Polars, groups with a missing key were dropped from the result; on pandas,
  `summarise()` ignored `drop=False` and `count()` used a full grid. All now
  match dplyr, and counts stay whole numbers.

## 0.5.0 — 2026-10-08

Notebook bare names follow dplyr's data mask: columns win.

### Changed

- Notebook bare names follow dplyr's data mask: a column beats a notebook
  variable with the same name, in expressions (`filter(x > limit)`) and in
  selections (`select(x)`, `count(g)`, `select([g, x])`). A name with no
  matching column is still your variable, and `select(cols)` with a list
  of names still works. Within one `mutate()` or `summarise()`, columns
  made by earlier assignments count. Before, any notebook variable won.

### Added

- `env.x`: the notebook variable `x` even when a column is named `x`, like
  dplyr's `.env$x`.

### Fixed

- After a notebook rebinds a tidy3 function name (`n = 7`), calls inside verbs
  still use the function (`n()` counts rows), as R skips non-functions.
- Lambda parameters and comprehension variables named like builtins
  (`lambda n: n > 0`) are no longer read as columns.

## 0.4.0 — 2026-10-08

dplyr 1.2 parity, dplyr's group functions, and more tidyr verbs, each checked
against R.

### Added

- dplyr 1.2's recoding helpers: `recode_values()` and `replace_values()`
  (cases as `(values, replacement)` pairs, or a `from_=`/`to=` lookup table;
  `None` matches missing values), `replace_when()`, and `when_any()` /
  `when_all()` with `na_rm=`. `recode_values(unmatched="error")` raises when
  a value has no case.
- `case_when(unmatched="error")`, as in dplyr 1.2: raises when a row matches
  no case (a missing condition counts as unmatched).
- dplyr's group functions: `group_data()`, `group_keys()`, `group_rows()`,
  `group_size()`, `group_indices()`, `groups()`, `group_trim()`,
  `group_walk()`, and `nest_by()`; `group_vars()` and `n_groups()` now also
  take a frame (`n_groups(df)`, `df >> n_groups()`). Empty groups for unused
  factor levels follow dplyr's `.drop = FALSE` rule. Row positions are
  0-based.
- `order_by()` and `with_order()`: compute a window expression as if the
  rows were sorted (`order_by("year", col("value").cum_sum())`), with results
  in the original row order; missing keys sort last and ties keep their order.
- tidyr's `expand_grid()`, `crossing()`, `uncount()`, and `full_seq()`.
  `expand()` and `complete()` take named values, as in
  `complete("country", year=full_seq("year", 5))`; the values keep the
  column's type (R would turn a whole-number column into decimals).
- tidyr's `separate_wider_regex()`, `separate_wider_position()`,
  `separate_longer_position()`, `chop()`, `unchop()`, and `unnest_auto()`.
  `separate_longer_position()` keeps a missing value as one missing row
  (tidyr 1.3.2 errors on it).

### Changed

- `group_split()`, `group_map()`, `group_modify()`, and `group_nest()`
  return groups in dplyr's order, sorted by key with missing last, instead
  of the order they first appear.

- `summarise()` follows dplyr 1.2: each expression must give one value per
  group, otherwise it raises "`r` must be size 1, not 2 … use reframe()".
  Polars used to return a list column silently and pandas failed with an
  unrelated error. `x * 2` still works where every group has one row.
- `pivot_wider()` handles values that are not uniquely identified like
  tidyr: they become list columns with tidyr's warning and advice, instead
  of a low-level Polars or pandas error. `values_fn="list"` asks for list
  columns without the warning.

### Fixed

- pandas backend: comparisons with a missing value give a missing result and
  `&` / `|` use three-valued logic, as in R and Polars. `mutate(big = x > 3)`
  used to give `False` where `x` is missing.
- `case_match()` matches missing values with `None`, like `NA ~` in R.
- In notebooks, columns named like Python builtins or tidy3 helpers (`id`,
  `type`, `max`, `min`, `sum`, `n`, …) work as bare names wherever only a
  column makes sense: `filter(id > 1)`, `select(id, type)`,
  `arrange(desc(id))`, `mutate(z = max * 2)`, `summarise(m = mean(max))`.
  They used to stay Python's `id`, `type`, `max` and fail. Functions passed
  as functions (`across(everything(), mean)`) and names you assign in the
  notebook are unchanged.

## 0.3.0 — 2026-10-08

First release on PyPI: `pip install tidy3`.

### Added

- Column-name helpers: `clean_names()` and `make_clean_names()` (janitor
  style), and `set_names()` for renaming every column at once.
- Inspection: `names()`, `colnames()` (paste-ready for `select()`), `dim()`,
  `columns`, `dtypes`, and `summary()` / `describe()` with full statistics;
  all of them pipe like `%>% f()` in R.
- R-style column masking in Jupyter: bare names, backticks for spaced names,
  and ``mutate(`new col` = expr)``.
- tidyselect: `!` / `-` negation, set operations between selectors, and
  `where()` with dtype predicates.
- `nb_export()` and `python -m tidy3 export` / `run`: turn an R-style
  notebook into a plain Python script.
- A standalone `%run tidy3.py` loader that also works on CRAFT remote kernels.
- Multi-line `>>` pipes in Jupyter no longer need the whole cell to
  themselves: a cell can hold comments (before, between, and after steps),
  imports and other statements, and several pipes. Error line numbers match
  the cell as typed. `nb_export()` reads these cells too.
- Every README example runs as written: each section builds the small
  tables it uses (no downloads), and `tests/test_readme.py` runs them all
  in order in an IPython shell, computing every table and plot they make.
- `separate_wider_delim(too_many="drop")`.
- The `plot3` extra now installs plot3 from PyPI.

### Changed

- `separate_wider_delim()` now stops with an error when a value has too few
  pieces, as tidyr does. Pass `too_few="align_start"` for the old behaviour.

### Fixed

- `separate_wider_delim()`: missing values stay missing (they became the text
  `"nan"`), `too_many="merge"` keeps the extra pieces in the last column
  (it dropped them), `too_few="error"` raises, and the new columns take the
  place of the split column.
- With pandas 2, `where(is_boolean)` and `where(is_categorical)` no longer
  pick text columns.
- With pandas 2, `fill()` no longer changes the frame it was given.
- `aes()` bare names work in the same notebook cell that imports plot3.
- Bare-name masking leaves lambda parameters and comprehension variables
  alone: `if_any(ends_with("_score"), lambda x: x > 0)` no longer reads a
  column named `x`.
- `pivot_longer(cols_vary="slowest")` keeps its row order on Polars 2.
- Multi-line `>>` pipes work under SolveIt and CRAFT, and Jupyter shell
  escapes (`!pip`, `!whoami`) are never rewritten.

## 0.2.0 — 2026-07-18

pandas backend for side-by-side comparison with datar, backend-neutral
expressions, and `tidy3.bench`.
