# Changelog

All notable changes to tidy3. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may change behaviour, and those changes are listed
under **Changed**.

## Unreleased

The R oracle tests now cover the 0.7.9 helpers against R, readr, and lubridate, and the differences they found are fixed. Where tidy3 still differs on purpose, the reference lists it.

### Fixed

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
