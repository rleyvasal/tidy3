# Changelog

All notable changes to tidy3. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may change behaviour, and those changes are listed
under **Changed**.

## Unreleased

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
