# Package review — 2026-10-21

This review covers the current working tree: 0.7.10 plus the unreleased
datetime parsers, timezone helpers, and `get_dupes`. This records the original
review findings; subsequent fixes are noted beside the relevant finding.

The main dplyr/tidyr workflow surface is broad. The most useful next work
is to close correctness gaps in existing operations, then expand janitor
and forcats helpers. Full tidyverse parity has not been established by
function-name coverage or the existing test suite.

## Evidence and scope

- Re-ran the full suite with the temporary R library enabled:
  `R_LIBS=/private/tmp/tidy3-r-library .venv/bin/python -m pytest -q -rs`.
  Result: **1,099 passed, 9 skipped**, with 17 IPython-directory warnings.
- Five skips are web-dependent README examples; four are explicit
  backend exceptions: three reader cases and the pandas NA/NaN distinction.
- Inspected all 351 exported names, public signatures, implementations,
  test classification, documented differences, and materialization paths.
  Not all exports are functions; classes and aliases are included.
- Compared the API with installed R exports and current official references.
  Local reference versions: dplyr 1.2.1, tidyr 1.3.2, stringr 1.6.0,
  forcats 1.0.1, lubridate 1.9.5, readr 2.2.0, janitor 2.2.1,
  purrr 1.2.2, tibble 3.3.1, and hms 1.1.4.
- Ran additional small reproductions against both backends and R.
  Those checks expose the findings below despite the green suite.
- No production code was changed during this review. This is not an
  exhaustive behavioral or performance certification of every argument.

## Confirmed correctness findings

### 1. High priority: pandas reshape operations discard frame metadata

**Follow-up: fixed in the working tree.** The fix covers the four reported
helpers plus `hoist`, `separate_wider_regex`, `separate_wider_position`, and
`separate_longer_position`, which had the same reconstruction bug. Regression
checks cover surviving groups, group-drop settings, recorded levels, original
parse failures, empty inputs, duplicate indexes, and later grouped summaries.
R comparisons cover all eight helpers on both backends.

Confirmed for `pack`, `unpack`, `separate_wider_delim`, and
`separate_longer_delim`. A frame grouped by `g`, with one recorded
`parse_integer` failure, comes back with `groups=None` and zero parsing
problems. The Polars equivalents retain `groups=['g']` and the failure.
R tidyr retains the grouping for these operations.

Impact: a later `summarise` can aggregate the entire table instead of
each group, and `stop_for_problems` can miss an earlier conversion failure.

The pandas paths reconstruct frames with `tidy(pdf, ...)` rather than
carrying the original metadata through `_with_pdf`.
Locations: `src/tidy3/tidyr.py:2408`, `src/tidy3/tidyr.py:2490`,
`src/tidy3/tidyr.py:3104`, and `src/tidy3/tidyr.py:3119`.

Acceptance check: on grouped input with a recorded parse failure, run each
operation and assert grouping, group-drop settings, category metadata, and
the original parsing problems survive. Check a subsequent grouped summary.

### 2. High priority: formatted datetime parsing fails on valid offsets

**Follow-up: fixed in the working tree.** `%z` is converted to a naive UTC
clock on both backends, the same contract as `as_datetime`. Mixed offsets,
a single offset, `Z`, a half-hour offset, missing values, strict mode, and
grouped use are covered. `ymd_hms` is unchanged and still timezone-aware.

Reproduction input:

```python
frame = tidy({"x": ["2024-01-01 12:00:00+0200", "2024-01-01 12:00:00-0500"]})
frame >> mutate(at=parse_datetime("x", "%Y-%m-%d %H:%M:%S%z"))
```

- Polars raises a schema error: the declared output is naive datetime,
  but the returned batch is UTC-aware.
- Pandas raises a timezone conversion error on the mixed offsets.
- R readr returns the instants `10:00 UTC` and `17:00 UTC`.

`_with_format` retains timezone-aware Python values, while output dtype
handling assumes the older datetime contract. Locations:
`src/tidy3/convert.py:509`, `src/tidy3/convert.py:1016`, and
`src/tidy3/convert.py:1063`.

Acceptance check: normalize `%z` inputs consistently, preserve both
instants, and test mixed offsets, a single offset, missing values, strict
mode, and grouped use. Choose and document the formatted parser's output
timezone contract explicitly; do not infer it from the newer order parsers.

Reference: [readr datetime parsing](https://readr.tidyverse.org/reference/parse_datetime.html).

### 3. High priority: pandas calendar rounding changes timezone semantics

**Follow-up: fixed in the working tree.** Pandas now rounds on the local
calendar and keeps the zone, including both repeated-hour occurrences.
R comparisons cover every unit except hour. Ceiling the first 01:30 of an
autumn fallback to an hour is the second 01:00 here and 02:00 in lubridate;
that difference is documented, and both backends agree.

For `2024-03-11 12:00 America/New_York`:

| Operation | Pandas result | R and Polars result |
|---|---|---|
| `floor_date(..., "month")` | March 1, timezone dropped | March 1 midnight, New York |
| `floor_date(..., "quarter")` | January 1, timezone dropped | January 1 midnight, New York |
| `floor_date(..., "year")` | January 1, timezone dropped | January 1 midnight, New York |
| `floor_date(..., "week")` | March 9 at 23:00 EST | March 10 at 00:00 EST |
| `ceiling_date(..., "month")` | April 1, timezone dropped | April 1 midnight, New York |

Flooring a resolved `2024-11-03 01:30:00.125-04:00` to a second also
raises an ambiguity error on pandas. R/Polars handle the tested occurrence.

Two causes: conversion to pandas periods drops zones, and subtracting
fixed 24-hour durations is not calendar-day arithmetic across DST.
Locations: `src/tidy3/dates.py:638` and `src/tidy3/dates.py:653`.

Acceptance check: verify zone, local boundary, and instant across spring
and autumn transitions, both repeated-hour occurrences, and all units.
The reference already documents part of this gap; the week-boundary
counterexample extends it.

Reference: [lubridate rounding](https://lubridate.tidyverse.org/reference/round_date.html).

### 4. Medium priority: imported factor metadata is lost on collection

**Follow-up: fixed in the working tree.** Polars collection to pandas restores
recorded levels, level order, and the ordered flag, including empty and
all-missing factors and integer levels. Factor expressions keep that flag
when they build an Enum. A later string overwrite is left as text.

For a pandas ordered categorical containing `['a', 'b']` with levels
`['b', 'a', 'unused']`, the Polars round trip through
`tidy(frame).collect(as_="pandas")` returns categories `['a', 'b']`
and `ordered=False`. The pandas backend retains the original metadata.

The frame records the original levels in `_category_levels`, but ordinary
collection does not restore them. This affects downstream modeling,
plotting, and unused-level reporting even when values look unchanged.
Location: `src/tidy3/frame.py:1030`; import metadata is recorded near
`src/tidy3/frame.py:1470`.

Acceptance check: ordinary input and backend round trips must preserve
unused levels, level order, and ordered/unordered status. Include empty
and all-missing factors. The explicit `fct_*` mutation path already uses
Enum levels and does not eliminate this import/collection gap.

### 5. Medium priority: general Polars arrange ignores imported factor order

**Follow-up: fixed in the working tree.** Polars `arrange` ranks recorded
factors by level index, including `col()`, `desc()`, and `by_group`. Missing
and unknown values stay last, and ties keep input order. `get_dupes` uses
that path.

For the same factor, `arrange("f")` returns `['a', 'b']` on Polars;
pandas and dplyr return `['b', 'a']` according to factor levels.
Location: `src/tidy3/verbs.py:1698`.

`get_dupes` already has a local factor-sort adaptation, but general
`arrange` does not. Shared sorting should respect the recorded level order
while keeping missing values last and ties stable.

### 6. Medium priority: Unicode equality differs from stringr

**Follow-up: NFC equality is fixed in the working tree.** `str_equal`
normalizes both sides to NFC after lowercasing when `ignore_case` is set.
`boundary("character")` still splits code points rather than grapheme
clusters. `str_equal` is not stringi collation: a joined family emoji is
not equal to the same emoji without ZWJ, and `ﬁ` is not equal to `fi` even
with `ignore_case`. Those limits are documented in the reference.

`str_equal(["é"], "e\u0301")` returns `[False]`; R stringr returns `TRUE`.
The strings use precomposed versus combining-accent encodings.
Both implementations compare code points directly, optionally after
lowercasing. Locations: `src/tidy3/stringr.py:984` and
`src/tidy3/stringr.py:1197`.

Related verified boundary gap: splitting `e\u0301` with
`boundary("character")` produces two pieces here and one grapheme in R;
a joined family emoji likewise splits into multiple code points here.
The existing title-case example `"don't" -> "Don'T"` is already documented.

Acceptance check: canonical Unicode equivalents, accented text, joined
emoji, and case handling should either match stringr or have an explicit
documented limitation. Locale collation is a separate broader decision.

Reference: [stringr Unicode equality](https://stringr.tidyverse.org/reference/str_equal.html).

## Coverage by family

These are workflow gaps, not percentages of R namespace exports. R exports
also contain superseded APIs, database internals, methods, and reexports;
some useful behavior is already available under Python methods.

| Family | Present | Most useful missing or partial coverage |
|---|---|---|
| dplyr | Main verbs, joins and join specifications, per-operation groups, ranking, row mutation, group context, set operations | Semantic consistency and metadata preservation are more pressing than more core verb names. Portable quantile summaries are missing: `col("x").quantile(.5)` works on Polars and raises `NotImplementedError` on pandas. |
| tidyr | 34 module exports including pivots/specs, missing values, nesting, packing, rectangling, modern separation | `names_ptypes`, `values_ptypes`, and `names_repair` controls; callable/per-value-column `pivot_wider(values_fn=...)`; broader multi-column unnest/separation contracts. Grouped pandas reshape behavior needs correction first. |
| forcats | Four ordering functions: `fct_relevel`, `fct_reorder`, `fct_infreq`, `fct_rev` | `fct_recode`, `fct_collapse`, `fct_lump_n`, `fct_lump_prop`, `fct_lump_min`, `fct_drop`, `fct_expand`, `fct_inorder`, and missing-value/level conversion. `as_factor` and `fct_reorder2` are also useful. |
| janitor | `clean_names`, `make_clean_names`, `get_dupes`, `remove_empty`, `remove_constant`, `row_to_names`, `find_header`, `compare_df_cols`, `compare_df_cols_same`, `describe_class`, `excel_numeric_to_date` | `tabyl`, `adorn_totals`, `adorn_percentages`, and `adorn_pct_formatting`. |
| stringr | 55 module exports; broad everyday detect/replace/split/format coverage | `str_rank`, `str_to_snake`, `str_to_camel`, `str_to_kebab`; locale-aware comparison/case conversion, graphemes, and regex portability. Options such as `str_dup(sep=...)` are absent even where the function name exists. |
| lubridate / hms | Date orders, four datetime orders, components, fiscal quarters, floor/ceiling, timezone helpers, month/year rollback | `round_date`, `isoyear`, `leap_year`, `make_date`, `make_datetime`, `ymd_h`, `mdy_hm`, `dmy_hm`, `parse_date_time`, time-of-day parsing, and interval/duration/period workflows (`interval`, `time_length`, `days`, `hours`, `ddays`, etc.). |
| readr / readxl | CSV/TSV/delimited readers and diagnostics, common expression parsers, Excel reader bridge; frame writer methods | `read_csv2`, `read_fwf`, `read_table`, `read_lines`, `parse_time`, `parse_factor`, `type_convert`, richer column specs and locale controls. Writers exist as frame methods, so missing free `write_csv` is an API convenience gap rather than missing CSV output. |
| purrr | Ordinary Python iteration and group callbacks can cover some use cases; no dedicated mapping API | `map`, typed mapping, `map2`, `pmap`, `pluck`, `keep`, `discard`, `reduce`, `safely`, `possibly`, and typed list-column mapping across both backends. A Polars expression escape hatch does not provide pandas parity. |
| tibble helpers | `tidy`/`TidyFrame` cover construction and display; `bind_rows`/`mutate` cover parts of common recipes | `enframe`, `deframe`, `rowid_to_column`, and `add_row` conveniences. These are lower priority than data-cleaning and factor helpers. |

Reference comparisons: [dplyr](https://dplyr.tidyverse.org/reference/index.html),
[tidyr longer](https://tidyr.tidyverse.org/reference/pivot_longer.html),
[tidyr wider](https://tidyr.tidyverse.org/reference/pivot_wider.html),
[forcats](https://forcats.tidyverse.org/reference/index.html),
[janitor](https://sfirke.github.io/janitor/reference/index.html),
[stringr changes](https://stringr.tidyverse.org/news/index.html),
[lubridate](https://lubridate.tidyverse.org/reference/index.html),
[hms](https://hms.tidyverse.org/reference/index.html),
[readr](https://readr.tidyverse.org/reference/index.html),
[purrr](https://purrr.tidyverse.org/reference/index.html), and
[tibble](https://tibble.tidyverse.org/reference/index.html).

The plotting bridge delegates to the separate plot3 package; this review
does not establish that package's ggplot2 coverage. SQL/database backends,
R metaprogramming syntax, and deprecated scoped verbs are separate scope
decisions, not the next missing Python data-cleaning helpers.

## Partial behavior and performance that need explicit contracts

- `coll` is currently literal matching without locale collation. Case
  helpers lack R's locale arguments. Rust regex on Polars and Python regex
  on pandas have different capabilities; look-around is a concrete example.
- Parsing conventions deliberately differ in places: two-digit years need
  explicit cutoffs, integer conversion refuses fractional values, and pandas
  cannot generally distinguish missing floats from NaN. Keep these declared
  differences distinct from newly reproduced failures.
- Factor mutation on Polars computes a complete snapshot to discover levels
  (`src/tidy3/forcats.py:212`). Some reshaping discovers output schema from
  values, and several string paths call Python per row. Broad API coverage
  therefore does not imply every helper is lazy or suitable for large scans.
- Only named reductions are accepted by `pivot_wider(values_fn=...)`;
  callable or per-column specifications need a separate implementation.
- Existing R normalization compares factor values/types but not all factor
  metadata, and formats timestamps without fractional seconds in general
  frame comparisons. Specialized tests cover some exceptions, but the new
  counterexamples show where general invariants are still needed.

## Recommended implementation order

1. **Correctness first:** reshape metadata, formatted offset parsing,
   timezone rounding, factor round trips, factor sorting, and NFC equality
   are fixed in the working tree. `boundary("character")` and stringi
   collation remain documented limits. The reproductions above are in
   dual-backend and R regression tests, except the one documented hour
   ceiling that differs from lubridate and the documented Unicode limits.
2. **Small cleaning helpers:** `remove_empty`, `remove_constant`,
   `compare_df_cols`, then `row_to_names` and Excel serial-date conversion.
   Document schema discovery when header or column removal depends on data.

   **Follow-up: added in the working tree.** `remove_empty`, `remove_constant`,
   `compare_df_cols`, `compare_df_cols_same`, `describe_class`, `row_to_names`,
   `find_header`, and `excel_numeric_to_date` run on both backends. Row-only
   `remove_empty` stays lazy. Dropping columns, constant detection, and
   `row_to_names` read the current rows to decide the schema. Blank strings
   are not empty. `compare_df_cols_same` does not print. Duplicate header
   names are made unique. A missing header cell becomes the string `"NA"`,
   including a NaN. janitor leaves that name missing, and
   `as.character(NaN)` is `"NaN"`. Each header cell is
   converted on its own, while janitor 2.2.1 coerces the whole row with
   `unlist` first. Excel serial 60 is missing on both date systems, matching
   janitor 2.2.1, and the Mac origin still shifts serials before 60.
   `excel_numeric_to_date` defaults to UTC and does not warn. Pandas float
   columns cannot tell null from NaN, so an all-missing float is constant
   there and is kept on Polars when null and NaN are both present.
   `cutoff` other than 1 raises when the denominator is zero. A Polars frame
   that loses every column also loses its rows. Multi-frame class comparisons
   sort names by code point.
3. **Factor value workflows:** `fct_recode`, `fct_collapse`, lumping,
   dropping/expanding levels, and explicit missing levels.

   **Follow-up: added in the working tree.** Recoding, collapsing, and
   lumping, then `fct_drop`, `fct_expand`, `fct_inorder`, `as_factor`,
   `fct_reorder2`, `fct_na_value_to_level`, and `fct_na_level_to_value`,
   checked against forcats 1.0.1.
4. **Common reporting:** `tabyl`, totals, and percentages. Keep numerical
   counts/percentages usable before adding formatted presentation strings.

   **Follow-up: added in the working tree.** `tabyl` (one and two
   variables), `adorn_totals`, and numeric `adorn_percentages`. Formatted
   strings (`adorn_pct_formatting`, `adorn_ns`) and three-variable tables
   remain.
5. **Date/time completeness:** time-of-day parsing, remaining common orders,
   `round_date`, constructors, then a deliberate calendar-period versus
   elapsed-duration and interval design.

   **Follow-up: rounding and constructors added in the working tree.**
   `round_date`, `make_date`, `make_datetime`, `isoyear`, and `leap_year`,
   checked against lubridate 1.9.5, with the differences in the reference.

   **Follow-up (parsing): added in the working tree.** `ymd_h`, `mdy_hm`,
   `mdy_h`, `dmy_hm`, and `dmy_h` complete the order family. `parse_date_time`
   reads several orders with lubridate's training: formats ranked by field
   count and by how many values each reads, on a prime-row sample for long
   columns. It supports `tz`, `truncated`, and `exact`. `parse_time` follows
   readr's default and explicit formats and returns a Polars `Time`, so
   readr's durations of a day or more are recorded as problems. All are
   checked against lubridate 1.9.5 and readr 2.2.0. Two-digit years still
   need `cutoff`, orders need a year, and negative truncation, custom
   `select_formats`, and week or day-of-year letters remain unsupported.

   **Follow-up:** the span design was approved on 2026-10-25 in
   [design-time-spans.md](design-time-spans.md).

   **Follow-up (spans phase 1): added in the working tree.** Durations, day and clock periods, zone-aware period arithmetic (skipped clocks missing for `+`, the transition for `add_with_rollback`), and `roll_to_first`, checked against lubridate 1.9.5; intervals (phase 2) and period columns (phase 3) remain.
6. **Remaining portability:** string case/ranking helpers, quantiles,
   advanced pivot arguments, and typed list-column mapping. Use Python's
   existing collection tools where a new purrr wrapper adds little value.

Track coverage as **name present**, **argument contract**, **backend parity**,
**R comparison**, **metadata preservation**, and **execution cost**. A single
function-count percentage would hide the most consequential gaps found here.
