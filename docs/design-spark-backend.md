# Design: running tidy3 pipelines on Spark

Status: **experimental, 2026-10-29.** `TidyFrame.to_spark()` and the
`tidy3[spark]` extra are in place; coverage grows from the list in
[What is not translated yet](#4-what-is-not-translated-yet).

Every result below was produced with **tidy3 0.12.4**, **Polars 1.42.1**,
**PySpark 4.2.1** in local mode, Java 21 (Temurin) and Python 3.13, on a
24-core Linux machine.

## Summary

```python notest
spark_df = (
    scan_parquet("s3://bucket/penguins/*.parquet")
    >> drop_na("sex")
    >> group_by("species", "sex")
    >> summarise(n=n(), mass_kg=(mean("body_mass_g") / 1000).round(2))
    >> arrange(desc("mass_kg"))
).to_spark(spark)
```

`to_spark()` returns a lazy PySpark DataFrame that computes the pipeline,
in tidy3's row order. A step with no Spark translation raises
`SparkExportError` (an `ExportError`) naming the step, before Spark runs
anything.

The rule behind the design: **translate the query tidy3 compiled, not the
verbs.** tidy3 already writes dplyr's rules into its lazy Polars query (a
`mean` with a missing value is missing, `arrange` puts missing values last,
summaries are sorted by their groups), and `to_polars_code()` already reads
that query from Polars' serialized plan. `to_spark()` reads the same plan
and builds Spark calls, so no verb is written twice and every verb tidy3
compiles to supported plan steps runs on Spark unchanged.

## 1. Alternatives considered

**Narwhals.** Narwhals offers a Polars-like API over Polars, pandas, DuckDB,
PySpark, Dask and Ibis. tidy3 would build Narwhals expressions where it
builds Polars ones today. That means rewriting the verb and helper modules (about 22,000
lines that build Polars expressions directly) on a smaller API, and
Narwhals' lazy backends have no row order, which dplyr's results depend on
(`distinct` keeps the first row, `slice_head`, `row_number`, `cumsum`, stable
`arrange`). Narwhals does accept compound aggregations such as
`(mean(x) / 1000).round(2)` on Spark and DuckDB; only its pandas and Dask
backends refuse them.

**A second backend.** tidy3 had a pandas backend until 0.8.0, with every
verb written twice; it was removed because the two drifted and the pandas
side was 2–25× slower. A plan translator avoids the second copy.

## 2. How the translator works

`tidy3/spark_export.py` walks the serialized plan, like the Polars printer
in `polars_export.py`, with one handler per plan step (`p_Scan`,
`p_GroupBy`, `p_Join`, ...) and per expression (`e_Agg`, `e_Over`,
`e_Function`, ...).

**Row order.** Polars keeps row order; Spark does not. Each scan numbers
its rows (`monotonically_increasing_id`, which follows file order within a
file). Each translated step carries the ordering of its rows as a list of
hidden key columns. A stable sort puts its keys in front of that list, so
tidy3's stacked sorts (group keys, then the `arrange` key) become one Spark
sort, applied last. A multi-file scan is read file by file, because Spark
orders a scan's files by size, and ordered by (file, row).
`distinct()` keeps the first row in that order; `row_number()`,
`slice_*()`, `cumsum()` and `lag()` are windows ordered by it.

**Aggregations** translate per context: inside `summarise` they are Spark
aggregates, in `mutate` and `filter` they are windows over the group (or
the whole frame). An aggregation of an aggregation (`sum(x - min(x))`),
which Polars allows within a group, is computed first as a column with the
inner one over the group's window.

**tidy3's checks.** `summarise` checks at collect time that each summary
is one value per group; on Spark that check is a `raise_error()`.

## 3. Where Spark and Polars differ, and what the translator does

| Difference | Translation | Checked by |
|---|---|---|
| Spark has no row order | Row numbers at each scan, ordering keys carried through every step, one sort at the end | stable-tie, distinct-first, slice and row-number tests; multi-file scan test |
| `7 %% -3` is `-2` in Polars and R; Spark's `%` gives `1`, `pmod` is never negative | floor-mod from `%` | `modulo` test |
| tidy3's `round()` is half to even; Spark's `round` is half up | `bround` | `round_ties` test |
| Spark refuses window functions in `WHERE` (grouped `filter`, `slice_max`) | the predicate is a hidden column, then filtered | `slice_max` test |
| Spark refuses aggregates or windows inside aggregates | inputs precomputed as columns | `an_aggregation_of_an_aggregation` |
| Polars sums booleans; Spark refuses | booleans cast to 0/1 in `sum`, `mean`, `cumsum` | `consecutive_id` benchmark case |
| No unsigned integers in Spark | held in the next wider signed type; `UInt64` as `Decimal(20, 0)` | `unsigned` test |
| Spark scans infer types differently | the Spark schema is Polars' schema for that scan; CSV options Spark would read differently are refused | penguins tests |
| Regular expressions: Rust's dialect in Polars, Java's in Spark | patterns get Java's `(?U)` (Unicode `\d`, `\s`, `\w` and case folding, as in Rust) and `(?d)` (`\n` the only line end); `\p{Han}` becomes `\p{IsHan}`, `(?P<name>` becomes `(?<name>`. Look-arounds work on Spark but not in Polars; `$` before a final newline differs | `string_functions` tests |
| Replacements: Rust writes `$1`, `${1}`, `$$`, and reads `$1x` as a group named "1x" (empty); Java reads `$12` as group 12 if it exists | rewritten for Java; a first-only replace (`n=1`), which Spark lacks, anchors the first match as `\A(?s:(.*?))(?:pattern)`. Polars 1 replaces a pattern with no regex characters literally, replacement and all; Polars 2 does not, and the translator follows the installed version | `polars_options` test |
| `regexp_extract` gives `""` where nothing matches, and for a group that took no part | null where nothing matches; a pattern whose group may be unset or empty (`\|`, `?`, `*`, `{0}`) is refused | `extract_whose_group_may_be_unset` test |
| `trim` strips spaces; `lpad`/`rpad` cut a longer string; `initcap` capitalizes after spaces only; `concat_ws` skips missing values | Unicode whitespace stripped by regex; padding only when shorter; title case letter by letter after any non-letter, in titlecase under Polars 2 (`ß` is `Ss`) and uppercase under Polars 1; `concat` with the separator between, so a missing value makes the result missing | `stringr` test |
| `date_trunc` reads a naive datetime in the session time zone, so a clock time daylight saving skipped moves an hour | truncation from the date and clock fields (`trunc`, `make_timestamp_ntz`); offsets add months (clamped to the month's end, as Polars does) then a day-time interval | session-time-zone test with 02:37 on 8 March 2020 in New York |
| Division by zero: Polars gives `inf`, `-inf` or `nan`; Spark gives null (ANSI off) or raises (ANSI on, Spark 4's default) | IEEE results written out, the sign of `-0.0` read from its text (Spark's `signum` and `atan2` drop it) | `arithmetic_edges` tests, both ANSI modes |
| `%%` by zero: null for integers, `nan` for floats in Polars | written out | same |
| A non-strict cast gives null for a value with no such form; a strict one fails | `try_cast`, and a `raise_error` for strict casts, so ANSI off does not turn a failure into null | `strict_cast` test, both ANSI modes |
| Integer overflow wraps in Polars | wraps with ANSI off; **raises** with ANSI on (Spark's `ARITHMETIC_OVERFLOW`), rather than hiding an overflow | `ansi_checks.py` on the cluster |
| A `select` of aggregations and literals (`adorn_totals`' totals row) is one row in Polars | one Spark `agg` when every column is inside an aggregation | `adorn_totals` benchmark case |

## 4. What is not translated yet

Of the 342 benchmark cases that collect a frame query, **200 run on Spark
and match tidy3** to within 1e-12, and the rest raise `SparkExportError`.
None returns different rows. (`to_polars_code()` exports 269.) The counts
are the same under Polars 1.42.1 and 2.1.0. The refusals, by count:

| Refused | Cases | Notes |
|---|---|---|
| A column computed by a Python function | 44 | Text conversions (`as_*`, `parse_*`, `ymd`), R `round`, per-value stringr, janitor rounding. Polars stores the functions cloudpickled in the plan, and they run outside Polars, so they could become Arrow UDFs, but only those tidy3 marks elementwise can be split across partitions. |
| Joins, binds, group order | 22 | right, full and cross joins (and the inequality joins built on a cross join), `bind_cols`, `bind_rows` of columns with different types, groups kept in order of appearance. |
| Factors (Enum columns and casts) | 18 | Spark has no Enum; a string column plus recorded levels would do. |
| Strings | 15 | Functions that give lists or structs (`str_split`, `str_extract_all`, `str_locate`, `str_match`, `word`, `separate`), `str_flatten`, and Unicode normalization (`str_equal`). |
| Dates and times | 14 | Durations (`ddays`, `as_duration`, `time_length`) and time zones (`with_tz`, `force_tz`, zoned datetimes). Spark's day-time intervals could hold durations; time zones need a session time-zone policy. |
| Reshape | 11 | `pivot_wider`, `pivot_longer`, `unnest`, `explode`, `pack`, `hoist`. Spark has `pivot`, `unpivot` and `explode`. |
| Other | 18 | `first`/`last` within groups, `lag`/`lead` with a default, `fill`, `slice_tail`, `nth`, `cume_dist`, `row_means`, `uncount`, IPC scans, and the row checks of `rows_*`. |

String and date functions with a Spark form are translated: replacing,
slicing, padding, trimming, case, counting and extracting matches, and
joining strings; date parts, ISO weeks and years, leap years, days in a
month, `floor_date`/`ceiling_date`, `make_date`, and adding periods.

## 5. Performance

Single machine, 50 million rows from a 0.78 GB Parquet file, 24 cores:

| Pipeline | Polars | Spark (local) | Same result |
|---|---|---|---|
| README summary (`drop_na`, two groups, rounded means, `arrange`) | 1.21 s | 2.63 s | exact |
| `left_join` then three-way `count` | 1.37 s | 1.54 s | exact |
| `filter` then grouped mean and `sd` | 0.22 s | 0.65 s | within 1e-9 |

On one machine Polars is faster; Spark is for data spread over a cluster or
beyond one machine's memory.

Three steps use an unpartitioned window and run on one Spark partition: an
ungrouped `row_number()` (a global row index), `slice()` with an offset,
and `bind_rows` of inputs ordered differently (inputs read the same way
carry their ordering keys instead). On the cluster below, an ungrouped
`row_number()` over 50M rows took 13.0 s against 7.95 s for the grouped
one, with Spark's "moving all data to a single partition" warning. A
pipeline that needs one at cluster scale is better numbered within groups.

## 6. Cluster verification

A Spark 4.2.1 standalone cluster on the same machine: one master and four
workers of 4 cores and 5 GB, queries spread over 8 executors (2 cores, 2 GB
each). Separate executor processes, so rows cross process boundaries in
every shuffle; not separate machines or network storage.

- The test suite against the cluster (`TIDY3_SPARK_MASTER=spark://host:7077`):
  every test passes, as in local mode.
- Row order under real splitting: 10M rows as one 388 MB CSV read in 93
  partitions (`spark.sql.files.maxPartitionBytes` = 4 MB), and as 12
  Parquet files of 4 KB to 23 MB read by one glob in 31 partitions, with
  adaptive query execution on. Every order-sensitive pipeline gave tidy3's
  rows in tidy3's order, with ANSI mode on and off:

| Check | Rows | Polars | Spark cluster |
|---|---|---|---|
| stable `arrange` on a 10-value key | 10,000,000 | 0.22 s | 3.89 s |
| `arrange(desc(s), k)`, `slice_head` | 200,000 | 0.42 s | 2.51 s |
| `distinct` keeping the first row | 1,000 | 0.14 s | 2.10 s |
| `row_number()` per group, `filter` | 2,000 | 0.22 s | 2.20 s |
| `cumsum` per group | 10,000,000 | 0.26 s | 3.17 s |
| `slice_head` over the Parquet glob | 100,000 | 0.00 s | 1.83 s |
| `slice_head` per group | 30 | 0.14 s | 1.75 s |
| `slice_max` per group | 30 | 0.34 s | 13.36 s |
| `left_join` keeps the left order, `slice_head` | 50,000 | 0.00 s | 1.20 s |
| grouped `mean`, `sd` | 40 | 0.19 s | 1.35 s |
| ungrouped `row_number()` | 10 | 0.10 s | 2.37 s |
| `bind_rows` of the glob and the CSV, `slice_head` | 100,000 | 0.01 s | 1.48 s |

The scripts are `cluster_checks.py`, `ansi_checks.py` and
`single_partition.py` in the gpudev workspace.

CI runs `tests/test_spark_export.py` in Spark local mode in a `spark` job
with two settings: the floors (PySpark 4.0, the lowest `tidy3[spark]`
allows, on Java 17) and the newest PySpark on Java 21. Both take the newest
Polars, so a change to Polars' plan format shows up there. Polars 2 already
writes join keys differently (`condition: {Equi, NonEqui}`); the translator
reads both forms, and coverage is the same on Polars 1.42 and 2.1.

## 7. Open questions

1. **Spark-only sources.** Some verbs read values from the data while the
   plan is built (factor levels, `pivot_wider` names, `complete()`,
   `compute()`); 14% of the test suite's plans do. That works when Spark
   reads the same files Polars can read. A Spark table with no file path
   would need those verbs to ask Spark for the values: a `scan_spark(table)`
   source and a backend hook in each such verb.
2. **Python steps** as Arrow UDFs (section 4), and which tidy3 functions
   can declare themselves elementwise.
3. **Separate machines.** The cluster above shares one machine's disk.
   Separate nodes reading network storage (and Spark Connect, which
   Databricks serverless uses) are not tested yet.
4. **Integer overflow under ANSI mode** raises where Polars wraps. Matching
   Polars there would mean hiding an overflow; raising looks right, and the
   reference says so.
