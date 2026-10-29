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
| Regular expressions | Java's dialect, not Rust's: most patterns agree; look-arounds work on Spark but not in Polars, and some classes differ | `str_detect` test |

## 4. What is not translated yet

Of the 342 benchmark cases that collect a frame query, **153 run on Spark
and match tidy3** to within 1e-12, and the rest raise `SparkExportError`.
None returns different rows. (`to_polars_code()` exports 269.) The
refusals, by count:

| Refused | Cases | Notes |
|---|---|---|
| A column computed by a Python function | 44 | Text conversions (`as_*`, `parse_*`, `ymd`), R `round`, per-value stringr, janitor rounding. Polars stores the functions cloudpickled in the plan, and they run outside Polars, so they could become Arrow UDFs, but only those tidy3 marks elementwise can be split across partitions. |
| Dates and times | 30 | `offset_by` (periods), time zones, `floor_date`, durations, week numbers. Time zones need a session time-zone policy. |
| Strings | 29 | `str_replace`, `str_sub`, `str_c`, `str_extract`, `str_split`, `str_trim`, padding, case. Mostly direct; `str_trim` must strip Unicode whitespace like Polars. |
| Factors (Enum columns and casts) | 18 | Spark has no Enum; a string column plus recorded levels would do. |
| Joins, binds, group order | 27 | right, full and cross joins, `bind_rows` with different columns, groups kept in order of appearance. |
| Reshape | 11 | `pivot_wider`, `pivot_longer`, `unnest`, `bind_cols`, `explode`. Spark has `pivot`, `unpivot` and `explode`. |
| Other | 30 | lists and structs, `first`/`last` within groups, `fill` (forward and backward), the row checks of `rows_*`, `between`. |

## 5. Performance

Single machine, 50 million rows from a 0.78 GB Parquet file, 24 cores:

| Pipeline | Polars | Spark (local) | Same result |
|---|---|---|---|
| README summary (`drop_na`, two groups, rounded means, `arrange`) | 1.21 s | 2.63 s | exact |
| `left_join` then three-way `count` | 1.37 s | 1.54 s | exact |
| `filter` then grouped mean and `sd` | 0.22 s | 0.65 s | within 1e-9 |

On one machine Polars is faster; Spark is for data spread over a cluster or
beyond one machine's memory. A global row index, and `slice()` with an
offset, use an unpartitioned window and run on one Spark partition; a
pipeline that needs them at cluster scale is better sliced within groups.

## 6. Open questions

1. **Spark-only sources.** Some verbs read values from the data while the
   plan is built (factor levels, `pivot_wider` names, `complete()`,
   `compute()`); 14% of the test suite's plans do. That works when Spark
   reads the same files Polars can read. A Spark table with no file path
   would need those verbs to ask Spark for the values: a `scan_spark(table)`
   source and a backend hook in each such verb.
2. **Python steps** as Arrow UDFs (section 4), and which tidy3 functions
   can declare themselves elementwise.
3. **Cluster checks.** Row order across many files and partitions, and the
   cost of the single-partition windows, need a real cluster run.
