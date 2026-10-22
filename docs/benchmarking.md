# Performance comparisons

Use **raw Polars** as the performance reference. Compare complete operations
on identical inputs, including the same output columns and materialization.
The previous tidy3 implementation is a separate reference for improvement,
and R comparisons validate semantics.

The [public API coverage audit](benchmarks/public-api-coverage-2026-10-21.md)
tracks all exported functions and public `TidyFrame` methods, the remaining
benchmark gaps, and the measured candidates for optimization.

## Rolling joins

```bash
python -m tidy3.bench_polars --repeat 3 --warmup 1 --memory --output report.json
```

This writes JSON samples and a Markdown report. It includes the original
200,000-row mixed-type workload, wide 200,000-row and 1,000,000-row inputs,
1,000 versus 100,000 lookup rows, and a join with two equality keys.
Use `--case wide_1m_dense` to run one case.

Both implementations materialize the complete Polars output, in the original
input order. Values, column names, dtypes and row order are checked before
timing. Generation and validation are excluded; execution includes sorting,
joining, restoring row order and collecting. Engine order alternates between
rounds. Reports show the best time, median, individual samples, and
`tidy3 / raw Polars`; a ratio of 2 means twice the native elapsed time.

The lookup keys in these controlled workloads are unique. Duplicate-match,
missing-value and join-validation semantics have separate correctness tests.

`--memory` runs each engine in a fresh process. Peak RSS includes interpreter
imports, input construction, execution and the complete output, including
native allocations. It is a process high-water measurement, not incremental
operator allocation, and requires macOS or Linux. Memory instrumentation is
outside the timing runs.

The [current Polars-relative report](benchmarks/rolling-joins-polars-2026-10-21.md)
is distinct from the [earlier before/after measurements](benchmarks/rolling-joins-2026-10-21.json).
Compare ratios from the same run; do not divide an older tidy3 timing by a
newly measured Polars timing.

## Speed guards

`tests/test_function_speed.py` uses independent raw Polars recipes for each
timed column function and grouped operation. The factor-scan baseline also
discovers levels and constructs the factor; it does more than read a file.
Results are checked before measurement. Sampling uses each engine's seeded
RNG, so it checks group sizes and source membership rather than identical
random draws. Other workloads compare values and ordering, including factor
levels.

Both sides produce the same output format. In particular, the pandas nesting
baseline performs the Polars grouping and exports a pandas DataFrame per
group, matching that backend's required representation. Its baseline time
includes that export; it is not the cost of Polars' native list-of-structs
output alone.

```bash
python -m pytest tests/test_function_speed.py --junitxml=timings.xml -o junit_family=legacy
```

JUnit properties contain `tidy3_ms`, `raw_polars_ms`, `tidy3_over_polars`, and
`ci_budget_ms`. The unit-test workflow uploads these results for every Python
version. CI thresholds are regression alarms with headroom, not claims of
acceptable overhead; use the measured ratios to prioritize improvements.

The performance workflow uploads the raw Polars rolling-join report. The
older `tidy3.bench_suite` remains a separate pandas adoption diagnostic with
explicit pandas-relative budgets; its `vs_pandas` values do not measure
overhead relative to Polars.

## Whole public API

```bash
python -m tidy3.bench_api --rows 100000 --repeat 3 --warmup 1 --output api-polars.json
```

Every name exported by tidy3, and every public `TidyFrame` method, is either
one timed case or an explicit reason. A case runs the tidy3 call and a raw
Polars recipe on the same table. Values, names, dtypes, and order are checked
before timing. File setup and other preparation sit outside the samples.
The command writes JSON and Markdown, ranked by excess time (tidy3 minus raw
Polars). `--case` and `--family` select a subset. The run stops if any export
or method is unclassified.

`tests/test_bench_api.py` checks that classification is complete and that
every case matches Polars on a 24-row table. The CI speed guards stay
regression alarms; this command is the parity measurement. It is not part of
the 15-minute performance workflow.
