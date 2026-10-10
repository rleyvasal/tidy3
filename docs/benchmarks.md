# tidy3 benchmarks and the R parity oracle

How tidy3 is timed against raw Polars, pandas, and datar, and how its results are checked against dplyr and tidyr.

Back to the [README](../README.md).

## Benchmark vs datar

```python notest
from tidy3 import bench
bench.run(rows=10_000_000)        # full pipeline; pip install datar datar-pandas for the datar row
bench.run_ops(rows=10_000_000)    # isolated verbs with each output boundary labelled
```

For the adoption-oriented suite, which covers everyday operations plus ML,
event-history, customer aggregation, and join/filter/aggregate workflows:

```bash
python -m tidy3.bench_suite --rows 1000000 --repeat 5 --output pandas
python -m tidy3.bench_suite --rows 1000000 --repeat 5 --output pandas-arrow
python -m tidy3.bench_suite --rows 1000000 --repeat 5 --output native

# Compare Polars execution modes (GPU mode is strict: no silent CPU fallback)
python -m tidy3.bench_suite --rows 1000000 --repeat 5 --output native --polars-engine auto
python -m tidy3.bench_suite --rows 1000000 --repeat 5 --output native --polars-engine streaming
python -m tidy3.bench_suite --rows 1000000 --repeat 5 --output native --polars-engine gpu

# Ratio-based CI budgets; workloads under 10ms are ignored as noise-prone
python -m tidy3.bench_suite --rows 1000000 --repeat 5 --output native \
  --max-tidy-pandas-geo 1.25 --max-tidy-polars-geo 1.25 \
  --budget-min-ms 10
```

The default `pandas` output makes every engine return a NumPy-backed pandas
DataFrame and therefore includes the Polars-to-pandas ML handoff.
`pandas-arrow` still returns pandas but usually avoids that copy; `native`
keeps results in Polars and isolates execution from conversion. Data
generation, warm-up, garbage collection, and correctness validation are
outside recorded times; every measured result is checked against raw pandas.
The suite reports medians and rotates engine order between repetitions.
`--polars-engine` accepts `auto`, `streaming`, and `gpu`; GPU benchmarking
uses strict Polars GPU execution and stops if a query would fall back to CPU.
Optional datar coverage remains in the smaller `tidy3.bench` benchmark as a
backup comparison.

Apple-silicon laptop, 10M rows × 100 groups, filter→mutate→group_by→summarise→arrange:

| engine | time | vs fastest |
|---|---|---|
| **tidy3[polars]** | 40.6ms | 1.0x |
| polars (raw lazy) | 41.5ms | 1.0x |
| pandas (raw) | 60.1ms | 1.5x |
| **tidy3[pandas]** | 60.8ms | 1.5x |
| datar[pandas] | 166.3ms | 4.1x |

An isolated one-expression mutate is a useful boundary stress test because
pandas Copy-on-Write makes the baseline unusually cheap. On the same laptop,
10M rows, median of 7 runs after 2 warm-ups:

| execution and output boundary | time | vs raw pandas |
|---|---:|---:|
| tidy3[pandas] → pandas | 5.0ms | 1.0x |
| raw pandas → pandas | 5.1ms | 1.0x |
| tidy3[polars] → native Polars | 10.4ms | 2.1x |
| raw Polars → native Polars | 10.7ms | 2.1x |
| tidy3[polars] → Arrow-backed pandas | 11.1ms | 2.2x |
| datar[pandas] → pandas | 11.3ms | 2.2x |
| tidy3[polars] → NumPy-backed pandas | 32.2ms | 6.4x |

This replaces the old `41.9ms (8.7x)` cell, which combined Polars execution
with a full NumPy conversion and presented the total as one engine number.
The remaining 2.1x on this deliberately tiny operation is Polars engine cost:
tidy3 tracks raw Polars within measurement noise. For a single cheap mutate
followed immediately by NumPy pandas, use the pandas backend. For longer lazy
pipelines, collect once at the end, project needed columns first, and prefer
native or Arrow-backed output. Aggregations, joins, sorts, and full workflows
are the adoption-relevant comparison in `bench_suite`.

Under `%gpu`, run the same benchmarks on the remote kernel. Install Polars GPU
support there first, then use `--polars-engine gpu`; the comprehensive suite
does not require datar.

### R semantic-parity oracle

`tests/test_r_oracle_parity.py` compares every public frame verb with dplyr or
tidyr, or assigns it to an explicit invariant/materialization category. The
suite includes missing and empty inputs, persistent and transient grouping,
categorical levels, duplicate names, and type coercion. Install R with dplyr,
tidyr, and jsonlite, then run:

```bash
pytest -q tests/test_r_oracle_parity.py
```

If R is kept in a Pixi environment, point the suite at its manifest:

```bash
TIDY3_R_ORACLE_MANIFEST=/path/to/pixi.toml \
  pytest -q tests/test_r_oracle_parity.py
```

All supported verb contracts in the oracle are strict tests on both backends;
there are no expected-failure parity cases. Random sampling and explicit
materialization boundaries use deterministic invariants where byte-for-byte
comparison with R would be inappropriate.

## Why not datar?

datar’s pandas/Python wrappers struggle on large frames. tidy3 compiles verbs to a **Polars Lazy** plan and only materializes at the edge. Previews use `LIMIT n`.
