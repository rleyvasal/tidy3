# tidy3

dplyr-style **lazy** data manipulation for Python, powered by [Polars](https://pola.rs/).

- Readable multi-line pipes (`>>` ≈ R’s `|>`)
- **Partial pipeline run in any Jupyter** (SolveIt, JupyterLab, classic, …) — no VS Code extension
- Lazy until `collect()` / plot / preview (`LIMIT n` only)
- Optional bridge to [plot3](https://github.com/rleyvasal/plot3) and [gpudev](https://github.com/rleyvasal/gpudev)

## Install

```bash
cd /path/to/tidy3
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```python
from tidy3 import tidy, filter, mutate, group_by, summarise, col, n, mean, collect

out = (
    tidy(cars)                          # pandas or polars
    >> filter(col("mpg") > 20)
    >> mutate(km=col("mpg") * 1.609)
    >> group_by("cyl")
    >> summarise(n=n(), avg=mean("mpg"))
    >> collect()                        # polars; collect(as_="pandas") for pandas
)
```

With the Jupyter extension loaded you can omit the outer parentheses — the **kernel** rewrites multi-line `>>` pipes before parse.

## Jupyter / SolveIt

### Load (same as other gpudev addons)

```text
%local
%run /path/to/gpudev/CRAFT.py
%run /path/to/gpudev/addons/tidy3.py
%gpu
```

Or without CRAFT: `%load_ext tidy3.jupyter` / `pip install -e .`

Under **`%gpu`**, cells run on the remote kernel — a separate namespace and
filesystem. The addon handles this automatically: on the first `%gpu` cell it
**pushes the local tidy3 source to the remote kernel** (`tidy3.craft`, ~10 KB
over the existing channel), installs polars there if missing, and loads the
Jupyter extension remotely (API names, `>>` rewriting, `%%tidy3_run`). It
re-seeds by itself after `%restart_kernel` and after local source edits
(content stamp). If something goes sideways: `seed_tidy3_remote(force=True)`.

Then just use normal tidy3 with **paths on the GPU host**:

```python
# after %gpu — file path is on the GPU box
scan_parquet("/home/gpudev/data/huge.parquet")
>> filter(col("year") >= 2020)
>> group_by("region")
>> summarise(n=n(), avg=mean("value"))
```

### Partial run

- **Run Selected Text** (if SolveIt has it): highlight a pipe prefix → run selection  
- Own cell with only the prefix  
- `%%tidy3_run` with the prefix pasted in  

```python
%%tidy3_run
tidy(cars)
>> filter(col("mpg") > 20)
```

```text
%tidy3_pipes on|off|status
```

## plot3

```python
from plot3 import aes, geom_point

(
    tidy(big)
    >> filter(col("intensity") > 10)
    >> select("x", "y", "z", "intensity")
).ggplot(aes(x="x", y="y", z="z", colour="intensity")) + geom_point()
```

## gpudev / CRAFT / SolveIt

```text
%local
%run /path/to/gpudev/CRAFT.py
%run /path/to/gpudev/addons/tidy3.py
%run /path/to/gpudev/addons/plot3.py
```

```bash
ln -s ../../tidy3 /path/to/gpudev/addons/tidy3   # if needed
```

## API (v0.1)

| Area | Symbols |
|------|---------|
| Frame | `tidy`, `scan_parquet`, `scan_csv`, `scan_ipc`, `TidyFrame` |
| Verbs | `filter`, `mutate`, `transmute`, `select`, `drop`, `rename`, `arrange`, `distinct`, `group_by`, `ungroup`, `summarise`, `count`, `head`, `sample_n`, `sample_frac`, `left_join`, `inner_join`, `collect` |
| Expr | `col`, `n`, `mean`, `sum`, `min`, `max`, `median`, `std`, `first`, `last`, `desc` |
| Jupyter | `%load_ext tidy3.jupyter`, `%tidy3_run`, `%%tidy3_run`, `%tidy3_pipes` |
| Partial | `partial_run`, `maybe_rewrite_cell`, `normalize_pipe_source` |
| Escape | `TidyFrame.with_polars(fn)` |

## Backends

The default engine is **Polars lazy**. For 1:1 engine comparisons (e.g.
against datar, which is pandas-only) the same pipeline also runs on an
**eager pandas backend**:

```python
tidy(df, backend="pandas") >> filter(col("x") > 0) >> ...   # per-frame
options(backend="pandas")                                    # session default
```

Expressions (`col("x") * 2`, `mean("y")`, comparisons, `cum_sum`, …) are
backend-neutral: they compile to `pl.Expr` on polars and evaluate natively on
pandas, with dplyr window semantics after `group_by` on both. The pandas
backend covers the documented verb/expression subset; anything
polars-specific raises a clear error pointing back to `backend="polars"`.

## Benchmark vs datar

```python
from tidy3 import bench
bench.run(rows=10_000_000)        # pip install datar datar-pandas for the datar row
```

Apple-silicon laptop, 10M rows × 100 groups, filter→mutate→group_by→summarise→arrange:

| engine | time | vs fastest |
|---|---|---|
| polars (raw lazy) | 39.7ms | 1.0x |
| **tidy3[polars]** | 45.5ms | 1.1x |
| pandas (raw) | 59.9ms | 1.5x |
| **tidy3[pandas]** | 103.9ms | 2.6x |
| datar[pandas] | 162.5ms | 4.1x |

Same engine, same pipeline: tidy3's pandas backend is ~1.6x faster than
datar; the polars backend is ~3.6x faster, with near-zero wrapper overhead.
Under `%gpu`, run the same `bench.run(...)` on the remote kernel
(`!uv pip install datar datar-pandas` there first for the datar row).

## Notes

- **Grouped semantics are dplyr's**: after `group_by`, `mutate`/`filter`/
  `sample_n`/`head` evaluate **per group** (Polars window `.over`).
  `summarise` aggregates per group.
- **Display is self-contained**: tables carry inline styles (no `<style>`
  block), so they render identically in local cells, republished `%gpu`
  output, and sslive exports. Under `%gpu`, bare polars DataFrames are
  restyled the same way (`seed_tidy3_remote(style_polars=False)` to opt out).
- **Builtins are shadowed** by design: injecting the API puts `filter`, `sum`,
  `min`, `max` into the notebook namespace (dplyr ergonomics). Use
  `builtins.filter` etc. when you need the Python originals.
- **Plotting big remote data**: aggregate remotely, then let plot3's own
  remote path pull the small result: `%plot3 res.to_pandas() x=... y=...`.

## Why not datar?

datar’s pandas/Python wrappers struggle on large frames. tidy3 compiles verbs to a **Polars Lazy** plan and only materializes at the edge. Previews use `LIMIT n`.

## License

MIT
