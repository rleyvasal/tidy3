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

## Jupyter / SolveIt (kernel-side partial run)

Works in **any** IPython kernel. Nothing editor-specific.

### Load once

```python
%load_ext tidy3.jupyter
# or gpudev: %run .../addons/tidy3.py  (loads the extension automatically)
```

This enables:

1. **Input transformer** — multi-line `>>` pipes (and partial prefixes) are auto-wrapped so they parse and display.
2. **Magics** — `%tidy3_run` / `%%tidy3_run` for explicit partial runs.
3. **Name inject** — `tidy`, `filter`, `col`, … available in the namespace.

### Partial run like RStudio

Write a full pipeline in a cell (optional outer `()`):

```python
tidy(cars)
>> filter(col("mpg") > 20)
>> mutate(km=col("mpg") * 1.609)
>> group_by("cyl")
>> summarise(n=n(), avg=mean("mpg"))
```

To inspect an intermediate:

**Option A — run only the prefix as its own cell** (simplest, works everywhere):

```python
tidy(cars)
>> filter(col("mpg") > 20)
```

Run the cell → filtered preview (transformer rewrites; `TidyFrame` shows head only).

**Option B — explicit magic** (same text, always via `partial_run`):

```python
%%tidy3_run
tidy(cars)
>> filter(col("mpg") > 20)
```

**Option C — line-by-line with `_`:**

```python
tidy(cars) >> filter(col("mpg") > 20)   # filtered preview
_ >> mutate(km=col("mpg") * 1.609)      # next step
```

**Option D — if the frontend can “run selected text” into the kernel**  
(SolveIt / some Jupyter UIs): select the prefix and run selection. The same input transformer rewrites the selection before exec — **no VS Code extension**.

```text
%tidy3_pipes on|off|status   # toggle auto-rewrite
```

### Large files

```python
from tidy3 import scan_parquet, filter, col, group_by, summarise, mean

scan_parquet("data/*.parquet")
>> filter(col("year") >= 2020)
>> group_by("region")
>> summarise(avg=mean("value"))
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

## Why not datar?

datar’s pandas/Python wrappers struggle on large frames. tidy3 compiles verbs to a **Polars Lazy** plan and only materializes at the edge. Previews use `LIMIT n`.

## License

MIT
