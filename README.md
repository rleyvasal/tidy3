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

### Load

```python
%load_ext tidy3.jupyter
# or gpudev:
#   %local
#   %run .../CRAFT.py
#   %gpu
#   %run .../addons/tidy3.py   # load on the remote kernel under %gpu
```

Under **`%gpu`**, every cell already runs on the remote machine — use normal tidy3
APIs with **paths on the GPU host**. No special `remote()` wrapper.

```python
# %gpu is on — this runs on the host box; file must be there
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

## Why not datar?

datar’s pandas/Python wrappers struggle on large frames. tidy3 compiles verbs to a **Polars Lazy** plan and only materializes at the edge. Previews use `LIMIT n`.

## License

MIT
