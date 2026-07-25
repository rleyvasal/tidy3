# tidy3

dplyr-style **lazy** data manipulation for Python, powered by [Polars](https://pola.rs/).

- Readable multi-line pipes (`>>` ≈ R’s `|>`)
- Works **locally** (VS Code, terminal, JupyterLab) or under **CRAFT / gpudev**
- Lazy until `collect()` / plot / preview (`LIMIT n` only)
- Handoffs for ML (`to_numpy`, pandas/Arrow) and optional [plot3](https://github.com/rleyvasal/plot3)

CRAFT/`%gpu` is an optional remote path. The default product surface is a normal
Python environment on your machine.

## Install (local)

```bash
cd /path/to/tidy3
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[jupyter]"        # IPython extension for multi-line >> pipes
pip install -e ".[excel]"          # write_excel
# plot3 is a separate package — clone it and pip install -e that repo if you plot
```

Verify:

```bash
python -c "import tidy3; print(tidy3.__version__)"
python -m pytest -q
```

## VS Code (local IDE)

tidy3 is a normal editable package. No CRAFT, SolveIt, or remote kernel is
required.

### One-time setup

1. Open the `tidy3` folder (or a workspace that includes it) in VS Code.
2. Create/select the project interpreter:
   - Command Palette → **Python: Select Interpreter** → `./.venv/bin/python`
3. Install the package into that environment (terminal in VS Code):

```bash
source .venv/bin/activate
pip install -e ".[dev,jupyter]"
```

4. Recommended extensions: **Python** (includes **Pylance**) and **Jupyter**.
   Optional: **Polars** is a library dependency, not a VS Code extension.

### Type checker / red squiggles under `>>` pipes

You do **not** need a special tidy3 VS Code extension. Pylance is enough once
tidy3 is installed editable from this repo (types ship with `py.typed`).

1. Interpreter = `tidy3/.venv` (Command Palette → **Python: Select Interpreter**).
2. Workspace settings in `.vscode/settings.json` point analysis at `src/`.
3. Import the names you use in that cell/file:

```python
from tidy3 import tidy, select, filter, arrange, slice_max, col, desc
```

4. Reload the window if squiggles linger: **Developer: Reload Window**.

If a name is still underlined, it is usually “not imported in this cell”, not a
pipe typing bug. Runtime green + import present ⇒ safe to ignore residual noise.

### Scripts (`.py`)

Parentheses around multi-line `>>` pipes are required (standard Python):

```python
# analysis.py
from tidy3 import tidy, filter, mutate, group_by, summarise, col, n, mean

result = (
    tidy({"cyl": [4, 4, 6], "mpg": [22.0, 24.0, 18.0]})
    >> filter(col("mpg") > 20)
    >> mutate(km=col("mpg") * 1.609)
    >> group_by("cyl")
    >> summarise(n=n(), avg=mean("mpg"))
)

print(result.collect())                 # Polars DataFrame
print(result.collect(as_="pandas"))     # for sklearn / export
# features = result.to_numpy(columns=["avg"], dtype="float32")
```

Run:

```bash
python analysis.py
# or VS Code: Run Python File
```

### Interactive window / notebook

1. Create `analysis.ipynb` (or open the Interactive Window).
2. Pick the same `.venv` kernel (**Select Kernel** → your tidy3 venv).
3. In the first cell, either import normally or load the pipe rewriter:

```python
# Option A — plain Python (parentheses required)
from tidy3 import tidy, filter, col

# Option B — multi-line >> without outer parentheses
%load_ext tidy3.jupyter
```

With the extension loaded:

```python
tidy(cars)
>> filter(col("mpg") > 20)
>> mutate(km=col("mpg") * 1.609)
```

Partial pipes for debugging:

```python
%%tidy3_run
tidy(cars)
>> filter(col("mpg") > 20)
```

```text
%tidy3_pipes on|off|status
```

### Handoffs from a local session

```python
# Algorithms / sklearn
df = result.collect(as_="pandas", columns=["cyl", "avg"], arrow_backed=True)

# NumPy → PyTorch (CPU)
import numpy as np
# import torch
X = result.to_numpy(columns=["avg"], dtype=np.float32, writable=True, order="c")
# t = torch.from_numpy(X)

# plot3 (optional; install plot3 separately)
# from plot3 import aes, geom_point, ggplot
# result >> ggplot(aes(x="cyl", y="avg")) + geom_point()
```

### Local vs CRAFT at a glance

| | Local (VS Code) | CRAFT / `%gpu` |
|--|-----------------|----------------|
| Install | `pip install -e .` in a venv | Addon seed to remote kernel |
| Paths | Your machine | Paths on the GPU host |
| Pipes | Parentheses in `.py`; extension optional in notebooks | Same extension after seed |
| Data size | Laptop RAM / local Polars | Remote GPU box + large files |
| Default for new users | **Yes** | Optional power path |

## Quick start

```python
from tidy3 import scan_parquet, filter, mutate, group_by, summarise, col, n, mean

result = (
    scan_parquet("cars.parquet")        # lazy: reads only when needed
    >> filter(col("mpg") > 20)
    >> mutate(km=col("mpg") * 1.609)
    >> group_by("cyl")
    >> summarise(n=n(), avg=mean("mpg"))
)

result                              # TidyFrame preview; keep piping if needed
polars_df = result.collect()        # materialize as Polars
pandas_df = result.collect(as_="pandas")
numpy_array = result.collect(as_="numpy")
```

For in-memory data, replace the scan with `tidy(cars)`. Starting from
`scan_parquet()`, `scan_csv()`, or `scan_ipc()` avoids first building a pandas
copy and lets Polars push projections and filters into the file scan.

With the Jupyter extension loaded you can omit the outer parentheses — the **kernel** rewrites multi-line `>>` pipes before parse.

## Save results

Write a pipeline directly without first calling `collect()`:

```python
result.write_parquet("summary.parquet")  # recommended for data pipelines
result.write_csv("summary.csv")
result.write_ipc("summary.arrow")        # Arrow IPC / Feather
```

With the default Polars backend, these execute the lazy plan and stream its
result to disk, avoiding a second fully materialized DataFrame in memory.
The same methods also work with `backend="pandas"`. GPU writers execute the
plan on the GPU, then materialize before serialization because current GPU
file sinks are not stable across all formats; `auto` and `streaming` retain
direct lazy sinks.

Choose how Polars executes when the plan is materialized or written:

```python
result.collect(engine="auto")             # default: let Polars choose
result.collect(engine="streaming")        # execute in streaming batches
result.collect(engine="gpu")              # GPU where supported; may fall back

result.write_parquet("summary.parquet", engine="streaming")
result.write_csv("summary.csv", engine="gpu")
result.write_ipc("summary.arrow", engine="auto")
```

The same `engine=` argument is available on `to_polars()`, `to_pandas()`,
`to_arrow()`, and `write_excel()`. It applies only to the default Polars
backend. For a GPU run that must not silently fall back to CPU, pass
`engine=polars.GPUEngine(raise_on_fail=True)`; the benchmark suite does this
automatically whenever `--polars-engine gpu` is selected.

Excel output is intended for smaller reporting datasets and must materialize
the result:

```bash
pip install "tidy3[excel]"
```

```python
result.write_excel(
    "summary.xlsx",
    worksheet="Summary",
    autofit=True,
    freeze_panes="A2",
)
```

You can still collect first when another library needs the result:

```python
result.collect().write_csv("summary.csv")          # Polars API
result.collect(as_="pandas").to_csv("summary.csv", index=False)
```

## plot3 (optional)

plot3 is a **separate** project. Install it in the same venv if you use it:

```bash
pip install -e /path/to/plot3
```

```python
from plot3 import aes, geom_point, ggplot

tidy(big)
>> filter(col("intensity") > 10)
>> select("x", "y", "z", "intensity")
>> ggplot(aes(x="x", y="y", z="z", colour="intensity"))
+ geom_point()
```

The method handoff remains available as ``tidy(df).ggplot(aes(...))``.
The bridge materializes to pandas for plot3; aggregate large data first.

## CRAFT / gpudev / SolveIt

Use this when you work in **SolveIt** (dialog notebook) and/or data lives on a
GPU host via CRAFT `%gpu`. Local VS Code setup above is enough for everyday
laptop work.

### SolveIt load (tidy3 + plot3)

```text
%local
%run /path/to/gpudev/CRAFT.py                 # if you need %gpu
%run /path/to/gpudev/addons/tidy3.py
%run /path/to/gpudev/addons/plot3.py          # ggplot / %plot3
```

Standalone SolveIt (no CRAFT) — preferred one-liner:

```text
%run /app/data/gpudevd/tidy3/tidy3.py
# or
%run /path/to/tidy3/tidy3.py
%run /path/to/tidy3/load.py          # same loader
```

That puts `src/` on the path, injects the API, and turns on multi-line `>>`
plus R-style bare names / backticks / `!` — same local experience as the
CRAFT addon, without `CRAFT.py` or `%gpu`.

After a normal editable install you can also use:

```text
%load_ext tidy3.jupyter
%load_ext plot3
```

### With `%gpu` (remote compute)

```text
%local
%run …/CRAFT.py
%run …/addons/tidy3.py
%run …/addons/plot3.py
%gpu
```

Under **`%gpu`**, cells run on the remote kernel (separate namespace +
filesystem). Both addons **push their source to the remote** over the CRAFT
channel (`tidy3.craft` / `plot3.craft`), install polars/pandas if missing, and
load Jupyter extensions there. Re-seed after kernel surgery with
`seed_tidy3_remote(force=True)` / `seed_plot3_remote(force=True)`.

```python
# after %gpu — paths are on the GPU box
scan_parquet("/home/gpudev/data/huge.parquet")
>> filter(col("year") >= 2020)
>> group_by("region")
>> summarise(n=n(), avg=mean("value"))
>> ggplot(aes(x="region", y="avg")) + geom_col()
```

`%plot3` is registered as a **host-local** magic (viewer + SolveIt red-eye stay
on the dialog machine) even while Python cells run remote.

### Partial run (any Jupyter / SolveIt)

- **Run Selected Text** (if the UI has it): highlight a pipe prefix → run  
- Own cell with only the prefix  
- `%%tidy3_run` with the prefix pasted in  

```python
%%tidy3_run
tidy(cars)
>> filter(col("mpg") > 20)
```

```text
%tidy3_pipes on|off|status
%tidy3_mask on|off|status    # R-style bare names / backticks
```

### EDA inspection

```python
names(cars)       # list[str]
colnames(cars)    # paste-ready selectors for select(...)
cars.columns      # same list as names
summary(cars)     # count, null_count, n_unique, mean, std, min, 25%, 50%, 75%, max
describe(cars)    # alias of summary (pandas-style name)
```

### R-style bare names & backticks

With the extension loaded (CRAFT addon or `%load_ext tidy3.jupyter`), cells may
omit many `col("…")` / quotes:

```python
cars >> filter(mpg > 20) >> mutate(z = if_else(cyl > 4, 1, 0))
cars_space >> mutate(x = `hp new` / cyl) >> select(`hp new`, x)
ggplot(df, aes(x=wt, y=mpg)) + geom_point()          # with plot3
ggplot(df, aes(x=`First Name`, y=mpg)) + geom_point()
```

- **Expression context** (`filter`, `mutate` RHS, …): bare name → `col("name")`
- **Selector context** (`select`, `group_by`, …): bare name → `"name"`
- **Backticks**: `` `any column name` `` for spaces / odd identifiers
- **plot3** `aes` / `facet_wrap` use the same style in Jupyter

#### Export notebook → plain Python script (`nb_export`)

R-style is the **authoring** form. For automation / CI, export rewrites it to
stock CPython (nbdev-style build artifact):

```python
from tidy3 import nb_export

nb_export("analysis.ipynb", "analysis_pipeline.py")
# only cells marked #| export (nbdev-style):
nb_export("analysis.ipynb", "lib.py", only_export=True)
```

```bash
python -m tidy3 export analysis.ipynb -o analysis_pipeline.py
python -m tidy3 export analysis.ipynb -o lib.py --only-export
```

Cell directives:

| Directive | Meaning |
|-----------|---------|
| `#\| export` | include when `--only-export` / `only_export=True` |
| `#\| skip` | never export (debug / interactive cells) |

What export does:

1. Collects code cells (skips markdown)
2. Applies the same bare-name / backtick / multi-line `>>` transforms as Jupyter
3. Applies plot3 `aes` masking when plot3 is installed
4. Turns internal sentinels into public API (`col("mpg")`, not `__tidy3_col__`)
5. Comments pure notebook magics (`%run`, …); best-effort rewrite of `%plot3`

The notebook stays the source of truth; re-export instead of hand-editing the
`.py`. Explicit `col("x")` / `aes(x="x")` still work everywhere as a compatible
subset.

Optional: run an unexported R-style script with the same transforms:

```bash
python -m tidy3 run job.py
```

### Symlink addons (CRAFT layout)

```bash
cd /path/to/gpudev/addons
ln -sfn /path/to/tidy3 tidy3
ln -sfn /path/to/plot3 plot3
```

## API (v0.2)

| Area | Symbols |
|------|---------|
| Export | `nb_export`, `transform_source`, `python -m tidy3 export` / `run` |
| Frame | `tidy`, `scan_parquet`, `scan_csv`, `scan_ipc`, `TidyFrame` |
| Output | `collect`, `to_numpy`, `TidyFrame.write_parquet`, `TidyFrame.write_csv`, `TidyFrame.write_ipc`, `TidyFrame.write_excel` |
| Rows | `filter`, `filter_out`, `arrange`, `distinct`, `slice`, `slice_head`, `slice_tail`, `slice_min`, `slice_max`, `slice_sample`, `head`, `sample_n`, `sample_frac` |
| Columns | `mutate`, `transmute`, `select`, `drop`, `rename`, `rename_with`, `relocate`, `pull`, `glimpse` |
| Groups | `group_by`, `rowwise`, `ungroup`, `with_groups`, `group_split`, `group_map`, `group_modify`, `group_nest`, `summarise`, `reframe`, `count`, `tally`, `add_count`, `add_tally` |
| Missing data | `drop_na`, `replace_na`, `fill`, `complete`, `expand`, `nesting` |
| Reshape | `pivot_longer`, `pivot_wider`, `separate`, `separate_longer_delim`, `separate_wider_delim`, `unite`, `nest`, `unnest`, `unnest_longer`, `unnest_wider`, `hoist`, `pack`, `unpack` |
| Joins | `left_join`, `right_join`, `inner_join`, `full_join`, `semi_join`, `anti_join`, `cross_join`, `nest_join` |
| Join specs | `join_by`, `eq`, `ge`, `gt`, `le`, `lt`, `closest`, `between`, `within`, `overlaps` |
| Bind/set | `bind_rows`, `bind_cols`, `union`, `union_all`, `intersect`, `setdiff`, `symdiff`, `setequal` |
| Row mutation | `rows_insert`, `rows_append`, `rows_update`, `rows_patch`, `rows_upsert`, `rows_delete` |
| Selectors | `everything`, `col_range`/`cols_between`, `last_col`, `group_cols`, `starts_with`, `ends_with`, `contains`, `matches`, `num_range`, `all_of`, `any_of`, `where`; set ops `\|` `&` `-` `~`/`!`/`-helper`; predicates `is_numeric`, `is_integer`, `is_float`, `is_string`/`is_character`, `is_bool`/`is_boolean`, `is_datetime`, `is_categorical`, `is_temporal` |
| Column-wise | `across`, `if_any`, `if_all`, `pick`, `c_across` |
| Materialize | `collect`, `pull`, `glimpse`, `peek` |
| Expr | `col`, `n`, `mean`, `sum`, `min`, `max`, `median`, `std`/`sd`, `var`, `any`, `all`, `first`, `last`, `nth`, `near`, `na_if`, `between`, `consecutive_id`, `case_match`, `recode`, ranking/window helpers, `n_distinct`, `coalesce`, `if_else`, `case_when` |
| Jupyter | `%load_ext tidy3.jupyter`, `%tidy3_run`, `%%tidy3_run`, `%tidy3_pipes` |
| Partial | `partial_run`, `maybe_rewrite_cell`, `normalize_pipe_source` |
| Escape | `TidyFrame.with_polars(fn)` |

### API maturity (v0.2)

tidy3 is alpha. Symbols work today, but not every verb is equally polished for
production pipelines or R byte-for-byte parity. Prefer the **stable core** when
you need predictable performance and semantics.

| Tier | Intent | Symbols / areas |
|------|--------|-----------------|
| **Stable core** | Daily driver: declarative dplyr on Polars, dual-backend tests, R oracle where installed | `tidy`, `scan_*`, `filter`, `filter_out`, `mutate`, `transmute`, `select`, `drop`, `rename`, `relocate`, `arrange`, `distinct`, `group_by`, `ungroup`, `summarise`/`summarize`, `count`, `tally`, `add_count`, `add_tally`, equality joins (`left`/`right`/`inner`/`full`/`semi`/`anti`/`cross`), `bind_rows`/`bind_cols`, set ops (`union`, `union_all`, `intersect`, `setdiff`, `symdiff`, `setequal`), `head`/`slice`/`slice_head`/`slice_tail`/`slice_min`/`slice_max`, core expr helpers (`col`, `n`, `mean`, `sum`, …), tidyselect basics, `collect` and file writers |
| **Growing** | Feature-complete enough for real work; more edge cases and schema-discovery cost | tidyr missing/reshape (`drop_na`, `replace_na`, `fill`, `complete`, `expand`, `pivot_*`, `separate`, `unite`, `nest`/`unnest*`), `across`/`if_any`/`if_all`/`pick`/`c_across`, `rowwise`, `reframe`, `rename_with`, `join_by` inequality joins, `nest_join`, `rows_*`, advanced ranking/window helpers, NumPy/`to_numpy` handoff |
| **Experimental** | Correctness-first or Python-callback paths; may materialize eagerly or lag R oracle coverage | `group_split`, `group_map`, `group_modify`, `group_nest`, `with_groups`, `hoist`, `pack`, `unpack`, `separate_longer_delim`, `separate_wider_delim`, stochastic sampling (`sample_n`, `sample_frac`, `slice_sample`) |

**Performance note:** the stable core is the path that tracks raw Polars and
beats pandas on realistic sizes. `group_nest` compiles to a Polars
`group_by().agg(struct)` plan and is typically **faster** than building
nested pandas DataFrames by hand; for counts alone use `count()`/`tally()`
instead of nesting. Python-callback verbs (`group_map` / `group_modify` /
`group_split`) still materialize and loop in Python — fine for small groups,
not for hot large-data paths. Prefer declarative `summarise` when
performance matters.

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

### Tidy-select and column-wise operations

Selectors work anywhere columns are selected by `select`, `drop`, `relocate`,
or `rename_with`. Combine them with `|` (union), `&` (intersection), `-`
(difference), or `~` (complement):

```python
tidy(df)
>> select("id", starts_with("measure_"), last_col())
>> select(!starts_with("tmp_"))            # Jupyter: ! → ~ ; or use ~starts_with
>> select(where(is_numeric) & !starts_with("id"))
>> select(cols_between("mpg", "hp"))       # inclusive column range

tidy(df)
>> select(everything() - ends_with("_raw"))

tidy(df)
>> relocate(where(is_numeric), after="label")
```

`where` receives column dtypes; portable predicates include `is_numeric`,
`is_string`, `is_boolean`, and `is_temporal`. `all_of(names)` is strict about
missing columns while `any_of(names)` silently ignores them.

### Reshape and missing data

The familiar tidyr operations work as pipe verbs or `TidyFrame` methods on
both backends:

```python
from tidy3 import (
    drop_na, fill, pivot_longer, pivot_wider, replace_na,
    separate, starts_with, tidy, unite, unnest_longer,
)

long = (
    tidy(measurements)
    >> fill("patient_id", direction="down")
    >> pivot_longer(
        starts_with("week_"),
        names_to="week",
        names_prefix="week_",
        values_to="reading",
        values_drop_na=True,
    )
)

wide = long >> pivot_wider(
    names_from="week",
    values_from="reading",
    names_prefix="week_",
    values_fill=0,
)

# Multiple measures use tidyr's value-first names (sales_Q1, cost_Q1).
wide_measures = tidy(long_measures) >> pivot_wider(
    names_from="quarter", values_from=["sales", "cost"]
)

# .value takes output value-column names from the input column names.
tidy(measurements) >> pivot_longer(
    starts_with(("mean_", "sd_")),
    names_to=[".value", "visit"],
    names_sep="_",
)

tidy(labels) >> separate("code", ["region", "id"], sep="-")
tidy(labels) >> unite("code", "region", "id", sep="-")
tidy(events) >> unnest_longer("items", indices_to="item_index")
tidy(values) >> replace_na({"score": 0}) >> drop_na("required_field")

tidy(observations) >> complete(
    "subject", "visit", fill={"score": 0}, explicit=False
)

nested = tidy(events) >> nest("records", cols=["time", "value"])
restored = nested >> unnest("records")
```

`fill(..., by=...)` provides temporary grouping, while an existing
`group_by()` is respected automatically. Most Polars operations add only lazy
plan nodes. `pivot_wider()` must know its output schema, so it discovers the
distinct `names_from` values unless `names=[...]` is supplied; providing
`names` avoids that metadata query for known categories. `unnest_wider()`
similarly discovers the width of unnamed list values.

`pivot_wider()` accepts multiple `names_from` and `values_from` columns.
`pivot_longer()` supports the `.value` sentinel, and `separate(convert=True)`
performs R-style logical/numeric inference on both backends. Schema-dependent
reshape and conversion operations issue a metadata-only query on Polars.

Use `across` as a positional argument to `mutate`, `transmute`, or
`summarise`. Python name templates accept `{col}` and `{fn}`; dplyr-style
`{.col}` and `{.fn}` are accepted too. Use `cur_column()` inside a function
when its calculation depends on the selected column name, and
`cur_group()["group_name"]` when it depends on a grouped key:

```python
tidy(df)
>> mutate(across(starts_with("x"), lambda x: x.round(2)))
>> filter(if_any(ends_with("_score"), lambda x: x > 0))
>> group_by("team")
>> summarise(
    across(
        where(is_numeric),
        {"mean": mean, "sd": std},
        names="{col}_{fn}",
    )
)

tidy(df) >> mutate(
    across(starts_with("x"), lambda x: x + len(cur_column()))
)

tidy(df) >> group_by("team") >> mutate(
    across(starts_with("x"), lambda x: x - cur_group()["team"])
)

tidy(df) >> group_by("team") >> mutate(
    across(starts_with("x"), lambda x: x + cur_group_id())
)
```

Functions passed to `across` can return a mapping of named expressions. Pass
`unpack=True` to expand those fields into regular columns; use an
`{outer}`/`{inner}` template to control the resulting names:

```python
tidy(df) >> mutate(
    across(
        starts_with("x"),
        lambda x: {"double": x * 2, "plus_one": x + 1},
        unpack="{outer}__{inner}",
    )
)
```

### Row-wise operations and reframing

`rowwise` evaluates ordinary window/aggregate expressions one row at a time.
Its optional tidy-select arguments are identifier columns preserved by
`summarise`. Use `c_across` for row-wise reductions; grouping identifiers are
automatically excluded:

```python
tidy(df)
>> rowwise("id")
>> mutate(
    total=sum(c_across(starts_with("score_"))),
    average=mean(c_across(starts_with("score_"))),
)
>> ungroup()
```

`pick` represents a selected set as one structured column and supports common
horizontal reductions without requiring `rowwise`:

```python
tidy(df) >> mutate(total=pick(starts_with("score_")).sum())
tidy(df) >> mutate(scores=pick(starts_with("score_")))
```

`reframe` accepts vector-valued expressions, recycles scalar results within
each group, and always returns an ungrouped frame:

```python
tidy(df)
>> group_by("team")
>> reframe(score=col("score"), team_mean=mean("score"))
```

### dplyr-compatible evaluation controls

Assignments in one `mutate()` are evaluated from left to right, so later
expressions can use columns created earlier. Independent assignments are
still fused into one backend operation:

```python
tidy(df) >> mutate(
    doubled=col("x") * 2,
    squared=col("doubled") ** 2,
    keep="used",                  # all, used, unused, or none
    before="x",
)
```

`distinct("id")` returns only the grouping columns and `id`, matching
dplyr. Use `distinct("id", keep_all=True)` to retain the first complete row.
Multi-level summaries default to dropping only the final group; override this
with `groups="drop"`, `"drop_last"`, `"keep"`, or `"rowwise"`.

Computed and additive groups and partial ungrouping are supported:

```python
tidy(df)
>> group_by("region")
>> group_by(add=True, decade=col("year") // 10)
>> ungroup("decade")
```

Aggregates accept `na_rm=` and default to `False`, matching dplyr: a missing
value propagates unless removal is requested explicitly. Use
`mean("x", na_rm=True)` to ignore missing values.

### Row mutation and advanced joins

The SQL-inspired row verbs use `y`'s first column as the key by default, or
accept explicit `by=` keys. `y` may contain any subset of `x`'s columns:

```python
tidy(accounts)
>> rows_insert(new_accounts, by="id", conflict="ignore")
>> rows_patch(corrections, by="id")       # only replaces missing values
>> rows_upsert(latest, by="id")           # update matches, append new keys
>> rows_delete(retired, by="id")
```

`rows_insert` supports `conflict="error"|"ignore"`; update, patch, and delete
support `unmatched="error"|"ignore"`. Polars validates these policies lazily
when the plan is collected.

`join_by` accepts same-name equality keys, `(left, right)` equality pairs, or
`(left, operator, right)` conditions. Named helpers are clearer for advanced
joins:

```python
sales
>> left_join(
    promos,
    by=join_by("id", ge("sale_date", "promo_date")),
)

# One nearest earlier promotion per sale
sales
>> left_join(
    promos,
    by=join_by("id", closest(ge("sale_date", "promo_date"))),
)

points >> inner_join(ranges, by=join_by(between("point", "lower", "upper")))
segments >> inner_join(regions, by=join_by(overlaps("lo", "hi", "start", "end")))
```

Advanced specifications work with left/right/inner/full/semi/anti joins.
Polars uses its native inequality-join plan when no equality partition is
needed; equality-partitioned inequalities first reduce candidates by key.

### Per-operation grouping and portable helpers

Use `by=` when grouping is needed for one operation only. It accepts the same
tidy-select specifications as `select()` and returns an ungrouped result:

```python
tidy(sales)
>> mutate(region_average=mean("amount"), by="region")
>> filter(col("amount") > mean("amount"), by="region")

tidy(sales) >> summarise(total=sum("amount"), by=["region", "year"])
tidy(sales) >> slice_max("amount", n=2, by="region")
```

`by=` is supported by `mutate`, `transmute`, `filter`, `filter_out`,
`summarise`, `reframe`, `slice`, and every `slice_*` variant. Like dplyr,
it cannot be combined with an already grouped or rowwise frame.

Portable helpers run with the same grouped semantics on both backends:

```python
tidy(events) >> mutate(
    row=row_number(),
    rank=dense_rank("score"),
    previous=lag("score"),
    running_average=cummean("score"),
    label=case_when(
        (col("score").is_null(), "missing"),
        (col("score") >= 90, "high"),
        default="other",
    ),
    by="team",
)
```

Ranking helpers are `row_number`, `min_rank`, `dense_rank`, `percent_rank`,
`cume_dist`, and `ntile`. Window and value helpers include `lead`, `lag`,
`cummean`, `cumall`, `cumany`, `n_distinct`, `coalesce`, `if_else`, and
`case_when`. `lead`, `lag`, `first`, `last`, and `nth` accept `order_by=`.

Mutating joins accept dplyr-style safety controls:

```python
orders >> left_join(
    customers,
    on="customer_id",
    relationship="many-to-one",
    unmatched="error",
    multiple="all",
    na_matches="never",
    keep=False,
)
```

`relationship` accepts `one-to-one`, `one-to-many`, `many-to-one`, or
`many-to-many`; `multiple` accepts `all`, `any`, `first`, or `last`.
Polars relationship and unmatched checks remain lazy and raise on collection.

Every frame-returning Phase 1–8 verb stays lazy on the Polars backend. `pull`,
`setequal`, plotting, previews, and explicit `collect` are intentional
materialization boundaries. Schema-dependent reshape operations may run a
small metadata query as described above. `slice_sample(weight_by=...)`,
including `replace=True`, is supported on both backends. Group-context helpers
and additional vector helpers remain future phases.

### Performance controls

Grouped `mutate()` expressions automatically reuse embedded or repeated
statistics. Expressions such as the following compute each group mean and
standard deviation once on both backends; temporary columns never appear in
the result:

```python
tidy(features) >> mutate(
    filled=coalesce(col("x"), mean("x", na_rm=True)),
    z=(
        coalesce(col("x"), mean("x", na_rm=True))
        - mean("x", na_rm=True)
    ) / std("x", na_rm=True),
    by="segment",
)
```

Project before a pandas/Arrow handoff without adding another pipeline verb:

```python
matrix = result.collect(
    as_="pandas",
    columns=["customer_id", starts_with("feature_")],
    arrow_backed=True,
)
```

`arrow_backed=True` still returns a pandas DataFrame, but avoids copying
Polars columns into NumPy buffers. Leave it off when a downstream library
specifically requires NumPy-backed pandas dtypes.

`distinct(..., maintain_order=True)` preserves dplyr row order by default.
Use `maintain_order=False` when output order is irrelevant and throughput is
more important.

### NumPy, Numba, and PyTorch

Build the matrix directly from the lazy pipeline and project away identifiers
or string columns before materializing:

```python
import numpy as np
from tidy3 import starts_with, to_numpy

features = result.to_numpy(
    columns=["feature_a", "feature_b", "feature_c"],
    dtype=np.float32,
    order="c",
    writable=True,
)

# Equivalent output boundaries
features = result.collect(as_="numpy", columns=starts_with("feature_"))
features = result >> to_numpy(columns=starts_with("feature_"))
features = np.asarray(result)  # all columns; materializes the lazy plan
```

Numba functions accept the returned array directly. For PyTorch,
`torch.from_numpy(features)` shares the CPU array's memory, so request
`writable=True`; use `order="c"` when a consumer requires C-contiguous input.
The default is Fortran order because tidy3/Polars are columnar and it offers
the best chance of avoiding a copy. Set `allow_copy=False` when an unexpected
copy should be an error.

This is a host-memory bridge even when `engine="gpu"`: Polars may execute the
query on the GPU, but the NumPy result resides in CPU memory. A future DLPack
bridge would be needed for a direct device-to-PyTorch handoff.

## Benchmark vs datar

```python
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

## Examples

Local end-to-end prep for algorithms (no CRAFT required):

```bash
pip install -e ".[dev]"
python examples/prep_for_ml.py
python examples/prep_for_ml.py --torch   # if torch is installed
```

That script builds a tidy feature frame, materializes a NumPy matrix with
`to_numpy`, optionally wraps it for PyTorch, and writes a plot3 scatter when
plot3 is on the path.

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q
```

CI (GitHub Actions):

- **Tests** — `pytest` on Python 3.10–3.12 for every push/PR; optional **R
  semantic oracle** job installs dplyr/tidyr/jsonlite and runs the differential
  suite
- **Performance budget** — `python -m tidy3.bench_suite` geometric ratios vs raw
  pandas (see `.github/workflows/`). Experimental Python-callback workloads
  (e.g. `group_modify`) are timed but excluded from the adoption geo-mean.

On a machine without R, oracle tests skip automatically. To run them locally:

```bash
# after installing R + dplyr, tidyr, jsonlite
pytest -q tests/test_r_oracle_parity.py
```

Uncommitted work on `expand-dplyr-parity` is staged in logical commits via
`docs/commit-plan-expand-dplyr-parity.md`.

## Notes

- **Grouped semantics are dplyr's**: after `group_by`, `mutate`/`filter`/
  `slice_*`/`sample_n`/`head` evaluate **per group** (Polars window `.over`).
  `summarise` aggregates per group.
- **Display is self-contained**: tables carry inline styles (no `<style>`
  block), so they render identically in local cells, republished `%gpu`
  output, and sslive exports. Under `%gpu`, bare polars DataFrames are
  restyled the same way (`seed_tidy3_remote(style_polars=False)` to opt out).
- **Builtins are shadowed** by design: injecting the API puts `filter`, `slice`,
  `sum`, `min`, `max`, `any`, and `all` into the notebook namespace. Use
  `builtins.filter` etc. when you need the Python originals.
- **Plotting big remote data**: aggregate remotely, then let plot3's own
  remote path pull the small result: `%plot3 res.to_pandas() x=... y=...`.
- **API maturity tiers** (stable / growing / experimental) are listed under
  [API maturity](#api-maturity-v02). Prefer the stable core for adoption and
  performance-sensitive work.

## Why not datar?

datar’s pandas/Python wrappers struggle on large frames. tidy3 compiles verbs to a **Polars Lazy** plan and only materializes at the edge. Previews use `LIMIT n`.

## License

MIT
