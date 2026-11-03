# tidy3 in VS Code and notebooks

Setting up a local environment, writing pipes in scripts and notebooks, R-style bare column names, and exporting a notebook to a plain Python script.

Back to the [README](../README.md).

The examples on this page share this setup:

```python
from tidy3 import *

# mtcars, read from the web; `model` holds the car names
cars = (
    scan_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv")
    >> rename(model="rownames")
    >> select("model", "mpg", "cyl", "hp", "wt")
    >> compute()
)
cars.write_parquet("cars.parquet")

result = (
    scan_parquet("cars.parquet")
    >> filter(col("mpg") > 20)
    >> mutate(km=col("mpg") * 1.609)
    >> group_by("cyl")
    >> summarise(n=n(), avg=mean("mpg"))
)
```

## VS Code (local IDE)

tidy3 is a normal package. No CRAFT, SolveIt, or remote kernel is required.

### One-time setup

1. Open your project folder in VS Code.
2. Create a virtual environment and install tidy3 (terminal in VS Code):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install tidy3
```

3. Select it: Command Palette → **Python: Select Interpreter** →
   `./.venv/bin/python`.
4. Recommended extensions: **Python** (includes **Pylance**) and **Jupyter**.
   Optional: **Polars** is a library dependency, not a VS Code extension.

### Type checker / red squiggles under `>>` pipes

You do **not** need a special tidy3 VS Code extension. Pylance is enough once
tidy3 is installed (types ship with `py.typed`).

1. Interpreter = your project's `.venv` (Command Palette → **Python: Select Interpreter**).
2. Import the names you use in that cell/file:

```python
from tidy3 import tidy, select, filter, arrange, slice_max, col, desc
```

3. Reload the window if squiggles linger: **Developer: Reload Window**.

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
3. Import tidy3. An ordinary import is all the core API needs, and ordinary
   Python syntax (parentheses around a multi-line pipe, quoted column names)
   works in the same cell as the import:

```python
from tidy3 import tidy, filter, mutate, col

(
    tidy(cars)
    >> filter(col("mpg") > 20)
    >> mutate(km=col("mpg") * 1.609)
)
```

#### Shorter notebook syntax needs an earlier cell

Multi-line `>>` pipes without the outer parentheses and bare column names
are not Python, so tidy3 rewrites each cell before IPython runs it. The
rewriting has to be on before the cell starts, and a cell can't turn it on
for itself: IPython parses the whole cell first, and the import inside it
runs too late. Once any cell that imports tidy3 has run, the rewriting is
on for the cells after it. To turn it on explicitly, run this in a cell of
its own:

```python
%load_ext tidy3.jupyter
```

From the next cell on:

```python
tidy(cars)
>> filter(col("mpg") > 20)
>> mutate(km=col("mpg") * 1.609)
```

A cell can mix pipes with comments and other code. Each line that starts
with `>>` continues the statement above it, even across comment and blank
lines; a `+` line continues a pipe into plot3 layers:

```python
from plot3 import aes, geom_point, ggplot

# Cars with good mileage
good = tidy(cars)
>> filter(col("mpg") > 20)   # fast enough
# keep what we plot
>> select("mpg", "hp")

good
>> ggplot(aes(x="hp", y="mpg"))
+ geom_point()
```

Inside an indented block (`if`, `for`, `def`), wrap the pipe in parentheses.

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

# plot3 (installed with tidy3)
# from plot3 import aes, geom_point, ggplot
# result >> ggplot(aes(x="cyl", y="avg")) + geom_point()
```

### Local vs CRAFT at a glance

| | Local (VS Code) | CRAFT / `%gpu` |
|--|-----------------|----------------|
| Install | `pip install tidy3` in a venv | Addon seed to remote kernel |
| Paths | Your machine | Paths on the GPU host |
| Pipes | Parentheses in `.py`; extension optional in notebooks | Same extension after seed |
| Data size | Laptop RAM / local Polars | Remote GPU box + large files |
| Default for new users | **Yes** | Optional power path |

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

### Paged tables

A frame in a notebook prints as a paged table, as R notebooks print a data
frame: ten rows a page, with Previous / page / Next buttons, through its
first 100 rows. Names and types stay in the header, and a wide frame shows
as many columns as fit, with arrows to the rest; the footer says which rows
and columns are showing, and how many rows there are in all.

```python
cars >> distinct("cyl", "hp")               # every combination, ten a page
options(preview_rows=25)                    # 25 a page (and in the text preview)
options(preview_max=1000)                   # page through the first 1,000 rows
options(preview_digits=4)                   # floats to four significant digits
options(preview_digits="full")              # every stored digit
```

Floats print as in the text preview, up to six decimal places, which shows
`0.000123456789` as `0.000123`; `preview_digits=4` keeps four significant
digits instead (`0.0001235`, as R's `signif()`). Every row the table holds
is saved with the notebook, about 100 KB for 100 rows of a dozen columns,
so raise `preview_max` for the cells that need it. Paging needs the notebook
to run output scripts; where it doesn't (an untrusted Jupyter notebook), the
same rows show in one box that scrolls. The table follows the notebook's
light or dark theme in VS Code, JupyterLab and SolveIt.

### Explore: `view()`

`view()` opens a data explorer, as R's `View()` does: one grid of up to
10,000 rows to scroll, with row numbers, a search box, headers that sort on a
click (ascending, descending, off), and a filter under each column:

```python
cars >> view()                           # or view(cars), or cars.view()
cars >> filter(mpg > 20) >> view() >> count("cyl")   # it passes the frame on
cars >> view(n=50_000, height=700)       # more rows, a taller grid
cars >> view(digits=4)                   # floats to four significant digits
```

| Column | Filter |
|---|---|
| numbers, dates | from and to, either blank; `2008` matches a whole year |
| text | contains, ignoring case |
| booleans, and text with at most 50 values | a pick-list, `null` included |

The status line counts the rows that pass ("42 of 344 rows | 8 columns") and
says when `n` cut the rows. The rows go into the notebook's output, so the
explorer still works when the notebook is reopened, about 1 MB for 10,000
rows of a dozen columns. It needs the notebook to run output scripts, as
paging does; where it doesn't, the first 100 rows show as a plain table.

In SolveIt, which sends every cell's code and output to its AI, `view()`
hides its cell from the AI (the red eye), as plot3 does for plots, so the
rows don't fill the model's context; the cell's code is hidden with them,
so keep `view()` on a line of its own. `view(hide=False)` keeps a cell
visible, and `table3.autohide(False)` stops hiding for the session. Printed
tables are not hidden.

### R-style bare names & backticks

Once the rewriting is on (an earlier cell imported tidy3 or ran
`%load_ext tidy3.jupyter`, or the CRAFT addon is loaded), later cells may
omit many `col("…")` / quotes. The cell that turns it on can't use them yet:

```python notest
from tidy3 import *
cars >> filter(mpg > 20)     # NameError: mpg; the rewriting starts next cell
```

```python
cars >> filter(mpg > 20) >> mutate(z = if_else(cyl > 4, 1, 0))
cars_space = cars >> rename(`hp new` = hp)             # a name with a space
cars_space >> mutate(ratio = `hp new` / cyl) >> select(`hp new`, ratio)
cars >> select(~starts_with("w"))      # prefer ~ for negation
cars >> select(!starts_with("w"))      # optional Jupyter sugar → ~ (tidy3 only)
ggplot(cars, aes(x=wt, y=mpg)) + geom_point()          # with plot3
ggplot(cars_space, aes(x=`hp new`, y=mpg)) + geom_point()
```

`!pip install …` and other notebook shell commands are **not** rewritten.

- **Expression context** (`filter`, `mutate` RHS, …): bare name → `col("name")`
- **Selector context** (`select`, `group_by`, …): bare name → `"name"`
- **Backticks**: `` `any column name` `` for spaces / odd identifiers
- **Columns named like Python builtins or tidy3 helpers** (`id`, `type`, `max`,
  `n`) work bare wherever only a column makes sense: `filter(id > 1)`,
  `select(id, type)`, `arrange(desc(id))`, `mutate(z = max * 2)`. Passed as a
  function, as in `across(everything(), mean)`, they stay functions.
- **Columns win over notebook variables**, as in dplyr: with `x = 100` in the
  notebook, `filter(x > limit)` still compares the column `x`, and `limit`
  is your variable because there is no column called `limit`. Use `env.x`
  for the variable when a column has the same name (dplyr's `.env$x`), or
  `col("x")` to insist on the column. Within one `mutate()`, a column made
  earlier counts: `mutate(y = x * 2, z = y + 1)`. Calls skip non-functions
  like R does, so `n()` still counts rows after `n = 7`.
- **plot3** `aes` / `facet_wrap` use the same style in Jupyter

#### `left ~ right` rules

Where R writes a rule as `condition ~ result`, a notebook cell may too, in
`case_when`, `replace_when`, `case_match`, `recode_values` and
`replace_values`. The rewriting turns each `left ~ right` into the
`(left, right)` pair those functions already take, so the two spellings can
share a call and give the same result:

```python notest
cars >> mutate(
    size = case_when(cyl > 6 ~ "big", cyl > 4 ~ "mid", default = "small"),
    gears = recode_values(gear, 3 ~ "low", 4 ~ "mid", default = "high"),
    cyl2 = replace_values(cyl, 8 ~ None),
)
cars >> mutate(known = case_when(!is_na(mpg) ~ mpg, default = 0))    # ! or ~ negates
cars >> mutate(size = case_when((cyl > 6, "big"), cyl > 4 ~ "mid", default = "small"))
```

- The separator is a `~` that follows a value. A `~` before a value is still
  negation, so `~is_na(mpg) ~ mpg` is the rule "not missing → mpg".
- `case_match`, `recode_values`, `replace_values` and `replace_when` take the
  column first, then the rules. Keyword arguments such as `default=` keep
  their spelling.
- A `~` inside a string, a backtick name, or brackets is left alone. A rule
  with no value on one side, or with two separators, is an error that names
  the function and the rule.
- Python itself has no binary `~`, so this works only where tidy3 rewrites
  the cell. A plain `.py` file needs the tuple spelling, and
  `nb_export` writes tuples into the script it produces.

#### Export notebook → plain Python script (`nb_export`)

R-style is the **authoring** form. For automation / CI, export rewrites it to
stock CPython (nbdev-style build artifact):

```python notest
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
