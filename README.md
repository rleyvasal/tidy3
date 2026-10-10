# tidy3

**dplyr for Python, on Polars.** Write R-style pipes in a notebook, one
verb per line with comments in between, see every column's type, and hand
the result to [plot3](https://github.com/rleyvasal/plot3), scikit-learn,
NumPy, or pandas without leaving the pipe.

<img src="https://raw.githubusercontent.com/rleyvasal/tidy3/main/docs/img/notebook_dark.png" alt="A notebook: multi-line tidy3 pipes with comments summarise the penguins into a table that shows each column's type under its name, then flow into a plot3 scatter with trend lines">

```bash
pip install "tidy3[jupyter,plot3]"
```

Contents: [In a notebook](#in-a-notebook) · [From R](#if-you-know-dplyr) ·
[Lazy and fast](#lazy-until-you-need-rows) · [Reshape](#reshape-for-plotting) ·
[Plot](#plot-with-plot3) · [scikit-learn](#hand-off-to-scikit-learn-numpy-and-pandas) ·
[Scripts](#in-a-python-script) · [Full reference](docs/reference.md)

## In a notebook

Load the extension once. After that, a pipe runs over as many lines as you
like, each line starting with `>>`, comments and blank lines included, and
column names go bare, as in R:

```python
%load_ext tidy3.jupyter
from tidy3 import *

penguins = scan_csv(
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv",
    null_values="NA",
) >> compute()

# Heaviest groups first
penguins
>> drop_na(sex)                        # 11 penguins have no recorded sex
>> group_by(species, sex)
>> summarise(
    n = n(),
    bill_mm = mean(bill_length_mm).round(1),
    mass_kg = (mean(body_mass_g) / 1000).round(2),
)
>> arrange(desc(mass_kg))
```

```text
TidyFrame (preview groups=['species'])
shape: (6, 5)
┌───────────┬────────┬─────┬─────────┬─────────┐
│ species   ┆ sex    ┆ n   ┆ bill_mm ┆ mass_kg │
│ ---       ┆ ---    ┆ --- ┆ ---     ┆ ---     │
│ str       ┆ str    ┆ u32 ┆ f64     ┆ f64     │
╞═══════════╪════════╪═════╪═════════╪═════════╡
│ Gentoo    ┆ male   ┆ 61  ┆ 49.5    ┆ 5.48    │
│ Gentoo    ┆ female ┆ 58  ┆ 45.6    ┆ 4.68    │
│ Adelie    ┆ male   ┆ 73  ┆ 40.4    ┆ 4.04    │
│ Chinstrap ┆ male   ┆ 34  ┆ 51.1    ┆ 3.94    │
│ Chinstrap ┆ female ┆ 34  ┆ 46.6    ┆ 3.53    │
│ Adelie    ┆ female ┆ 73  ┆ 37.3    ┆ 3.37    │
└───────────┴────────┴─────┴─────────┴─────────┘
```

Every table shows each column's type under its name, and the caption
keeps the grouping that's left (`summarise` drops the last group, as in
dplyr). `glimpse(penguins)` prints one line per column. To check part of a
long pipe, run its first lines with `%%tidy3_run`.

## If you know dplyr

The verbs, helpers, and arguments are dplyr's and tidyr's, with `>>` for
R's `|>`:

| R (dplyr) | tidy3 |
|---|---|
| `penguins \|> filter(species == "Gentoo")` | `penguins >> filter(species == "Gentoo")` |
| `mutate(mass_kg = body_mass_g / 1000)` | `mutate(mass_kg = body_mass_g / 1000)` |
| `group_by(species) \|> summarise(n = n(), bill = mean(bill_length_mm, na.rm = TRUE))` | `group_by(species) >> summarise(n = n(), bill = mean(bill_length_mm, na_rm = True))` |
| `count(species, island, sort = TRUE)` | `count(species, island, sort = True)` |
| `select(species, ends_with("_mm"))` | `select(species, ends_with("_mm"))` |
| `select(-year)` | `select(-year)` |
| `pivot_longer(ends_with("_mm"), names_to = "measure")` | `pivot_longer(ends_with("_mm"), names_to = "measure")` |
| `` rename(`body mass` = body_mass_g) `` | `` rename(`body mass` = body_mass_g) `` |
| `across(all_of(vars), mean)` | `across(all_of(vars), mean)` |

As in dplyr, a column beats a notebook variable of the same name; write
`env.x` for the variable. Python's own spellings differ in a few places:
`na_rm` for `na.rm`, `True` for `TRUE`, and `_x`/`_y` join suffixes for
`.x`/`.y`.

## Lazy until you need rows

A pipe is a plan, not a computation. Polars optimises the whole plan, reads
only the columns it uses, and pushes filters into the file scan. Rows are
computed when something needs them:

- a preview (a notebook table shows its first rows only),
- a plot, `to_numpy()`, or `write_parquet()` / `write_csv()`,
- `collect()`, which returns a Polars, pandas, Arrow, or NumPy result,
- `compute()`, which runs the plan so far and keeps piping from the
  in-memory result. Use it for a table you'll reuse, like `penguins`
  above: the CSV downloads once, not once per step.

```python
penguins.collect()                  # Polars DataFrame
penguins.collect(as_="pandas")      # pandas DataFrame
```

On 10 million rows, a filter → mutate → group_by → summarise → arrange
pipeline takes **41 ms** in tidy3, the same as hand-written Polars (42 ms),
against 60 ms in pandas and 166 ms in datar
([benchmarks](docs/benchmarks.md)).

## Reshape for plotting

`pivot_longer` turns the three measurement columns into one, the shape
facets want:

```python
long = penguins
>> mutate(id = row_number())
>> select(id, species, ends_with("_mm"))
>> pivot_longer(ends_with("_mm"), names_to = "measure", values_to = "mm",
                values_drop_na = True)

long >> head(4)
```

```text
┌─────┬─────────┬───────────────────┬───────┐
│ id  ┆ species ┆ measure           ┆ mm    │
│ --- ┆ ---     ┆ ---               ┆ ---   │
│ i64 ┆ str     ┆ str               ┆ f64   │
╞═════╪═════════╪═══════════════════╪═══════╡
│ 1   ┆ Adelie  ┆ bill_length_mm    ┆ 39.1  │
│ 1   ┆ Adelie  ┆ bill_depth_mm     ┆ 18.7  │
│ 1   ┆ Adelie  ┆ flipper_length_mm ┆ 181.0 │
│ 2   ┆ Adelie  ┆ bill_length_mm    ┆ 39.5  │
└─────┴─────────┴───────────────────┴───────┘
```

A pipe flows into [plot3](https://github.com/rleyvasal/plot3)'s `ggplot()`,
and a line starting with `+` adds a layer, as in R:

```python
from plot3 import *

long
>> ggplot(aes(x = species, y = mm, fill = species))
+ geom_boxplot()
+ facet_wrap("measure", scales = "free_y")   # one panel per measurement
+ scale_fill_hue()                           # ggplot2's default colours
+ theme_grey()                               # and its grey panel
+ theme(legend_position = "none")
```

<img src="https://raw.githubusercontent.com/rleyvasal/tidy3/main/docs/img/facets_ggplot.png" alt="Boxplots of bill depth, bill length, and flipper length by species, one panel each, in ggplot2's grey theme and default colours">

plot3 has ggplot2's themes and scales, so a figure can look exactly like
one from R. The other figures here use `theme_dark()`, the look of plot3's
interactive viewer.

`pivot_wider`, `separate`, `unite`, `nest`/`unnest`, `complete`, and the
rest of tidyr are in the [reference](docs/reference.md#reshape-and-missing-data).

## Plot with plot3

One scatter shows Simpson's paradox: over all penguins, longer bills are
shallower (grey), but within each species they are deeper:

```python
penguins
>> drop_na(bill_length_mm, bill_depth_mm)
>> ggplot(aes(x = bill_length_mm, y = bill_depth_mm, colour = species))
+ geom_point(alpha = 0.6)
+ geom_smooth(method = "lm")                     # one trend per species
+ geom_smooth(method = "lm", colour = "grey70")  # and one for all penguins
```

<img src="https://raw.githubusercontent.com/rleyvasal/tidy3/main/docs/img/simpson_dark.png" alt="Bill depth against bill length: each species trends upward, all penguins together trend downward">

Figures are interactive (zoom, hover, orbit in 3D) and save for papers with
`ggsave("fig.pdf", p)`. plot3's animations work the same way, here
gapminder's life expectancy against income, year by year:

```python
gapminder = scan_csv("https://raw.githubusercontent.com/kirenz/datasets/master/gapminder.csv")

gapminder
>> mutate(pop_millions = pop / 1e6)
>> ggplot(aes(x = gdpPercap, y = lifeExp, size = pop_millions,
              colour = continent, group = country))
+ geom_point(alpha = 0.7)
+ scale_x_log10()
+ transition_time("year")
+ labs(title = "{frame_time}", x = "GDP per capita", y = "Life expectancy")
```

<img src="https://raw.githubusercontent.com/rleyvasal/tidy3/main/docs/img/gapminder_dark.gif" width="60%" alt="Gapminder bubbles moving from 1952 to 2007">

## Hand off to scikit-learn, NumPy, and pandas

tidy3 does the cleaning and scaling, `to_numpy()` hands scikit-learn a plain
matrix, and the model's output comes straight back as a column. Here k-means
finds three groups in the penguins' bill and flipper sizes:

```python notest
from sklearn.cluster import KMeans

features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]
complete = penguins >> drop_na(all_of(features)) >> compute()

X = complete
>> mutate(across(all_of(features), lambda x: (x - mean(x)) / sd(x)))   # R's scale()
>> to_numpy(columns = features)

kmeans = KMeans(n_clusters=3, n_init=10, random_state=1).fit(X)

clustered = complete >> mutate(cluster = kmeans.labels_ + 1)   # labels come back
clustered >> count(species, cluster)
```

```text
┌───────────┬─────────┬─────┐
│ species   ┆ cluster ┆ n   │
│ ---       ┆ ---     ┆ --- │
│ str       ┆ i32     ┆ u32 │
╞═══════════╪═════════╪═════╡
│ Adelie    ┆ 1       ┆ 146 │
│ Adelie    ┆ 3       ┆ 5   │
│ Chinstrap ┆ 1       ┆ 5   │
│ Chinstrap ┆ 3       ┆ 63  │
│ Gentoo    ┆ 2       ┆ 123 │
└───────────┴─────────┴─────┘
```

The clusters match the species for 332 of 342 penguins. The centres, in
millimetres, are a `group_by`, not an inverse transform:

```python notest
centres = clustered >> group_by(cluster) >> summarise(across(all_of(features), mean))

clustered
>> ggplot(aes(x = flipper_length_mm, y = bill_length_mm, colour = factor(cluster)))
+ geom_point(aes(shape = species), alpha = 0.75)
+ geom_point(data = centres, size = 16, shape = "cross", colour = "grey70")
```

<p>
<img src="https://raw.githubusercontent.com/rleyvasal/tidy3/main/docs/img/clusters_dark.png" width="54%" alt="Penguins coloured by k-means cluster, shaped by species, with grey crosses at the cluster centres">
<img src="https://raw.githubusercontent.com/rleyvasal/tidy3/main/docs/img/clusters_3d_dark.png" width="44%" alt="The three features in 3D, coloured by cluster">
</p>

The whole example, as a script, is
[examples/cluster_penguins.py](examples/cluster_penguins.py)
(`pip install scikit-learn`). Other handoffs:

| To | Call |
|---|---|
| NumPy, PyTorch, Numba | `to_numpy(columns=..., dtype=np.float32, order="c", writable=True)` |
| pandas | `collect(as_="pandas")`, or `arrow_backed=True` to skip the copy |
| Arrow, Polars | `collect(as_="arrow")`, `collect()` |
| Files | `write_parquet`, `write_csv`, `write_ipc`, `write_excel` |
| GPU | `collect(engine="gpu")` with `cudf-polars` installed |

## In a Python script

A plain `.py` file is ordinary Python: wrap a multi-line pipe in
parentheses and name columns with `col("…")` in expressions and strings
elsewhere. This form works everywhere, notebooks included, and mixes with
bare names:

```python
result = (
    penguins
    >> filter(col("species") == "Gentoo")
    >> group_by("island")
    >> summarise(heaviest=max("body_mass_g"))
)
```

Or keep the notebook style and run the script with `python -m tidy3 run
job.py`, which applies the same rewriting. `nb_export` turns a notebook into
a plain script, and VS Code setup, `%%tidy3_run`, and the bare-name rules
are in [docs/notebooks.md](docs/notebooks.md). Running on a remote GPU
kernel under CRAFT or SolveIt: [docs/craft.md](docs/craft.md).

## More

- **[Reference](docs/reference.md)**: every verb family, the API table,
  backends (Polars or eager pandas), joins, `across`, row-wise work, and
  performance controls.
- **[Benchmarks and R parity](docs/benchmarks.md)**: how tidy3 is timed,
  and the suite that checks its results against dplyr and tidyr.
- **[Changelog](CHANGELOG.md)**

## Development

```bash
git clone https://github.com/rleyvasal/tidy3 && cd tidy3
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,jupyter]"
python -m pytest -q            # includes every example in this README and docs/
```

Releases go to PyPI from a version tag; see [docs/releasing.md](docs/releasing.md).

## License

MIT. See [LICENSE](LICENSE).
