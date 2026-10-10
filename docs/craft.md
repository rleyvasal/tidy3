# tidy3 under CRAFT, gpudev, and SolveIt

Loading tidy3 and plot3 in SolveIt, and running pipes on a remote GPU kernel with CRAFT `%gpu`. A normal local install needs none of this.

Back to the [README](../README.md).

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

**One command** — works local *and* under `%gpu` (CRAFT is auto-detected):

```text
%run /app/data/gpudevd/tidy3/tidy3.py
# or
%run /path/to/tidy3/tidy3.py
%run /path/to/tidy3/load.py          # same loader
```

That puts `src/` on the path, injects the API, and turns on multi-line `>>`
plus R-style bare names / backticks / `~` (optional `!` sugar only inside
tidy3 verb calls — shell cells like `!pip install` are never rewritten). If
CRAFT is already loaded (or becomes connected on `%gpu`), the same cell also
registers remote seeding — no second command.

```text
%local
%run /app/data/gpudevd/tidy3/tidy3.py
%run /app/data/gpudevd/plot3/plot3.py
# only when you need the GPU:
%run /path/to/gpudev/CRAFT.py
%gpu
```

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

```python notest
# after %gpu — paths are on the GPU box
scan_parquet("/home/gpudev/data/huge.parquet")
>> filter(col("year") >= 2020)
>> group_by("region")
>> summarise(n=n(), avg=mean("value"))
>> ggplot(aes(x="region", y="avg")) + geom_col()
```

`%plot3` is registered as a **host-local** magic (viewer + SolveIt red-eye stay
on the dialog machine) even while Python cells run remote.

### Symlink addons (CRAFT layout)

```bash
cd /path/to/gpudev/addons
ln -sfn /path/to/tidy3 tidy3
ln -sfn /path/to/plot3 plot3
```
