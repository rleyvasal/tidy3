"""Chained verbs must not copy the source table into the plan more than linearly.

A Polars query is a tree. A verb that uses its input in more than one place (a
check that re-reads the input, a self-join, a unique-values query joined back)
puts that subtree in the plan several times, and chaining such verbs multiplies
the copies: ``r`` uses per step give ``r ** k`` copies after ``k`` steps. Polars
can sometimes merge identical copies, but planning and schema resolution still
walk them all, so a handful of chained steps gets slow (see CHANGELOG, 0.12.10).

Each case below applies one verb four times and counts the copies of the source
table in the unoptimized plan. Linear growth keeps the count within four times
its value after one step. ``KNOWN_SUPERLINEAR`` lists any verb that does not yet;
it is an expected failure, and one that gets fixed makes this test fail until it
is removed from the list. Verbs that read their input more than once either
avoid the second read (the joins, ``slice_sample``) or mark the input shared
(``cache()``), which prints it once.

The count only sees printed copies. A shared input is still walked once per
reader by Polars' optimizer, so ``complete``, ``expand`` and the ``rows_*`` verbs,
which must read their input twice (the rows, and the keys or values), still take
about twice as long for each extra step in a chain. Verbs that read it once, like
``mutate``, grow by a small constant.
"""

import builtins
import datetime as dt
import time

import polars as pl
import pytest

import tidy3 as t
from tidy3 import (
    add_count,
    anti_join,
    across,
    arrange,
    bind_cols,
    bind_rows,
    case_when,
    col,
    clean_names,
    complete,
    count,
    cross_join,
    crossing,
    desc,
    distinct,
    drop_na,
    everything,
    expand,
    fill,
    filter,
    first,
    full_join,
    ge,
    get_dupes,
    group_by,
    inner_join,
    intersect,
    join_by,
    lag,
    le,
    left_join,
    mean,
    mutate,
    nest_join,
    pivot_longer,
    pivot_wider,
    relocate,
    rename,
    replace_na,
    right_join,
    row_number,
    rows_append,
    rows_delete,
    rows_insert,
    rows_patch,
    rows_update,
    rows_upsert,
    rowwise,
    sample_frac,
    sample_n,
    select,
    semi_join,
    separate,
    setdiff,
    slice,
    slice_head,
    slice_max,
    slice_min,
    slice_sample,
    slice_tail,
    summarise,
    symdiff,
    tabyl,
    tidy,
    tally,
    transmute,
    uncount,
    ungroup,
    union,
    union_all,
    unite,
)
from tidy3 import max as t_max, min as t_min, n as t_n  # noqa: F401

x, y, g, s, d = col("x"), col("y"), col("g"), col("s"), col("d")
id_ = col("id")
N = 200
base = pl.DataFrame({
    "id": list(range(N)), "g": [f"g{i % 5}" for i in range(N)], "x": [i * 0.5 for i in range(N)],
    "y": [float(i % 7) for i in range(N)], "s": [f"a-{i % 9}-b" for i in range(N)],
    "d": [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(N)],
})
side = pl.DataFrame({"id": list(range(0, N, 2)), "w": [float(i) for i in range(N // 2)]})
side_g = pl.DataFrame({"g": [f"g{i}" for i in range(5)], "gv": list(range(5))})
many = pl.DataFrame({"id": [i // 2 for i in range(N)], "m": list(range(N))})
extra = base.head(20)

def nm(prefix, i): return f"{prefix}{i}"
C = {}

# --- single-table verbs ---
C["filter"] = lambda f, i: f >> filter(x > i)
C["mutate"] = lambda f, i: f >> mutate(**{nm("m", i): x * 2})
C["mutate grouped"] = lambda f, i: f >> group_by("g") >> mutate(**{nm("m", i): mean(x)}) >> ungroup()
C["transmute"] = lambda f, i: f >> transmute(id=id_, g=g, x=x + 1, y=y, s=s, d=d)
C["select"] = lambda f, i: f >> select(everything())
C["rename"] = lambda f, i: f >> rename(**{nm("r", i): "y"}) >> rename(y=nm("r", i))
C["relocate"] = lambda f, i: f >> relocate("y")
C["arrange"] = lambda f, i: f >> arrange(desc(x))
C["distinct"] = lambda f, i: f >> distinct()
C["distinct by"] = lambda f, i: f >> distinct("g", keep_all=True)
C["count"] = lambda f, i: f >> count("g") >> mutate(id=t_n(), g=g, x=t_n() * 1.0, y=t_n() * 1.0, s=g, d=pl.lit(dt.date(2024, 1, 1)))
C["add_count"] = lambda f, i: f >> add_count("g", name=nm("c", i))
C["tally"] = lambda f, i: f >> group_by("g") >> tally() >> mutate(id=t_n(), x=t_n() * 1.0, y=t_n() * 1.0, s=g, d=pl.lit(dt.date(2024, 1, 1)))
C["summarise"] = lambda f, i: f >> group_by("g") >> summarise(id=t_min(id_), x=mean(x), y=mean(y), s=first(s), d=first(d))
C["group_by+mutate"] = lambda f, i: f >> group_by("g") >> mutate(**{nm("r", i): row_number()})
C["slice_head"] = lambda f, i: f >> slice_head(n=150)
C["slice_tail"] = lambda f, i: f >> slice_tail(n=150)
C["slice_min"] = lambda f, i: f >> slice_min("x", n=150)
C["slice_max"] = lambda f, i: f >> slice_max("x", n=150)
C["slice"] = lambda f, i: f >> slice(range(0, 150))
C["slice_sample"] = lambda f, i: f >> slice_sample(n=150, seed=1)
C["slice_sample weight_by"] = lambda f, i: f >> slice_sample(n=150, weight_by="y", seed=1)
C["sample_n"] = lambda f, i: f >> sample_n(150)
C["sample_frac"] = lambda f, i: f >> sample_frac(0.9)
C["drop_na"] = lambda f, i: f >> drop_na()
C["replace_na"] = lambda f, i: f >> replace_na({"x": 0.0})
C["fill"] = lambda f, i: f >> fill("x")
C["complete"] = lambda f, i: f >> complete("g", "id")
C["expand"] = lambda f, i: f >> expand("g", "id") >> mutate(x=1.0, y=1.0, s="a", d=pl.lit(dt.date(2024, 1, 1)))
C["separate"] = lambda f, i: f >> separate("s", into=["p", "q"], sep="-", remove=False, convert=False) >> select("id", "g", "x", "y", "s", "d")
C["unite"] = lambda f, i: f >> unite(nm("u", i), "g", "s", remove=False)
C["uncount"] = lambda f, i: f >> mutate(n=1) >> uncount("n")
C["pull"] = lambda f, i: f
C["with_order/lag"] = lambda f, i: f >> mutate(**{nm("l", i): lag("x")})
C["across"] = lambda f, i: f >> mutate(across(["x", "y"], lambda v: v + 1))
C["if_else/case_when"] = lambda f, i: f >> mutate(**{nm("c", i): case_when((x > 5, "hi"), default="lo")})
C["rowwise"] = lambda f, i: f >> rowwise() >> mutate(**{nm("r", i): x + y}) >> ungroup()
C["pivot_longer+wider"] = lambda f, i: f >> pivot_longer(["x", "y"], names_to="k", values_to="v") >> pivot_wider(names_from="k", values_from="v")
C["bind_rows"] = lambda f, i: f >> bind_rows(extra)
C["bind_cols"] = lambda f, i: f >> bind_cols(pl.DataFrame({nm("b", i): list(range(N))}))
C["union"] = lambda f, i: f >> union(extra)
C["union_all"] = lambda f, i: f >> union_all(extra)
C["intersect"] = lambda f, i: f >> intersect(base)
C["setdiff"] = lambda f, i: f >> setdiff(extra)
C["symdiff"] = lambda f, i: f >> symdiff(extra)
C["get_dupes"] = lambda f, i: f >> get_dupes("g") >> select("id", "g", "x", "y", "s", "d")
C["clean_names"] = lambda f, i: f >> clean_names()
C["tabyl"] = lambda f, i: f >> tabyl("g") >> mutate(id=t_n(), g=g, x=t_n() * 1.0, y=t_n() * 1.0, s=g, d=pl.lit(dt.date(2024, 1, 1)))
C["crossing"] = lambda f, i: f >> select("id") >> mutate(g="a", x=1.0, y=1.0, s="a", d=pl.lit(dt.date(2024, 1, 1)))

# --- joins ---
J = {"left": left_join, "inner": inner_join, "right": right_join, "full": full_join}
for how, fn in J.items():
    C[f"{how}_join by="] = (lambda fn: lambda f, i: f >> fn(side.rename({"w": nm("w", i)}), on="id"))(fn)
    C[f"{how}_join join_by"] = (lambda fn: lambda f, i: f >> fn(side.rename({"w": nm("w", i)}), on=join_by("id")))(fn)
    C[f"{how}_join relationship=many-to-one"] = (lambda fn: lambda f, i: f >> fn(side.rename({"w": nm("w", i)}), on=join_by("id"), relationship="many-to-one"))(fn)
    C[f"{how}_join relationship=one-to-one"] = (lambda fn: lambda f, i: f >> fn(side.rename({"w": nm("w", i)}), on=join_by("id"), relationship="one-to-one"))(fn)
    C[f"{how}_join unmatched=error"] = (lambda fn: lambda f, i: f >> fn(side_g.rename({"gv": nm("gv", i)}), on=join_by("g"), unmatched="error"))(fn)
    C[f"{how}_join na_matches=never"] = (lambda fn: lambda f, i: f >> fn(side.rename({"w": nm("w", i)}), on=join_by("id"), na_matches="never"))(fn)
    C[f"{how}_join multiple=first"] = (lambda fn: lambda f, i: f >> fn(many.rename({"m": nm("m", i)}), on=join_by("id"), multiple="first"))(fn)
bins = pl.DataFrame({"lo": list(range(0, N, 10)), "hi": list(range(9, N + 9, 10)), "bin": list(range(N // 10))})
C["inequality join (between)"] = lambda f, i: f >> left_join(bins.rename({"bin": nm("bin", i)}), on=join_by(ge("id", "lo"), le("id", "hi"))) >> select("id", "g", "x", "y", "s", "d")
C["semi_join"] = lambda f, i: f >> semi_join(side, on="id")
C["anti_join"] = lambda f, i: f >> anti_join(side, on="id")
C["semi_join join_by"] = lambda f, i: f >> semi_join(side, on=join_by("id"))
C["cross_join"] = lambda f, i: f >> cross_join(pl.DataFrame({nm("z", i): [1]}))
C["nest_join"] = lambda f, i: f >> nest_join(many, on="id", name=nm("nj", i))

# --- rows_* ---
upd = base.head(10)
C["rows_update"] = lambda f, i: f >> rows_update(upd, by="id")
C["rows_patch"] = lambda f, i: f >> rows_patch(upd, by="id")
C["rows_upsert"] = lambda f, i: f >> rows_upsert(upd, by="id")
C["rows_insert"] = lambda f, i: f >> rows_insert(pl.DataFrame({k: [v[0] if k != "id" else 10_000 + i] for k, v in base.to_dict(as_series=False).items()}), by="id")
C["rows_delete"] = lambda f, i: f >> rows_delete(pl.DataFrame({"id": [N + 10_000]}), by="id", unmatched="ignore")
C["rows_append"] = lambda f, i: f >> rows_append(extra)

KNOWN_SUPERLINEAR: set[str] = set()


def _copies(name, steps=4):
    frame = tidy(base)
    counts = []
    for k in builtins.range(1, steps + 1):
        frame = C[name](frame, k)
        counts.append(frame._lf.explain(optimized=False).count("DF ["))
    return counts


def _case(name):
    marks = []
    if name in KNOWN_SUPERLINEAR:
        marks.append(pytest.mark.xfail(strict=True, reason="plan grows faster than linearly"))
    return pytest.param(name, id=name, marks=marks)


@pytest.mark.parametrize("name", [_case(name) for name in C])
def test_chained_verb_copies_the_source_linearly(name):
    counts = _copies(name)
    assert counts[-1] <= 4 * counts[0], counts


def test_slice_sample_weight_checks_still_raise():
    df = pl.DataFrame({"g": ["a"] * 3 + ["b"] * 3, "w": [1.0, 0.0, 2.0, 1.0, 1.0, 1.0], "v": list(range(6))})
    bad = {
        "negative": (df.with_columns(w=pl.Series([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])), {"n": 2}),
        "missing": (df.with_columns(w=pl.Series([1.0, None, 1.0, 1.0, 1.0, 1.0])), {"n": 2}),
        "not finite": (df.with_columns(w=pl.Series([1.0, float("inf"), 1.0, 1.0, 1.0, 1.0])), {"n": 2}),
    }
    for name, (frame, options) in bad.items():
        with pytest.raises(ValueError, match="finite, non-missing, and non-negative"):
            (tidy(frame) >> slice_sample(weight_by="w", seed=1, **options)).collect()
    with pytest.raises(ValueError, match="larger sample than the number of positive weights"):
        (tidy(df) >> slice_sample(n=6, weight_by="w", seed=1)).collect()
    with pytest.raises(ValueError, match="too few positive weights"):
        (tidy(df.with_columns(w=pl.lit(0.0))) >> slice_sample(n=2, weight_by="w", replace=True, seed=1)).collect()
    # Each group is checked on its own.
    with pytest.raises(ValueError, match="larger sample than the number of positive weights"):
        (tidy(df) >> group_by("g") >> slice_sample(n=3, weight_by="w", seed=1)).collect()
    ok = (tidy(df) >> group_by("g") >> slice_sample(n=2, weight_by="w", seed=1)).collect()
    assert ok.height == 4


# Verbs that read their input twice by nature (the rows, and the keys or values):
# a shared input prints once, but Polars still walks it once per reader, so the
# time to plan a chain of them multiplies. Not strict: a fast machine may hide it.
TIME_GROWS = {"complete", "rows_insert"}


def _chain_ms(name, steps, repeats=5):
    """Fastest time, in ms, to build and collect `steps` chained applications."""
    best = float("inf")
    for _ in builtins.range(repeats):
        frame = tidy(base)
        start = time.perf_counter()
        for k in builtins.range(1, steps + 1):
            frame = C[name](frame, k)
        frame.collect()
        best = builtins.min(best, (time.perf_counter() - start) * 1e3)
    return best


def _time_case(name):
    marks = []
    if name in TIME_GROWS:
        marks.append(
            pytest.mark.xfail(strict=False, reason="reads its input twice, so planning time multiplies")
        )
    return pytest.param(name, id=name, marks=marks)


@pytest.mark.parametrize("name", [_time_case(name) for name in C])
def test_chained_verb_time_grows_gently(name):
    # Eight steps of a verb that costs a fixed amount per step take about twice
    # as long as four. One that multiplies its work each step takes many times
    # longer (complete: ~14x, rows_insert: ~23x). The slack keeps tiny timings
    # from failing on noise.
    four, eight = _chain_ms(name, 4), _chain_ms(name, 8)
    assert eight <= 8 * four + 2.0, (round(four, 2), round(eight, 2))
