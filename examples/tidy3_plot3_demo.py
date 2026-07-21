#!/usr/bin/env python3
"""End-to-end tidy3 + plot3 demo for local VS Code.

Setup (once, in the tidy3 venv)::

    cd /Users/admin/tidy3
    source .venv/bin/activate
    pip install -e ".[dev,jupyter]"
    pip install -e /Users/admin/plot3

VS Code::

    1. Open this file (or a notebook that pastes cells below).
    2. Select interpreter: tidy3/.venv/bin/python
    3. Run Python File, or open Interactive Window / .ipynb and run cells.

Figures write HTML under examples/output/ — open in a browser, or use a
notebook cell ending in the figure object for an inline iframe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from tidy3 import (
    col,
    filter,
    group_by,
    mean,
    mutate,
    n,
    select,
    summarise,
    tidy,
)
from plot3 import (
    aes,
    facet_wrap,
    geom_boxplot,
    geom_col,
    geom_point,
    geom_point3d,
    ggplot,
    labs,
    scale_colour_viridis_c,
    theme_light,
)

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)


def make_cars(n_per_cyl: int = 40, seed: int = 0):
    """mtcars-like sample as a TidyFrame from the start (no CSV required)."""
    rng = np.random.default_rng(seed)
    rows = []
    for cyl, mpg0, wt0 in ((4, 26.0, 2.2), (6, 20.0, 3.0), (8, 15.0, 3.8)):
        mpg = rng.normal(mpg0, 2.5, n_per_cyl)
        wt = rng.normal(wt0, 0.35, n_per_cyl)
        hp = rng.normal(80 + 15 * (cyl - 4), 20, n_per_cyl)
        for i in range(n_per_cyl):
            rows.append(
                {
                    "cyl": cyl,
                    "mpg": float(mpg[i]),
                    "wt": float(wt[i]),
                    "hp": float(max(hp[i], 40.0)),
                    "gear": int(rng.choice([3, 4, 5])),
                }
            )
    return tidy(pl.DataFrame(rows))


def main() -> None:
    cars = make_cars()
    print("cars:", cars.collect().shape)

    # ── 1) tidy3 pipeline: filter → mutate → aggregate ─────────────────────
    summary = (
        cars
        >> filter(col("mpg") > 12)
        >> mutate(
            km_l=col("mpg") * 0.425144,  # rough mpg → km/L
            heavy=col("wt") > 3.0,
        )
        >> group_by("cyl")
        >> summarise(
            n=n(),
            avg_mpg=mean("mpg"),
            avg_wt=mean("wt"),
            avg_hp=mean("hp"),
        )
    )
    print("\nsummary (Polars):")
    print(summary.collect())

    # ── 2) Pipe tidy3 straight into plot3 (>> ggplot + geom_*) ─────────────
    # Parentheses required in a .py script (not needed in notebooks after
    # %load_ext tidy3.jupyter).
    bar = (
        summary
        >> ggplot(aes(x="cyl", y="avg_mpg", colour="cyl"))
        + geom_col(width=0.7)
        + labs(title="Mean MPG by cylinders", x="cyl", y="avg mpg")
        + theme_light()
    )
    path_bar = bar.save(OUT / "01_mpg_by_cyl.html")
    print(f"\nwrote {path_bar}")

    # ── 3) Row-level scatter after a tidy select/filter ─────────────────────
    scatter = (
        cars
        >> filter(col("hp") < 250)
        >> select("wt", "mpg", "cyl", "hp")
        >> ggplot(aes(x="wt", y="mpg", colour="cyl"))
        + geom_point(size=5, alpha=0.85)
        + labs(title="Weight vs MPG", x="weight", y="mpg", colour="cyl")
        + theme_light()
    )
    path_scatter = scatter.save(OUT / "02_wt_mpg.html")
    print(f"wrote {path_scatter}")

    # ── 4) Method handoff: .ggplot(...) on a TidyFrame ─────────────────────
    box = (
        cars
        >> mutate(cyl_f=col("cyl").cast(pl.Utf8))
    ).ggplot(aes(x="cyl_f", y="mpg", colour="cyl_f")) + geom_boxplot(
        width=0.55
    ) + labs(title="MPG distribution by cylinders", x="cyl", y="mpg") + theme_light()
    path_box = box.save(OUT / "03_mpg_boxplot.html")
    print(f"wrote {path_box}")

    # ── 5) Faceted panels (one panel per gear) ──────────────────────────────
    facet = (
        cars
        >> mutate(gear_f=col("gear").cast(pl.Utf8), cyl_f=col("cyl").cast(pl.Utf8))
        >> ggplot(aes(x="wt", y="mpg", colour="cyl_f"))
        + geom_point(size=4, alpha=0.8)
        + facet_wrap("gear_f", ncol=3)
        + labs(title="Weight vs MPG by gear", colour="cyl")
        + theme_light()
    )
    path_facet = facet.save(OUT / "04_facet_gear.html")
    print(f"wrote {path_facet}")

    # ── 6) 3D points: aes(z=...) → orbit viewer ─────────────────────────────
    cloud = (
        cars
        >> ggplot(aes(x="wt", y="mpg", z="hp", colour="mpg"))
        + geom_point3d(size=0.06, alpha=0.9)
        + scale_colour_viridis_c(option="turbo")
        + labs(title="wt · mpg · hp", x="wt", y="mpg", z="hp", colour="mpg")
    )
    path_3d = cloud.save(OUT / "05_point3d.html")
    print(f"wrote {path_3d}")

    print("\nOpen any HTML above in a browser (or Simple Browser in VS Code).")
    print("In a notebook, drop the .save(...) and leave the figure as the last")
    print("expression in a cell for an inline interactive iframe.")


if __name__ == "__main__":
    main()
