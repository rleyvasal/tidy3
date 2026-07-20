#!/usr/bin/env python3
"""Local VS Code dogfood: tidy3 prep → feature matrix → optional PyTorch.

Run from the repo root with the project venv active::

    pip install -e ".[dev]"
    python examples/prep_for_ml.py

Optional::

    pip install torch
    python examples/prep_for_ml.py --torch
"""

from __future__ import annotations

import argparse

import numpy as np

from tidy3 import (
    col,
    filter,
    group_by,
    if_else,
    mean,
    mutate,
    n,
    select,
    std,
    summarise,
    tidy,
    ungroup,
)


def make_events(rows: int = 20_000, seed: int = 0):
    """Synthetic customer events (same spirit as the adoption bench)."""
    rng = np.random.default_rng(seed)
    customer_id = rng.integers(0, 500, rows)
    return {
        "customer_id": customer_id,
        "segment": customer_id % 8,
        "amount": rng.lognormal(3.2, 0.6, rows),
        "feature_a": rng.normal(size=rows),
        "feature_b": rng.normal(5.0, 2.0, rows),
        "score": rng.normal(size=rows),
        "active": rng.random(rows) < 0.85,
    }


def build_feature_frame(events: dict):
    """dplyr-style prep: filter → group features → tidy frame ready for ML."""
    return (
        tidy(events)
        >> filter(col("active") & (col("amount") > 5))
        >> mutate(
            log_amount=(col("amount") + 1).log(),
            high_score=if_else(col("score") > 0.5, 1, 0),
        )
        >> group_by("customer_id", "segment")
        >> summarise(
            n_events=n(),
            revenue=mean("amount", na_rm=True),
            mean_score=mean("score", na_rm=True),
            sd_a=std("feature_a", na_rm=True),
            mean_b=mean("feature_b", na_rm=True),
            share_high=mean("high_score", na_rm=True),
        )
        >> ungroup()
        >> mutate(
            revenue_z=(col("revenue") - mean("revenue", na_rm=True))
            / std("revenue", na_rm=True),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument(
        "--torch",
        action="store_true",
        help="also wrap the feature matrix with torch.from_numpy",
    )
    args = parser.parse_args()

    features = build_feature_frame(make_events(args.rows))
    feature_cols = [
        "n_events",
        "revenue",
        "mean_score",
        "sd_a",
        "mean_b",
        "share_high",
        "revenue_z",
    ]

    # Keep identifiers out of the matrix; project only model inputs.
    preview = features.collect(
        as_="pandas",
        columns=["customer_id", "segment", *feature_cols],
    )
    print("feature frame (head):")
    print(preview.head())
    print(f"\nrows={len(preview):,}  columns={list(preview.columns)}")

    X = features.to_numpy(
        columns=feature_cols,
        dtype=np.float32,
        order="c",
        writable=True,
    )
    print(f"\nNumPy matrix: shape={X.shape} dtype={X.dtype} C_CONTIGUOUS={X.flags['C_CONTIGUOUS']}")

    if args.torch:
        try:
            import torch
        except ImportError as exc:
            raise SystemExit(
                "torch is not installed in this environment; "
                "pip install torch  (or omit --torch)"
            ) from exc
        tensor = torch.from_numpy(X)
        print(f"torch.Tensor: shape={tuple(tensor.shape)} dtype={tensor.dtype}")
        print("shared memory with NumPy:", tensor.data_ptr() != 0)

    # Optional plot3 handoff (skip if plot3 is not installed).
    try:
        from plot3 import aes, geom_point, ggplot, labs

        sample = (
            features
            >> select("revenue", "mean_score", "segment")
        ).collect(as_="pandas")
        # Cap points for a light interactive figure.
        if len(sample) > 2_000:
            sample = sample.sample(2_000, random_state=0)
        fig = (
            ggplot(sample, aes(x="revenue", y="mean_score", colour="segment"))
            + geom_point(size=4, alpha=0.7)
            + labs(title="Customer features (sample)", x="revenue", y="mean score")
        )
        out = fig.save("examples/prep_for_ml_scatter.html")
        print(f"\nplot3 figure written to {out}")
    except ImportError:
        print("\nplot3 not installed — skipped scatter; pip install -e /path/to/plot3")


if __name__ == "__main__":
    main()
