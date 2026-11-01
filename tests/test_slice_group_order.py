"""slice_sample() returns its rows group by group, as the other slices do.

The R oracle checks the deterministic slices against dplyr 1.2.1. Samples
draw different rows in R, so here only the group order is checked: key order
with the missing key last after group_by(), first appearance with by=.
"""

from __future__ import annotations

import pytest

import tidy3 as t3

DATA = {"g": ["b", "a", "b", None, "a", "c"], "x": [0.1, 0.9, 0.5, 0.3, 0.2, 0.4]}
BY_KEY = ["a", "b", "c", None]
BY_APPEARANCE = ["b", "a", None, "c"]

SAMPLES = {
    "draw": dict(n=1),
    "proportion": dict(prop=1.0),
    "weighted": dict(n=1, weight_by="x"),
    "weighted_replace": dict(n=2, weight_by="x", replace=True),
    "replace": dict(n=2, replace=True),
}


def _groups(frame):
    keys = frame.collect()["g"].to_list()
    return [key for index, key in enumerate(keys) if index == 0 or keys[index - 1] != key]


@pytest.mark.parametrize("args", SAMPLES.values(), ids=SAMPLES.keys())
def test_grouped_samples_list_groups_in_key_order(args):
    frame = t3.tidy(DATA) >> t3.group_by("g") >> t3.slice_sample(seed=1, **args)
    assert _groups(frame) == BY_KEY


@pytest.mark.parametrize("args", SAMPLES.values(), ids=SAMPLES.keys())
def test_by_samples_list_groups_by_first_appearance(args):
    frame = t3.tidy(DATA) >> t3.slice_sample(seed=1, by="g", **args)
    assert _groups(frame) == BY_APPEARANCE


def test_weighted_sample_with_replacement_draws_from_the_missing_key_group():
    frame = t3.tidy(DATA) >> t3.slice_sample(n=2, weight_by="x", replace=True, seed=1, by="g")
    assert frame.collect()["g"].to_list().count(None) == 2
