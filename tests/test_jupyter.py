"""Jupyter namespace injection, including remote re-seed behavior."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import tidy3
from tidy3.jupyter import (
    disable_pipe_transform,
    enable_pipe_transform,
    inject_api,
    tidy3_input_transformer,
)


def _old_tidy3_function():
    return "old"


_old_tidy3_function.__module__ = "tidy3.verbs"


def test_inject_api_refreshes_tidy3_names_but_preserves_user_values():
    user_filter = object()
    old_module = ModuleType("tidy3")
    ipython = SimpleNamespace(
        user_ns={
            "filter": _old_tidy3_function,
            "mutate": user_filter,
            "tidy3": old_module,
        }
    )

    inject_api(ipython)

    assert ipython.user_ns["filter"] is tidy3.filter
    assert ipython.user_ns["mutate"] is user_filter
    assert ipython.user_ns["tidy3"] is tidy3
    assert ipython.user_ns["tidy"] is tidy3.tidy


def test_inject_api_populates_an_empty_namespace():
    ipython = SimpleNamespace(user_ns={})

    inject_api(ipython)

    assert ipython.user_ns["tidy"] is tidy3.tidy
    assert ipython.user_ns["tidy3"] is tidy3


def test_pipe_transform_registration_replaces_stale_module_copy():
    def stale_transformer(lines):
        return lines

    stale_transformer.__module__ = "tidy3.jupyter"
    stale_transformer.__name__ = "tidy3_input_transformer"
    unrelated = lambda lines: lines
    ipython = SimpleNamespace(
        input_transformers_cleanup=[unrelated, stale_transformer],
        input_transformers_post=[stale_transformer],
    )

    assert enable_pipe_transform(ipython)
    # tidy3 is inserted at the front so it runs before CRAFT's %gpu router.
    assert ipython.input_transformers_cleanup == [
        tidy3_input_transformer,
        unrelated,
    ]
    assert ipython.input_transformers_post == [tidy3_input_transformer]

    disable_pipe_transform(ipython)
    assert ipython.input_transformers_cleanup == [unrelated]
    assert ipython.input_transformers_post == []
