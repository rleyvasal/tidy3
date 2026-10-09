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


def test_inject_api_refreshes_tidy3_names_and_reclaims_conflicts():
    """Stale tidy3 + foreign (datar-like) symbols are replaced by default."""
    user_mean = object()  # stands in for datar/pipda mean
    old_module = ModuleType("tidy3")
    ipython = SimpleNamespace(
        user_ns={
            "filter": _old_tidy3_function,
            "mean": user_mean,
            "tidy3": old_module,
        }
    )

    inject_api(ipython)  # force=True default

    assert ipython.user_ns["filter"] is tidy3.filter
    assert ipython.user_ns["mean"] is tidy3.mean
    assert ipython.user_ns["tidy3"] is tidy3
    assert ipython.user_ns["tidy"] is tidy3.tidy


def test_inject_api_force_false_preserves_non_tidy3_user_values():
    user_mutate = object()
    ipython = SimpleNamespace(
        user_ns={
            "filter": _old_tidy3_function,
            "mutate": user_mutate,
        }
    )

    inject_api(ipython, force=False)

    assert ipython.user_ns["filter"] is tidy3.filter  # stale tidy3 refreshed
    assert ipython.user_ns["mutate"] is user_mutate  # user value kept


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

    from tidy3.jupyter import _R_STYLE_ON
    from tidy3.masking import tidy3_backtick_transform

    # Default R-style on: backticks then pipe rewriter at front.
    import tidy3.jupyter as jmod

    jmod._R_STYLE_ON = True
    assert enable_pipe_transform(ipython)
    assert ipython.input_transformers_cleanup == [
        tidy3_backtick_transform,
        tidy3_input_transformer,
        unrelated,
    ]
    assert ipython.input_transformers_post == [
        tidy3_backtick_transform,
        tidy3_input_transformer,
    ]

    disable_pipe_transform(ipython)
    assert ipython.input_transformers_cleanup == [unrelated]
    assert ipython.input_transformers_post == []

def test_aes_bare_names_work_in_the_cell_that_imports_plot3():
    pytest = __import__("pytest")
    pytest.importorskip("IPython")
    pytest.importorskip("plot3")
    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    try:
        shell.run_line_magic("load_ext", "tidy3.jupyter")
        shell.run_cell('cars = tidy({"mpg": [21.0, 22.8], "wt": [2.62, 2.32]})')
        first = shell.run_cell(
            "from plot3 import aes, geom_point, ggplot\n"
            "p = ggplot(cars, aes(x=wt, y=mpg)) + geom_point()"
        )
        later = shell.run_cell("q = ggplot(cars, aes(x=wt, y=mpg)) + geom_point()")

        assert first.success and later.success
        maskers = [
            t for t in shell.ast_transformers
            if type(t).__name__ == "Plot3MaskTransformer"
        ]
        assert len(maskers) == 1
    finally:
        shell.run_line_magic("unload_ext", "tidy3.jupyter")
        InteractiveShell.clear_instance()
