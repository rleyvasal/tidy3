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


def test_multiline_pipe_inside_a_function_runs_in_ipython():
    pytest = __import__("pytest")
    pytest.importorskip("IPython")
    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    try:
        shell.run_line_magic("load_ext", "tidy3.jupyter")
        shell.run_cell(
            'penguins = tidy({"species": ["Adelie", "Adelie", "Gentoo"],'
            ' "island": ["Dream", "Dream", "Biscoe"]})'
        )
        defined = shell.run_cell(
            "def duplicate_rows(panel):\n"
            "    counted = panel\n"
            '    >> add_count(species, island, name="Duplicate Count")\n'
            "    return counted\n"
        )
        called = shell.run_cell("result = duplicate_rows(penguins).collect()")
        assert defined.success, defined.error_before_exec or defined.error_in_exec
        assert called.success, called.error_in_exec
        assert shell.user_ns["result"]["Duplicate Count"].to_list() == [2, 2, 1]
    finally:
        shell.run_line_magic("unload_ext", "tidy3.jupyter")
        InteractiveShell.clear_instance()


def _fresh_shell():
    pytest = __import__("pytest")
    pytest.importorskip("IPython")
    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    shell.run_line_magic("load_ext", "tidy3.jupyter")
    return shell


def _close_shell(shell):
    from IPython.core.interactiveshell import InteractiveShell

    shell.run_line_magic("unload_ext", "tidy3.jupyter")
    InteractiveShell.clear_instance()


def test_masking_leaves_a_notebook_defined_verb_alone():
    shell = _fresh_shell()
    try:
        shell.run_cell("w = 3\ndef mutate(**kw):\n    return kw")
        result = shell.run_cell("out = mutate(z = w)")
        assert result.success, result.error_in_exec
        assert shell.user_ns["out"] == {"z": 3}
    finally:
        _close_shell(shell)


def test_masking_leaves_another_packages_verbs_alone(monkeypatch):
    import sys
    from types import ModuleType

    # Another package exporting tidy3's verb names, as tidy3a does.
    other = ModuleType("otherverbs")

    def filter(condition):  # noqa: A001
        return ("otherverbs", condition)

    filter.__module__ = "otherverbs"
    other.filter = filter
    other.__all__ = ["filter"]
    monkeypatch.setitem(sys.modules, "otherverbs", other)

    shell = _fresh_shell()
    try:
        shell.run_cell("w = 3")
        same_cell = shell.run_cell("from otherverbs import *\nfirst = filter(w > 1)")
        later = shell.run_cell("second = filter(w > 2)")
        assert same_cell.success, same_cell.error_in_exec
        assert later.success, later.error_in_exec
        assert shell.user_ns["first"] == ("otherverbs", True)
        assert shell.user_ns["second"] == ("otherverbs", True)

        # Importing tidy3's filter again makes it tidy3's, and masking resumes.
        back = shell.run_cell(
            'from tidy3 import filter, tidy\n'
            'cars = tidy({"mpg": [21.0, 30.0]})\n'
            "kept = (cars >> filter(mpg > 25)).collect()"
        )
        assert back.success, back.error_in_exec
        assert shell.user_ns["kept"]["mpg"].to_list() == [30.0]
    finally:
        _close_shell(shell)
