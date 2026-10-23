"""tidy() on objects that know their own table (stats3's fitted models)."""

import polars as pl

from tidy3 import TidyFrame, tidy


class Fit:
    """Stands in for a fitted model with broom's tidy table."""

    def __tidy__(self, **kwargs):
        return {"table": True, **kwargs}


def test_tidy_calls_the_objects_own_table_with_its_arguments():
    assert tidy(Fit(), conf_int=True, conf_level=0.9) == {"table": True, "conf_int": True, "conf_level": 0.9}


def test_tidy_with_no_arguments_is_a_pipe_step():
    assert (Fit() >> tidy()) == {"table": True}
    assert isinstance(pl.DataFrame({"x": [1, 2]}) >> tidy(), TidyFrame)
    assert isinstance(tidy()(pl.DataFrame({"x": [1, 2]})), TidyFrame)


def test_data_is_wrapped_as_before():
    assert isinstance(tidy({"x": [1, 2]}), TidyFrame)
    assert isinstance(tidy(x=[1, 2], y=[3, 4]), TidyFrame)
    frame = tidy(x=[1, 2])
    assert tidy(frame) is frame
