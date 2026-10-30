

def test_seq_dates_per_row_window():
    import polars as pl

    from tidy3 import mutate, months, seq, tidy, ym

    df = tidy({"sale_month": ["2024-03", None, "2024-01"]})
    out = (
        df >> mutate(w=seq(ym("sale_month") - months(11), ym("sale_month"), by="month"))
    ).collect()["w"]
    assert len(out[0]) == 12 and out[0][0].isoformat() == "2023-04-01"
    assert out[0][-1].isoformat() == "2024-03-01"
    assert out[1] is None
    assert out[2][0].isoformat() == "2023-02-01"
    two = (df >> mutate(w=seq(ym("sale_month"), ym("sale_month") + months(4), by="2 months"))).collect()["w"]
    assert [d.isoformat() for d in two[0]] == ["2024-03-01", "2024-05-01", "2024-07-01"]
    same = (df >> mutate(w=seq(ym("sale_month"), ym("sale_month"), by=months(1)))).collect()["w"]
    assert len(same[0]) == 1


def test_seq_rejects_bad_by():
    import pytest

    from tidy3 import months, seq

    for bad in ("-1 month", "fortnight", "0 days", months(-1), 2.5):
        with pytest.raises((ValueError, TypeError)):
            seq("a", "b", by=bad)


def test_date_seq_total_size_and_argument_rules():
    import datetime as dt

    import pytest

    from tidy3 import date_seq, mutate, months, tidy

    df = tidy({"d": [dt.date(2024, 1, 31), None]})
    out = (df >> mutate(w=date_seq("d", by="month", total_size=3), v=date_seq("d", by=7, total_size=2))).collect()
    assert [x.isoformat() for x in out["w"][0]] == ["2024-01-31", "2024-02-29", "2024-03-31"]
    assert [x.isoformat() for x in out["v"][0]] == ["2024-01-31", "2024-02-07"]
    assert out["w"][1] is None
    assert len((df >> mutate(w=date_seq("d", by=months(1), total_size=0))).collect()["w"][0]) == 0
    for kwargs in ({}, {"to": "d"}, {"to": "d", "by": "day", "total_size": 2}):
        with pytest.raises(ValueError):
            date_seq("d", **kwargs)
    with pytest.raises(NotImplementedError):
        date_seq("d", to="d", total_size=2)
