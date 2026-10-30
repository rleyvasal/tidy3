

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

    for bad in ("-1 month", "fortnight", "0 days", months(-1), 3):
        with pytest.raises((ValueError, TypeError)):
            seq("a", "b", by=bad)
