"""Timezone kernels shared by the date expression backends."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import polars as pl


_POLICIES = frozenset({"NA", "pre", "post", "boundary"})
_SCALE = {"ms": 1000, "us": 1_000_000, "ns": 1_000_000_000}


def validate_zone(tzone: str) -> str:
    if not isinstance(tzone, str) or not tzone:
        raise TypeError("tzone must be a nonempty timezone name")
    ZoneInfo(tzone)
    return tzone


def validate_roll(roll_dst: str | tuple[str, str]) -> tuple[str, str]:
    if isinstance(roll_dst, str):
        policies = (roll_dst, roll_dst)
    elif isinstance(roll_dst, (tuple, list)):
        policies = tuple(roll_dst)
        if len(policies) == 1:
            policies *= 2
    else:
        raise TypeError("roll_dst must be a policy string or a pair of policies")
    if len(policies) != 2 or any(not isinstance(p, str) or p not in _POLICIES for p in policies):
        raise ValueError("roll_dst policies must be 'NA', 'pre', 'post', or 'boundary'")
    return policies


def _transition(earlier: int, later: int, scale: int, zone: ZoneInfo) -> int:
    """The whole-second timezone boundary between two candidate instants."""
    low, high = earlier // scale, later // scale
    before = datetime.fromtimestamp(low, timezone.utc).astimezone(zone).utcoffset()
    while high - low > 1:
        middle = (low + high) // 2
        offset = datetime.fromtimestamp(middle, timezone.utc).astimezone(zone).utcoffset()
        if offset == before:
            low = middle
        else:
            high = middle
    return high * scale


def transform(
    series: pl.Series, name: str, *, tzone: str,
    roll_dst: tuple[str, str] = ("NA", "post"),
) -> pl.Series:
    if series.dtype in {pl.Date, pl.Null}:
        series = series.cast(pl.Datetime("us"))
    if not isinstance(series.dtype, pl.Datetime):
        raise TypeError(f"{name}() needs dates or datetimes; parse text first")
    if name == "with_tz":
        return series.dt.convert_time_zone(tzone)
    source_zone = series.dtype.time_zone
    if source_zone == tzone:
        return series.clone()
    unit = series.dtype.time_unit
    clocks = series.dt.replace_time_zone(None)
    earlier = clocks.dt.replace_time_zone(tzone, ambiguous="earliest", non_existent="null")
    later = clocks.dt.replace_time_zone(tzone, ambiguous="latest", non_existent="null")
    skipped = series.is_not_null() & earlier.is_null()
    repeated = (earlier != later).fill_null(False)
    rows = (skipped | repeated).arg_true()
    if not len(rows):
        return later
    scale = _SCALE[unit]
    zone = ZoneInfo(tzone)
    original_zone = ZoneInfo(source_zone) if source_zone else None
    ticks = clocks.gather(rows).cast(pl.Int64).to_list()
    missing = skipped.gather(rows).to_list()
    resolved = []
    for value, clock_ticks, is_skipped in zip(series.gather(rows).to_list(), ticks, missing):
        local = value.replace(tzinfo=None)
        candidates = [
            clock_ticks - int(local.replace(tzinfo=zone, fold=fold).utcoffset().total_seconds()) * scale
            for fold in (0, 1)
        ]
        pre, post = min(candidates), max(candidates)
        policy = roll_dst[0 if is_skipped else 1]
        # A resolved source repeat carries its occurrence into a target
        # repeat, even when a different roll policy was requested in R.
        if not is_skipped and original_zone is not None:
            first = local.replace(tzinfo=original_zone, fold=0).utcoffset()
            last = local.replace(tzinfo=original_zone, fold=1).utcoffset()
            if first != last:
                policy = "post" if value.fold else "pre"
        if policy == "NA":
            resolved.append(None)
        elif policy == "boundary":
            resolved.append(_transition(pre, post, scale, zone))
        else:
            resolved.append(pre if policy == "pre" else post)
    return later.cast(pl.Int64).scatter(rows, resolved).cast(pl.Datetime(unit, tzone))
