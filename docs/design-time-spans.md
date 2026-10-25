# Design: durations, periods, and intervals

Status: **approved 2026-10-25.** The open questions were decided as
recommended; see [Decisions](#5-decisions). This note closes the last part of item 5
in the [2026-10-21 package review](package-review-2026-10-21.md): "a
deliberate calendar-period versus elapsed-duration and interval design."

Every R output below was produced locally with **lubridate 1.9.5** (the
version the review records). Every Polars output was produced with
**Polars 1.42.1** and **pandas 3.0.3**. Times in `America/New_York` use the
2024 changes: clocks skip 02:00–03:00 on 10 March and repeat 01:00–02:00 on
3 November.

## Summary

| Span | Meaning | tidy3 value | Column dtype | Lands in |
|---|---|---|---|---|
| Duration | Exact elapsed seconds | `ddays(1)` and friends, `date - date` | `pl.Duration("us")`; pandas `timedelta64[us]` | Phase 1 |
| Period | Calendar and clock units | the existing `Period`, widened to days, hours, minutes, seconds | Not a column at first; an operand of `+` and `-` | Phase 1 (arithmetic), Phase 3 (columns) |
| Interval | A start and an end instant | `interval(start, end)` | `pl.Struct({"start", "end"})`; pandas Arrow struct | Phase 2 |

The rule behind every decision: **durations count seconds, periods move the
clock and calendar, intervals remember where they started.** tidy3 already
has the calendar half of periods (`months`, `years`) with lubridate's
`%m+%` rollback. This note keeps that rollback and fills in the rest.

## 1. lubridate's three span types

### Durations: exact seconds

`dseconds`, `dminutes`, `dhours`, `ddays`, `dweeks`, `dmonths`, `dyears`,
`duration()`, and `as.duration()` build a number of seconds. A day is always
86,400 seconds, a year is 365.25 days, and a month is a twelfth of that.

| R | Output |
|---|---|
| `ddays(1)` | `86400s (~1 days)` |
| `dmonths(1)` | `2629800s (~4.35 weeks)` |
| `dyears(1)` | `31557600s (~1 years)` |
| `ymd_hms("2024-03-09 12:00:00", tz = ny) + ddays(1)` | `2024-03-10 13:00:00` (spring: 24 real hours) |
| `ymd_hms("2024-11-02 12:00:00", tz = ny) + ddays(1)` | `2024-11-03 11:00:00` (autumn) |
| `ymd_hms("2024-03-09 02:30:00", tz = ny) + ddays(1)` | `2024-03-10 03:30:00` (never missing) |
| `ymd("2012-02-29") + ddays(365)` | `2013-02-28`, class `Date` |
| `ymd("2012-02-29") + dyears(1)` | `2013-02-28 06:00:00`, class `POSIXct`, UTC |
| `ymd("2012-02-29") + ddays(1.5)` | `2012-03-01 12:00:00`, class `POSIXct`, UTC |
| `ymd("2012-03-01") - ymd("2012-01-31")` | `30 days`, class `difftime` |
| `ymd_hms("2024-03-11", tz = ny) - ymd_hms("2024-03-10", tz = ny)` | `23 hours` |
| `as.duration(ymd("2012-03-01") - ymd("2012-01-31"))` | `2592000s (~4.29 weeks)` |
| `sum(ddays(1:3))` | `518400` |

A Date plus a duration stays a Date only when the whole vector is whole
days; otherwise the result becomes a UTC datetime. Subtracting dates gives
base R's `difftime`, whose unit (`days`, `hours`, `mins`) is chosen from the
size of the gap, not lubridate's duration.

### Periods: clock and calendar units

`years`, `months`, `weeks`, `days`, `hours`, `minutes`, `seconds`, and
`period()` count units. Adding one moves the local clock or calendar, so a
day may be 23 or 25 real hours. lubridate's `+` gives `NA` when the target
time does not exist.

| R | Output |
|---|---|
| `ymd_hms("2024-03-09 12:00:00", tz = ny) + days(1)` | `2024-03-10 12:00:00` (23 real hours) |
| `ymd_hms("2024-03-09 12:00:00", tz = ny) + hours(24)` | `2024-03-10 12:00:00` (also clock time) |
| `ymd_hms("2024-03-09 02:30:00", tz = ny) + days(1)` | `NA` (02:30 does not exist on 10 March) |
| `ymd_hms("2024-03-10 01:30:00", tz = ny) + hours(1)` | `NA` |
| `ymd_hms("2024-11-03 00:30:00", tz = ny) + hours(2)` | `02:30 EST`, three real hours later |
| `ymd_hms("2024-11-04 01:30:00", tz = ny) - days(1)` | `2024-11-03 01:30 EDT` (earlier repeat) |
| `ymd_hms("2024-11-03 01:30 EST") + minutes(1)` | `01:31 EDT`, 59 minutes **earlier** |
| `ymd_hms("2024-11-03 01:30 EST") + seconds(0)` | `01:30 EDT`, a different instant |
| `ymd("2010-01-31") + months(1)` | `NA` |
| `ymd("2010-01-31") %m+% months(1)` | `2010-02-28` |
| `ymd("2012-02-29") + years(1)` | `NA`; `%m+%` gives `2013-02-28` |
| `ymd("2010-01-31") + (months(1) + days(1))` | `NA` |
| `ymd_hms("2024-02-10 02:30:00", tz = ny) + months(1)` | `NA`; `%m+%` gives `2024-03-10 03:00:00 EDT` |
| `add_with_rollback(ymd("2010-01-31"), months(1), roll_to_first = TRUE)` | `2010-03-01` |
| `ymd("2012-02-29") + hours(1)` | `2012-02-29 01:00:00`, class `POSIXct` |
| `days(1.5)` | error: "periods must have integer values" |
| `period(hours = 25)` | `25H 0M 0S` (not normalized to a day) |
| `days(1) + hours(25)` | `1d 25H 0M 0S` |
| `ddays(1) + months(1)` | error: "Incompatible classes" |
| `ddays(1) == days(1)` | `TRUE` |
| `months(1) / days(1)` | `30.4375` (via average lengths) |
| `sum(days(1:3))`, `mean(c(days(1), days(2)))` | `0`, `0` (the period slots are ignored) |

A repeated autumn clock produced by period arithmetic resolves to the
**earlier** occurrence, even when the input was the later one. That is why
adding `seconds(0)` changes the instant: the input's occurrence is not kept.

### Intervals: anchored spans

`interval(start, end)` and `start %--% end` keep both instants. Dates become
UTC datetimes.

| R | Output |
|---|---|
| `i <- interval(ymd("2012-01-31"), ymd("2012-03-01"))` | `2012-01-31 UTC--2012-03-01 UTC` |
| `int_start(i)` | `2012-01-31`, class `POSIXct` |
| `int_length(i)` | `2592000` (seconds) |
| `time_length(i, "day")`, `"week"` | `30`, `4.285714` |
| `time_length(i, "month")` | `1` |
| `time_length(i, "year")` | `0.08196721` |
| `as.period(i)` | `1m 1d 0H 0M 0S` |
| `as.period(i, unit = "day")` | `30d 0H 0M 0S` |
| `i / months(1)` | `1.016667` (not `time_length`'s `1`) |
| `i %/% months(1)` | `1` |
| `i / ddays(1)` | `30` |
| `time_length(interval(ymd("2012-01-31"), ymd("2012-02-29")), "month")` | `0.9666667` |
| `time_length(interval(ymd("2012-02-29"), ymd("2013-02-28")), "year")` | `0.9972678` |
| `time_length(interval(ymd("2000-02-29"), ymd("2024-02-28")), "year")` | `23.99726`; `%/% years(1)` is `23` |
| `time_length(int_flip(i), "month")` | `-1.032258` (not `-1`) |
| `ymd("2012-03-01") %within% i` | `TRUE` (both ends included) |
| `ymd("2012-03-01") %within% int_flip(i)` | `TRUE` (a reversed interval is standardized) |
| `int_overlaps(i, interval(ymd("2012-03-01"), ymd("2012-04-01")))` | `TRUE` (touching ends overlap) |
| 10–11 March 2024 in New York: `int_length`, `time_length("day")` | `82800`, `0.9583333` |
| same interval: `as.period()`, `as.period(unit = "hour")` | `1d 0H 0M 0S`, `23H 0M 0S` |
| `interval(NA, ymd("2012-03-01"))` | `NA--NA` |

`time_length` for months and years is anchored at the start. lubridate takes
the whole number of units from `as.period`, finds the anniversaries before
and after the end with `add_with_rollback(roll_to_first = TRUE)`, and adds
the fraction of that span that has elapsed. That is why 31 January to
1 March 2012 is exactly one month (the anniversary rolls to 1 March) and
29 February 2012 to 28 February 2013 is just short of a year. A reversed
interval is measured backward from its own start, so flipping it does not
simply negate the result. Units shorter than a month divide the exact
duration.

## 2. What tidy3 and Polars have today

### tidy3

| Piece | Where | Behaviour |
|---|---|---|
| `Period(months=, years=)` | `src/tidy3/dates.py:105` | Holds integer or column counts. Supports `-p` and `p + q`. |
| `months(n)`, `years(n)` | `dates.py:152`, `:163` | Reject non-integers; `n` may be a column. |
| `add_with_rollback(x, p)` | `dates.py:173` | Same result as `x + p`; months and years only. No `roll_to_first`. |
| `total_months`, `shift_polars` | `dates.py:397`, `:414` | `12 * years + months` in one `dt.offset_by("Nmo")`, so 29 Feb 2024 + `years(1) + months(1)` is 29 Mar 2025, as lubridate does. |
| `compile_period_bin` | `dates.py:465`, called from `expr.py:886` | Intercepts `+`/`-` with a period literal before it reaches `pl.lit`; `period - date` and `period + period` inside an expression raise. |
| `shift_pandas` | `dates.py:948`, called from `pandas_engine.py:154` | `pd.DateOffset(months=n)`; per-row counts loop in Python. |

The month rule deliberately differs from lubridate's `+`: tidy3 rolls back
where lubridate gives `NA` (documented in `docs/reference.md`, "Column
helpers, dates, and readers", and the `dates.py` module docstring). There
are no day, hour, minute, or second periods, no durations, and no intervals.
`date - date` passes straight through to the backend.

One existing gap surfaced while writing this note: on a timezone-aware
column, `datetime + months(1)` into a skipped hour gives Polars' answer, not
lubridate's.

| Input (New York) | tidy3 0.9.0, both backends | lubridate `+` | lubridate `%m+%` |
|---|---|---|---|
| `2024-02-10 02:30 EST` + one month | `2024-03-10 01:30 EST` | `NA` | `2024-03-10 03:00 EDT` |

The 01:30 EST result is an hour *before* the requested clock time and
matches neither R operator. Phase 1 fixes it (decision 3).

### Polars

| Need | Polars | Matches lubridate? |
|---|---|---|
| Exact duration column | `pl.Duration(time_unit)`, `pl.duration(days=..., hours=...)` (accepts columns) | Yes |
| `date - date` | `Duration("us")`; `30 days, 0:00:00` | Same value as `as.duration`; R gives a `difftime` |
| Duration to number | `dt.total_days()`, `total_hours()`, … | Truncates to an integer unless `fractional=True` (36 hours: `1` versus `1.5`) |
| `sum`, `mean` of durations | Native; `mean` of 1 s and 2 s is 1.5 s | Yes |
| Date + `pl.duration(hours=36)` | **`Date` 2012-02-01**: the 12 hours are silently dropped | No; R gives a datetime |
| Calendar months/years | `dt.offset_by("1mo")`, `"1y"`, also per row via `pl.format` | Yes, as `%m+%` (31 Jan 2012 + `1mo` is 29 Feb) |
| Calendar days on a zoned datetime | `dt.offset_by("1d")` keeps 12:00 across the spring change | Yes |
| Clock hours on a zoned datetime | `dt.offset_by("1h")`, `"2h"` are **exact** hours: 01:30 + `1h` on 10 March is 03:30 EDT | No; `hours(1)` is clock time, giving `NA` |
| Calendar day into a skipped hour | `2024-03-09 02:30` + `"1d"` is `2024-03-10 01:30 EST` | No; R gives `NA` |
| Calendar day into a repeated hour | `2024-11-02 01:30` + `"1d"` is `01:30 EDT` (earlier) | Yes |
| Resolving a local clock | `dt.replace_time_zone(tz, ambiguous=, non_existent=)`, where `ambiguous` may be a per-row expression | Building block |
| Start/end pair | `pl.struct("start", "end")` | Building block |
| Struct to pandas | Default: `object` column of dicts. `use_pyarrow_extension_array=True`: `struct<start: timestamp[us], end: timestamp[us]>[pyarrow]` | — |
| Duration to pandas | `timedelta64[us]` | — |

pandas behaves similarly: `pd.DateOffset(days=1)` moves the calendar,
`pd.Timedelta(days=1)` is exact, and `DateOffset(days=1)` into the skipped
hour returns the impossible wall time `2024-03-10 02:30:00-05:00`.

**Conclusion.** Polars already does exact durations and calendar months,
years, and days. tidy3 must add clock-time hours, minutes, and seconds;
missing results for skipped clocks; a rule for repeated clocks; whole-day
checks when a Date meets a duration; and the anchored `time_length`.

## 3. Proposed API and representation

### Durations

| Item | Design |
|---|---|
| Constructors | `dseconds`, `dminutes`, `dhours`, `ddays`, `dweeks`, `dmonths`, `dyears`, `duration(num, units)`, `as_duration(x)` |
| Arguments | Integer, float, or column. Fractions are allowed, as in R. Missing counts give missing durations. |
| Year and month | 365.25 days and 30.4375 days, as in R |
| Scalar | `Duration` Python object holding microseconds; compiles to `pl.duration(microseconds=...)` |
| Column dtype | `pl.Duration("us")` on Polars, `timedelta64[us]` on pandas, which survives `collect(as_="pandas")` unchanged |
| `date - date`, `datetime - datetime` | Stay a `Duration` column (`30d`), the value of `as.duration`. No `difftime` and no automatic units. |
| `datetime + ddays(1)` | Exact: `2024-03-09 12:00 EST` + `ddays(1)` is `2024-03-10 13:00 EDT` |
| `date + duration` | Whole-day literal stays a Date. Any other literal, or a duration column, gives a naive datetime (UTC clocks, as `as_datetime` does). Polars' silent truncation is never exposed. |
| Comparison | `ddays(1) == days(1)` is true, as in R; other duration–period comparisons convert the period with average lengths |
| To numbers | `time_length(d, "hour")` and `as_numeric(d, "hour")`, always fractional |
| Summaries | `sum`, `mean`, `min`, `max`, `median` of a duration column are durations |

```python
cars >> mutate(quarter_mile=dseconds(col("qsec")))   # Duration column
cars >> filter(dseconds(col("qsec")) < dseconds(17))
```

### Periods

| Item | Design |
|---|---|
| Constructors | Existing `months`, `years`; new `weeks`, `days`, `hours`, `minutes`, `seconds`, `period(**units)`, and `as_period` |
| Arguments | Integers or columns only; `days(1.5)` raises, as in R |
| Python object | `Period` with slots `years, months, days, hours, minutes, seconds` (weeks become 7 days, as R prints `7d`). `+`, `-`, and `* int` combine slots; no normalization, so `hours(25)` stays 25 hours. |
| Column dtype | None in Phase 1; a struct column in Phase 3 (decision 4) |
| Arithmetic order | One step: all months (`12 * years + months`) with rollback, then days on the calendar, then hours/minutes/seconds on the local clock. Then the local clock is placed back in the zone. |
| `date + days(1)` | Date. A period with clock units gives a naive datetime, as R gives a UTC `POSIXct`. |
| `datetime + days(1)` across the spring change | `2024-03-10 12:00 EDT` (23 real hours), matching R |
| `datetime + hours(1)` at 01:30 on 10 March | missing, matching R |
| Skipped clock | Missing, matching lubridate's `+` (decision 3) |
| Repeated clock | Keep the input's UTC offset when it is valid for the result; otherwise the earlier occurrence (see section 4) |
| `period + duration` | Raises, as R does |

Polars plan for a zoned column (one expression, stays lazy):

```python
local = x.dt.replace_time_zone(None)
moved = local.dt.offset_by("1mo2d3h")          # calendar, then clock
moved.dt.replace_time_zone(tz, non_existent="null", ambiguous=same_offset_or_earliest)
```

Naive datetimes and Dates skip the zone steps, so Date and naive arithmetic
stays a single `offset_by`, as it is today.

### Intervals

| Item | Design |
|---|---|
| Constructor | `interval(start, end)`; no `%--%` operator in Python |
| Column dtype | `pl.Struct({"start": T, "end": T})`, where `T` is the inputs' common dtype. A Date stays a Date (decision 5); a zoned start keeps its zone and the end is converted to it, as R does. |
| Recognition | The frame records interval columns by name, as it records factor levels, so a user's unrelated `{start, end}` struct is not treated as one |
| Missing | Either end missing makes the whole struct missing, so `is_na(col("season"))` works |
| Accessors | `int_start`, `int_end`, `int_length` (a Duration; R gives seconds), `int_flip`, `int_shift(i, period_or_duration)`, `int_standardize` |
| Predicates | `within(x, i)` for `%within%` (closed, reversed intervals standardized; `i` may also be an interval); `int_overlaps(a, b)` (touching ends overlap) |
| Measuring | `time_length(i, unit)` with lubridate's anchored months and years; exact division for shorter units. `as_period(i, unit=None)` returns a period struct (Phase 3). |
| Division | `i // months(1)` and `i // years(1)` give whole units (`%/%`, the age idiom). `i / period` raises and points to `time_length` (section 4). `i / ddays(1)` is exact. |
| pandas backend | `pd.ArrowDtype` struct column. `int_start` uses `Series.struct.field("start")`. |
| Collection to pandas | Interval columns are converted with Arrow struct dtype even when `arrow_backed=False`, so a pandas user gets `.struct.field` rather than an `object` column of dicts. Duration columns become `timedelta64[us]`. |

Example with `penguins_raw` (Palmer penguins, from Rdatasets), whose
`Date Egg` column is the clutch completion date:

```python
raw = (
    read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins_raw.csv")
    >> clean_names()                            # date_egg, study_name, species, ...
    >> mutate(date_egg=as_date(col("date_egg")))
)

# Durations: days since the first clutch in each study
raw >> group_by("study_name") >> mutate(since_first=col("date_egg") - min("date_egg"))

# Periods: a follow-up visit one month later, rolled back at month ends
raw >> mutate(follow_up=col("date_egg") + months(1), next_day=col("date_egg") + days(1))

# Intervals in summarise, then measured in mutate
seasons = (
    raw
    >> group_by("study_name")
    >> summarise(season=interval(min("date_egg"), max("date_egg")))
    >> mutate(weeks=time_length(col("season"), "week"),
              months_=time_length(col("season"), "month"))
)

# Intervals in filter
raw >> filter(within(col("date_egg"), interval(date(2008, 11, 1), date(2008, 11, 30))))
```

### What the verbs need

| Verb | Requirement |
|---|---|
| `mutate` | Span arithmetic compiles to native Polars expressions; a period with column counts uses `pl.format` offsets, as `months(col("n"))` does now. Interval columns are registered in frame metadata. |
| `summarise` | `sum`/`mean` of durations native; `interval(min(...), max(...))` builds one per group; `min`/`max` on an interval column raise rather than comparing structs field by field. |
| `filter` | Duration comparisons are native; `within` and `int_overlaps` are plain Boolean expressions. |
| `arrange` | Durations sort natively. Interval columns sort by start, then end (Polars' struct order). |
| Joins, `bind_rows`, `pivot_*` | Interval metadata follows the column, as factor levels do. Struct columns already survive reshaping. |
| `collect(as_="pandas")` | Durations as `timedelta64[us]`; intervals as Arrow structs (above); export to Polars code emits `pl.Struct`. |

## 4. Deliberate differences from lubridate

| Topic | lubridate | tidy3 | Reason |
|---|---|---|---|
| `+ months(n)`, `+ years(n)` at month ends | `NA` | Roll back (`%m+%`) | Existing documented contract since `months()` landed; Polars' native rule; one-step `12 * years + months`. Changing it would break users silently. |
| Mixed period with an impossible month step | `ymd("2010-01-31") + (months(1) + days(1))` is `NA` | 1 March 2010 (28 February + 1 day) | Follows from rollback; months move first, then days. |
| Repeated clock after period arithmetic | Always the earlier repeat; `x + seconds(0)` can change the instant | Keep the input's offset when still valid, else earlier | Adding zero must be a no-op. `force_tz` already preserves an already resolved repeat. |
| `date - date` | `difftime` with automatic units | `Duration` column | Python and Polars have no `difftime`; a duration converts to any unit with `time_length`. |
| `int_length` | Seconds as a double | `Duration` | Typed result; `time_length(i, "second")` gives the number. |
| Interval ends from Dates | Converted to UTC datetimes | Stay Dates (decision 5) | No invented time zone; dates compare and print as dates. |
| `interval / months(1)` | `1.016667`, unlike `time_length`'s `1` | Raises; points to `time_length` | Two answers for one question invite errors. `//` keeps the whole-unit idiom. |
| `sum(days(1:3))` | `0` | Not available until period columns exist; then field-wise sums | R's result is a slot artefact, not a design. |
| Date + duration, mixed vector | Per vector: all whole days stay Date | Decided by the literal, or always datetime for a duration column | A lazy plan must know its dtype before reading rows. |
| `%within%`, `%--%` | Infix operators | `within()`, `interval()` | Python cannot define infix names. |

Every other case should match lubridate, including the anchored and
asymmetric `time_length` for reversed intervals, 365.25-day `dyears`, and
`NA` for periods landing in a skipped clock.

## 5. Decisions

All seven questions were decided on 2026-10-25, each as recommended. The
trade-off of each is kept as its rationale.

1. **`+ months(1)` keeps rollback.** Rollback stays the only `+`
   behaviour; `add_with_rollback` gains lubridate's `roll_to_first=True`.
   There is no NA mode.
   *Trade-off accepted:* scripts ported from R that rely on `NA` to drop
   31sts behave differently; an `invalid="NA"` option could come later
   without breaking anything.
2. **`date - date` returns a `Duration` column**, unchanged from Polars.
   *Trade-off accepted:* R prints `30 days` and `as.numeric` gives 30;
   tidy3 users write `time_length(d, "day")`. A difftime-like class would
   add a dtype Polars does not have.
3. **Skipped clock after adding a period to a zoned datetime:** missing
   for `+` with any period, matching lubridate's `+`; `add_with_rollback`
   gives the transition (03:00 EDT), matching `%m+%`. Both replace today's
   01:30 EST.
   *Trade-off accepted:* `x + months(1)` and `add_with_rollback(x,
   months(1))` differ for zoned datetimes only. The alternative, the
   transition for both, never loses data but departs from lubridate's `+`.
4. **Period columns wait for Phase 3.** Phase 1 keeps periods as operands
   only. Phase 3 adds a struct column (`years, months, days, hours,
   minutes, seconds`) for `as_period(interval)` and `period(col(...))`.
   *Trade-off accepted:* until then `as.period` results cannot be stored;
   the struct needs its own printing and pandas conversion.
5. **Interval ends built from Dates stay Dates.**
   *Trade-off accepted:* `int_start` returns a Date, not R's UTC datetime,
   and `int_length` of a Date interval is whole days only.
6. **Interval columns collect to pandas as an Arrow struct dtype**, even
   without `arrow_backed=True`.
   *Trade-off accepted:* one column type differs from the default
   NumPy-backed conversion. The alternatives were an `object` column of
   dicts, which pandas cannot compute on, or splitting into
   `name_start`/`name_end`, which round-trips poorly.
7. **A repeated clock keeps the input's offset**, otherwise the earlier
   occurrence, so `x + seconds(0)` is `x`.
   *Trade-off accepted:* a few autumn results differ from lubridate (its
   `x + seconds(0)` moves the instant); matching R exactly is simpler to
   test but breaks `x + 0 == x`.

## 6. Phased implementation plan

| Phase | Scope | Notes |
|---|---|---|
| 1 | Duration constructors, `as_duration`, `time_length`/`as_numeric` for durations; `days`, `weeks`, `hours`, `minutes`, `seconds`, `period()`; widened `Period`; clock-time arithmetic and zone resolution; fix the skipped-hour month result; `add_with_rollback(roll_to_first=)` | Extends `compile_period_bin` and `shift_pandas`; no new dtypes |
| 2 | `interval`, accessors, `within`, `int_overlaps`, `int_shift`, `int_flip`, `int_standardize`, `time_length` for intervals, `//` with periods, interval metadata and pandas collection | The struct column and metadata land together |
| 3 | Period columns, `as_period`, `period_to_seconds`, `seconds_to_period` | Struct column as decided (decision 4) |

Each phase updates `docs/reference.md`, `CHANGELOG.md`, the "Where these
differ from R on purpose" list, and the `bench_api` classification.

### R oracle cases

Add to `tests/r_oracle_cases.R`, each run on both backends:

- Durations: every constructor at 1 and 1.5; `dyears(1)` and `ddays(365)`
  from 29 February 2012; `ddays(1)` over both 2024 New York changes; `date
  + ddays(1)` versus `ddays(1.5)` dtype; duration `sum` and `mean`.
- Periods: `days(1)` and `hours(24)` over both changes; `days(1)` landing in
  the skipped hour; `hours(1)` from 01:30 on 10 March; `hours(2)` from 00:30
  on 3 November; `days(1)` landing in the repeated hour from each side;
  `months(1)` into the skipped hour for `+` and `%m+%`; 31 January and
  29 February with `months`, `years`, and `months(1) + days(1)` (recorded as
  deliberate differences); `roll_to_first`; column counts with missing values.
- Intervals: every row of the interval table in section 1, including
  reversed intervals, touching overlaps, missing ends, the DST day, and
  the age case 29 February 2000 to 28 February 2024.
- A half-hour zone (`Australia/Lord_Howe`) for one period and one interval
  case, as the `force_tz` tests already do.

### Benchmark expectations against raw Polars

| Case | Raw Polars recipe | Expected tidy3 cost |
|---|---|---|
| `date + ddays(1)`, `date - date`, duration `sum` | Same expression | Within 1.1×; compile only |
| `date + days(1)`, `+ months(1)` on Date | `dt.offset_by` | Within 1.1× (today's months path) |
| `datetime + hours(1)` on a zoned column | `replace_time_zone(None)`, `offset_by`, `replace_time_zone(tz, ...)` | Within 1.3×; the offset-preserving `ambiguous` expression is the extra cost |
| `time_length(i, "month")` | Hand-written anniversary recipe | Within 1.5×; several `offset_by` calls per row, all lazy |
| `within`, `int_overlaps` | `is_between` / two comparisons on struct fields | Within 1.1× |
| Per-row period counts on pandas | — | Python loop today (`shift_pandas`); vectorize by grouping equal counts |

No span operation should read source rows to decide a dtype, so every
Polars plan stays lazy, as the review requires.
