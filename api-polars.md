# Polars-relative public API timings

Python 3.12.12; Polars 2.1.0; Darwin arm64.

100,000 rows. One correctness comparison, 1 warm-up(s), best of 3 alternating rounds.
Preparation is outside the samples. Cases are ranked by absolute excess time (tidy3 minus raw Polars).

| Case | Family | Rows | Raw Polars | tidy3 | Ratio | Excess |
|---|---|---:|---:|---:|---:|---:|
| parse_date_time | convert | 100,000 | 1.07 ms | 18.82 ms | 17.52x | 17.75 ms |
| parse_logical | convert | 100,000 | 0.65 ms | 3.74 ms | 5.79x | 3.09 ms |
| parse_time | convert | 100,000 | 0.77 ms | 3.07 ms | 3.96x | 2.30 ms |
| stop_for_problems | convert | 100,000 | 0.35 ms | 2.14 ms | 6.10x | 1.79 ms |
| ymd_h | convert | 100,000 | 1.17 ms | 2.95 ms | 2.52x | 1.78 ms |
| dmy_h | convert | 100,000 | 1.09 ms | 2.70 ms | 2.49x | 1.62 ms |
| mdy_h | convert | 100,000 | 1.10 ms | 2.68 ms | 2.43x | 1.58 ms |
| parse_double | convert | 100,000 | 0.44 ms | 1.96 ms | 4.41x | 1.51 ms |
| parse_number | convert | 100,000 | 1.92 ms | 3.38 ms | 1.75x | 1.45 ms |
| as_datetime | convert | 100,000 | 1.84 ms | 3.25 ms | 1.77x | 1.41 ms |
| as_logical | convert | 100,000 | 0.61 ms | 1.93 ms | 3.19x | 1.33 ms |
| parse_datetime | convert | 100,000 | 1.17 ms | 2.48 ms | 2.13x | 1.32 ms |
| as_numeric | convert | 100,000 | 0.40 ms | 1.62 ms | 4.07x | 1.22 ms |
| as_double | convert | 100,000 | 0.30 ms | 1.45 ms | 4.80x | 1.15 ms |
| mdy | convert | 100,000 | 0.43 ms | 1.54 ms | 3.58x | 1.11 ms |
| as_string | convert | 100,000 | 0.49 ms | 1.46 ms | 2.99x | 0.97 ms |
| problems | convert | 100,000 | 0.38 ms | 1.23 ms | 3.23x | 0.85 ms |
| mdy_hms | convert | 100,000 | 1.24 ms | 2.08 ms | 1.68x | 0.84 ms |
| ymd | convert | 100,000 | 0.46 ms | 1.28 ms | 2.82x | 0.83 ms |
| ymd_hm | convert | 100,000 | 1.02 ms | 1.80 ms | 1.77x | 0.78 ms |
| as_date | convert | 100,000 | 0.40 ms | 1.15 ms | 2.90x | 0.76 ms |
| ym | convert | 100,000 | 0.44 ms | 1.18 ms | 2.71x | 0.75 ms |
| ymd_hms | convert | 100,000 | 1.26 ms | 1.98 ms | 1.57x | 0.72 ms |
| dmy_hm | convert | 100,000 | 1.02 ms | 1.73 ms | 1.70x | 0.71 ms |
| mdy_hm | convert | 100,000 | 0.99 ms | 1.69 ms | 1.71x | 0.70 ms |
| dmy | convert | 100,000 | 0.54 ms | 1.24 ms | 2.31x | 0.70 ms |
| parse_date | convert | 100,000 | 0.40 ms | 1.08 ms | 2.71x | 0.68 ms |
| parse_integer | convert | 100,000 | 0.19 ms | 0.85 ms | 4.53x | 0.66 ms |
| dmy_hms | convert | 100,000 | 1.00 ms | 1.65 ms | 1.65x | 0.65 ms |
| as_character | convert | 100,000 | 0.34 ms | 0.94 ms | 2.76x | 0.60 ms |
| as_integer | convert | 100,000 | 0.35 ms | 0.88 ms | 2.55x | 0.54 ms |

33 exports and 7 methods have no raw Polars comparison.
