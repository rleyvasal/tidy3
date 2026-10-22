# Polars-relative closest() timings

Python 3.14.6; Polars 1.42.1; Darwin arm64.

One correctness comparison, 1 warm-up(s), best of 3 alternating timing rounds.
Input construction and validation are excluded from timing. Both engines materialize the same complete Polars frame, including input row order and dtypes.

| Workload | Rows | Lookup rows | Width | Raw Polars | tidy3 | tidy3 / Polars |
|---|---:|---:|---:|---:|---:|---:|
| original_sweep | 200,000 | 100 | 30 | 19.36 ms | 37.32 ms | 1.93x |
| wide_200k | 200,000 | 1,000 | 21 | 2.78 ms | 10.37 ms | 3.73x |
| wide_1m_sparse | 1,000,000 | 1,000 | 21 | 18.95 ms | 57.18 ms | 3.02x |
| wide_1m_dense | 1,000,000 | 100,000 | 21 | 19.36 ms | 61.97 ms | 3.20x |
| grouped_1m | 1,000,000 | 100,000 | 3 | 13.86 ms | 28.69 ms | 2.07x |

Peak resident memory is measured separately in fresh processes, once per engine. It includes interpreter imports, input construction, execution and the complete output; it is not incremental operator allocation. These RSS measurements require macOS or Linux.

| Workload | Raw Polars peak RSS | tidy3 peak RSS |
|---|---:|---:|
| original_sweep | 471.0 MiB | 627.2 MiB |
| wide_200k | 276.8 MiB | 320.5 MiB |
| wide_1m_sparse | 691.4 MiB | 807.7 MiB |
| wide_1m_dense | 697.0 MiB | 803.0 MiB |
| grouped_1m | 352.5 MiB | 439.2 MiB |

These measured ratios are separate from CI regression limits. The benchmark covers unique lookup keys and default `multiple="all"`; duplicate-match correctness is covered by the rolling-join tests.
