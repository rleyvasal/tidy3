# Public API benchmark coverage audit

Snapshot: 2026-10-21. This inventories current coverage; it is not a completed all-functions benchmark.

The public `tidy3.__all__` exports **351 functions** and 9 other objects. `TidyFrame` exposes **98 public methods**. Aliases are listed separately so public entry points cannot disappear from the audit.

| Function status | Names | Meaning |
|---|---:|---|
| Validated | 28 | Dedicated operation has a checked, matched raw Polars timing case. |
| Legacy native comparison | 2 | Native timings exist in `bench.run_ops`; that runner does not validate outputs before timing. |
| Pending | 321 | Needs a checked benchmark mapping, contextual measurement, or an explicit non-comparable reason. |

These counts concern individual functions, not every parameter or dtype. `closest`, `join_by`, `ge`, aggregation expressions, and selectors can participate in another operation without having an isolated benchmark. A helper should be measured in its consuming operation and, where useful, in query construction. Such participation does not count as complete function coverage here.

`bench_suite` has pandas references, so its workflow coverage does not fill the raw Polars gaps. The existing `nest_by` timing label invokes `nest(by=...)`; it does not measure the distinct `nest_by()` function. The 98 method entry points also need explicit case mappings; timing `.collect()` as part of another case does not isolate collection or method dispatch.

## Current timing evidence

Python 3.14.6; Polars 1.42.1; macOS-27.0.1-arm64-arm-64bit-Mach-O. Existing speed guards run alone: 110 passed; 58 timing comparisons. Each uses one warm-up and the best of three samples. Full case values and CI limits are in the [JSON snapshot](public-api-coverage-2026-10-21.json).

Largest absolute gaps among the existing **Polars-input, Polars-output** cases:

| Case | Raw Polars | tidy3 | Ratio | Excess time |
|---|---:|---:|---:|---:|
| `test_many_groups_speed[fct_reorder-polars]` | 4.78 ms | 36.10 ms | 7.56x | 31.33 ms |
| `test_function_speed_is_close_to_native[ymd_hms-polars]` | 6.38 ms | 35.71 ms | 5.60x | 29.33 ms |
| `test_many_groups_speed[fct_infreq-polars]` | 3.63 ms | 31.73 ms | 8.74x | 28.10 ms |
| `test_function_speed_is_close_to_native[ymd-polars]` | 1.56 ms | 19.05 ms | 12.17x | 17.48 ms |
| `test_function_speed_is_close_to_native[mdy-polars]` | 1.57 ms | 18.67 ms | 11.88x | 17.10 ms |
| `test_function_speed_is_close_to_native[as_datetime-polars]` | 6.53 ms | 22.81 ms | 3.49x | 16.28 ms |
| `test_many_groups_speed[fct_relevel-polars]` | 3.41 ms | 19.23 ms | 5.64x | 15.82 ms |
| `test_function_speed_is_close_to_native[parse_number-polars]` | 16.45 ms | 31.36 ms | 1.91x | 14.91 ms |

These are workload-specific gaps, not proof of their cause. Canonical parser inputs use a fixed native format; tidy3 also supports guessing and diagnostics. Grouped factors operate on 100,000 rows and 20,000 groups. Column cases use 200,000 rows. The scan case times construction of a factor plan, not final collection. Pandas-input column cases have a different input representation from their raw Polars baseline and are recorded separately in the JSON; their ratios include that backend path.

The [rolling-join report](rolling-joins-polars-2026-10-21.md) separately measures five cases through one million rows and process peak RSS. Memory has not yet been measured for the other function families.

## Benchmark completion criteria

- Every function and method has an explicit case mapping, a verified alias mapping, or a reason a raw Polars comparison is not meaningful. New public names must appear as pending until mapped.
- Independent raw Polars recipes implement the same semantics: full values, names, dtypes, ordering, missing values, factor levels, join ties, and output representation. Recipes must not call tidy3 internally. APIs without a direct native equivalent need a composed recipe or an explicit non-comparable status.
- Data operations run at 1,000, 100,000, and 1,000,000 rows, with an opt-in 10,000,000-row stress profile where output size permits. Select representative wide, many-group, high-cardinality, null-heavy, and tie-heavy cases per family instead of taking an unbounded Cartesian product.
- Report query construction and complete materialization separately. Use identical input representations for wrapper-overhead comparisons. Record native Polars, Arrow, pandas, and NumPy output boundaries as separate cases.
- Preserve samples, medians, absolute excess time, ratios, environment, input shape, and source revision. Alternate timing order after warm-up. Measure process peak RSS in separate workers. Keep expensive scaling runs outside ordinary unit tests.
- Rank optimizations by repeatable absolute excess time at scale, then profile the cause. Check semantics and rerun the same cases before and after each change. Performance targets apply to meaningful elapsed times; fixed wrapper costs need absolute budgets on tiny operations.

## First optimization candidate

Grouped `fct_reorder` and `fct_infreq` have the largest measured Polars-input excess times in this audit. Profile their level reconciliation and grouping paths before changing them. The canonical date parser cases are the next measured candidates. Unmeasured families may reveal larger gaps when covered.

## Function inventory

Source links point to this working-tree snapshot. `Validated` means a workload exists; it does not mean native-speed parity or full semantic coverage.

### convert

25 exported functions; 11 validated, 0 legacy native comparisons, 14 pending.

| Function | Status | Existing cases |
|---|---|---|
| [as_character](../../src/tidy3/convert.py#L1650) | pending | — |
| [as_date](../../src/tidy3/convert.py#L1716) | validated | column/as_date |
| [as_datetime](../../src/tidy3/convert.py#L1726) | validated | column/as_datetime |
| [as_double](../../src/tidy3/convert.py#L1615) | validated | column/as_double |
| [as_integer](../../src/tidy3/convert.py#L1632) | pending | — |
| [as_logical](../../src/tidy3/convert.py#L1702) | pending | — |
| [as_numeric](../../src/tidy3/convert.py#L1626) | pending | — |
| [as_string](../../src/tidy3/convert.py#L1691) | validated | column/as_string |
| [dmy](../../src/tidy3/convert.py#L1852) | pending | — |
| [dmy_hms](../../src/tidy3/convert.py#L1918) | pending | — |
| [locale](../../src/tidy3/convert.py#L209) | pending | — |
| [mdy](../../src/tidy3/convert.py#L1847) | validated | column/mdy |
| [mdy_hms](../../src/tidy3/convert.py#L1910) | pending | — |
| [parse_date](../../src/tidy3/convert.py#L1786) | pending | — |
| [parse_datetime](../../src/tidy3/convert.py#L1810) | validated | column/parse_datetime |
| [parse_double](../../src/tidy3/convert.py#L1745) | pending | — |
| [parse_integer](../../src/tidy3/convert.py#L1757) | validated | column/parse_integer |
| [parse_logical](../../src/tidy3/convert.py#L1767) | validated | column/parse_logical |
| [parse_number](../../src/tidy3/convert.py#L1775) | validated | column/parse_number |
| [problems](../../src/tidy3/convert.py#L1573) | pending | — |
| [stop_for_problems](../../src/tidy3/convert.py#L1583) | pending | — |
| [ym](../../src/tidy3/convert.py#L1857) | pending | — |
| [ymd](../../src/tidy3/convert.py#L1835) | validated | column/ymd |
| [ymd_hm](../../src/tidy3/convert.py#L1898) | pending | — |
| [ymd_hms](../../src/tidy3/convert.py#L1878) | validated | column/ymd_hms |

### dates

22 exported functions; 0 validated, 0 legacy native comparisons, 22 pending.

| Function | Status | Existing cases |
|---|---|---|
| [add_with_rollback](../../src/tidy3/dates.py#L173) | pending | — |
| [ceiling_date](../../src/tidy3/dates.py#L328) | pending | — |
| [day](../../src/tidy3/dates.py#L195) | pending | — |
| [days_in_month](../../src/tidy3/dates.py#L340) | pending | — |
| [excel_numeric_to_date](../../src/tidy3/dates.py#L638) | pending | — |
| [floor_date](../../src/tidy3/dates.py#L315) | pending | — |
| [force_tz](../../src/tidy3/dates.py#L269) | pending | — |
| [hour](../../src/tidy3/dates.py#L243) | pending | — |
| [isoweek](../../src/tidy3/dates.py#L225) | pending | — |
| [minute](../../src/tidy3/dates.py#L248) | pending | — |
| [month](../../src/tidy3/dates.py#L190) | pending | — |
| [months](../../src/tidy3/dates.py#L152) | pending | — |
| [now](../../src/tidy3/dates.py#L370) | pending | — |
| [quarter](../../src/tidy3/dates.py#L230) | pending | — |
| [second](../../src/tidy3/dates.py#L253) | pending | — |
| [today](../../src/tidy3/dates.py#L361) | pending | — |
| [wday](../../src/tidy3/dates.py#L204) | pending | — |
| [week](../../src/tidy3/dates.py#L220) | pending | — |
| [with_tz](../../src/tidy3/dates.py#L258) | pending | — |
| [yday](../../src/tidy3/dates.py#L215) | pending | — |
| [year](../../src/tidy3/dates.py#L184) | pending | — |
| [years](../../src/tidy3/dates.py#L163) | pending | — |

### eda

8 exported functions; 0 validated, 0 legacy native comparisons, 8 pending.

| Function | Status | Existing cases |
|---|---|---|
| [colnames](../../src/tidy3/eda.py#L169) | pending | — |
| [describe](../../src/tidy3/eda.py#L316) | pending | — |
| [dim](../../src/tidy3/eda.py#L207) | pending | — |
| [dtypes](../../src/tidy3/eda.py#L218) | pending | — |
| [names](../../src/tidy3/eda.py#L99) | pending | — |
| [ncol](../../src/tidy3/eda.py#L200) | pending | — |
| [nrow](../../src/tidy3/eda.py#L193) | pending | — |
| [summary](../../src/tidy3/eda.py#L282) | pending | — |

### export

4 exported functions; 0 validated, 0 legacy native comparisons, 4 pending.

| Function | Status | Existing cases |
|---|---|---|
| [collect_known_names](../../src/tidy3/export.py#L94) | pending | — |
| [nb_export](../../src/tidy3/export.py#L392) | pending | — |
| [transform_cell](../../src/tidy3/export.py#L308) | pending | — |
| [transform_source](../../src/tidy3/export.py#L240) | pending | — |

### expr

46 exported functions; 0 validated, 0 legacy native comparisons, 46 pending.

| Function | Status | Existing cases |
|---|---|---|
| [all](../../src/tidy3/expr.py#L216) | pending | — |
| [any](../../src/tidy3/expr.py#L216) | pending | — |
| [between_values](../../src/tidy3/expr.py#L487) | pending | — |
| [case_match](../../src/tidy3/expr.py#L581) | pending | — |
| [case_when](../../src/tidy3/expr.py#L528) | pending | — |
| [coalesce](../../src/tidy3/expr.py#L518) | pending | — |
| [col](../../src/tidy3/expr.py#L200) | pending | — |
| [consecutive_id](../../src/tidy3/expr.py#L495) | pending | — |
| [cumall](../../src/tidy3/expr.py#L393) | pending | — |
| [cumany](../../src/tidy3/expr.py#L397) | pending | — |
| [cume_dist](../../src/tidy3/expr.py#L341) | pending | — |
| [cummean](../../src/tidy3/expr.py#L389) | pending | — |
| [dense_rank](../../src/tidy3/expr.py#L333) | pending | — |
| [desc](../../src/tidy3/expr.py#L205) | pending | — |
| [first](../../src/tidy3/expr.py#L285) | pending | — |
| [if_else](../../src/tidy3/expr.py#L524) | pending | — |
| [lag](../../src/tidy3/expr.py#L370) | pending | — |
| [last](../../src/tidy3/expr.py#L299) | pending | — |
| [lead](../../src/tidy3/expr.py#L351) | pending | — |
| [max](../../src/tidy3/expr.py#L216) | pending | — |
| [mean](../../src/tidy3/expr.py#L216) | pending | — |
| [median](../../src/tidy3/expr.py#L216) | pending | — |
| [min](../../src/tidy3/expr.py#L216) | pending | — |
| [min_rank](../../src/tidy3/expr.py#L329) | pending | — |
| [n](../../src/tidy3/expr.py#L210) | pending | — |
| [n_distinct](../../src/tidy3/expr.py#L445) | pending | — |
| [na_if](../../src/tidy3/expr.py#L483) | pending | — |
| [near](../../src/tidy3/expr.py#L477) | pending | — |
| [nth](../../src/tidy3/expr.py#L455) | pending | — |
| [ntile](../../src/tidy3/expr.py#L345) | pending | — |
| [order_by](../../src/tidy3/expr.py#L709) | pending | — |
| [percent_rank](../../src/tidy3/expr.py#L337) | pending | — |
| [recode](../../src/tidy3/expr.py#L770) | pending | — |
| [recode_values](../../src/tidy3/expr.py#L644) | pending | — |
| [replace_values](../../src/tidy3/expr.py#L675) | pending | — |
| [replace_when](../../src/tidy3/expr.py#L696) | pending | — |
| [rescale](../../src/tidy3/expr.py#L422) | pending | — |
| [row_number](../../src/tidy3/expr.py#L324) | pending | — |
| [scale](../../src/tidy3/expr.py#L401) | pending | — |
| [sd](../../src/tidy3/expr.py#L216) | pending | — |
| [std](../../src/tidy3/expr.py#L216) | pending | — |
| [sum](../../src/tidy3/expr.py#L233) | pending | — |
| [var](../../src/tidy3/expr.py#L216) | pending | — |
| [when_all](../../src/tidy3/expr.py#L762) | pending | — |
| [when_any](../../src/tidy3/expr.py#L753) | pending | — |
| [with_order](../../src/tidy3/expr.py#L722) | pending | — |

### forcats

4 exported functions; 4 validated, 0 legacy native comparisons, 0 pending.

| Function | Status | Existing cases |
|---|---|---|
| [fct_infreq](../../src/tidy3/forcats.py#L71) | validated | grouped/fct_infreq, lazy-scan/fct_infreq |
| [fct_relevel](../../src/tidy3/forcats.py#L37) | validated | grouped/fct_relevel |
| [fct_reorder](../../src/tidy3/forcats.py#L52) | validated | grouped/fct_reorder |
| [fct_rev](../../src/tidy3/forcats.py#L83) | validated | grouped/fct_rev |

### frame

1 exported functions; 0 validated, 0 legacy native comparisons, 1 pending.

| Function | Status | Existing cases |
|---|---|---|
| [tidy](../../src/tidy3/frame.py#L1535) | pending | — |

### groups

9 exported functions; 0 validated, 0 legacy native comparisons, 9 pending.

| Function | Status | Existing cases |
|---|---|---|
| [group_data](../../src/tidy3/groups.py#L179) | pending | — |
| [group_indices](../../src/tidy3/groups.py#L222) | pending | — |
| [group_keys](../../src/tidy3/groups.py#L197) | pending | — |
| [group_rows](../../src/tidy3/groups.py#L206) | pending | — |
| [group_size](../../src/tidy3/groups.py#L214) | pending | — |
| [group_trim](../../src/tidy3/groups.py#L241) | pending | — |
| [group_walk](../../src/tidy3/groups.py#L273) | pending | — |
| [groups](../../src/tidy3/groups.py#L234) | pending | — |
| [nest_by](../../src/tidy3/groups.py#L293) | pending | — |

### helpers

12 exported functions; 4 validated, 0 legacy native comparisons, 8 pending.

| Function | Status | Existing cases |
|---|---|---|
| [abs](../../src/tidy3/helpers.py#L159) | pending | — |
| [ceiling](../../src/tidy3/helpers.py#L169) | pending | — |
| [floor](../../src/tidy3/helpers.py#L164) | pending | — |
| [is_finite](../../src/tidy3/helpers.py#L149) | pending | — |
| [is_infinite](../../src/tidy3/helpers.py#L154) | pending | — |
| [is_na](../../src/tidy3/helpers.py#L128) | validated | column/is_na |
| [is_nan](../../src/tidy3/helpers.py#L139) | pending | — |
| [pmax](../../src/tidy3/helpers.py#L204) | pending | — |
| [pmin](../../src/tidy3/helpers.py#L190) | validated | column/pmin |
| [round](../../src/tidy3/helpers.py#L174) | validated | column/round |
| [row_means](../../src/tidy3/helpers.py#L237) | pending | — |
| [row_sums](../../src/tidy3/helpers.py#L212) | validated | column/row_sums |

### io

8 exported functions; 0 validated, 0 legacy native comparisons, 8 pending.

| Function | Status | Existing cases |
|---|---|---|
| [excel_sheets](../../src/tidy3/io.py#L416) | pending | — |
| [read_csv](../../src/tidy3/io.py#L343) | pending | — |
| [read_delim](../../src/tidy3/io.py#L398) | pending | — |
| [read_excel](../../src/tidy3/io.py#L425) | pending | — |
| [read_tsv](../../src/tidy3/io.py#L392) | pending | — |
| [scan_csv](../../src/tidy3/io.py#L304) | pending | — |
| [scan_ipc](../../src/tidy3/io.py#L53) | pending | — |
| [scan_parquet](../../src/tidy3/io.py#L48) | pending | — |

### janitor

7 exported functions; 0 validated, 0 legacy native comparisons, 7 pending.

| Function | Status | Existing cases |
|---|---|---|
| [compare_df_cols](../../src/tidy3/janitor.py#L528) | pending | — |
| [compare_df_cols_same](../../src/tidy3/janitor.py#L607) | pending | — |
| [describe_class](../../src/tidy3/janitor.py#L395) | pending | — |
| [find_header](../../src/tidy3/janitor.py#L675) | pending | — |
| [remove_constant](../../src/tidy3/janitor.py#L268) | pending | — |
| [remove_empty](../../src/tidy3/janitor.py#L163) | pending | — |
| [row_to_names](../../src/tidy3/janitor.py#L733) | pending | — |

### join_spec

10 exported functions; 0 validated, 0 legacy native comparisons, 10 pending.

| Function | Status | Existing cases |
|---|---|---|
| [between](../../src/tidy3/join_spec.py#L105) | pending | — |
| [closest](../../src/tidy3/join_spec.py#L92) | pending | — |
| [eq](../../src/tidy3/join_spec.py#L72) | pending | — |
| [ge](../../src/tidy3/join_spec.py#L80) | pending | — |
| [gt](../../src/tidy3/join_spec.py#L76) | pending | — |
| [join_by](../../src/tidy3/join_spec.py#L57) | pending | — |
| [le](../../src/tidy3/join_spec.py#L88) | pending | — |
| [lt](../../src/tidy3/join_spec.py#L84) | pending | — |
| [overlaps](../../src/tidy3/join_spec.py#L150) | pending | — |
| [within](../../src/tidy3/join_spec.py#L136) | pending | — |

### options

1 exported functions; 0 validated, 0 legacy native comparisons, 1 pending.

| Function | Status | Existing cases |
|---|---|---|
| [options](../../src/tidy3/options.py#L25) | pending | — |

### partial_run

4 exported functions; 0 validated, 0 legacy native comparisons, 4 pending.

| Function | Status | Existing cases |
|---|---|---|
| [looks_like_tidy_pipe](../../src/tidy3/partial_run.py#L38) | pending | — |
| [maybe_rewrite_cell](../../src/tidy3/partial_run.py#L371) | pending | — |
| [normalize_pipe_source](../../src/tidy3/partial_run.py#L72) | pending | — |
| [partial_run](../../src/tidy3/partial_run.py#L453) | pending | — |

### stringr

55 exported functions; 0 validated, 0 legacy native comparisons, 55 pending.

| Function | Status | Existing cases |
|---|---|---|
| [boundary](../../src/tidy3/stringr.py#L106) | pending | — |
| [coll](../../src/tidy3/stringr.py#L77) | pending | — |
| [fixed](../../src/tidy3/stringr.py#L72) | pending | — |
| [invert_match](../../src/tidy3/stringr.py#L340) | pending | — |
| [regex](../../src/tidy3/stringr.py#L82) | pending | — |
| [str_c](../../src/tidy3/stringr.py#L599) | pending | — |
| [str_conv](../../src/tidy3/stringr.py#L779) | pending | — |
| [str_count](../../src/tidy3/stringr.py#L258) | pending | — |
| [str_detect](../../src/tidy3/stringr.py#L243) | pending | — |
| [str_dup](../../src/tidy3/stringr.py#L505) | pending | — |
| [str_ends](../../src/tidy3/stringr.py#L253) | pending | — |
| [str_equal](../../src/tidy3/stringr.py#L293) | pending | — |
| [str_escape](../../src/tidy3/stringr.py#L774) | pending | — |
| [str_extract](../../src/tidy3/stringr.py#L533) | pending | — |
| [str_extract_all](../../src/tidy3/stringr.py#L543) | pending | — |
| [str_flatten](../../src/tidy3/stringr.py#L622) | pending | — |
| [str_flatten_comma](../../src/tidy3/stringr.py#L637) | pending | — |
| [str_glue](../../src/tidy3/stringr.py#L671) | pending | — |
| [str_glue_data](../../src/tidy3/stringr.py#L682) | pending | — |
| [str_ilike](../../src/tidy3/stringr.py#L288) | pending | — |
| [str_length](../../src/tidy3/stringr.py#L352) | pending | — |
| [str_like](../../src/tidy3/stringr.py#L277) | pending | — |
| [str_locate](../../src/tidy3/stringr.py#L330) | pending | — |
| [str_locate_all](../../src/tidy3/stringr.py#L335) | pending | — |
| [str_match](../../src/tidy3/stringr.py#L548) | pending | — |
| [str_match_all](../../src/tidy3/stringr.py#L558) | pending | — |
| [str_order](../../src/tidy3/stringr.py#L711) | pending | — |
| [str_pad](../../src/tidy3/stringr.py#L496) | pending | — |
| [str_remove](../../src/tidy3/stringr.py#L445) | pending | — |
| [str_remove_all](../../src/tidy3/stringr.py#L450) | pending | — |
| [str_replace](../../src/tidy3/stringr.py#L417) | pending | — |
| [str_replace_all](../../src/tidy3/stringr.py#L422) | pending | — |
| [str_replace_na](../../src/tidy3/stringr.py#L455) | pending | — |
| [str_sort](../../src/tidy3/stringr.py#L722) | pending | — |
| [str_split](../../src/tidy3/stringr.py#L564) | pending | — |
| [str_split_1](../../src/tidy3/stringr.py#L573) | pending | — |
| [str_split_fixed](../../src/tidy3/stringr.py#L580) | pending | — |
| [str_split_i](../../src/tidy3/stringr.py#L588) | pending | — |
| [str_squish](../../src/tidy3/stringr.py#L491) | pending | — |
| [str_starts](../../src/tidy3/stringr.py#L248) | pending | — |
| [str_sub](../../src/tidy3/stringr.py#L362) | pending | — |
| [str_sub_all](../../src/tidy3/stringr.py#L369) | pending | — |
| [str_subset](../../src/tidy3/stringr.py#L313) | pending | — |
| [str_to_lower](../../src/tidy3/stringr.py#L470) | pending | — |
| [str_to_sentence](../../src/tidy3/stringr.py#L480) | pending | — |
| [str_to_title](../../src/tidy3/stringr.py#L475) | pending | — |
| [str_to_upper](../../src/tidy3/stringr.py#L465) | pending | — |
| [str_trim](../../src/tidy3/stringr.py#L485) | pending | — |
| [str_trunc](../../src/tidy3/stringr.py#L387) | pending | — |
| [str_unique](../../src/tidy3/stringr.py#L733) | pending | — |
| [str_view](../../src/tidy3/stringr.py#L752) | pending | — |
| [str_which](../../src/tidy3/stringr.py#L320) | pending | — |
| [str_width](../../src/tidy3/stringr.py#L357) | pending | — |
| [str_wrap](../../src/tidy3/stringr.py#L513) | pending | — |
| [word](../../src/tidy3/stringr.py#L398) | pending | — |

### tidyr

34 exported functions; 1 validated, 0 legacy native comparisons, 33 pending.

| Function | Status | Existing cases |
|---|---|---|
| [build_longer_spec](../../src/tidy3/tidyr.py#L1282) | pending | — |
| [build_wider_spec](../../src/tidy3/tidyr.py#L1423) | pending | — |
| [check_pivot_spec](../../src/tidy3/tidyr.py#L1275) | pending | — |
| [chop](../../src/tidy3/tidyr.py#L2935) | pending | — |
| [complete](../../src/tidy3/tidyr.py#L626) | pending | — |
| [crossing](../../src/tidy3/tidyr.py#L540) | pending | — |
| [drop_na](../../src/tidy3/tidyr.py#L200) | pending | — |
| [expand](../../src/tidy3/tidyr.py#L369) | pending | — |
| [expand_grid](../../src/tidy3/tidyr.py#L527) | pending | — |
| [fill](../../src/tidy3/tidyr.py#L253) | pending | — |
| [full_seq](../../src/tidy3/tidyr.py#L39) | pending | — |
| [hoist](../../src/tidy3/tidyr.py#L3035) | pending | — |
| [nest](../../src/tidy3/tidyr.py#L2056) | validated | grouped/nest_by |
| [nesting](../../src/tidy3/tidyr.py#L23) | pending | — |
| [pack](../../src/tidy3/tidyr.py#L3177) | pending | — |
| [pivot_longer](../../src/tidy3/tidyr.py#L944) | pending | — |
| [pivot_longer_spec](../../src/tidy3/tidyr.py#L1346) | pending | — |
| [pivot_wider](../../src/tidy3/tidyr.py#L1019) | pending | — |
| [pivot_wider_spec](../../src/tidy3/tidyr.py#L1467) | pending | — |
| [replace_na](../../src/tidy3/tidyr.py#L225) | pending | — |
| [separate](../../src/tidy3/tidyr.py#L1794) | pending | — |
| [separate_longer_delim](../../src/tidy3/tidyr.py#L2480) | pending | — |
| [separate_longer_position](../../src/tidy3/tidyr.py#L2906) | pending | — |
| [separate_wider_delim](../../src/tidy3/tidyr.py#L2502) | pending | — |
| [separate_wider_position](../../src/tidy3/tidyr.py#L2831) | pending | — |
| [separate_wider_regex](../../src/tidy3/tidyr.py#L2750) | pending | — |
| [unchop](../../src/tidy3/tidyr.py#L2967) | pending | — |
| [uncount](../../src/tidy3/tidyr.py#L549) | pending | — |
| [unite](../../src/tidy3/tidyr.py#L1985) | pending | — |
| [unnest](../../src/tidy3/tidyr.py#L2246) | pending | — |
| [unnest_auto](../../src/tidy3/tidyr.py#L3020) | pending | — |
| [unnest_longer](../../src/tidy3/tidyr.py#L2136) | pending | — |
| [unnest_wider](../../src/tidy3/tidyr.py#L2381) | pending | — |
| [unpack](../../src/tidy3/tidyr.py#L3191) | pending | — |

### tidyselect

34 exported functions; 0 validated, 0 legacy native comparisons, 34 pending.

| Function | Status | Existing cases |
|---|---|---|
| [across](../../src/tidy3/tidyselect.py#L676) | pending | — |
| [all_of](../../src/tidy3/tidyselect.py#L260) | pending | — |
| [any_of](../../src/tidy3/tidyselect.py#L274) | pending | — |
| [c_across](../../src/tidy3/tidyselect.py#L803) | pending | — |
| [col_range](../../src/tidy3/tidyselect.py#L232) | pending | — |
| [cols_between](../../src/tidy3/tidyselect.py#L253) | pending | — |
| [contains](../../src/tidy3/tidyselect.py#L197) | pending | — |
| [cur_column](../../src/tidy3/tidyselect.py#L692) | pending | — |
| [cur_group](../../src/tidy3/tidyselect.py#L704) | pending | — |
| [cur_group_id](../../src/tidy3/tidyselect.py#L718) | pending | — |
| [cur_group_rows](../../src/tidy3/tidyselect.py#L774) | pending | — |
| [ends_with](../../src/tidy3/tidyselect.py#L181) | pending | — |
| [everything](../../src/tidy3/tidyselect.py#L152) | pending | — |
| [group_cols](../../src/tidy3/tidyselect.py#L308) | pending | — |
| [group_vars](../../src/tidy3/tidyselect.py#L729) | pending | — |
| [if_all](../../src/tidy3/tidyselect.py#L786) | pending | — |
| [if_any](../../src/tidy3/tidyselect.py#L782) | pending | — |
| [is_bool](../../src/tidy3/tidyselect.py#L412) | pending | — |
| [is_boolean](../../src/tidy3/tidyselect.py#L399) | pending | — |
| [is_categorical](../../src/tidy3/tidyselect.py#L454) | pending | — |
| [is_character](../../src/tidy3/tidyselect.py#L394) | pending | — |
| [is_datetime](../../src/tidy3/tidyselect.py#L417) | pending | — |
| [is_float](../../src/tidy3/tidyselect.py#L363) | pending | — |
| [is_integer](../../src/tidy3/tidyselect.py#L347) | pending | — |
| [is_numeric](../../src/tidy3/tidyselect.py#L328) | pending | — |
| [is_string](../../src/tidy3/tidyselect.py#L379) | pending | — |
| [is_temporal](../../src/tidy3/tidyselect.py#L433) | pending | — |
| [last_col](../../src/tidy3/tidyselect.py#L295) | pending | — |
| [matches](../../src/tidy3/tidyselect.py#L213) | pending | — |
| [n_groups](../../src/tidy3/tidyselect.py#L747) | pending | — |
| [num_range](../../src/tidy3/tidyselect.py#L224) | pending | — |
| [pick](../../src/tidy3/tidyselect.py#L799) | pending | — |
| [starts_with](../../src/tidy3/tidyselect.py#L163) | pending | — |
| [where](../../src/tidy3/tidyselect.py#L284) | pending | — |

### verbs

67 exported functions; 8 validated, 2 legacy native comparisons, 57 pending.

| Function | Status | Existing cases |
|---|---|---|
| [add_count](../../src/tidy3/verbs.py#L2983) | pending | — |
| [add_tally](../../src/tidy3/verbs.py#L2997) | pending | — |
| [anti_join](../../src/tidy3/verbs.py#L4164) | pending | — |
| [arrange](../../src/tidy3/verbs.py#L1753) | legacy-native | — |
| [bind_cols](../../src/tidy3/verbs.py#L4358) | pending | — |
| [bind_rows](../../src/tidy3/verbs.py#L4277) | pending | — |
| [clean_names](../../src/tidy3/verbs.py#L1560) | pending | — |
| [collect](../../src/tidy3/verbs.py#L4840) | pending | — |
| [compute](../../src/tidy3/verbs.py#L4868) | pending | — |
| [count](../../src/tidy3/verbs.py#L2894) | pending | — |
| [cross_join](../../src/tidy3/verbs.py#L4180) | pending | — |
| [distinct](../../src/tidy3/verbs.py#L1800) | pending | — |
| [drop](../../src/tidy3/verbs.py#L1280) | pending | — |
| [filter](../../src/tidy3/verbs.py#L832) | validated | grouped/filter |
| [filter_out](../../src/tidy3/verbs.py#L893) | pending | — |
| [full_join](../../src/tidy3/verbs.py#L3970) | pending | — |
| [get_dupes](../../src/tidy3/verbs.py#L1853) | pending | — |
| [glimpse](../../src/tidy3/verbs.py#L1711) | pending | — |
| [group_by](../../src/tidy3/verbs.py#L2222) | pending | — |
| [group_map](../../src/tidy3/verbs.py#L2395) | pending | — |
| [group_modify](../../src/tidy3/verbs.py#L2428) | pending | — |
| [group_nest](../../src/tidy3/verbs.py#L2461) | validated | grouped/group_nest |
| [group_split](../../src/tidy3/verbs.py#L2386) | pending | — |
| [head](../../src/tidy3/verbs.py#L3008) | pending | — |
| [inner_join](../../src/tidy3/verbs.py#L3938) | pending | — |
| [intersect](../../src/tidy3/verbs.py#L4524) | pending | — |
| [left_join](../../src/tidy3/verbs.py#L3922) | validated | closest/wide-speed-guard, closest/five-scaling-cases |
| [make_clean_names](../../src/tidy3/verbs.py#L1446) | pending | — |
| [mutate](../../src/tidy3/verbs.py#L947) | legacy-native | — |
| [nest_join](../../src/tidy3/verbs.py#L3987) | pending | — |
| [peek](../../src/tidy3/verbs.py#L4905) | pending | — |
| [pull](../../src/tidy3/verbs.py#L1685) | pending | — |
| [reframe](../../src/tidy3/verbs.py#L2749) | validated | grouped/reframe |
| [relocate](../../src/tidy3/verbs.py#L1359) | pending | — |
| [rename](../../src/tidy3/verbs.py#L1296) | pending | — |
| [rename_with](../../src/tidy3/verbs.py#L1397) | pending | — |
| [right_join](../../src/tidy3/verbs.py#L3954) | pending | — |
| [rows_append](../../src/tidy3/verbs.py#L4797) | pending | — |
| [rows_delete](../../src/tidy3/verbs.py#L4829) | pending | — |
| [rows_insert](../../src/tidy3/verbs.py#L4786) | pending | — |
| [rows_patch](../../src/tidy3/verbs.py#L4812) | pending | — |
| [rows_update](../../src/tidy3/verbs.py#L4801) | pending | — |
| [rows_upsert](../../src/tidy3/verbs.py#L4823) | pending | — |
| [rowwise](../../src/tidy3/verbs.py#L2208) | pending | — |
| [sample_frac](../../src/tidy3/verbs.py#L3037) | pending | — |
| [sample_n](../../src/tidy3/verbs.py#L3015) | pending | — |
| [select](../../src/tidy3/verbs.py#L1181) | pending | — |
| [semi_join](../../src/tidy3/verbs.py#L4148) | pending | — |
| [set_names](../../src/tidy3/verbs.py#L1607) | pending | — |
| [setdiff](../../src/tidy3/verbs.py#L4528) | pending | — |
| [setequal](../../src/tidy3/verbs.py#L4536) | pending | — |
| [slice](../../src/tidy3/verbs.py#L1917) | pending | — |
| [slice_head](../../src/tidy3/verbs.py#L2011) | validated | grouped/slice_head |
| [slice_max](../../src/tidy3/verbs.py#L2183) | validated | grouped/slice_max |
| [slice_min](../../src/tidy3/verbs.py#L2158) | pending | — |
| [slice_sample](../../src/tidy3/verbs.py#L3059) | validated | grouped/slice_sample |
| [slice_tail](../../src/tidy3/verbs.py#L2032) | pending | — |
| [summarise](../../src/tidy3/verbs.py#L2609) | validated | grouped/summarise_mean, grouped/summarise_nth |
| [summarize](../../src/tidy3/verbs.py#L2609) | pending | — |
| [symdiff](../../src/tidy3/verbs.py#L4532) | pending | — |
| [tally](../../src/tidy3/verbs.py#L2928) | pending | — |
| [to_numpy](../../src/tidy3/verbs.py#L4881) | pending | — |
| [transmute](../../src/tidy3/verbs.py#L1174) | pending | — |
| [ungroup](../../src/tidy3/verbs.py#L2307) | pending | — |
| [union](../../src/tidy3/verbs.py#L4516) | pending | — |
| [union_all](../../src/tidy3/verbs.py#L4520) | pending | — |
| [with_groups](../../src/tidy3/verbs.py#L2327) | pending | — |

## Public TidyFrame methods

All entries below need a dedicated method case or a validated mapping to the corresponding function workload.

| Method | Status |
|---|---|
| [add_count](../../src/tidy3/frame.py#L604) | pending method benchmark |
| [add_tally](../../src/tidy3/frame.py#L615) | pending method benchmark |
| [anti_join](../../src/tidy3/frame.py#L944) | pending method benchmark |
| [arrange](../../src/tidy3/frame.py#L466) | pending method benchmark |
| [bind_cols](../../src/tidy3/frame.py#L968) | pending method benchmark |
| [bind_rows](../../src/tidy3/frame.py#L963) | pending method benchmark |
| [clean_names](../../src/tidy3/frame.py#L439) | pending method benchmark |
| [collect](../../src/tidy3/frame.py#L1090) | pending method benchmark |
| [colnames](../../src/tidy3/frame.py#L278) | pending method benchmark |
| [compare_df_cols](../../src/tidy3/frame.py#L526) | pending method benchmark |
| [compare_df_cols_same](../../src/tidy3/frame.py#L531) | pending method benchmark |
| [complete](../../src/tidy3/frame.py#L736) | pending method benchmark |
| [compute](../../src/tidy3/frame.py#L1070) | pending method benchmark |
| [count](../../src/tidy3/frame.py#L579) | pending method benchmark |
| [cross_join](../../src/tidy3/frame.py#L958) | pending method benchmark |
| [describe](../../src/tidy3/frame.py#L270) | pending method benchmark |
| [distinct](../../src/tidy3/frame.py#L471) | pending method benchmark |
| [drop](../../src/tidy3/frame.py#L419) | pending method benchmark |
| [drop_na](../../src/tidy3/frame.py#L714) | pending method benchmark |
| [expand](../../src/tidy3/frame.py#L731) | pending method benchmark |
| [explain](../../src/tidy3/frame.py#L1424) | pending method benchmark |
| [fill](../../src/tidy3/frame.py#L724) | pending method benchmark |
| [filter](../../src/tidy3/frame.py#L379) | pending method benchmark |
| [filter_out](../../src/tidy3/frame.py#L384) | pending method benchmark |
| [find_header](../../src/tidy3/frame.py#L521) | pending method benchmark |
| [full_join](../../src/tidy3/frame.py#L902) | pending method benchmark |
| [get_dupes](../../src/tidy3/frame.py#L487) | pending method benchmark |
| [ggplot](../../src/tidy3/frame.py#L1440) | pending method benchmark |
| [glimpse](../../src/tidy3/frame.py#L461) | pending method benchmark |
| [group_by](../../src/tidy3/frame.py#L536) | pending method benchmark |
| [head](../../src/tidy3/frame.py#L626) | pending method benchmark |
| [inner_join](../../src/tidy3/frame.py#L888) | pending method benchmark |
| [intersect](../../src/tidy3/frame.py#L983) | pending method benchmark |
| [lazy](../../src/tidy3/frame.py#L1421) | pending method benchmark |
| [left_join](../../src/tidy3/frame.py#L881) | pending method benchmark |
| [mutate](../../src/tidy3/frame.py#L389) | pending method benchmark |
| [names_list](../../src/tidy3/frame.py#L284) | pending method benchmark |
| [nest](../../src/tidy3/frame.py#L833) | pending method benchmark |
| [nest_join](../../src/tidy3/frame.py#L909) | pending method benchmark |
| [pivot_longer](../../src/tidy3/frame.py#L746) | pending method benchmark |
| [pivot_wider](../../src/tidy3/frame.py#L771) | pending method benchmark |
| [preview](../../src/tidy3/frame.py#L1429) | pending method benchmark |
| [problems](../../src/tidy3/frame.py#L1364) | pending method benchmark |
| [pull](../../src/tidy3/frame.py#L456) | pending method benchmark |
| [reframe](../../src/tidy3/frame.py#L574) | pending method benchmark |
| [relocate](../../src/tidy3/frame.py#L446) | pending method benchmark |
| [remove_constant](../../src/tidy3/frame.py#L497) | pending method benchmark |
| [remove_empty](../../src/tidy3/frame.py#L492) | pending method benchmark |
| [rename](../../src/tidy3/frame.py#L424) | pending method benchmark |
| [rename_with](../../src/tidy3/frame.py#L429) | pending method benchmark |
| [replace_na](../../src/tidy3/frame.py#L719) | pending method benchmark |
| [right_join](../../src/tidy3/frame.py#L895) | pending method benchmark |
| [row_to_names](../../src/tidy3/frame.py#L502) | pending method benchmark |
| [rows_append](../../src/tidy3/frame.py#L1014) | pending method benchmark |
| [rows_delete](../../src/tidy3/frame.py#L1048) | pending method benchmark |
| [rows_insert](../../src/tidy3/frame.py#L1003) | pending method benchmark |
| [rows_patch](../../src/tidy3/frame.py#L1030) | pending method benchmark |
| [rows_update](../../src/tidy3/frame.py#L1019) | pending method benchmark |
| [rows_upsert](../../src/tidy3/frame.py#L1041) | pending method benchmark |
| [rowwise](../../src/tidy3/frame.py#L549) | pending method benchmark |
| [sample_frac](../../src/tidy3/frame.py#L708) | pending method benchmark |
| [sample_n](../../src/tidy3/frame.py#L703) | pending method benchmark |
| [select](../../src/tidy3/frame.py#L414) | pending method benchmark |
| [semi_join](../../src/tidy3/frame.py#L930) | pending method benchmark |
| [separate](../../src/tidy3/frame.py#L796) | pending method benchmark |
| [set_names](../../src/tidy3/frame.py#L434) | pending method benchmark |
| [setdiff](../../src/tidy3/frame.py#L988) | pending method benchmark |
| [setequal](../../src/tidy3/frame.py#L998) | pending method benchmark |
| [show](../../src/tidy3/frame.py#L1436) | pending method benchmark |
| [slice](../../src/tidy3/frame.py#L631) | pending method benchmark |
| [slice_head](../../src/tidy3/frame.py#L636) | pending method benchmark |
| [slice_max](../../src/tidy3/frame.py#L666) | pending method benchmark |
| [slice_min](../../src/tidy3/frame.py#L650) | pending method benchmark |
| [slice_sample](../../src/tidy3/frame.py#L682) | pending method benchmark |
| [slice_tail](../../src/tidy3/frame.py#L643) | pending method benchmark |
| [stop_for_problems](../../src/tidy3/frame.py#L1370) | pending method benchmark |
| [summarise](../../src/tidy3/frame.py#L559) | pending method benchmark |
| [summarize](../../src/tidy3/frame.py#L559) | pending method benchmark |
| [summary](../../src/tidy3/frame.py#L260) | pending method benchmark |
| [symdiff](../../src/tidy3/frame.py#L993) | pending method benchmark |
| [tally](../../src/tidy3/frame.py#L593) | pending method benchmark |
| [to_arrow](../../src/tidy3/frame.py#L1189) | pending method benchmark |
| [to_numpy](../../src/tidy3/frame.py#L1192) | pending method benchmark |
| [to_pandas](../../src/tidy3/frame.py#L1175) | pending method benchmark |
| [to_polars](../../src/tidy3/frame.py#L1170) | pending method benchmark |
| [transmute](../../src/tidy3/frame.py#L409) | pending method benchmark |
| [ungroup](../../src/tidy3/frame.py#L554) | pending method benchmark |
| [union](../../src/tidy3/frame.py#L973) | pending method benchmark |
| [union_all](../../src/tidy3/frame.py#L978) | pending method benchmark |
| [unite](../../src/tidy3/frame.py#L819) | pending method benchmark |
| [unnest](../../src/tidy3/frame.py#L861) | pending method benchmark |
| [unnest_longer](../../src/tidy3/frame.py#L844) | pending method benchmark |
| [unnest_wider](../../src/tidy3/frame.py#L874) | pending method benchmark |
| [with_polars](../../src/tidy3/frame.py#L1059) | pending method benchmark |
| [write_csv](../../src/tidy3/frame.py#L1280) | pending method benchmark |
| [write_excel](../../src/tidy3/frame.py#L1376) | pending method benchmark |
| [write_ipc](../../src/tidy3/frame.py#L1340) | pending method benchmark |
| [write_parquet](../../src/tidy3/frame.py#L1316) | pending method benchmark |

## Other public exports

| Name | Kind |
|---|---|
| `TidyFrame` | type |
| `ColumnNotFoundError` | type |
| `JoinCondition` | type |
| `JoinSpec` | type |
| `Selector` | type |
| `env` | _Env |
| `ExportResult` | type |
| `ConversionError` | type |
| `__version__` | str |

Classes, exception types, the environment proxy, and the version string are tracked for scope but are not counted as exported functions. Constructor or context costs belong in the consuming operation where meaningful.
