suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(jsonlite))

column_type <- function(x) {
  if (is.factor(x)) return("factor")
  if (inherits(x, "Date")) return("date")
  if (inherits(x, "POSIXt")) return("datetime")
  if (is.list(x)) return("list")
  if (is.logical(x)) return("logical")
  if (is.integer(x)) return("integer")
  if (is.numeric(x)) return("double")
  if (is.character(x)) return("character")
  class(x)[[1]]
}

normalise_value <- function(x) {
  if (is.data.frame(x)) return(normalise(x))
  if (inherits(x, "Date")) return(format(x, "%Y-%m-%d"))
  if (inherits(x, "POSIXt")) return(format(x, "%Y-%m-%dT%H:%M:%S", tz="UTC"))
  if (is.factor(x)) return(as.character(x))
  if (is.list(x)) return(lapply(x, normalise_value))
  x
}

normalise <- function(x) {
  if (is.data.frame(x)) {
    return(list(
      kind="dataframe",
      columns=unname(names(x)),
      types=unname(vapply(x, column_type, character(1))),
      data=unname(lapply(x, function(column) {
        cells <- lapply(as.list(column), normalise_value)
        # Keep list-column cells as arrays: auto_unbox would turn list(3)
        # into a bare 3.
        if (is.list(column)) {
          cells <- lapply(cells, function(v) if (is.atomic(v) && !is.null(v)) I(v) else v)
        }
        unname(cells)
      }))
    ))
  }
  if (is.list(x)) return(unname_or_names(x))
  if (length(x) == 1) return(normalise_value(x))
  list(kind="vector", data=unname(lapply(as.list(x), normalise_value)))
}

unname_or_names <- function(x) {
  values <- lapply(x, normalise)
  if (is.null(names(x))) unname(values) else values
}

case_name <- commandArgs(trailingOnly=TRUE)[[1]]

base <- tibble(
  id=1:4,
  g=c("b", "a", "b", "a"),
  x=c(2, NA_real_, 4, 1),
  y=c(20, 10, 40, 30)
)

result <- switch(
  case_name,
  janitor_totals = {
    if (!requireNamespace("janitor", quietly=TRUE)) {
      list(skip=TRUE, reason="janitor is not installed")
    } else {
      source <- data.frame(label=c("a", "b"), yes=c(1L, NA_integer_), no=c(2L, 3L), note=c("x", "y"))
      result <- list(
        row=janitor::adorn_totals(source),
        col=janitor::adorn_totals(source, "col"),
        both=janitor::adorn_totals(source, c("row", "col")),
        propagate=janitor::adorn_totals(source, c("row", "col"), na.rm=FALSE),
        custom=janitor::adorn_totals(source, where=c("row", "col"), fill="-", na.rm=TRUE, name=c("All", "Grand"), yes),
        typed_fill=janitor::adorn_totals(source, where="row", fill=NA, na.rm=TRUE, name="Total", yes),
        row_then_col=janitor::adorn_totals(janitor::adorn_totals(source), "col"),
        col_then_row=janitor::adorn_totals(janitor::adorn_totals(source, "col"))
      )
      factor <- janitor::adorn_totals(data.frame(g=ordered(c("b", "a"), levels=c("b", "a", "unused")), n=c(1L, 2L)))
      result$factor <- factor
      result$factor_levels <- levels(factor$g)
      result$factor_ordered <- is.ordered(factor$g)
      numeric <- data.frame(g=c(1L, 2L), n=c(3L, 4L))
      result$numeric_label <- janitor::adorn_totals(numeric)
      result$numeric_first_selected <- janitor::adorn_totals(numeric, where=c("row", "col"), fill="-", na.rm=TRUE, name="Total", dplyr::everything())
      result$empty <- janitor::adorn_totals(data.frame(label=character(), n=integer()), c("row", "col"))
      result$missing <- janitor::adorn_totals(data.frame(label=c("a", "b"), n=c(NA_real_, NaN)), c("row", "col"))
      report <- janitor::tabyl(data.frame(g=c("a", "a", "b"), answer=c("yes", "no", "yes")), g, answer)
      result$tabyl <- janitor::adorn_totals(report, c("row", "col"))
      result
    }
  },
  janitor_percentages = {
    if (!requireNamespace("janitor", quietly=TRUE)) {
      list(skip=TRUE, reason="janitor is not installed")
    } else {
      source <- data.frame(label=c("a", "b", "c"), yes=c(1L, NA, 0L), no=c(2L, 3L, 0L), w=c(0.5, NaN, 1), note=c("x", "y", "z"))
      strict <- data.frame(label=c("a", "b", "c"), yes=c(1L, NA, 0L), no=c(2L, 3L, 0L))
      totals <- janitor::adorn_totals(strict, c("row", "col"))
      result <- list()
      for (denominator in c("row", "col", "all")) {
        result[[denominator]] <- janitor::adorn_percentages(source, denominator)
        result[[paste0(denominator, "_keep_na")]] <- janitor::adorn_percentages(strict, denominator, na.rm=FALSE)
        result[[paste0(denominator, "_totals")]] <- janitor::adorn_percentages(totals, denominator)
        result[[paste0(denominator, "_row_totals")]] <- janitor::adorn_percentages(janitor::adorn_totals(strict), denominator)
        result[[paste0(denominator, "_col_totals")]] <- janitor::adorn_percentages(janitor::adorn_totals(strict, "col"), denominator)
        result[[paste0(denominator, "_selected")]] <- janitor::adorn_percentages(totals, denominator, TRUE, yes)
        result[[paste0(denominator, "_selected_totals")]] <- janitor::adorn_percentages(totals, denominator, TRUE, c(no, Total))
      }
      numeric <- data.frame(g=c(1L, 2L), n=c(3L, 1L))
      result$first_column <- janitor::adorn_percentages(numeric, "col", TRUE, dplyr::everything())
      result$text_selected <- suppressMessages(janitor::adorn_percentages(source, "row", TRUE, c(yes, note)))
      result$empty <- janitor::adorn_percentages(data.frame(label=character(), n=integer()), "col")
      report <- janitor::tabyl(data.frame(g=c("a", "a", "b", "b"), answer=c("yes", "no", "yes", "yes")), g, answer)
      result$tabyl <- janitor::adorn_percentages(janitor::adorn_totals(report, c("row", "col")), "all")
      factor <- data.frame(g=ordered(c("b", "a"), levels=c("b", "a", "unused")), n=c(1L, 3L))
      result$factor <- janitor::adorn_percentages(janitor::adorn_totals(factor), "col")
      # JSON keeps about 16 digits, so recurring shares are compared at 12.
      lapply(result, function(df) { df[] <- lapply(df, function(x) if (is.double(x)) signif(x, 12) else x); df })
    }
  },
  janitor_tabyl = {
    if (!requireNamespace("janitor", quietly=TRUE)) {
      list(skip=TRUE, reason="janitor is not installed")
    } else {
      source <- data.frame(x=c("b", "a", "a", NA_character_))
      f <- data.frame(x=ordered(c("b", NA, "b"), levels=c("c", "b", "a")))
      result <- list(
        ordinary=janitor::tabyl(source, x),
        without_na=janitor::tabyl(source, x, show_na=FALSE)
      )
      for (na in c(FALSE, TRUE)) for (missing in c(FALSE, TRUE)) {
        result[[paste0("factor_", as.integer(na), "_", as.integer(missing))]] <-
          janitor::tabyl(f, x, show_na=na, show_missing_levels=missing)
      }
      result$empty <- janitor::tabyl(data.frame(x=character()), x)
      result$empty_factor <- janitor::tabyl(data.frame(x=factor(character(), levels=c("b", "a"))), x)
      result$all_missing <- janitor::tabyl(data.frame(x=rep(NA_character_, 2)), x)
      result$nan_null <- janitor::tabyl(data.frame(x=c(NA_real_, NaN, 1, NaN)), x)
      two_way <- data.frame(x=c("b", "a", "a", NA, "b"), y=c("", "z", NA, "m", "z"))
      result$two_way <- janitor::tabyl(two_way, x, y)
      result$two_way_no_na <- janitor::tabyl(two_way, x, y, show_na=FALSE)
      factors <- data.frame(x=ordered(c("b", NA, "b"), levels=c("c", "b", "a")),
                            y=factor(c("z", "m", NA), levels=c("q", "z", "m")))
      for (na in c(FALSE, TRUE)) for (missing in c(FALSE, TRUE)) {
        result[[paste0("two_way_factor_", as.integer(na), "_", as.integer(missing))]] <-
          janitor::tabyl(factors, x, y, show_na=na, show_missing_levels=missing)
      }
      result$two_way_empty <- janitor::tabyl(data.frame(x=factor(character(), levels=c("b", "a")),
                                                       y=factor(character(), levels=c("z", "m"))), x, y)
      result$two_way_empty_after_drop <- janitor::tabyl(data.frame(x=c("a", NA), y=c(NA, "b")), x, y, show_na=FALSE)
      result$two_way_numeric <- janitor::tabyl(data.frame(x=c("b", "a", "b"), y=c(10, 2, 1)), x, y)
      result$two_way_boolean <- janitor::tabyl(data.frame(x=c("b", "a", "b"), y=c(TRUE, FALSE, TRUE)), x, y)
      result$two_way_nan <- janitor::tabyl(data.frame(x=c("b", "a", "b"), y=c(1, NaN, NA_real_)), x, y)
      result
    }
  },
  rolling_joins = {
    x <- tibble(g=c("b", "a", NA, "a", "z"), x=c(4, 2, 2, 0, 9), lid=as.double(1:5))
    y <- tibble(h=c("a", "a", NA, "b", "q"), y=c(1, 1, 1, 4, 8), rid=as.double(1:5))
    specs <- list(
      ge=join_by(g == h, closest(x >= y)),
      gt=join_by(g == h, closest(x > y)),
      le=join_by(g == h, closest(x <= y)),
      lt=join_by(g == h, closest(x < y))
    )
    result <- list()
    for (op in names(specs)) {
      for (how in c("left", "inner", "right", "full", "semi", "anti")) {
        join <- get(paste0(how, "_join"), asNamespace("dplyr"))
        result[[paste(op, how, sep="_")]] <- suppressWarnings(join(x, y, by=specs[[op]]))
      }
    }
    result$never <- left_join(x, y, by=specs$ge, na_matches="never")
    result$keep <- left_join(x, y, by=specs$ge, keep=TRUE)
    result
  },
  reshape_metadata = {
    x <- tibble(g=factor(c("b", "a", "b"), levels=c("a", "b", "unused"), ordered=TRUE),
                code=c("x-y", "p-q", "r-s"), x=c(2L, 4L, 8L)) %>% group_by(g, .drop=FALSE)
    operations <- list(
      pack=function(d) pack(d, packed=x), unpack=function(d) unpack(d, record),
      hoist=function(d) hoist(d, record, nested_value="nested_value"),
      wider_delim=function(d) separate_wider_delim(d, code, names=c("left", "right"), delim="-"),
      wider_regex=function(d) separate_wider_regex(d, code, c(left="[a-z]", "-", right="[a-z]")),
      wider_position=function(d) separate_wider_position(d, code, widths=c(left=1L, 1L, right=1L)),
      longer_delim=function(d) separate_longer_delim(d, code, delim="-"),
      longer_position=function(d) separate_longer_position(d, code, width=2)
    )
    result <- list()
    for (name in names(operations)) for (empty in c(FALSE, TRUE)) {
      data <- x
      data$record <- if (name == "unpack") tibble(nested_value=c(10L,20L,30L)) else
        list(list(nested_value=10L), list(nested_value=20L), list(nested_value=30L))
      if (empty) data <- head(data, 0)
      out <- operations[[name]](data)
      result[[paste0(name, if (empty) "_empty" else "_rows")]] <- list(
        groups=group_vars(out), drop=group_by_drop_default(out), levels=levels(out$g),
        # Compare summary labels/counts; factor levels are checked above.
        summary=summarise(out, rows=n(), .groups="drop") %>% mutate(g=as.character(g))
      )
    }
    result
  },
  janitor_dupes = {
    if (!requireNamespace("janitor", quietly=TRUE)) {
      list(skip=TRUE, reason="janitor is not installed")
    } else {
      x <- tibble(g=c("x", "x", "y", "z", "y", "x", "a", "b"),
                  key_id=c("b", "a", "b", "c", "a", "a", NA, NA),
                  key_sub=c(1L, 2L, 1L, 3L, 2L, 9L, 4L, 4L), row=1:8)
      full <- tibble(a=c("b", "a", "b", "a", "x"), b=c(1L, 2L, 1L, 2L, 3L))
      grouped <- x %>% group_by(g) %>% janitor::get_dupes(key_id)
      factors <- tibble(key=factor(c("a", "b", "a", "b", NA, NA), levels=c("b", "a", "unused")), row=0:5)
      list(
        single=janitor::get_dupes(x, key_id),
        composite=janitor::get_dupes(x, key_id, key_sub),
        selector=janitor::get_dupes(x, starts_with("key")),
        all=suppressMessages(janitor::get_dupes(full)),
        empty=suppressMessages(janitor::get_dupes(x[FALSE, ], key_id)),
        none=suppressMessages(janitor::get_dupes(x, row)),
        empty_selection=janitor::get_dupes(x, any_of(character())),
        grouped=grouped, group_names=group_vars(grouped),
        factors=janitor::get_dupes(factors, key)
      )
    }
  },
  timezone_conversion = {
    if (!requireNamespace("lubridate", quietly=TRUE)) {
      list(skip=TRUE, reason="lubridate is not installed")
    } else {
      result <- list()
      x <- lubridate::ymd_hms(c("2024-01-02 12:34:56.125", "2024-07-02 12:34:56.5", "2024-11-03 05:30:00", "2024-11-03 06:30:00", NA_character_))
      zones <- c("UTC", "America/New_York", "Europe/Dublin", "Australia/Sydney", "Asia/Kathmandu")
      for (source in zones) for (target in zones) for (name in c("with_tz", "force_tz")) {
        f <- getExportedValue("lubridate", name)
        output <- f(lubridate::with_tz(x, source), target)
        result[[paste(source, target, name, sep="_")]] <- list(
          epochs=as.numeric(output), clocks=format(output, "%Y-%m-%d %H:%M:%S %z", tz=target)
        )
      }
      dates <- as.Date(c("2024-01-02", "2024-03-10", "2024-11-03", NA_character_))
      for (target in zones) for (name in c("with_tz", "force_tz")) {
        output <- getExportedValue("lubridate", name)(dates, target)
        result[[paste("dates", target, name, sep="_")]] <- list(
          epochs=as.numeric(output), clocks=format(output, "%Y-%m-%d %H:%M:%S %z", tz=target)
        )
      }
      result
    }
  },
  timezone_dst = {
    if (!requireNamespace("lubridate", quietly=TRUE)) {
      list(skip=TRUE, reason="lubridate is not installed")
    } else {
      inputs <- list(
        "America/New_York"=c("2024-03-10 01:59:59.125", "2024-03-10 02:30:00.125", "2024-03-10 03:00:00", "2024-11-03 01:30:00.125", "2024-11-03 02:00:00", NA_character_),
        "Europe/Dublin"=c("2024-03-31 01:30:00.125", "2024-10-27 01:30:00.125", "2024-01-01 12:34:56", NA_character_),
        "Australia/Lord_Howe"=c("2024-04-07 01:45:00.125", "2024-10-06 02:15:00.125", "2024-01-01 12:34:56", NA_character_),
        "Pacific/Apia"=c("2011-12-29 12:30:00.125", "2011-12-30 12:30:00.125", "2011-12-31 12:30:00.125", NA_character_)
      )
      policies <- list(NA_both="NA", pre="pre", post="post", boundary="boundary", default=c("NA", "post"), mixed=c("boundary", "pre"))
      result <- list()
      for (target in names(inputs)) for (policy in names(policies)) {
        x <- lubridate::ymd_hms(inputs[[target]])
        # R retains subseconds for a skipped "boundary" when the repeated
        # policy is "pre". Compare whole-second inputs for that pair;
        # tidy3's exact-boundary rule is tested separately.
        if (policy == "mixed") x <- lubridate::floor_date(x, "second")
        output <- lubridate::force_tz(x, target, roll_dst=policies[[policy]])
        result[[paste(target, policy, sep="_")]] <- list(
          epochs=as.numeric(output), clocks=format(output, "%Y-%m-%d %H:%M:%S %z", tz=target)
        )
      }
      result
    }
  },
  timezone_resolved_repeats = {
    if (!requireNamespace("lubridate", quietly=TRUE)) {
      list(skip=TRUE, reason="lubridate is not installed")
    } else {
      x <- lubridate::ymd_hms(c("2024-11-03 04:30:00.125", "2024-11-03 05:30:00.125", "2024-11-03 06:30:00.125", "2024-10-27 00:30:00.125", "2024-10-27 01:30:00.125", NA_character_))
      result <- list()
      for (source in c("UTC", "America/Toronto", "America/Halifax", "Europe/Dublin")) {
        for (target in c("UTC", "America/New_York", "America/Toronto", "Europe/Dublin")) {
          for (policy in c("NA", "pre", "post", "boundary")) {
            # R discards a resolved repeat's subseconds with "boundary".
            # Compare whole seconds here; tidy3 preserves that occurrence
            # and its original precision, covered by regression tests.
            values <- if (policy == "boundary") lubridate::floor_date(x, "second") else x
            output <- lubridate::force_tz(lubridate::with_tz(values, source), target, roll_dst=policy)
            result[[paste(source, target, policy, sep="_")]] <- list(
              epochs=as.numeric(output), clocks=format(output, "%Y-%m-%d %H:%M:%S %z", tz=target)
            )
          }
        }
      }
      result
    }
  },
  formatted_offsets = {
    if (!requireNamespace("readr", quietly=TRUE)) {
      list(skip=TRUE, reason="readr is not installed")
    } else {
      suppressPackageStartupMessages(library(readr))
      x <- parse_datetime(
        c("2024-01-01 12:00:00+0200", "2024-01-01 12:00:00-0500",
          "2024-01-01 12:00:00+02:00", "2024-01-01 12:00:00Z",
          "2024-06-01 00:30:00-0530", "bad", NA_character_),
        format="%Y-%m-%d %H:%M:%S%z"
      )
      list(clocks=format(x, "%Y-%m-%d %H:%M:%S", tz="UTC"))
    }
  },
  calendar_rounding = {
    if (!requireNamespace("lubridate", quietly=TRUE)) {
      list(skip=TRUE, reason="lubridate is not installed")
    } else {
      suppressPackageStartupMessages(library(lubridate))
      zone <- "America/New_York"
      x <- ymd_hms(c(
        "2024-03-11 12:00:00", "2024-03-10 01:30:00", "2024-11-03 01:30:00.125",
        "2024-11-03T01:30:00.125-04:00", "2024-11-03T01:00:00-04:00", NA_character_
      ), tz=zone)
      # Hour ceiling of the first fallback 01:30 is 02:00 in lubridate and
      # the second 01:00 in tidy3. That unit is covered by the regression
      # tests, not this comparison.
      units <- c("second", "minute", "day", "week", "month", "quarter", "year")
      result <- list()
      for (unit in units) for (name in c("floor_date", "ceiling_date")) {
        output <- getExportedValue("lubridate", name)(x, unit)
        result[[paste(unit, name, sep="_")]] <- list(
          epochs=as.numeric(output),
          clocks=format(output, "%Y-%m-%d %H:%M:%S %z", tz=zone)
        )
      }
      monday <- floor_date(x, "week", week_start=1)
      result[["week_monday"]] <- list(
        epochs=as.numeric(monday),
        clocks=format(monday, "%Y-%m-%d %H:%M:%S %z", tz=zone)
      )
      result
    }
  },
  date_helpers = {
    if (!requireNamespace("lubridate", quietly=TRUE)) {
      list(skip=TRUE, reason="lubridate is not installed")
    } else {
      suppressPackageStartupMessages(library(lubridate))
      zone <- "America/New_York"
      instants <- function(x) list(epochs=as.numeric(x), clocks=format(x, "%Y-%m-%d %H:%M:%S %z", tz=tz(x)))
      d <- as.Date(c("2010-02-01", "2010-02-14", "2010-02-15", "2010-02-16", "2024-02-29",
                     "2023-12-31", "2024-01-03", "2024-08-16", NA))
      # Ties (a Wednesday noon, the middle of February 2024, half a second,
      # half of a 23- or 25-hour day) round up. The first fallback 01:30 is
      # left out of the hour unit: see calendar_rounding.
      x <- ymd_hms(c("2024-01-03 12:00:00", "2024-02-15 12:00:00", "2024-07-02 00:00:00.5",
                     "2024-07-02 23:59:59.4", "2024-03-10 12:00:00", "2024-03-10 12:30:00",
                     "2024-11-03 12:00:00", "2024-11-03 11:30:00", "2024-03-10 01:30:00",
                     "2024-11-03T01:30:00-05:00", "2024-08-16 06:00:00", NA_character_), tz=zone)
      result <- list()
      for (unit in c("day", "week", "month", "quarter", "year")) {
        result[[paste("date", unit, sep="_")]] <- as.numeric(round_date(d, unit))
      }
      result[["date_week_monday"]] <- as.numeric(round_date(d, "week", week_start=1))
      for (unit in c("second", "minute", "hour", "day", "week", "month", "quarter", "year")) {
        result[[paste("time", unit, sep="_")]] <- instants(round_date(x, unit))
      }
      result[["time_week_monday"]] <- instants(round_date(x, "week", week_start=1))
      year <- c(2023, 2023, 2023, 2023, 2023, 2023.7, -5, 0, NA, 2024, 10000, -2023.5, 2024)
      month <- c(2, 13, 0, 1, 1, 1.9, 1, 1, 1, 2, 1, 1, NA)
      day <- c(29, 1, 1, 0, 32, 1.5, 1, 1, 1, 29, 1, 1, 1)
      result[["make_date"]] <- as.numeric(make_date(year, month, day))
      result[["make_date_defaults"]] <- as.numeric(make_date(c(2020, 2021)))
      parts <- data.frame(
        year=c(2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, NA, 2024),
        month=c(2, 1, 1, 1, 13, 0, -1, 1, 1, 1, 1, 1, 2),
        day=c(31, 1, 1, 1, 1, 1, 1, -5, 1, 1, 1, 1, 29),
        hour=c(0, 25, 0, 23, 0, 0, 0, 0, 1000, 0, 0, 0, 12),
        min=c(0, 0, -61, 59, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        sec=c(0, 0, 0, 60, 0, 0, 0, 0, 0, -0.5, 1.25, 0, 30)
      )
      result[["make_datetime"]] <- with(parts, instants(make_datetime(year, month, day, hour, min, sec)))
      dst <- data.frame(day=c(10, 3, 10, 10, 9, 3, 3), hour=c(2, 1, 1, 26, 26, 0, 1), min=c(30, 30, 90, 0, 30, 90, 0))
      dst$month <- c(3, 11, 3, 3, 3, 11, 11)
      result[["make_datetime_new_york"]] <- with(dst, instants(make_datetime(2024, month, day, hour, min, tz=zone)))
      result[["make_datetime_kolkata"]] <- instants(make_datetime(2023, 1, c(1, 2), 0, 0, 1, tz="Asia/Kolkata"))
      span <- c(as.Date("2019-12-25") + 0:1849, as.Date(NA))
      result[["isoyear"]] <- isoyear(span)
      result[["isoyear_time"]] <- isoyear(x)
      result[["leap_number"]] <- leap_year(c(1900, 2000, 2023, 2024, NA, -4, 0, 2000.5, -1.5, 2100))
      result[["leap_integer"]] <- leap_year(c(1900L, 2000L, 2024L, NA))
      result[["leap_date"]] <- leap_year(span)
      result[["leap_time"]] <- leap_year(x)
      result
    }
  },
  time_spans = {
    if (!requireNamespace("lubridate", quietly=TRUE)) {
      list(skip=TRUE, reason="lubridate is not installed")
    } else {
      suppressPackageStartupMessages(library(lubridate))
      ny <- "America/New_York"
      lh <- "Australia/Lord_Howe"
      at <- function(x, tz) list(epochs=as.numeric(x), clocks=format(x, "%Y-%m-%d %H:%M:%S %z", tz=tz))
      day_text <- function(x) format(x, "%Y-%m-%d")
      utc_text <- function(x) format(x, "%Y-%m-%d %H:%M:%OS3", tz="UTC")
      # JSON keeps 15 digits; compare fractional lengths at 10 places.
      r10 <- function(x) round(as.numeric(x), 10)
      n <- c(1, 1.5, NA)
      leap <- ymd("2012-02-29")
      ends <- ymd(c("2010-01-31", "2012-02-29", "2010-03-31"))
      noon <- ymd_hms(c("2024-03-09 12:00:00", "2024-11-02 12:00:00"), tz=ny)
      later <- with_tz(ymd_hms("2024-11-03 06:30:00"), ny)  # 01:30 EST, the second 01:30
      list(
        constructors=lapply(list(dseconds=dseconds, dminutes=dminutes, dhours=dhours, ddays=ddays,
                                 dweeks=dweeks, dmonths=dmonths, dyears=dyears), function(f) r10(f(n))),
        duration_units=r10(c(duration(90, "minutes"), duration(1.5, "days"), duration(2, "weeks"), duration(1, "month"))),
        as_duration=r10(c(as.duration(c(90, NA)), as.duration(months(1)), as.duration(days(1) + hours(1)), as.duration(weeks(1)))),
        leap_dates=list(
          days365=day_text(leap + ddays(365)), year=utc_text(leap + dyears(1)),
          day_and_half=utc_text(leap + ddays(1.5)), minus=utc_text(leap - ddays(1.5)),
          classes=c(class(leap + ddays(365))[1], class(leap + dyears(1))[1], class(leap + ddays(1.5))[1])
        ),
        ddays_dst=at(c(noon, ymd_hms("2024-03-09 02:30:00", tz=ny)) + ddays(1), ny),
        summaries=r10(c(sum(ddays(1:3)), mean(ddays(c(1, 2))), median(ddays(c(1, 2, 4))))),
        time_length=lapply(setNames(nm=c("second", "minute", "hour", "day", "week", "month", "year")),
                           function(u) r10(time_length(ddays(c(1.5, NA)), u))),
        date_difference=r10(c(as.duration(ymd("2012-03-01") - ymd("2012-01-31")),
                              as.duration(ymd("2024-03-11", tz=ny) - ymd("2024-03-10", tz=ny)))),
        comparisons=c(ddays(1) == days(1), dhours(1) < minutes(61), days(1) == hours(24), months(1) == days(30)),
        compare_column=dseconds(c(3600, 90000)) > days(1),
        days_hours_dst=list(days=at(noon + days(1), ny), hours=at(noon + hours(24), ny)),
        skipped=at(c(ymd_hms("2024-03-09 02:30:00", tz=ny) + days(1),
                     ymd_hms("2024-03-10 01:30:00", tz=ny) + hours(1),
                     ymd_hms("2024-03-10 01:30:00", tz=ny) + minutes(30),
                     ymd_hms("2024-03-10 01:30:00", tz=ny) + minutes(90)), ny),
        autumn_hours=at(ymd_hms("2024-11-03 00:30:00", tz=ny) + hours(c(1, 2)), ny),
        repeat_from_before=at(ymd_hms("2024-11-02 01:30:00", tz=ny) + days(1), ny),
        # Deliberate difference: tidy3 keeps the input's offset when it is
        # still valid; lubridate takes the earlier repeat. The R side of the
        # tidy3 rule is force_tz with the later occurrence.
        repeat_from_after=list(
          tidy3=at(force_tz(ymd_hms("2024-11-03 01:30:00"), ny, roll_dst=c("NA", "post")), ny),
          lubridate=format(ymd_hms("2024-11-04 01:30:00", tz=ny) - days(1), "%H:%M %Z")
        ),
        repeat_identity=list(
          tidy3=at(c(later, later + hours(1), later - hours(1), later + dminutes(1)), ny),
          lubridate=format(c(later + seconds(0), later + minutes(1)), "%H:%M %Z")
        ),
        month_skipped=list(plus=at(ymd_hms("2024-02-10 02:30:00", tz=ny) + months(1), ny),
                           rollback=at(ymd_hms("2024-02-10 02:30:00", tz=ny) %m+% months(1), ny),
                           first=at(add_with_rollback(ymd_hms("2024-01-31 02:30:00", tz=ny), months(1), roll_to_first=TRUE), ny)),
        # Deliberate difference: tidy3's + rolls back, as %m+% does.
        month_ends=list(
          rollback=list(months=day_text(ends %m+% months(1)), years=day_text(ends %m+% years(1)),
                        mixed=day_text(ends %m+% (months(1) + days(1)))),
          lubridate_plus=list(months=day_text(ends + months(1)), years=day_text(ends + years(1)),
                              mixed=day_text(ends + (months(1) + days(1))))
        ),
        roll_to_first=list(
          months=day_text(add_with_rollback(ends, months(c(1, 12, -1)), roll_to_first=TRUE)),
          years=day_text(add_with_rollback(ends, years(1), roll_to_first=TRUE)),
          mixed=day_text(add_with_rollback(ends, months(1) + days(1), roll_to_first=TRUE))
        ),
        column_counts=list(
          days=day_text(ends + days(c(1L, NA, -1L))),
          months=day_text(ends %m+% months(c(1L, NA, -1L))),
          hours=utc_text(ends + hours(c(1L, NA, 30L)))
        ),
        date_classes=c(class(leap + days(1))[1], class(leap + hours(1))[1], class(leap + weeks(2))[1]),
        date_clock=list(hours=utc_text(leap + hours(1)), weeks=day_text(leap + weeks(2)),
                        seconds=utc_text(ymd_hms("2024-01-01 00:00:00") + seconds(1.5)),
                        period=at(noon + period(hours=25), ny), mixed_sign=utc_text(leap + (days(1) - hours(2)))),
        lord_howe=at(c(ymd_hms("2024-10-05 02:15:00", tz=lh) + days(1),
                       ymd_hms("2024-04-06 01:45:00", tz=lh) + days(1),
                       ymd_hms("2024-10-06 01:45:00", tz=lh) + minutes(30)), lh)
      )
    }
  },
  datetime_parsers = {
    if (!requireNamespace("lubridate", quietly=TRUE)) {
      list(skip=TRUE, reason="lubridate is not installed")
    } else {
      result <- list()
      calendars <- c(ymd_hms="2024-02-29", ymd_hm="2024-02-29", mdy_hms="02/29/2024", dmy_hms="29/02/2024")
      named <- c(ymd_hms="2024 February 29", ymd_hm="2024 February 29", mdy_hms="February 29th 2024", dmy_hms="29th February 2024")
      compact <- c(ymd_hms="20240229", ymd_hm="20240229", mdy_hms="02292024", dmy_hms="29022024")
      for (name in names(calendars)) {
        parser <- getExportedValue("lubridate", name)
        clock <- if (name == "ymd_hm") "12:34" else "12:34:56"
        digits <- if (name == "ymd_hm") "1234" else "123456"
        x <- c(paste(calendars[[name]], clock), paste(named[[name]], clock), paste0(compact[[name]], digits),
               paste0(calendars[[name]], "T", clock, "Z"), paste0(calendars[[name]], " ", clock, "+05:30"),
               paste(calendars[[name]], clock, "PM"), paste(calendars[[name]], clock, "AM"), "bad", NA_character_)
        if (name != "ymd_hm") x <- c(x, paste0(calendars[[name]], " ", clock, ".125"))
        for (zone in c("UTC", "America/New_York", "Asia/Kolkata")) {
          parsed <- suppressMessages(suppressWarnings(parser(x, tz=zone, locale="C")))
          if (name == "ymd_hm") {
            # R's hm parser drops Z and rejects some offsets. Compare the
            # intended instants to hms with zero seconds for these two rows.
            offset_text <- sub(clock, paste0(clock, ":00"), x[c(4, 5)], fixed=TRUE)
            parsed[c(4, 5)] <- suppressMessages(lubridate::ymd_hms(offset_text, tz=zone))
          }
          result[[paste(name, zone, sep="_")]] <- as.numeric(parsed)
        }
        result[[paste0(name, "_numeric")]] <- as.numeric(suppressWarnings(parser(as.numeric(paste0(compact[[name]], digits)))))
      }
      result
    }
  },
  date_time_orders = {
    if (!requireNamespace("lubridate", quietly=TRUE) || !requireNamespace("readr", quietly=TRUE)) {
      list(skip=TRUE, reason="lubridate or readr is not installed")
    } else {
      q <- function(x) suppressMessages(suppressWarnings(x))
      result <- list()
      # Hour and hour-minute orders: separators, names, compact digits,
      # AM/PM, 24:00, impossible dates, extra fields, and time zones.
      calendars <- c(ymd="2024-02-29", mdy="02/29/2024", dmy="29/02/2024")
      named <- c(ymd="2024 February 29", mdy="February 29th 2024", dmy="29th February 2024")
      compact <- c(ymd="20240229", mdy="02292024", dmy="29022024")
      impossible <- c(ymd="2023-02-29", mdy="02/29/2023", dmy="29/02/2023")
      # A year ending in 99 cannot be read as a two-digit year and an hour.
      bare <- c(ymd="1999-01-02", mdy="01/02/1999", dmy="02/01/1999")
      for (order in names(calendars)) for (clock in c("h", "hm")) {
        name <- paste(order, clock, sep="_")
        parser <- getExportedValue("lubridate", name)
        time <- if (clock == "h") c("13", "1 PM", "12 AM", "24", "13:05") else c("13:05", "1:05 PM", "12:05 AM", "24:00", "13:05:09")
        digits <- if (clock == "h") "13" else "1305"
        cal <- calendars[[order]]
        x <- c(paste(cal, time[1]), paste(named[[order]], time[1]), paste0(compact[[order]], digits),
               paste0(cal, "T", time[1]), paste(cal, time[2]), paste(cal, time[3]), paste(cal, time[4]),
               paste(impossible[[order]], time[1]), paste(cal, time[5]), bare[[order]], "bad", NA_character_)
        for (zone in c("UTC", "America/New_York", "Asia/Kolkata")) {
          result[[paste(name, zone, sep="_")]] <- as.numeric(q(parser(x, tz=zone)))
        }
        for (truncated in seq_len(nchar(clock))) {
          result[[paste(name, "truncated", truncated, sep="_")]] <- as.numeric(q(parser(x, truncated=truncated)))
        }
      }
      # parse_date_time: one order, mixed orders, training on the column,
      # truncation, AM/PM, offsets, daylight saving, names, and exact formats.
      long_sparse <- rep("2024-01-15", 150)
      long_sparse[c(1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
                    79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149)] <- "garbage"
      i <- 1:300
      long_majority <- ifelse(i %% 10 == 0, "05/06/2024", ifelse(i %% 10 == 5, "06/14/2024", "13/06/2024"))
      specs <- list(
        single=list(x=c("2024-02-29 12:34:56", "2024/2/29 1:2:3", "20240229123456", "on 2024-02-29 12:34:56",
                        "2024-02-29 12:34:56.125", "Thursday 2024-02-29 12:34:56", "2024-02-29T12:34:56Z",
                        "2024-02-29 12:34:56+05:30", "x2024-02-29 12:34:56", "2024-02-30 12:00:00",
                        "2024-02-29 24:00:00", "2024-02-29 12:60:00", "2024-02-29 12:34", "2024_02_29 12.34.56",
                        "bad", "", NA), orders="ymd HMS"),
        single_ny=list(x=c("2024-02-29 12:34:56", "2024-07-04 08:00:00", "2024-02-29T12:34:56Z", NA),
                       orders="Ymd HMS", tz="America/New_York"),
        mixed=list(x=c("02/29/2024", "29/02/2024", "2024-02-29", "Feb 29, 2024", "29 February 2024", "2024 Feb 29",
                       "2024 Sept 29", "2024 FEB 29", "13/01/2024", "01/02/2024", "29/02/2024 10:00", "2024-02", NA),
                   orders=c("mdy", "dmy", "ymd", "dmy HM")),
        majority_dmy=list(x=c("01/02/2024", "13/01/2024", "14/01/2024"), orders=c("mdy", "dmy")),
        majority_mdy=list(x=c("01/02/2024", "01/13/2024", "01/14/2024"), orders=c("dmy", "mdy")),
        separators=list(x=c("01/02/2024", "13-01-2024", "14-01-2024", "02/13/2024"), orders=c("mdy", "dmy")),
        truncated=list(x=c("2024-02-29 1:02:03", "2024-02-29 13:02", "2024-02-29 13", "2024-02-29", "2024-02"),
                       orders="ymd HMS", truncated=3),
        ampm=list(x=c("2024-02-29 1:02:03 PM", "2024-02-29 01:02:03PM", "2024-02-29 12:02:03 AM",
                      "2024-02-29 1:02:03", "2024-02-29 01:02:03.5 pm", "2024-02-29 13:02:03 PM"), orders="ymd IMS p"),
        ampm_24=list(x=c("2024-02-29 1:02:03 PM", "2024-02-29 13:02:03", "2024-02-29 13:02:03 PM",
                         "2024-02-29 12:00:00 AM"), orders="ymd HMS p"),
        clocks=list(x=c("2024-02-29 1:02:03 PM", "2024-02-29 13:02:03", "2024-02-29 13:02"), orders=c("ymd T", "ymd R")),
        offsets=list(x=c("2024-02-29 12:34:56Z", "2024-02-29 12:34:56+05:30", "2024-02-29 12:34:56+0530",
                         "2024-02-29 12:34:56-05", "2024-02-29 12:34:56"), orders="ymd HMS z", tz="America/New_York"),
        dst=list(x=c("2024-03-10 02:30:00", "2024-11-03 01:30:00", "2024-03-10 03:30:00"), orders="ymd HMS",
                 tz="America/New_York"),
        names=list(x=c("Feb 29 2024", "February 29 2024", "02 29 2024"), orders="bdY"),
        time_first=list(x=c("12:34:56 2024-02-29", "12:34 29/02/2024"), orders=c("HMS ymd", "HM dmy")),
        two_digit=list(x=c("24-02-29", "02/29/24", "29.02.69", "2024-02-29"), orders=c("ymd", "mdy", "dmy")),
        long_sparse=list(x=long_sparse, orders="ymd"),
        long_majority=list(x=long_majority, orders=c("mdy", "dmy")),
        exact=list(x=c("2024-02-29 12:34:56", "29/02/2024", "bad"), orders=c("%Y-%m-%d %H:%M:%S", "%d/%m/%Y"),
                   exact=TRUE)
      )
      for (name in names(specs)) {
        spec <- specs[[name]]
        parsed <- q(lubridate::parse_date_time(spec$x, spec$orders, tz=if (is.null(spec$tz)) "UTC" else spec$tz,
                                               truncated=if (is.null(spec$truncated)) 0 else spec$truncated,
                                               exact=isTRUE(spec$exact)))
        result[[paste0("parse_date_time_", name)]] <- as.numeric(parsed)
      }
      result[["parse_date_time_numeric"]] <- as.numeric(q(lubridate::parse_date_time(c(20240229, 2024022913), c("ymd", "ymd H"))))
      # parse_time: readr's default format and explicit formats. tidy3
      # returns a time of day, so readr's durations of a day or more are
      # missing there.
      seconds <- function(x) { v <- as.numeric(x); ifelse(v >= 86400, NA, v) }
      result[["parse_time_default"]] <- seconds(q(readr::parse_time(c(
        "12:34", "12:34:56", "1:02", "01:02:03.25", "12:34 PM", "12:34pm", "12:00 AM", "1:02:03 am", " 12:34 ",
        "23:60", "23:59:60", "12", "1234", "12.34", "abc", "", "NA", NA, "12:34:56.125", "0:00 AM", "-1:00",
        "12:34 X", "T12:34", "12:34:56Z", "100:00", "12:3", "12:345", "12:034", "12:34:056", "12:34.5",
        "1:02:03PM", "12:34   PM", "12:34 a.m.", "24:00", "13:00 PM"))))
      formats <- list(
        list("%Hh%M", c("12h34", "07.05", "1205", "bad")),
        list("%H.%M", c("12h34", "07.05", "1205")),
        list("%H%M", c("12h34", "1205", "0705")),
        list("%I:%M:%S %p", c("7:05:03 PM", "12:00:00 am", "13:00:00 PM", "7:05:03")),
        list("%H:%M", c("7:05", "07:05:03", "7:5", "24:00", "23:59", "0:00")),
        list("%H:%M:%OS", c("07:05:03.5", "7:5:3", "07:05:3.25", "07:05:03")),
        list("%H:%M:%S", c("07:05:03.5", "07:05:003", "07:05:3")),
        list("%T", c("07:05:03", "07:05")),
        list("%R", c("07:05", "07:05:03")),
        list("%I:%M %p", c("7:05 pm", "07:05PM", "0:05 AM", "7:05", "12:05 am")),
        list("%H:%M %p", c("7:05 pm", "7:05")),
        list("%Y-%m-%dT%H:%M", c("2024-01-01T07:05", "07:05")),
        list("%H:%M %Z", c("07:05 UTC", "07:05:03 Z")),
        list("%H%.%M", c("07h05", "07-05", "07 05", "0705")),
        list("%H%*%M", c("07h05", "07--05", "0705", "07abc05")),
        list("%H%+%M", c("07h05", "07--05", "0705")),
        list("%H %M", c("0705", " 0705", "07 05")),
        list("%H:%M:%S%z", c("12:34:56+01", "12:34:56")),
        list("%H:%M:%S %z", c("12:34:56 +0100", "12:34:56 Z")),
        list("%b %H:%M", c("Jan 12:34", "January 12:34", "jan 12:34"))
      )
      result[["parse_time_formats"]] <- lapply(formats, function(f) seconds(q(readr::parse_time(f[[2]], f[[1]]))))
      result
    }
  },
  datetime_truncation = {
    if (!requireNamespace("lubridate", quietly=TRUE)) {
      list(skip=TRUE, reason="lubridate is not installed")
    } else {
      result <- list()
      for (name in c("ymd_hms", "ymd_hm", "mdy_hms", "dmy_hms")) {
        parser <- getExportedValue("lubridate", name)
        # A year ending in 99 cannot be reinterpreted as an hour by R's
        # two-digit-year fallback; tidy3 never makes that reinterpretation.
        calendar <- switch(name, mdy_hms="01/02/1999", dmy_hms="02/01/1999", "1999-01-02")
        x <- c(paste(calendar, "12:34:56"), paste(calendar, "12:34"), paste(calendar, "12"), calendar, "bad", NA_character_)
        maximum <- if (name == "ymd_hm") 2 else 3
        for (truncated in 0:maximum) {
          result[[paste0(name, "_", truncated)]] <- as.numeric(suppressWarnings(parser(x, truncated=truncated)))
        }
      }
      result
    }
  },
  datetime_dst = {
    if (!requireNamespace("lubridate", quietly=TRUE)) {
      list(skip=TRUE, reason="lubridate is not installed")
    } else {
      x <- c("2024-03-10 01:59:59.5", "2024-03-10 02:30:00", "2024-03-10 03:30:00", "2024-11-03 01:30:00",
             "2024-11-03T01:30:00-04:00", "2024-11-03T01:30:00-05", "2024-01-01T12:00:00Z",
             "2024-02-29 24:00:00", "2024-02-29 23:59:60", "2023-02-29 12:00:00", NA_character_)
      as.numeric(suppressMessages(suppressWarnings(lubridate::ymd_hms(x, tz="America/New_York"))))
    }
  },
  date_components = {
    if (!requireNamespace("lubridate", quietly=TRUE)) {
      list(skip=TRUE, reason="lubridate is not installed")
    } else {
      d <- c(as.Date("2019-12-25") + 0:1849, as.Date(NA))
      list(
        weekday=lubridate::wday(d), monday=lubridate::wday(d, week_start=1),
        yearday=lubridate::yday(d), week=lubridate::week(d), iso=lubridate::isoweek(d),
        q=lubridate::quarter(d), hour=lubridate::hour(d), minute=lubridate::minute(d), second=lubridate::second(d)
      )
    }
  },
  weekday_labels = {
    if (!requireNamespace("lubridate", quietly=TRUE)) {
      list(skip=TRUE, reason="lubridate is not installed")
    } else {
      d <- c(as.Date("2024-01-01") + 0:6, as.Date(NA))
      result <- list()
      for (start in 1:7) for (abbr in c(TRUE, FALSE)) {
        f <- lubridate::wday(d, label=TRUE, abbr=abbr, week_start=start, locale="C")
        key <- paste0(start, "_", if (abbr) "abbr" else "full")
        result[[key]] <- list(values=as.character(f), levels=levels(f), ordered=is.ordered(f))
      }
      result
    }
  },
  fiscal_quarters = {
    if (!requireNamespace("lubridate", quietly=TRUE)) {
      list(skip=TRUE, reason="lubridate is not installed")
    } else {
      d <- c(as.Date(sprintf("%d-%02d-15", rep(c(2023, 2024), each=12), rep(1:12, 2))), as.Date(NA))
      result <- list()
      for (start in 1:12) for (kind in c("quarter", "year.quarter", "date_first", "date_last", "year_start/end")) {
        value <- lubridate::quarter(d, type=kind, fiscal_start=start)
        if (kind %in% c("date_first", "date_last")) value <- as.character(value)
        result[[paste0(start, "_", kind)]] <- value
      }
      result
    }
  },
  clock_components = {
    if (!requireNamespace("lubridate", quietly=TRUE)) {
      list(skip=TRUE, reason="lubridate is not installed")
    } else {
      d <- as.POSIXct(c("2024-01-01 23:59:59.125", "2024-03-10 01:59:59.5", "2024-03-10 03:00:00.25", "2024-11-03 01:30:00", NA), tz="America/Los_Angeles")
      list(h=lubridate::hour(d), m=lubridate::minute(d), s=lubridate::second(d), weekday=lubridate::wday(d))
    }
  },
  factor_ordering = {
    if (!requireNamespace("forcats", quietly=TRUE)) {
      list(skip=TRUE, reason="forcats is not installed")
    } else {
      f <- factor(c("Gentoo", "Adelie", "Gentoo", NA, "Chinstrap"), levels=c("Chinstrap", "Adelie", "Gentoo", "unused"))
      mass <- c(5000, 3500, 5200, 4000, 3700)
      payload <- function(f) list(values=as.character(f), levels=levels(f))
      lapply(list(
        reverse=forcats::fct_rev(f), frequency=forcats::fct_infreq(f),
        weighted=forcats::fct_infreq(f, w=mass),
        relevel=forcats::fct_relevel(f, "Gentoo", "Adelie"),
        after=forcats::fct_relevel(f, "Adelie", after=2),
        last=forcats::fct_relevel(f, "Adelie", after=Inf),
        callable=forcats::fct_relevel(f, sort),
        median=forcats::fct_reorder(f, mass),
        mean_desc=forcats::fct_reorder(f, mass, mean, .desc=TRUE)
      ), payload)
    }
  },
  factor_values = {
    if (!requireNamespace("forcats", quietly=TRUE)) {
      list(skip=TRUE, reason="forcats is not installed")
    } else {
      f <- ordered(c("b", "a", "a", "c", NA), levels=c("b", "a", "c", "unused"))
      w <- c(1, 1, 1, 4, 10)
      payload <- function(f) list(
        values=list(kind="vector", data=unname(as.list(as.character(f)))),
        levels=list(kind="vector", data=unname(as.list(levels(f)))),
        ordered=is.ordered(f)
      )
      result <- list(
        recode=payload(forcats::fct_recode(f, renamed="a", spare="unused")),
        remove=payload(forcats::fct_recode(f, b="a", NULL="c")),
        collapse=payload(forcats::fct_collapse(f, pair=c("a", "b"))),
        remainder=payload(forcats::fct_collapse(f, pair=c("a", "b"), other_level="rest")),
        existing=payload(forcats::fct_collapse(f, b="a", other_level="rest")),
        overlap=payload(forcats::fct_collapse(f, first=c("a", "b"), last="a")),
        other_collision=payload(forcats::fct_lump_min(f, 2, other_level="b"))
      )
      for (n in c(0, 1, 2, -1, -2, 10)) {
        result[[paste0("n_", n)]] <- payload(forcats::fct_lump_n(f, n))
        result[[paste0("weighted_n_", n)]] <- payload(forcats::fct_lump_n(f, n, w=w))
      }
      props <- c(zero=0, low=.2, high=.4, negative_low=-.2, negative_high=-.4, all=1, negative_all=-1)
      for (label in names(props)) {
        result[[paste0("prop_", label)]] <- payload(forcats::fct_lump_prop(f, props[[label]]))
        result[[paste0("weighted_prop_", label)]] <- payload(forcats::fct_lump_prop(f, props[[label]], w=w))
      }
      for (minimum in c(0, 1, 2, 3)) {
        result[[paste0("min_", minimum)]] <- payload(forcats::fct_lump_min(f, minimum))
        result[[paste0("weighted_min_", minimum)]] <- payload(forcats::fct_lump_min(f, minimum, w=w))
      }
      for (method in c("min", "average", "first", "last", "max")) {
        result[[paste0("ties_", method)]] <- payload(forcats::fct_lump_n(f, 2, ties.method=method))
      }
      grouped <- tibble(g=c(2L, 2L, 2L, 1L, 1L, 1L), species=c("a", "a", "b", "b", "b", "a"))
      persistent <- grouped |> group_by(g) |> mutate(species=forcats::fct_lump_n(species, 1))
      transient <- grouped |> mutate(species=forcats::fct_lump_n(species, 1), .by=g)
      result$persistent <- payload(persistent$species)[c("values", "levels")]
      result$transient <- payload(transient$species)[c("values", "levels")]
      result
    }
  },
  factor_levels = {
    if (!requireNamespace("forcats", quietly=TRUE)) {
      list(skip=TRUE, reason="forcats is not installed")
    } else {
      f <- ordered(c("b", "a", "a", "c", NA), levels=c("b", "a", "c", "unused"))
      x <- c(1, 2, NA, 3, 4)
      y <- c(5, 1, 2, 7, 3)
      ties <- c(2, 2, 2, 1, 1)
      text <- c("z", "y", NA, "z", "x")
      payload <- function(f) list(
        values=list(kind="vector", data=unname(as.list(as.character(f)))),
        levels=list(kind="vector", data=unname(as.list(levels(f)))),
        ordered=is.ordered(f)
      )
      # fct_reorder2(.na_rm=TRUE) also drops the missing rows from its result;
      # tidy3 keeps every row, so compare its levels on the original values.
      all_rows <- function(out) factor(as.character(f), levels=levels(out), ordered=is.ordered(out))
      grouped <- tibble(g=c(2L, 2L, 1L, 1L, 1L), s=c("q", "p", "r", "p", NA), x=c(1, 2, 3, 1, 2), y=c(1, 2, 3, 4, 5))
      persistent <- grouped |> group_by(g) |> mutate(
        inorder=forcats::fct_inorder(s), reorder2=forcats::fct_reorder2(s, x, y, .na_rm=TRUE)
      )
      transient <- grouped |> mutate(
        inorder=forcats::fct_inorder(s), reorder2=forcats::fct_reorder2(s, x, y, .na_rm=TRUE), .by=g
      )
      list(
        drop=payload(forcats::fct_drop(f)),
        drop_only=payload(forcats::fct_drop(f, only=c("unused", "a", "zz"))),
        expand=payload(forcats::fct_expand(f, "z", "a", "y")),
        expand_after=payload(forcats::fct_expand(f, c("z", "y"), after=1)),
        expand_front=payload(forcats::fct_expand(f, "z", after=0)),
        inorder=payload(forcats::fct_inorder(f)),
        inorder_unordered=payload(forcats::fct_inorder(f, ordered=FALSE)),
        inorder_text=payload(forcats::fct_inorder(text)),
        value_to_level=payload(forcats::fct_na_value_to_level(f, "(Missing)")),
        value_to_existing=payload(forcats::fct_na_value_to_level(f, "c")),
        level_to_value=payload(forcats::fct_na_level_to_value(f, c("a", "zz"))),
        reorder2=payload(all_rows(forcats::fct_reorder2(f, x, y, .na_rm=TRUE))),
        reorder2_first=payload(all_rows(forcats::fct_reorder2(f, x, y, forcats::first2, .na_rm=TRUE, .desc=FALSE))),
        reorder2_ties=payload(forcats::fct_reorder2(f, ties, y)),
        reorder2_tied_scores=payload(forcats::fct_reorder2(f, ties, c(1, 1, 1, 1, 1))),
        as_factor=payload(forcats::as_factor(f)),
        as_factor_text=payload(forcats::as_factor(text)),
        as_factor_numbers=payload(forcats::as_factor(c(3, 1.5, NA, 10, 3))),
        as_factor_integers=payload(forcats::as_factor(c(2L, 1L, NA, 2L, 7L))),
        as_factor_logical=payload(forcats::as_factor(c(TRUE, NA, TRUE, TRUE, TRUE))),
        persistent_inorder=payload(persistent$inorder)[c("values", "levels")],
        persistent_reorder2=payload(persistent$reorder2)[c("values", "levels")],
        transient_inorder=payload(transient$inorder)[c("values", "levels")],
        transient_reorder2=payload(transient$reorder2)[c("values", "levels")]
      )
    }
  },
  factor_grouped = {
    if (!requireNamespace("forcats", quietly=TRUE)) {
      list(skip=TRUE, reason="forcats is not installed")
    } else {
      penguins <- tibble(island=c("b", "a", "b", "a"), species=c("Gentoo", "Adelie", "Gentoo", "Adelie"))
      persistent <- penguins |> group_by(island) |> mutate(species=forcats::fct_rev(forcats::fct_infreq(species)))
      transient <- penguins |> mutate(species=forcats::fct_rev(forcats::fct_infreq(species)), .by=island)
      payload <- function(df) list(values=as.character(df$species), levels=levels(df$species))
      list(persistent=payload(persistent), transient=payload(transient))
    }
  },
  factor_missing = {
    if (!requireNamespace("forcats", quietly=TRUE)) {
      list(skip=TRUE, reason="forcats is not installed")
    } else {
      f <- factor(c("Gentoo", "Adelie", "Gentoo", NA), levels=c("Gentoo", "Adelie", "unused"))
      mass <- c(NA, 3500, 5000, NA)
      # Explicit vector wrappers retain empty values in the JSON oracle.
      payload <- function(f) list(
        values=list(kind="vector", data=unname(as.list(as.character(f)))),
        levels=list(kind="vector", data=unname(as.list(levels(f))))
      )
      list(
        remove=payload(forcats::fct_reorder(f, mass, .na_rm=TRUE)),
        keep=payload(forcats::fct_reorder(f, mass, .na_rm=FALSE, .desc=TRUE)),
        default=payload(forcats::fct_reorder(f, mass, .na_rm=TRUE, .default=-Inf)),
        empty=payload(forcats::fct_rev(factor(character(), levels=c("Adelie", "Gentoo"))))
      )
    }
  },
  filter_missing = base |> filter(x > 1),
  filter_out_missing = base |> filter(is.na(x) | !(x > 1)),
  mutate_sequential = base |> mutate(a=y * 2, b=a + 1, .before=x),
  transmute = base |> transmute(g, a=y * 2, b=a + 1),
  select_drop = list(
    selected=base |> select(g, x),
    dropped=base |> select(-y)
  ),
  rename = base |> rename(score=x),
  rename_with = base |> rename_with(toupper, c(x, y)),
  relocate = base |> relocate(y, .before=x),
  pull = base |> pull(y),
  arrange = list(
    ascending=base |> arrange(x),
    descending=base |> arrange(desc(x))
  ),
  distinct = tibble(g=c("b", "a", "b", "a"), x=c(1, 2, 1, 3)) |>
    distinct(g),
  slices = list(
    positions=base |> slice(c(3, 1, 1)),
    first_rows=base |> head(2),
    head=base |> slice_head(n=2),
    tail=base |> slice_tail(n=2),
    minimum=base |> slice_min(x, n=2),
    maximum=base |> slice_max(x, n=2)
  ),
  grouping = list(
    persistent=base |> group_by(g) |> summarise(n=n(), avg=mean(y), .groups="drop"),
    transient=base |> summarise(n=n(), avg=mean(y), .by=g),
    ungrouped=base |> group_by(g) |> ungroup() |> summarise(n=n()),
    by_col=base |> group_by(g, x) |> ungroup(x) |> summarise(n=n(), .groups="drop")
  ),
  rowwise = tibble(id=1:2, x=c(1, 2), y=c(10, 20)) |>
    rowwise(id) |>
    mutate(total=sum(c_across(c(x, y)))) |>
    ungroup(),
  reframe = tibble(g=c("b", "b", "a"), x=c(2, 4, 10)) |>
    group_by(g) |>
    reframe(value=x, avg=mean(x)),
  counts = list(
    counted=base |> count(g),
    tallied=base |> group_by(g) |> tally(),
    added=base |> add_count(g),
    add_tallied=base |> group_by(g) |> add_tally() |> ungroup(),
    counted_computed=base |> count(big=y > 15),
    added_computed=base |> add_count(g, big=y > 15, name="k")
  ),
  joins = {
    left <- tibble(k=c(1, 2, NA_real_), x=c("a", "b", "c"))
    right <- tibble(k=c(2, NA_real_, 3), y=c(20, 99, 30))
    list(
      left=left |> left_join(right, by="k"),
      right=left |> right_join(right, by="k"),
      inner=left |> inner_join(right, by="k"),
      full=left |> full_join(right, by="k"),
      semi=left |> semi_join(right, by="k"),
      anti=left |> anti_join(right, by="k"),
      cross=tibble(x=1:2) |> cross_join(tibble(y=c("a", "b")))
    )
  },
  nest_join = {
    left <- tibble(k=c(1L, 2L), x=c("a", "b"))
    right <- tibble(k=c(1L, 1L, 2L), y=c(10L, 11L, 20L))
    left |> nest_join(right, by="k", name="matches") |>
      unnest(matches)
  },
  binds = list(
    rows=bind_rows(tibble(x=1L, y="a"), tibble(x=2L, z=3.5)),
    cols=bind_cols(tibble(x=1:2), tibble(y=c("a", "b")))
  ),
  sets = {
    x <- tibble(a=c(1, 2, 2))
    y <- tibble(a=c(2, 3))
    list(
      union=union(x, y),
      union_all=union_all(x, y),
      intersect=intersect(x, y),
      setdiff=setdiff(x, y),
      symdiff=symdiff(x, y),
      setequal=setequal(x, tibble(a=c(2, 1)))
    )
  },
  rows = {
    x <- tibble(id=1:3, value=c(10, NA_real_, 30), old=c("a", "b", "c"))
    list(
      inserted=rows_insert(x, tibble(id=4L, value=40, old="d"), by="id"),
      appended=bind_rows(x, tibble(id=4L, value=40, old="d")),
      updated=rows_update(x, tibble(id=2L, value=20, old="B"), by="id"),
      patched=rows_patch(x, tibble(id=2L, value=20, old="B"), by="id"),
      upserted=rows_upsert(x, tibble(id=c(2L, 4L), value=c(20, 40), old=c("B", "d")), by="id"),
      deleted=rows_delete(x, tibble(id=2L), by="id")
    )
  },
  missing_data = {
    x <- tibble(g=c("a", "a", "b"), id=c(1L, 2L, 1L), value=c(NA, 2, NA))
    list(
      dropped=x |> drop_na(value),
      replaced=x |> replace_na(list(value=0)),
      filled=x |> fill(value, .direction="downup"),
      expanded=x |> select(g, id) |> expand(g, id),
      completed=x |> complete(g, id, fill=list(value=0))
    )
  },
  pivots = {
    wide <- tibble(id=1:2, a=c(10L, 20L), b=c(30L, 40L))
    long <- wide |> pivot_longer(c(a, b), names_to="name", values_to="value")
    dup <- tibble(id=c(1L, 1L, 2L), key=c("a", "a", "b"), val=c(1, 2, 3))
    list(
      long=long,
      wide=long |> pivot_wider(names_from=name, values_from=value),
      duplicated=suppressWarnings(dup |> pivot_wider(names_from=key, values_from=val)),
      as_list=dup |> pivot_wider(names_from=key, values_from=val, values_fn=list),
      averaged=dup |> pivot_wider(names_from=key, values_from=val, values_fn=mean)
    )
  },
  separate_unite = {
    x <- tibble(id=1:2, code=c("a-10", "b-20"))
    separated <- x |> separate(code, c("group", "number"), sep="-")
    list(separated=separated, united=separated |> unite("code", group, number, sep="-"))
  },
  nest_roundtrip = {
    x <- tibble(g=c("a", "a", "b"), value=1:3)
    x |> nest(data=value) |> unnest(data)
  },
  unnest_longer = tibble(id=1:2, values=list(c(10L, 11L), 20L)) |>
    unnest_longer(values, indices_to="position"),
  unnest_wider = tibble(id=1:2, values=list(list(a=10L, b=20L), list(a=30L, b=40L))) |>
    unnest_wider(values),
  empty_inputs = list(
    filtered=tibble(x=double()) |> filter(x > 0),
    summary=tibble(x=double()) |> summarise(n=n(), avg=mean(x)),
    grouped=tibble(g=character(), x=double()) |> group_by(g) |> summarise(n=n(), .groups="drop"),
    joined=tibble(k=integer(), x=character()) |> left_join(tibble(k=integer(), y=double()), by="k")
  ),
  categorical = tibble(
    g=factor("a", levels=c("a", "b")), x=1
  ) |> group_by(g, .drop=FALSE) |> summarise(n=n(), .groups="drop"),
  categorical_count = tibble(
    g=factor("a", levels=c("a", "b")), x=1
  ) |> count(g, .drop=FALSE),
  duplicate_names = {
    # tidy3 spells dplyr's repaired x...1 as x_1.
    out <- suppressMessages(bind_cols(tibble(x=1L), tibble(x=2L)))
    names(out) <- sub("...", "_", names(out), fixed=TRUE)
    out
  },
  descending_rank = tibble(x=c(3, 1, 2, NA_real_)) |>
    mutate(rank=min_rank(desc(x))),
  sequential_summary = tibble(x=1:3) |>
    summarise(a=mean(x), b=a * 2),
  sequential_summary_nested = tibble(x=1:3) |>
    summarise(a=sum(x), b=sum(a), c=sum(a + x)),
  mutate_delete = tibble(x=1:2, y=3:4) |> mutate(x=NULL),
  computed_distinct = tibble(x=c(1L, 3L, 2L), y=10:12) |>
    distinct(parity=x %% 2L),
  n_distinct_multi = tibble(x=c(1L, 1L, 2L), y=c("a", "b", "b")) |>
    summarise(n=n_distinct(x, y)),
  bind_cols_recycle = bind_cols(tibble(x=1:3), tibble(z=9L)),
  pivot_value = tibble(id=1:2, x_a=1:2, y_a=3:4) |>
    pivot_longer(-id, names_to=c(".value", "set"), names_sep="_"),
  pivot_wider_multi = tibble(
    id=c(1L, 1L), name=c("a", "b"), v1=c(10L, 20L), v2=c(30L, 40L)
  ) |> pivot_wider(names_from=name, values_from=c(v1, v2)),
  pivot_wider_multi_names = tibble(
    id=c(1L, 1L), axis=c("x", "x"), period=c("q1", "q2"), value=c(10L, 20L)
  ) |> pivot_wider(names_from=c(axis, period), values_from=value),
  separate_convert = tibble(code=c("a-10", "b-20")) |>
    separate(code, c("group", "number"), sep="-", convert=TRUE),
  separate_convert_types = tibble(code=c("TRUE-1.5", "FALSE-2.0")) |>
    separate(code, c("flag", "number"), sep="-", convert=TRUE),
  arrange_by_group = tibble(g=c("b", "a", "b"), x=c(2, 3, 1)) |>
    group_by(g) |>
    arrange(x, .by_group=TRUE) |>
    ungroup(),
  type_coercion = list(
    rows=bind_rows(tibble(x=1L), tibble(x=2.5)),
    mutated=tibble(x=1:2) |> mutate(y=x / 2, flag=x > 1, label=if_else(flag, "yes", "no"))
  ),
  new_helpers = {
    x <- tibble(code=c(1L, 2L, 3L, NA_integer_), text=c("a|b", "c", "d|e", NA_character_))
    list(
      labels=x |> mutate(
        bucket=case_match(code, c(1L, 2L) ~ "low", 3L ~ "high", .default="other"),
        recoded=recode(code, `1`="one", `2`="two", .default="other", .missing="missing")
      ),
      longer=tibble(text=c("a|b", "c")) |> separate_longer_delim(text, delim="|"),
      wider=tibble(text=c("a|b", "c")) |> separate_wider_delim(text, names=c("left", "right"), delim="|", too_few="align_start"),
      merged=tibble(id=1:3, text=c("a|b|c", "d", NA), z=1:3) |>
        separate_wider_delim(text, names=c("left", "right"), delim="|", too_few="align_end", too_many="merge"),
      dropped=tibble(id=1:3, text=c("a|b|c", "d", NA), z=1:3) |>
        separate_wider_delim(text, names=c("left", "right"), delim="|", too_few="align_start", too_many="drop")
    )
  },
  dplyr_12_helpers = {
    x <- tibble(
      state=c("NC", "NYC", "CA", NA, "NYC", "Unknown"),
      score=c(1, 2, 3, 4, 5, NA)
    )
    b <- tibble(
      x=c(TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, NA, NA, NA),
      y=c(TRUE, FALSE, NA, TRUE, FALSE, NA, TRUE, FALSE, NA)
    )
    pets <- tibble(type=c("dog", "dog", "cat", "dog", "cat"), age=c(1, 3, 5, 2, 4))
    list(
      values=x |> mutate(
        full=recode_values(state, "NC" ~ "North Carolina", "NYC" ~ "New York", "CA" ~ "California"),
        grouped=recode_values(state, "NC" ~ "North Carolina", c("NYC", "CA") ~ "elsewhere", default="<not recorded>"),
        label=recode_values(score, from=c(1, 2, 3, 4, 5), to=c("SD", "D", "N", "A", "SA")),
        replaced=replace_values(state, "NYC" ~ "NY"),
        no_missing=replace_values(state, NA ~ "Unknown (NA)"),
        tidied=replace_values(state, c(NA, "Unknown") ~ "<not recorded>"),
        capped=replace_when(score, score > 3 ~ 0),
        big=score > 3
      ),
      logic=b |> mutate(
        any_p=when_any(x, y), any_r=when_any(x, y, na_rm=TRUE),
        all_p=when_all(x, y), all_r=when_all(x, y, na_rm=TRUE)
      ),
      pets=pets |> mutate(type=replace_when(type, type == "dog" & age <= 2 ~ "puppy")),
      filtered=x |> filter(when_any(state == "CA", score >= 5)),
      oldest=pets |> group_by(type) |> filter(age == max(age)) |> summarise(age_next=age + 1),
      strict=x |> mutate(band=case_when(
        score < 3 ~ "low", score >= 3 ~ "high", is.na(score) ~ NA_character_,
        .unmatched="error"
      ))
    )
  },
  group_info = {
    d <- tibble(g=c("b", "a", "b", "a", "b"), h=c(1, 1, 2, 1, 2), x=1:5)
    gd <- group_by(d, g, h)
    f <- tibble(k=factor(c("x", "x", NA), levels=c("x", "y")), g=c("b", "a", "b"))
    fd <- group_by(f, g, k, .drop=FALSE)
    list(
      data=group_data(gd) |> mutate(.rows=lapply(.rows, as.integer)),
      keys=group_keys(gd),
      sizes=tibble(size=group_size(gd)),
      indices=tibble(index=group_indices(gd)),
      counts=tibble(n=n_groups(gd), vars=paste(group_vars(gd), collapse=",")),
      ungrouped=group_data(d) |> mutate(.rows=lapply(.rows, as.integer), n=n_groups(d)),
      empty=group_data(fd) |> mutate(.rows=lapply(.rows, as.integer)),
      trimmed=tibble(size=group_size(group_trim(fd))),
      split=bind_rows(lapply(group_split(gd), function(part) slice(part, 1))),
      modified=gd |> group_modify(~ head(.x, 1)) |> ungroup(),
      nested=nest_by(d, g) |> mutate(rows=nrow(data)) |> ungroup() |> select(g, rows),
      group_nested=group_nest(gd) |> mutate(rows=vapply(data, nrow, integer(1))) |> select(g, h, rows)
    )
  },
  ordering = {
    d <- tibble(
      g=c("a", "a", "b", "b", "a", "b", "a"),
      t=c(3, 1, 2, NA, 2, 1, 1),
      x=c(10, 20, 30, 40, 50, 60, 70)
    )
    list(
      ungrouped=d |> mutate(
        running=order_by(t, cumsum(x)),
        previous=with_order(t, lag, x),
        latest_first=order_by(desc(t), cumsum(x))
      ),
      grouped=d |> group_by(g) |> mutate(
        running=order_by(t, cumsum(x)),
        rank=with_order(desc(t), row_number, x)
      ) |> ungroup()
    )
  },
  grids = {
    df <- tibble(a=c(2L, 1L, 2L), b=c("v", "u", "v"))
    counts <- tibble(x=c("a", "b", "c"), n=c(1, 0, 2))
    obs <- tibble(country=c("A", "A", "B"), year=c(1952L, 1962L, 1957L), pop=c(1, 2, 3))
    list(
      expand_grid=expand_grid(x=c(3, 1, 3), y=c("b", "a")),
      expand_grid_frame=expand_grid(df, z=1:2),
      crossing=crossing(x=c(3, 1, 3, NA), y=c("b", "a")),
      crossing_frame=crossing(df, z=c(2L, 1L)),
      uncounted=uncount(counts, n),
      uncounted_id=uncount(counts, n, .id="copy"),
      uncounted_kept=uncount(counts, n, .remove=FALSE),
      sequence=tibble(value=full_seq(c(1, 2, 4, 5, 10), 1)),
      # tidy3 keeps year's whole-number type; R's full_seq() would make it double.
      completed=obs |> complete(country, year=as.integer(full_seq(year, 5))),
      expanded=obs |> expand(country, year=as.integer(full_seq(year, 5)))
    )
  },
  splitting = {
    d <- tibble(id=1:3, code=c("ab-12", "cd-3", NA))
    pattern <- c(letters="[a-z]+", "-", number="[0-9]+")
    dates <- tibble(id=1:3, code=c("20240115", "2023", NA))
    chopped <- chop(tibble(g=c("a", "a", "b"), x=1:3, y=c(10, 20, 30)), c(x, y))
    lists <- tibble(g=c("a", "b", "c"), x=list(1:2, integer(), NULL))
    list(
      regex=separate_wider_regex(d, code, pattern),
      regex_partial=separate_wider_regex(tibble(code=c("ab-12", "ab")), code, pattern, too_few="align_start"),
      regex_prefixed=separate_wider_regex(d, code, pattern, names_sep="_"),
      position=separate_wider_position(dates, code, c(year=4, month=2, day=2), too_few="align_start"),
      position_skip=separate_wider_position(tibble(code="20240115"), code, c(year=4, 2, day=2)),
      longer=separate_longer_position(tibble(id=1:3, x=c("abcde", "fg", "h")), x, width=2),
      chopped=chopped,
      unchopped=unchop(chopped, c(x, y)),
      unchop_drop=unchop(lists, x),
      unchop_keep=unchop(lists, x, keep_empty=TRUE),
      auto_wider=suppressMessages(unnest_auto(tibble(id=1:2, v=list(list(a=1, b=2), list(a=3, b=4))), v)),
      auto_longer=suppressMessages(unnest_auto(tibble(id=1:2, v=list(c(1, 2), c(3))), v))
    )
  },
  empty_groups = {
    d <- tibble(k=factor(c("x", "x", NA), levels=c("x", "y")), g=c("b", "a", "b"), v=c(1, 2, 3))
    list(
      kg=d |> group_by(k, g, .drop=FALSE) |> summarise(n=n(), s=sum(v), m=mean(v), .groups="drop"),
      gk=d |> group_by(g, k, .drop=FALSE) |> summarise(n=n(), s=sum(v), m=mean(v), .groups="drop"),
      k=d |> group_by(k, .drop=FALSE) |> summarise(n=n(), .groups="drop"),
      counted=count(d, k, g, .drop=FALSE),
      sorted=count(d, k, .drop=FALSE, sort=TRUE)
    )
  },
  join_names = {
    # tidy3 spells dplyr's default .x/.y as _x/_y.
    sx <- c("_x", "_y")
    a <- tibble(id=1:3, v=c(1, 2, 3), w=c(5, 6, 7))
    b <- tibble(id=c(1L, 2L, 4L), v=c(10, 20, 40))
    clash <- tibble(id=1:2, v=c(1, 2), v_x=c(9, 9))
    list(
      left=left_join(a, b, by="id", suffix=sx),
      inner=inner_join(a, b, by="id", suffix=sx),
      full=full_join(a, b, by="id", suffix=sx),
      right=right_join(a, b, by="id", suffix=sx),
      pair=left_join(a, b, by="id", suffix=c("_l", "_r")),
      kept=left_join(a, b, by="id", suffix=sx, keep=TRUE),
      clash=left_join(clash, b, by="id", suffix=sx),
      empty_left=left_join(a, b, by="id", suffix=c("", "_right")),
      crossed=cross_join(a[1:2, ], b[1:2, ], suffix=sx)
    )
  },
  pivot_options = {
    d <- tibble(id=c(1, 1, 2, 2), extra=c("p", "q", "r", "s"), key=c("a", "b", "a", "b"),
                sub=c("x", "y", "x", "x"), v=c(1, 2, 3, 4), w=c(5, 6, 7, 8))
    f <- tibble(id=c(1, 2), key=factor(c("a", "a"), levels=c("a", "b", "c")), v=c(1, 2))
    g <- tibble(id=factor(c("i1", "i1"), levels=c("i1", "i2")), key=c("a", "b"), v=c(1, 2))
    l <- tibble(id=1:2, week_1=c(5L, 6L), week_2=c(7L, 8L))
    list(
      sep=pivot_wider(d, id_cols=id, names_from=c(key, sub), values_from=v, names_sep="."),
      glue=pivot_wider(d, id_cols=id, names_from=key, values_from=c(v, w), names_glue="{key}_{.value}"),
      slowest=pivot_wider(d, id_cols=id, names_from=key, values_from=c(v, w), names_vary="slowest"),
      names_expanded=pivot_wider(f, names_from=key, values_from=v, names_expand=TRUE),
      ids_expanded=pivot_wider(g, names_from=key, values_from=v, id_expand=TRUE),
      unused=pivot_wider(d, id_cols=id, names_from=key, values_from=v, unused_fn=list(w=max)),
      transformed=pivot_longer(l, starts_with("week_"), names_to="week", names_prefix="week_",
                               names_transform=list(week=as.integer), values_transform=list(value=as.character))
    )
  },
  list_options = {
    d <- tibble(id=1:2, code=c("a-1", "b-2"))
    h <- tibble(id=1:2, info=list(list(name="x", n="1", k=5), list(name="y", n="2", k=6)))
    n <- tibble(g=c(1, 1, 2), x_a=1:3, x_b=4:6)
    list(
      sep=separate_wider_delim(d, code, "-", names=c("l", "n"), names_sep="_"),
      kept=separate_wider_delim(d, code, "-", names=c("l", "n"), cols_remove=FALSE),
      indexed=unnest_longer(tibble(id=1:2, x=list(c(10, 11), 20)), x, indices_include=TRUE),
      longer_int=unnest_longer(tibble(id=1:2, x=list(c("1", "2"), "3")), x, transform=as.integer),
      wider_int=unnest_wider(tibble(id=1:2, x=list(list(a="1", b="2"), list(a="3", b="4"))), x,
                             transform=list(a=as.integer)),
      hoisted=hoist(h, info, nm="name", n="n", .transform=list(n=as.integer)) |> select(-info),
      hoisted_all=hoist(h, info, name="name", n="n", k="k"),
      nested=nest(n, x=c(x_a, x_b), .names_sep="_") |> unnest(x)
    )
  },
  specs = {
    wide <- tibble(id=1:2, mean_1=c(5, 6), mean_2=c(7, 8), sd_1=c(0.1, 0.2), sd_2=c(0.3, 0.4))
    long <- tibble(id=c(1, 1, 2, 2), key=c("a", "b", "a", "b"), v=1:4, w=5:8)
    s1 <- build_longer_spec(wide, -id, names_to=c(".value", "visit"), names_sep="_")
    s2 <- build_longer_spec(wide, starts_with("mean"), names_to="visit", names_prefix="mean_", values_to="score")
    s3 <- build_wider_spec(long, names_from=key, values_from=c(v, w))
    s4 <- build_wider_spec(long, names_from=key, values_from=v)
    s4$.name <- c("first", "second")
    list(
      longer_spec=s1, longer=pivot_longer_spec(wide, s1), longer_values=pivot_longer_spec(wide, s2),
      wider_spec=s3, wider=pivot_wider_spec(long, s3), renamed=pivot_wider_spec(long, s4, id_cols=id),
      glued=build_wider_spec(long, names_from=key, values_from=v, names_glue="col_{key}", names_sort=TRUE)
    )
  },
  helpers = {
    rows <- cbind(c(1, NA, NA), c(NA, NA, 4), c(2, NA, 6))
    list(
      rounded=round(c(0.5, 1.5, -1.5, 1.25)),
      rounded_1=round(c(1.25, 1.35), 1),
      rounded_tens=round(c(123, -15), -1),
      pmin=pmin(c(1, NA, 5), c(2, 3, NA)),
      pmin_rm=pmin(c(1, NA, 5), c(2, 3, NA), na.rm=TRUE),
      pmax=pmax(c(1, NA, 5), c(0, 4, 4)),
      row_sums=rowSums(rows),
      row_sums_rm=rowSums(rows, na.rm=TRUE),
      row_means_rm=rowMeans(rows, na.rm=TRUE)
    )
  },
  lubridate_dates = {
    if (!requireNamespace("lubridate", quietly=TRUE)) {
      list(skip=TRUE, reason="lubridate is not installed")
    } else {
      suppressPackageStartupMessages(library(lubridate))
      d <- as.Date(c("2010-01-31", "2012-02-29", "2010-01-25", "2010-02-03", "2010-01-01"))
      tibble(
        rollback=format(add_with_rollback(d, months(1)), "%Y-%m-%d"),
        year_roll=format(add_with_rollback(d, years(1)), "%Y-%m-%d"),
        floor_week=format(floor_date(d, "week"), "%Y-%m-%d"),
        ceiling_week=format(ceiling_date(d, "week"), "%Y-%m-%d"),
        ceiling_month=format(ceiling_date(d, "month"), "%Y-%m-%d"),
        days=as.integer(days_in_month(d))
      )
    }
  },
  readr_parse = {
    if (!requireNamespace("readr", quietly=TRUE)) {
      list(skip=TRUE, reason="readr is not installed")
    } else {
      suppressPackageStartupMessages(library(readr))
      list(
        number=parse_number(c("$1,234.50", "abc")),
        plain=parse_double(c("1.5", "x")),
        integer=parse_integer(c("1234", "2.5")),
        logical=parse_logical(c("true", "1", "0", "yes"))
      )
    }
  },
  helper_edges = {
    v <- c(1, NA, NaN, Inf)
    list(
      is_na=is.na(v),
      is_finite=is.finite(v),
      is_infinite=is.infinite(v),
      abs=abs(c(-2.5, 2.5, NA)),
      floor=floor(c(-2.5, 2.5, NA)),
      ceiling=ceiling(c(-2.5, 2.5, NA)),
      round_2=round(c(2.675, 0.125, 1.005), 2),
      round_1=round(c(0.15, 0.25, 0.35), 1),
      round_tens=round(c(125, 135, -15), -1),
      pmin_text=pmin(c("nan", "b"), c("c", "a")),
      pmin_nan=pmin(c(1, NaN), c(2, 3))
    )
  },
  is_nan = is.nan(c(1, NA, NaN, Inf)),
  conversions = list(
    as_double=suppressWarnings(as.numeric(c("1.5", " 2 ", "1e5", "-.5", "NaN", "abc", "1,234"))),
    as_integer=as.integer(c("2", " 7 ", "-3", "007")),
    as_integer_num=as.integer(c(2, -3, NA)),
    as_logical_text=as.logical(c("TRUE", "true", "T", "True", "FALSE", "false", "F", "tRuE", "yes", "1")),
    as_logical_num=as.logical(c(0, 2, -1, NaN, NA)),
    as_logical_exact=as.logical(c(" TRUE", "TRUE ", "yes", "True", "T", "false", "F")),
    # tidy3 writes a float as its type shows it ("21.0" where R gives "21"),
    # so compare integers and fractional doubles, where the two agree.
    as_character_int=as.character(c(1L, 123456L, -3L, 100001L)),
    as_character=as.character(c(1.5, 0.25, -3.75, 22.8)),
    # R spells logical values TRUE / FALSE; tidy3 does when asked.
    as_character_logical=as.character(c(TRUE, FALSE, NA)),
    as_date=format(as.Date(c("2010-01-31", NA)), "%Y-%m-%d")
  ),
  lubridate_more = {
    if (!requireNamespace("lubridate", quietly=TRUE)) {
      list(skip=TRUE, reason="lubridate is not installed")
    } else {
      suppressPackageStartupMessages(library(lubridate))
      day_text <- function(x) format(x, "%Y-%m-%d")
      time_text <- function(x) format(x, "%Y-%m-%d %H:%M:%S", tz="UTC")
      d <- as.Date(c("2024-02-29", "2024-01-31", "2010-03-31"))
      t <- ymd_hms(c("2026-10-21 15:30:00", "2026-10-18 09:00:00", "2026-10-18 00:00:00"))
      list(
        ymd=day_text(suppressWarnings(ymd(c("2010-01-31", "20100131", "2010/1/31", "2010.01.31", "2010 01 31", "bad")))),
        ymd_with_time=day_text(suppressWarnings(ymd("2010-01-31 10:00:00"))),
        mdy=day_text(suppressWarnings(mdy(c("01/31/2010", "1-31-2010", "Jan 31 2010")))),
        dmy=day_text(dmy(c("31.01.2010", "31012010"))),
        ym=day_text(ym(c("2010-02", "201002"))),
        parts=list(year=year(d), month=month(d), day=day(d), days=as.integer(days_in_month(d))),
        both=day_text(add_with_rollback(d, years(1) + months(1))),
        back=day_text(add_with_rollback(d, -(years(1) + months(1)))),
        month_column=day_text(d %m+% months(c(1L, NA, -1L))),
        year_minus=day_text(d %m-% years(1)),
        floor_week=time_text(floor_date(t, "week")),
        ceiling_week=time_text(ceiling_date(t, "week")),
        floor_week_monday=time_text(floor_date(t, "week", week_start=1)),
        floor_month=time_text(floor_date(t, "month")),
        ceiling_hour=time_text(ceiling_date(t, "hour")),
        floor_quarter=day_text(floor_date(d, "quarter")),
        ceiling_day=day_text(ceiling_date(d, "day")),
        ceiling_month=day_text(ceiling_date(as.Date(c("2010-01-31", "2010-02-01")), "month"))
      )
    }
  },
  readr_more = {
    if (!requireNamespace("readr", quietly=TRUE)) {
      list(skip=TRUE, reason="readr is not installed")
    } else {
      suppressPackageStartupMessages(library(readr))
      q <- function(x) suppressWarnings(x)
      kept <- tibble(x=c("1", "a", "2", "b")) |>
        filter(x != "1") |>
        mutate(n=q(parse_double(x)))
      found <- problems(kept$n)
      list(
        number=q(parse_number(c("-$1,234", "$-5", "USD -7.5", "Price - 12", "50%", "abc"))),
        double=q(parse_double(c("1.5", " 1.5 ", "1e3", "abc"))),
        double_grouped=q(parse_double("1,234.5")),
        double_euro=q(parse_double(c("1,5", "-0,25"), locale=locale(decimal_mark=",", grouping_mark="."))),
        integer=q(parse_integer(c("1234", " 7 ", "2.5", "x"))),
        logical=q(parse_logical(c("TRUE", "T", "true", "t", "1", "FALSE", "f", "0", "yes"))),
        date=format(q(parse_date(c("31/01/2010", "bad"), "%d/%m/%Y")), "%Y-%m-%d"),
        date_cutoff=format(parse_date(c("68/01/02", "69/01/02"), "%y/%m/%d"), "%Y-%m-%d"),
        problem_rows=found$row,
        problem_values=found$actual
      )
    }
  },
  readr_files = {
    if (!requireNamespace("readr", quietly=TRUE)) {
      list(skip=TRUE, reason="readr is not installed")
    } else {
      suppressPackageStartupMessages(library(readr))
      q <- function(x) suppressWarnings(suppressMessages(x))
      cars <- q(read_csv(I("model,mpg,cyl,am\nMazda RX4,21.0, 6,TRUE\nDatsun 710,NA,4 ,false\nValiant, NA , 6,\n"), show_col_types=FALSE))
      bad <- q(read_csv(I("model,hp\nMazda RX4,110\nValiant,abc\nDatsun 710,2.5\n"), col_types=cols(model=col_character(), hp=col_integer())))
      found <- problems(bad)
      list(
        mpg=cars$mpg,
        cyl=as.numeric(cars$cyl),
        am=cars$am,
        hp=bad$hp,
        problem_rows=found$row,
        problem_values=found$actual
      )
    }
  },
  round_extremes = {
    x <- c(1.25, -1.25, 0, 1.25e-310, -1.25e-310, 1.75e-310, 5e-324)
    digits <- c(309, 310, 323, 324, -309)
    setNames(lapply(digits, function(d) {
      list(kind="vector", data=as.list(sprintf("%.17g", round(x, d))))
    }), as.character(digits))
  },
  string_summaries = {
    if (!requireNamespace("stringr", quietly=TRUE)) {
      list(skip=TRUE, reason="stringr is not installed")
    } else {
      cars <- tibble(cyl=c(4L, 4L, 6L), model=c("Datsun", "Toyota", "Mazda"))
      list(
        whole=cars |> summarise(
          length=stringr::str_length(first(model)),
          upper=stringr::str_to_upper(first(model)),
          label=stringr::str_c(first(model), "!")
        ),
        grouped=cars |> group_by(cyl) |> summarise(
          length=stringr::str_length(first(model)),
          upper=stringr::str_to_upper(first(model)),
          label=stringr::str_c(first(model), "!")
        )
      )
    }
  },
  inferred_csv_failures = {
    if (!requireNamespace("readr", quietly=TRUE)) {
      list(skip=TRUE, reason="readr is not installed")
    } else {
      # Once these types are chosen, late failures follow readr's parser
      # contract. Sampling positions and default numeric types differ.
      text <- paste0("hp,mpg,am\n", paste(rep("110,21.5,TRUE\n", 100), collapse=""), "oops,oops,oops\n")
      cars <- suppressWarnings(readr::read_csv(I(text), col_types=readr::cols(
        hp=readr::col_integer(), mpg=readr::col_double(), am=readr::col_logical()
      )))
      found <- readr::problems(cars)
      list(
        hp=cars$hp, mpg=cars$mpg, am=cars$am,
        problems=tibble(
          row=found$row, column=c("hp", "mpg", "am"),
          original=found$actual, expected=c("integer", "double", "logical")
        )
      )
    }
  },
  round_grid = {
    # Exact text, not JSON numbers: jsonlite keeps 15 decimals, which would
    # hide a result that is off in the last bit.
    x <- c((0:2000) / 1000, -(1:500) / 1000 * 3.7, (0:400) / 8,
           2.675, 1.005, 5.015, 0.285, 2^900, 123456789.123456789)
    exact <- function(v) sprintf("%.17g", v)
    list(
      d0=exact(round(x, 0)),
      d1=exact(round(x, 1)),
      d2=exact(round(x, 2)),
      d3=exact(round(x, 3)),
      tens=exact(round(x * 1000, -1)),
      hundreds=exact(round(x * 1000, -2))
    )
  },
  readr_grid = {
    if (!requireNamespace("readr", quietly=TRUE)) {
      list(skip=TRUE, reason="readr is not installed")
    } else {
      suppressPackageStartupMessages(library(readr))
      q <- function(x) suppressWarnings(suppressMessages(x))
      euro <- locale(decimal_mark=",", grouping_mark=".")
      list(
        number=q(parse_number(c("-$1,234", "$-5", "a-5", "- 5", "--5", "+5", "$+5", "1-2",
          "1,234,", "(5)", ".5", "-.5", "12abc34", "abc", "1e3", "1E3", "50%", "1,23",
          "1,2345", "12,34,567", "$1,234.50", "USD -7.5", "-", "Price 12", "1.2.3",
          "1,234.5.6", " 7 ", "", "Price - 12", "3eggs"))),
        number_euro=q(parse_number(c("1.234,5", "€1.234,5", "1,5"), locale=euro)),
        double=q(parse_double(c("1,234", "1234", " 1e3", "1e+3", "1E-2", ".5", "5.", "+5",
          "-.5", "Inf", "inf", "NaN", "1 234", "", "1.2.3"))),
        double_euro=q(parse_double(c("1,5", "1.234,5", "1e3", ",5"), locale=euro)),
        integer=q(parse_integer(c("1,234", "+5", "007", "1e3", "-0", " 12 ", "1.0", ""))),
        logical=q(parse_logical(c("True", "TRUE", "yes", "Y", "N", "no", " true", "F", "0")))
      )
    }
  },
  readr_rows = {
    if (!requireNamespace("readr", quietly=TRUE)) {
      list(skip=TRUE, reason="readr is not installed")
    } else {
      suppressPackageStartupMessages(library(readr))
      q <- function(x) suppressWarnings(suppressMessages(x))
      rows <- function(text, ...) {
        q(problems(q(read_csv(I(text), col_types=cols(model=col_character(), hp=col_integer()), ...))))$row
      }
      list(
        rows_plain=rows("model,hp\nMazda RX4,110\nValiant,abc\nDatsun 710,2.5\n"),
        rows_skip=rows("junk line\nmodel,hp\nMazda RX4,110\nValiant,abc\n", skip=1),
        rows_comment=rows("model,hp\n# note\nMazda RX4,110\n# note\nValiant,abc\n", comment="#")
      )
    }
  },
  lubridate_grid = {
    if (!requireNamespace("lubridate", quietly=TRUE)) {
      list(skip=TRUE, reason="lubridate is not installed")
    } else {
      suppressPackageStartupMessages(library(lubridate))
      day_text <- function(x) format(x, "%Y-%m-%d")
      q <- function(x) suppressWarnings(x)
      d <- as.Date(c("2010-01-31", "2010-01-30", "2010-01-01", "2010-04-01", "2010-02-17"))
      list(
        ymd=day_text(q(ymd(c("2010-01-31T10:00:00", "2010-01-31 10:00", "2010 Jan 31", "2010-jan-31")))),
        mdy=day_text(q(mdy(c("January 31, 2010", "jan 31 2010", "Sept 1 2010", "31 Jan 2010")))),
        dmy=day_text(q(dmy(c("31 Jan 2010", "31-January-2010", "1st March 2010")))),
        ym=day_text(q(ym(c("2010 Feb", "2010-February")))),
        ceiling_week=day_text(ceiling_date(d, "week")),
        ceiling_month=day_text(ceiling_date(d, "month")),
        ceiling_quarter=day_text(ceiling_date(d, "quarter")),
        ceiling_year=day_text(ceiling_date(d, "year")),
        floor_week=day_text(floor_date(d, "week"))
      )
    }
  },
  factor_sort = {
    payload <- function(d) list(
      values=as.character(d$f),
      rows=d$row,
      levels=levels(d$f),
      ordered=is.ordered(d$f)
    )
    f <- factor(c("a", "b", "a", NA, "b"), levels=c("b", "a", "unused"), ordered=TRUE)
    d <- tibble(f=f, row=1:5)
    u <- tibble(f=factor(c("a", "b", "a", NA, "b"), levels=c("b", "a", "unused")), row=1:5)
    empty <- tibble(f=factor(character(), levels=c("b", "a", "unused"), ordered=TRUE), row=integer())
    missing <- tibble(f=factor(c(NA, NA), levels=c("b", "a")), row=1:2)
    list(
      ascending=payload(arrange(d, f)),
      descending=payload(arrange(d, desc(f))),
      grouped=payload(ungroup(arrange(group_by(d, f), row, .by_group=TRUE))),
      unordered=payload(arrange(u, f)),
      empty=payload(empty),
      missing=payload(arrange(missing, f))
    )
  },
  string_equal = {
    if (!requireNamespace("stringr", quietly=TRUE)) {
      list(skip=TRUE, reason="stringr is not installed")
    } else {
      tibble(
        x=c("\u00e9", "\u00c9", "e", NA, "a", "\u212b"),
        y=c("e\u0301", "e\u0301", "e", NA, "b", "\u00c5")
      ) %>% mutate(
        equal=stringr::str_equal(x, y),
        ignore=stringr::str_equal(x, y, ignore_case=TRUE)
      )
    }
  },
  janitor_cleaning = {
    if (!requireNamespace("janitor", quietly=TRUE)) {
      list(skip=TRUE, reason="janitor is not installed")
    } else {
      fmt_date <- function(x) format(x, "%Y-%m-%d")
      fmt_time <- function(x) format(x, "%Y-%m-%dT%H:%M:%OS6")
      suppressWarnings(suppressMessages({
        cleaning <- tibble(
          a=c(1, NA_real_, 2, NA_real_),
          b=c(NA_real_, NA_real_, NA_real_, NA_real_),
          c=c("x", NA_character_, "", "y"),
          d=c(NA_real_, NA_real_, 1, NA_real_)
        )
        constant <- tibble(
          mix=c(1, 1, NA_real_), vary=c(1, 2, 1),
          const=c(5, 5, 5), allna=c(NA_real_, NA_real_, NA_real_)
        )
        zero <- tibble(a=numeric(), b=character())
        old <- tibble(Z=1.5, A=2.5, B=1L)
        new <- tibble(M=3.5, A=4.5, B=5.0, C="x")
        one <- tibble(Z=1.5, A=1L)
        fac <- tibble(f=factor(c("y", "x"), levels=c("y", "x"), ordered=TRUE))
        headers <- tibble(a=c("id", "1"), b=c("code", "x"))
        numbers <- tibble(a=c(1.0, 9))
        flags <- tibble(a=c(TRUE, FALSE))
        sheet <- tibble(a=c(NA, "h1", "1"), b=c("a", "h2", "2"))
        seek <- tibble(a=c("skip", "header", "data"), b=c("no", "yes", "row"))
        list(
          empty=janitor::remove_empty(cleaning, which=c("rows", "cols")),
          rows_cutoff=janitor::remove_empty(cleaning, which="rows", cutoff=0.5),
          cols_cutoff=janitor::remove_empty(cleaning, which="cols", cutoff=0.5),
          constant=janitor::remove_constant(constant),
          constant_na=janitor::remove_constant(constant, na.rm=TRUE),
          zero_cols=janitor::remove_empty(zero, which="cols"),
          zero_constant=janitor::remove_constant(zero),
          classes=janitor::compare_df_cols(old=old, new=new),
          matched=janitor::compare_df_cols(old=old, new=new, return="match"),
          mismatched=janitor::compare_df_cols(old=old, new=new, return="mismatch"),
          bound=janitor::compare_df_cols(old=old, new=new, return="mismatch", bind_method="rbind"),
          one_classes=janitor::compare_df_cols(one=one),
          strict=janitor::compare_df_cols(fac=fac, strict_description=TRUE),
          loose=janitor::compare_df_cols(fac=fac, strict_description=FALSE),
          same_bind=janitor::compare_df_cols_same(data.frame(A=1), data.frame(B=2), verbose=FALSE),
          same_rbind=janitor::compare_df_cols_same(data.frame(A=1), data.frame(B=2), bind_method="rbind", verbose=FALSE),
          same_one=janitor::compare_df_cols_same(one, verbose=FALSE),
          headers=janitor::row_to_names(headers, 1),
          numbers=janitor::row_to_names(numbers, 1),
          flags=janitor::row_to_names(flags, 1),
          found=janitor::row_to_names(sheet, "find_header"),
          complete=janitor::find_header(seek),
          header_value=janitor::find_header(seek, "header"),
          by_name=janitor::find_header(seek, yes="b"),
          by_pos=janitor::find_header(seek, yes=2),
          excel_modern=tibble(value=fmt_date(janitor::excel_numeric_to_date(c(-1, 0, 1, 59, 60, 61, 42370, 40000, 40000.5)))),
          excel_mac=tibble(value=fmt_date(janitor::excel_numeric_to_date(c(0, 1, 60, 40908), date_system="mac pre-2011"))),
          excel_time=tibble(value=fmt_time(janitor::excel_numeric_to_date(c(40000.5, 40000.521, 60.5, 1.999999999), include_time=TRUE))),
          excel_noround=tibble(value=fmt_time(janitor::excel_numeric_to_date(40000.521, include_time=TRUE, round_seconds=FALSE)))
        )
      }))
    }
  },
  factor_combine = {
    if (!requireNamespace("forcats", quietly=TRUE)) {
      list(skip=TRUE, reason="forcats is not installed")
    } else {
      suppressPackageStartupMessages(library(forcats))
      q <- function(x) suppressMessages(suppressWarnings(x))
      # One column described by its type, values, and levels, so factor
      # metadata is compared, not only the values.
      describe <- function(x) list(
        type=if (is.factor(x)) "factor" else class(x)[[1]],
        values=I(as.character(x)),
        levels=I(if (is.factor(x)) levels(x) else character())
      )
      p <- tibble(species=c("Adelie", "Gentoo", "Adelie", "Chinstrap", NA),
                  mass=c(3750, 5000, 3800, 3700, 4000)) |>
        mutate(species=fct_infreq(species))
      text_lookup <- tibble(species=c("Adelie", "Gentoo", "Emperor"), w=c(1, 2, 3))
      other_levels <- tibble(species=factor(c("Gentoo", "Emperor"), levels=c("Gentoo", "Emperor")), w=c(2, 3))
      same_levels <- tibble(species=factor(c("Gentoo", "Adelie"), levels=levels(p$species)), w=c(2, 1))
      emperor <- tibble(species=fct_rev(factor("Emperor")), mass=1)
      years <- tibble(species=factor(c("Adelie", "Gentoo", NA), levels=c("Gentoo", "Chinstrap", "Adelie")),
                      year=c(2007, 2008, 2008), n=c(1, 2, 3))
      key <- function(x, name="species") describe(x[[name]])
      list(
        left_text=key(q(left_join(p, text_lookup, by="species"))),
        left_text_w=I(q(left_join(p, text_lookup, by="species"))$w),
        inner_text=key(q(inner_join(p, text_lookup, by="species"))),
        full_text=key(q(full_join(p, text_lookup, by="species"))),
        right_text=key(q(right_join(text_lookup, p, by="species"))),
        semi_text=key(q(semi_join(p, text_lookup, by="species"))),
        anti_text=key(q(anti_join(p, text_lookup, by="species"))),
        left_other=key(q(left_join(p, other_levels, by="species"))),
        full_other=key(q(full_join(p, other_levels, by="species"))),
        left_same=key(q(left_join(p, same_levels, by="species"))),
        bind_factors=key(q(bind_rows(p, emperor))),
        bind_factors_reversed=key(q(bind_rows(emperor, p))),
        bind_text=key(q(bind_rows(p, tibble(species="Emperor", mass=1)))),
        bind_missing_column=key(q(bind_rows(p, tibble(mass=1)))),
        complete_species_year=key(q(complete(years, species, year))),
        complete_species_year_n=I(q(complete(years, species, year))$n),
        complete_species=key(q(complete(years, species))),
        complete_nesting=key(q(complete(years, nesting(species), year))),
        expand_species=key(q(expand(years, species))),
        complete_grouped=key(q(years |> group_by(year) |> complete(species) |> ungroup()))
      )
    }
  },
  right_join_order = {
    q <- function(x) suppressMessages(suppressWarnings(x))
    x <- tibble(island=c("Torgersen", "Biscoe", "Dream", "Biscoe"), from_x=c("x1", "x2", "x3", "x4"))
    y <- tibble(island=c("Biscoe", "Anvers", "Torgersen", "Biscoe", NA), from_y=c("y1", "y2", "y3", "y4", "y5"))
    list(
      plain=q(right_join(x, y, by="island")),
      # tidy3 spells dplyr's .x / .y suffixes as _x / _y.
      keep=q(right_join(x, y, by="island", keep=TRUE, suffix=c("_x", "_y"))),
      first=q(right_join(x, y, by="island", multiple="first")),
      by_spec=q(right_join(x, y, by=join_by(island))),
      swapped=q(right_join(y, x, by="island"))
    )
  },
  many_groups = {
    if (!requireNamespace("forcats", quietly=TRUE)) {
      list(skip=TRUE, reason="forcats is not installed")
    } else {
      suppressPackageStartupMessages(library(forcats))
      q <- function(x) suppressMessages(suppressWarnings(x))
      # Arithmetic data, identical in R and Python: ~100 groups, ties, NAs.
      id <- 1:900
      p <- tibble(
        id=id,
        g=(id * 7L) %% 97L,
        species=c("Adelie", "Gentoo", "Chinstrap", "Emperor", "King")[(id * 3) %% 5 + 1],
        mass=ifelse(id %% 23 == 0, NA, ((id * 13) %% 50) * 100),
        bill=(id * 17) %% 41 + 0.5
      )
      levels_of <- function(x) list(values=I(as.character(x)), levels=I(levels(x)))
      # tidy3 lists grouped slice rows in first-appearance group order and
      # dplyr in sorted group order; compare which rows, ordered by id.
      by_id <- function(x) x |> ungroup() |> arrange(id)
      list(
        slice_max=by_id(q(p |> group_by(g) |> slice_max(mass, n=1, na_rm=TRUE))),
        slice_max_no_ties=by_id(q(p |> group_by(g) |> slice_max(mass, n=2, with_ties=FALSE, na_rm=TRUE))),
        slice_min_prop=by_id(q(p |> group_by(g) |> slice_min(bill, prop=0.3))),
        slice_head=by_id(q(p |> group_by(g) |> slice_head(n=2))),
        slice_tail=by_id(q(p |> group_by(g) |> slice_tail(n=1))),
        slice_positions=by_id(q(p |> group_by(g) |> slice(c(1, 3)))),
        reframe=q(p |> reframe(q=median(mass, na.rm=TRUE), k=n(), .by=g) |> arrange(g)),
        nth=q(p |> group_by(g) |> summarise(a=nth(mass, 2), b=nth(bill, -1), c=nth(species, 2, order_by=bill)) |> ungroup()),
        filter=q(p |> group_by(g) |> filter(bill > mean(bill)) |> ungroup()),
        reorder=levels_of(q(p |> group_by(g) |> mutate(s=fct_reorder(species, bill)) |> ungroup())$s),
        reorder_desc=levels_of(q(p |> group_by(g) |> mutate(s=fct_reorder(species, bill, .desc=TRUE)) |> ungroup())$s),
        infreq=levels_of(q(p |> group_by(g) |> mutate(s=fct_infreq(species)) |> ungroup())$s),
        relevel=levels_of(q(p |> group_by(g) |> mutate(s=fct_relevel(species, "King", after=1)) |> ungroup())$s),
        rev=levels_of(q(p |> group_by(g) |> mutate(s=fct_rev(species)) |> ungroup())$s)
      )
    }
  },
  stop(sprintf("unknown oracle case: %s", case_name))
)

cat(toJSON(normalise(result), auto_unbox=TRUE, na="null", null="null", digits=15))
