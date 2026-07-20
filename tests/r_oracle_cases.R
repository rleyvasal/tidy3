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
        unname(lapply(as.list(column), normalise_value))
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
    ungrouped=base |> group_by(g) |> ungroup() |> summarise(n=n())
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
    add_tallied=base |> group_by(g) |> add_tally() |> ungroup()
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
    list(long=long, wide=long |> pivot_wider(names_from=name, values_from=value))
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
  duplicate_names = bind_cols(tibble(x=1L), tibble(x=2L)),
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
      wider=tibble(text=c("a|b", "c")) |> separate_wider_delim(text, names=c("left", "right"), delim="|")
    )
  },
  stop(sprintf("unknown oracle case: %s", case_name))
)

cat(toJSON(normalise(result), auto_unbox=TRUE, na="null", null="null", digits=15))
