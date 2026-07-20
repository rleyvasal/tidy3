"""Backend-neutral column expressions.

``col("x") > 10`` builds a small AST. On the polars backend it compiles 1:1
to a ``pl.Expr`` (``to_polars``); on the pandas backend it is evaluated
directly against the DataFrame (``tidy3.pandas_engine``). Unknown methods
are recorded symbolically, so the full Polars expression API keeps working
under ``backend="polars"``, while the pandas backend supports a documented
subset and fails with a clear message otherwise.

Node forms (tuples): ("col", name) | ("lit", v) | ("pl", pl.Expr) |
("bin", op, l, r) | ("neg", x) | ("not", x) | ("n",) | ("desc", x) |
("call", name, base_node, args, kwargs)
"""

from __future__ import annotations

import builtins
import operator
from typing import Any

import polars as pl

__all__ = [
    "Expr", "col", "desc", "n", "mean", "sum", "min", "max",
    "median", "std", "sd", "var", "any", "all", "first", "last",
    "nth", "near", "na_if", "between_values", "consecutive_id",
    "row_number", "min_rank",
    "dense_rank", "percent_rank", "cume_dist", "ntile", "lead", "lag",
    "cummean", "cumall", "cumany", "n_distinct", "coalesce", "if_else",
    "case_when", "case_match", "recode", "cur_group_id", "n_groups", "to_polars",
]

_PL_BIN = {
    "+": operator.add, "-": operator.sub, "*": operator.mul,
    "/": operator.truediv, "//": operator.floordiv, "%": operator.mod,
    "**": operator.pow, "==": operator.eq, "!=": operator.ne,
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
    "&": operator.and_, "|": operator.or_,
}


def _node(v: Any):
    if isinstance(v, Expr):
        return v.node
    if isinstance(v, pl.Expr):
        return ("pl", v)
    return ("lit", v)


class Expr:
    """Deferred column expression; combine with operators, pipe into verbs."""

    __slots__ = ("node",)

    def __init__(self, node: tuple):
        self.node = node

    def __repr__(self) -> str:
        return f"<tidy3.Expr {self.node!r}>"

    def __bool__(self) -> bool:
        raise TypeError(
            "tidy3 Expr is not a boolean; use & | ~ (with parentheses), "
            "not `and`/`or`/`not`"
        )

    # ── operators ───────────────────────────────────────────────────────
    def _bin(self, op: str, other: Any, swap: bool = False) -> "Expr":
        l, r = (_node(other), self.node) if swap else (self.node, _node(other))
        return Expr(("bin", op, l, r))

    def __add__(self, o): return self._bin("+", o)
    def __radd__(self, o): return self._bin("+", o, swap=True)
    def __sub__(self, o): return self._bin("-", o)
    def __rsub__(self, o): return self._bin("-", o, swap=True)
    def __mul__(self, o): return self._bin("*", o)
    def __rmul__(self, o): return self._bin("*", o, swap=True)
    def __truediv__(self, o): return self._bin("/", o)
    def __rtruediv__(self, o): return self._bin("/", o, swap=True)
    def __floordiv__(self, o): return self._bin("//", o)
    def __rfloordiv__(self, o): return self._bin("//", o, swap=True)
    def __mod__(self, o): return self._bin("%", o)
    def __rmod__(self, o): return self._bin("%", o, swap=True)
    def __pow__(self, o): return self._bin("**", o)
    def __rpow__(self, o): return self._bin("**", o, swap=True)
    def __eq__(self, o): return self._bin("==", o)  # type: ignore[override]
    def __ne__(self, o): return self._bin("!=", o)  # type: ignore[override]
    def __lt__(self, o): return self._bin("<", o)
    def __le__(self, o): return self._bin("<=", o)
    def __gt__(self, o): return self._bin(">", o)
    def __ge__(self, o): return self._bin(">=", o)
    def __and__(self, o): return self._bin("&", o)
    def __rand__(self, o): return self._bin("&", o, swap=True)
    def __or__(self, o): return self._bin("|", o)
    def __ror__(self, o): return self._bin("|", o, swap=True)
    def __neg__(self): return Expr(("neg", self.node))
    def __invert__(self): return Expr(("not", self.node))

    __hash__ = None  # type: ignore[assignment]  # __eq__ builds an Expr

    # ── symbolic method forwarding (.mean(), .cum_sum(), .round(2), …) ──
    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def _call(*args: Any, **kwargs: Any) -> "Expr":
            return Expr(("call", name, self.node, args, kwargs))

        _call.__name__ = name
        return _call


# ── constructors / helpers (dplyr-style) ────────────────────────────────


def col(name: str) -> Expr:
    """Column reference."""
    return Expr(("col", name))


def desc(x: str | Expr | pl.Expr) -> Expr:
    """Descending sort key for ``arrange``."""
    return Expr(("desc", _node(col(x) if isinstance(x, str) else x)))


def n() -> Expr:
    """Row count (within group if grouped)."""
    return Expr(("n",))


def _agg(name: str):
    def f(x: "str | Expr | pl.Expr", *, na_rm: bool = False) -> Expr:
        if not isinstance(na_rm, bool):
            raise TypeError(f"{name}() na_rm must be a boolean")
        aggregate = getattr(x, "_tidy3_aggregate", None)
        if callable(aggregate):
            return aggregate(name)
        base = col(x) if isinstance(x, str) else x
        return Expr(("call", name, _node(base), (), {"na_rm": na_rm}))

    f.__name__ = name
    f.__doc__ = f"Aggregate: ``{name}`` of a column or expression."
    return f


mean = _agg("mean")
sum = _agg("sum")  # noqa: A001
min = _agg("min")  # noqa: A001
max = _agg("max")  # noqa: A001
median = _agg("median")
std = _agg("std")
sd = std
var = _agg("var")
any = _agg("any")  # noqa: A001
all = _agg("all")  # noqa: A001
_first_agg = _agg("first")
_last_agg = _agg("last")


def first(
    x: Any,
    *,
    order_by: Any = None,
    default: Any = None,
    na_rm: bool = False,
) -> Expr:
    if order_by is None and default is None:
        return _first_agg(x, na_rm=na_rm)
    return nth(
        x, 1, order_by=order_by, default=default, na_rm=na_rm
    )


def last(
    x: Any,
    *,
    order_by: Any = None,
    default: Any = None,
    na_rm: bool = False,
) -> Expr:
    if order_by is None and default is None:
        return _last_agg(x, na_rm=na_rm)
    return nth(
        x, -1, order_by=order_by, default=default, na_rm=na_rm
    )


def _func(name: str, *args: Any, **kwargs: Any) -> Expr:
    return Expr(
        (
            "func",
            name,
            tuple(_node(arg) for arg in args),
            {key: _node(value) for key, value in kwargs.items()},
        )
    )


def row_number(x: Any = None) -> Expr:
    """Sequential row number, or ordinal rank of *x*, within each group."""
    return _func("row_number") if x is None else _func("row_number", col(x) if isinstance(x, str) else x)


def min_rank(x: Any) -> Expr:
    return _func("min_rank", col(x) if isinstance(x, str) else x)


def dense_rank(x: Any) -> Expr:
    return _func("dense_rank", col(x) if isinstance(x, str) else x)


def percent_rank(x: Any) -> Expr:
    return _func("percent_rank", col(x) if isinstance(x, str) else x)


def cume_dist(x: Any) -> Expr:
    return _func("cume_dist", col(x) if isinstance(x, str) else x)


def ntile(x: Any, n: int) -> Expr:
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("ntile() n must be a positive integer")
    return _func("ntile", col(x) if isinstance(x, str) else x, n=n)


def lead(
    x: Any,
    n: int = 1,
    default: Any = None,
    *,
    order_by: Any = None,
) -> Expr:
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("lead() n must be a non-negative integer")
    order = col(order_by) if isinstance(order_by, str) else order_by
    return _func(
        "lead",
        col(x) if isinstance(x, str) else x,
        order,
        n=n,
        default=default,
    )


def lag(
    x: Any,
    n: int = 1,
    default: Any = None,
    *,
    order_by: Any = None,
) -> Expr:
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("lag() n must be a non-negative integer")
    order = col(order_by) if isinstance(order_by, str) else order_by
    return _func(
        "lag",
        col(x) if isinstance(x, str) else x,
        order,
        n=n,
        default=default,
    )


def cummean(x: Any) -> Expr:
    return _func("cummean", col(x) if isinstance(x, str) else x)


def cumall(x: Any) -> Expr:
    return _func("cumall", col(x) if isinstance(x, str) else x)


def cumany(x: Any) -> Expr:
    return _func("cumany", col(x) if isinstance(x, str) else x)


def n_distinct(*values: Any, na_rm: bool = False) -> Expr:
    if not values:
        raise TypeError("n_distinct() requires at least one value")
    return _func(
        "n_distinct",
        *(col(value) if isinstance(value, str) else value for value in values),
        na_rm=na_rm,
    )


def nth(
    x: Any,
    n: int,
    *,
    order_by: Any = None,
    default: Any = None,
    na_rm: bool = False,
) -> Expr:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("nth() n must be an integer")
    value = col(x) if isinstance(x, str) else x
    order = col(order_by) if isinstance(order_by, str) else order_by
    return _func(
        "nth",
        value,
        order,
        n=n,
        default=default,
        na_rm=na_rm,
    )


def near(x: Any, y: Any, *, tolerance: float = 1.490116e-8) -> Expr:
    if tolerance < 0:
        raise ValueError("near() tolerance must be non-negative")
    return _func("near", x, y, tolerance=tolerance)


def na_if(x: Any, y: Any) -> Expr:
    return _func("na_if", x, y)


def between_values(
    x: Any, left: Any, right: Any, *, bounds: str = "[]"
) -> Expr:
    if bounds not in {"[]", "[)", "(]", "()"}:
        raise ValueError("bounds must be one of '[]', '[)', '(]', or '()'")
    return _func("between", x, left, right, bounds=bounds)


def consecutive_id(*values: Any) -> Expr:
    if not values:
        raise TypeError("consecutive_id() requires at least one value")
    return _func(
        "consecutive_id",
        *(col(value) if isinstance(value, str) else value for value in values),
    )


def cur_group_id(groups: Any = None) -> Expr:
    """Return the 1-based identifier of the current group.

    ``groups`` is an internal context argument supplied by ``cur_group_id()``
    when used inside ``across``; callers normally invoke it without arguments.
    """
    return _func("cur_group_id", groups=tuple(groups or ()))


def n_groups(groups: Any = None) -> Expr:
    """Return the number of groups in the current grouped context."""
    return _func("n_groups", groups=tuple(groups or ()))


def coalesce(*values: Any) -> Expr:
    if not values:
        raise TypeError("coalesce() requires at least one value")
    return _func("coalesce", *values)


def if_else(condition: Any, true: Any, false: Any, *, missing: Any = None) -> Expr:
    return _func("if_else", condition, true, false, missing=missing)


def case_when(*cases: tuple[Any, Any], default: Any = None) -> Expr:
    if not cases:
        raise TypeError("case_when() requires at least one (condition, value) pair")
    if builtins.any(
        not isinstance(case, tuple) or len(case) != 2 for case in cases
    ):
        raise TypeError("case_when() cases must be (condition, value) pairs")
    return Expr(
        (
            "case_when",
            tuple((_node(condition), _node(value)) for condition, value in cases),
            _node(default),
        )
    )


def case_match(x: Any, *cases: tuple[Any, Any], default: Any = None) -> Expr:
    """Match values of ``x`` against scalar values or iterables.

    Each case is ``(values, replacement)``; a sequence on the left matches
    any of its values. Cases are evaluated in order and ``default`` is used
    when no case matches.
    """
    if not cases:
        raise TypeError("case_match() requires at least one case")
    value = col(x) if isinstance(x, str) else x
    normalized: list[tuple[Any, Any]] = []
    for case in cases:
        if not isinstance(case, tuple) or len(case) != 2:
            raise TypeError("case_match() cases must be (values, replacement) pairs")
        values, replacement = case
        if isinstance(values, (list, tuple, set, frozenset)):
            values = tuple(values)
        else:
            values = (values,)
        condition = None
        for candidate in values:
            current = value == candidate
            condition = current if condition is None else condition | current
        normalized.append((condition, replacement))
    return case_when(*normalized, default=default)


def recode(
    x: Any,
    mapping: dict[Any, Any],
    *,
    default: Any = None,
    missing: Any = None,
) -> Expr:
    """Recode values using a mapping, with optional default and missing value."""
    if not isinstance(mapping, dict) or not mapping:
        raise TypeError("recode() mapping must be a non-empty dictionary")
    value = col(x) if isinstance(x, str) else x
    cases = [(value_key, replacement) for value_key, replacement in mapping.items()]
    result = case_match(value, *cases, default=default)
    if missing is not None:
        result = if_else(value.is_null(), missing, result)
    return result


# ── polars compiler ─────────────────────────────────────────────────────


def to_polars(e: Any) -> Any:
    """Compile an Expr (or pass through pl.Expr / literals) to polars."""
    if isinstance(e, Expr):
        return _compile_pl(e.node)
    return e


def _compile_pl(node: tuple) -> Any:
    kind = node[0]
    if kind == "col":
        return pl.col(node[1])
    if kind == "lit":
        return pl.lit(node[1])
    if kind == "pl":
        return node[1]
    if kind == "n":
        return pl.len()
    if kind == "desc":
        raise ValueError("desc() is only valid inside arrange()")
    if kind == "neg":
        return -_compile_pl(node[1])
    if kind == "not":
        return ~_compile_pl(node[1])
    if kind == "bin":
        return _PL_BIN[node[1]](_compile_pl(node[2]), _compile_pl(node[3]))
    if kind == "horizontal":
        _, operation, columns = node
        expressions = [pl.col(name) for name in columns]
        if not expressions:
            identities = {"sum": 0, "any": False, "all": True}
            return pl.lit(identities.get(operation))
        if operation == "sum":
            return pl.sum_horizontal(expressions)
        if operation == "mean":
            return pl.mean_horizontal(expressions)
        if operation == "min":
            return pl.min_horizontal(expressions)
        if operation == "max":
            return pl.max_horizontal(expressions)
        if operation == "median":
            return pl.concat_list(expressions).list.median()
        if operation == "std":
            return pl.concat_list(expressions).list.std(ddof=1)
        if operation == "any":
            return pl.any_horizontal(expressions)
        if operation == "all":
            return pl.all_horizontal(expressions)
        if operation == "first":
            return expressions[0]
        if operation == "last":
            return expressions[-1]
        raise ValueError(f"unknown horizontal operation: {operation}")
    if kind == "column_set":
        _, columns, as_list = node
        expressions = [pl.col(name) for name in columns]
        if as_list:
            return pl.concat_list(expressions)
        return pl.struct(expressions)
    if kind == "func":
        _, name, raw_args, raw_kwargs = node
        descending = bool(
            name
            in {"row_number", "min_rank", "dense_rank", "percent_rank", "cume_dist", "ntile"}
            and raw_args
            and raw_args[0][0] == "desc"
        )
        if descending:
            raw_args = (raw_args[0][1], *raw_args[1:])
        args = tuple(_compile_pl(arg) for arg in raw_args)
        kwargs = {key: _compile_pl(value) for key, value in raw_kwargs.items()}
        literal_kwargs = {
            key: value[1] if value[0] == "lit" else kwargs[key]
            for key, value in raw_kwargs.items()
        }
        if name == "row_number":
            return (
                args[0].rank(method="ordinal", descending=descending)
                if args
                else pl.int_range(1, pl.len() + 1)
            )
        if name == "cur_group_id":
            group_names = literal_kwargs.get("groups", ())
            if not group_names:
                return pl.lit(1)
            return pl.struct([pl.col(group) for group in group_names]).rank(
                method="dense"
            )
        if name == "n_groups":
            group_names = literal_kwargs.get("groups", ())
            if not group_names:
                return pl.lit(1)
            return pl.struct([pl.col(group) for group in group_names]).n_unique()
        if name in {"min_rank", "dense_rank"}:
            method = "min" if name == "min_rank" else "dense"
            return args[0].rank(method=method, descending=descending)
        if name in {"percent_rank", "cume_dist", "ntile"}:
            value = args[0]
            count = value.count()
            if name == "percent_rank":
                return (
                    value.rank(method="min", descending=descending) - 1
                ) / (count - 1)
            if name == "cume_dist":
                return value.rank(method="max", descending=descending) / count
            return (
                (
                    (
                        value.rank(method="ordinal", descending=descending)
                        - 1
                    )
                    * literal_kwargs["n"]
                    / count
                )
                .floor()
                + 1
            ).cast(pl.UInt32)
        if name in {"lead", "lag"}:
            periods = (
                -literal_kwargs["n"]
                if name == "lead"
                else literal_kwargs["n"]
            )
            value, order = args
            raw_order = raw_args[1]
            if raw_order[0] != "lit" or raw_order[1] is not None:
                value = (
                    value.sort_by(order)
                    .shift(periods, fill_value=kwargs["default"])
                    .sort_by(order.arg_sort())
                )
                return value
            return value.shift(periods, fill_value=kwargs["default"])
        if name == "cummean":
            value = args[0]
            result = (
                value.cum_sum()
                / value.is_not_null().cast(pl.UInt32).cum_sum()
            )
            missing_seen = value.is_null().cast(pl.UInt8).cum_max() > 0
            return pl.when(missing_seen).then(None).otherwise(result)
        if name == "cumall":
            return args[0].cum_min()
        if name == "cumany":
            return args[0].cum_max()
        if name == "n_distinct":
            value = (
                pl.struct(
                    argument.alias(f"__tidy3_n_distinct_{index}")
                    for index, argument in enumerate(args)
                )
                if len(args) > 1
                else args[0]
            )
            if literal_kwargs["na_rm"]:
                if len(args) > 1:
                    value = value.filter(
                        pl.all_horizontal(
                            [argument.is_not_null() for argument in args]
                        )
                    )
                else:
                    value = value.drop_nulls()
            return value.n_unique()
        if name == "nth":
            value, order = args
            raw_order = raw_args[1]
            if raw_order[0] != "lit" or raw_order[1] is not None:
                value = value.sort_by(order, nulls_last=True)
            if literal_kwargs["na_rm"]:
                value = value.drop_nulls()
            position = literal_kwargs["n"]
            index = position - 1 if position > 0 else position
            result = value.get(index, null_on_oob=True)
            default = kwargs["default"]
            valid = (
                value.len() > index
                if position > 0
                else value.len() >= -position
            )
            if position == 0:
                valid = pl.lit(False)
            return pl.when(valid).then(result).otherwise(default)
        if name == "near":
            return (args[0] - args[1]).abs() <= kwargs["tolerance"]
        if name == "na_if":
            return pl.when(args[0] == args[1]).then(None).otherwise(args[0])
        if name == "between":
            value, left, right = args
            bounds = literal_kwargs["bounds"]
            lower = value >= left if bounds[0] == "[" else value > left
            upper = value <= right if bounds[1] == "]" else value < right
            return lower & upper
        if name == "consecutive_id":
            value = pl.struct(list(args)) if len(args) > 1 else args[0]
            changed = value.ne_missing(value.shift()) | (
                pl.int_range(pl.len()) == 0
            )
            return changed.cum_sum().cast(pl.UInt32)
        if name == "coalesce":
            return pl.coalesce(args)
        if name == "if_else":
            condition, true, false = args
            missing = kwargs["missing"]
            return (
                pl.when(condition.is_null())
                .then(missing)
                .when(condition)
                .then(true)
                .otherwise(false)
            )
        raise ValueError(f"unknown expression function: {name}")
    if kind == "case_when":
        _, raw_cases, raw_default = node
        expression = None
        for raw_condition, raw_value in raw_cases:
            condition = _compile_pl(raw_condition)
            value = _compile_pl(raw_value)
            expression = (
                pl.when(condition).then(value)
                if expression is None
                else expression.when(condition).then(value)
            )
        return expression.otherwise(_compile_pl(raw_default))
    if kind == "call":
        _, name, base, args, kwargs = node
        args = tuple(to_polars(a) if isinstance(a, Expr) else a for a in args)
        kwargs = {k: to_polars(v) if isinstance(v, Expr) else v for k, v in kwargs.items()}
        if name in {"mean", "sum", "min", "max", "median", "std", "var", "any", "all", "first", "last"}:
            na_rm = bool(kwargs.pop("na_rm", False))
            value = _compile_pl(base)
            if name in {"first", "last"}:
                if na_rm:
                    value = value.drop_nulls()
                return getattr(value, name)(*args, **kwargs)
            result = getattr(value, name)(*args, **kwargs)
            if na_rm:
                return result
            missing = value.is_null().any()
            if name == "any":
                return (
                    pl.when(result)
                    .then(True)
                    .when(missing)
                    .then(None)
                    .otherwise(False)
                )
            if name == "all":
                return (
                    pl.when(~result)
                    .then(False)
                    .when(missing)
                    .then(None)
                    .otherwise(True)
                )
            return pl.when(missing).then(None).otherwise(result)
        value = _compile_pl(base)
        result = getattr(value, name)(*args, **kwargs)
        if name in {"cum_sum", "cumsum", "cum_max", "cum_min", "cum_prod"}:
            missing_seen = value.is_null().cast(pl.UInt8).cum_max() > 0
            return pl.when(missing_seen).then(None).otherwise(result)
        return result
    raise ValueError(f"unknown expression node: {node!r}")
