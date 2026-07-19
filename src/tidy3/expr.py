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

import operator
from typing import Any

import polars as pl

__all__ = [
    "Expr", "col", "desc", "n", "mean", "sum", "min", "max",
    "median", "std", "first", "last", "row_number", "min_rank",
    "dense_rank", "percent_rank", "cume_dist", "ntile", "lead", "lag",
    "cummean", "cumall", "cumany", "n_distinct", "coalesce", "if_else",
    "case_when", "to_polars",
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
    def f(x: "str | Expr | pl.Expr") -> Expr:
        aggregate = getattr(x, "_tidy3_aggregate", None)
        if callable(aggregate):
            return aggregate(name)
        base = col(x) if isinstance(x, str) else x
        return Expr(("call", name, _node(base), (), {}))

    f.__name__ = name
    f.__doc__ = f"Aggregate: ``{name}`` of a column or expression."
    return f


mean = _agg("mean")
sum = _agg("sum")  # noqa: A001
min = _agg("min")  # noqa: A001
max = _agg("max")  # noqa: A001
median = _agg("median")
std = _agg("std")
first = _agg("first")
last = _agg("last")


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


def lead(x: Any, n: int = 1, default: Any = None) -> Expr:
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("lead() n must be a non-negative integer")
    return _func("lead", col(x) if isinstance(x, str) else x, n=n, default=default)


def lag(x: Any, n: int = 1, default: Any = None) -> Expr:
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("lag() n must be a non-negative integer")
    return _func("lag", col(x) if isinstance(x, str) else x, n=n, default=default)


def cummean(x: Any) -> Expr:
    return _func("cummean", col(x) if isinstance(x, str) else x)


def cumall(x: Any) -> Expr:
    return _func("cumall", col(x) if isinstance(x, str) else x)


def cumany(x: Any) -> Expr:
    return _func("cumany", col(x) if isinstance(x, str) else x)


def n_distinct(x: Any, *, na_rm: bool = False) -> Expr:
    return _func(
        "n_distinct", col(x) if isinstance(x, str) else x, na_rm=na_rm
    )


def coalesce(*values: Any) -> Expr:
    if not values:
        raise TypeError("coalesce() requires at least one value")
    return _func("coalesce", *values)


def if_else(condition: Any, true: Any, false: Any, *, missing: Any = None) -> Expr:
    return _func("if_else", condition, true, false, missing=missing)


def case_when(*cases: tuple[Any, Any], default: Any = None) -> Expr:
    if not cases:
        raise TypeError("case_when() requires at least one (condition, value) pair")
    if any(not isinstance(case, tuple) or len(case) != 2 for case in cases):
        raise TypeError("case_when() cases must be (condition, value) pairs")
    return Expr(
        (
            "case_when",
            tuple((_node(condition), _node(value)) for condition, value in cases),
            _node(default),
        )
    )


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
        args = tuple(_compile_pl(arg) for arg in raw_args)
        kwargs = {key: _compile_pl(value) for key, value in raw_kwargs.items()}
        literal_kwargs = {
            key: value[1] if value[0] == "lit" else kwargs[key]
            for key, value in raw_kwargs.items()
        }
        if name == "row_number":
            return (
                args[0].rank(method="ordinal")
                if args
                else pl.int_range(1, pl.len() + 1)
            )
        if name in {"min_rank", "dense_rank"}:
            method = "min" if name == "min_rank" else "dense"
            return args[0].rank(method=method)
        if name in {"percent_rank", "cume_dist", "ntile"}:
            value = args[0]
            count = value.count()
            if name == "percent_rank":
                return (value.rank(method="min") - 1) / (count - 1)
            if name == "cume_dist":
                return value.rank(method="max") / count
            return (
                ((value.rank(method="ordinal") - 1) * literal_kwargs["n"] / count)
                .floor()
                + 1
            ).cast(pl.UInt32)
        if name in {"lead", "lag"}:
            periods = (
                -literal_kwargs["n"]
                if name == "lead"
                else literal_kwargs["n"]
            )
            return args[0].shift(periods, fill_value=kwargs["default"])
        if name == "cummean":
            value = args[0]
            return value.cum_sum() / value.is_not_null().cast(pl.UInt32).cum_sum()
        if name == "cumall":
            return args[0].cum_min()
        if name == "cumany":
            return args[0].cum_max()
        if name == "n_distinct":
            value = (
                args[0].drop_nulls() if literal_kwargs["na_rm"] else args[0]
            )
            return value.n_unique()
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
        return getattr(_compile_pl(base), name)(*args, **kwargs)
    raise ValueError(f"unknown expression node: {node!r}")
