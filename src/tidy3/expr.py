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
    "median", "std", "first", "last", "to_polars",
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
        return _compile_pl(node[1]).desc()
    if kind == "neg":
        return -_compile_pl(node[1])
    if kind == "not":
        return ~_compile_pl(node[1])
    if kind == "bin":
        return _PL_BIN[node[1]](_compile_pl(node[2]), _compile_pl(node[3]))
    if kind == "call":
        _, name, base, args, kwargs = node
        args = tuple(to_polars(a) if isinstance(a, Expr) else a for a in args)
        kwargs = {k: to_polars(v) if isinstance(v, Expr) else v for k, v in kwargs.items()}
        return getattr(_compile_pl(base), name)(*args, **kwargs)
    raise ValueError(f"unknown expression node: {node!r}")
