"""Python-friendly join specifications corresponding to dplyr ``join_by()``."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class JoinCondition:
    left: str
    operator: str
    right: str
    rolling: bool = False

    def __post_init__(self):
        if not self.left or not self.right:
            raise ValueError("join columns must be non-empty names")
        if self.operator not in {"==", ">", ">=", "<", "<="}:
            raise ValueError(f"unsupported join operator: {self.operator!r}")


@dataclass(frozen=True)
class JoinConditions:
    conditions: tuple[JoinCondition, ...]


@dataclass(frozen=True)
class JoinSpec:
    conditions: tuple[JoinCondition, ...]

    @property
    def equality(self) -> list[JoinCondition]:
        return [condition for condition in self.conditions if condition.operator == "=="]

    @property
    def inequality(self) -> list[JoinCondition]:
        return [condition for condition in self.conditions if condition.operator != "=="]


def _condition(value: Any) -> JoinCondition:
    if isinstance(value, JoinCondition):
        return value
    if isinstance(value, str):
        return JoinCondition(value, "==", value)
    if isinstance(value, tuple):
        if len(value) == 2:
            return JoinCondition(str(value[0]), "==", str(value[1]))
        if len(value) == 3:
            return JoinCondition(str(value[0]), str(value[1]), str(value[2]))
    raise TypeError(
        "join_by() conditions must be names, (left, right), "
        "(left, operator, right), or join helper results"
    )


def join_by(*conditions: Any) -> JoinSpec:
    if not conditions:
        raise TypeError("join_by() requires at least one condition")
    expanded: list[JoinCondition] = []
    for condition in conditions:
        if isinstance(condition, JoinConditions):
            expanded.extend(condition.conditions)
        else:
            expanded.append(_condition(condition))
    rolling = [condition for condition in expanded if condition.rolling]
    if len(rolling) > 1:
        raise ValueError("join_by() supports at most one closest() condition")
    return JoinSpec(tuple(expanded))


def eq(left: str, right: str | None = None) -> JoinCondition:
    return JoinCondition(left, "==", right or left)


def gt(left: str, right: str) -> JoinCondition:
    return JoinCondition(left, ">", right)


def ge(left: str, right: str) -> JoinCondition:
    return JoinCondition(left, ">=", right)


def lt(left: str, right: str) -> JoinCondition:
    return JoinCondition(left, "<", right)


def le(left: str, right: str) -> JoinCondition:
    return JoinCondition(left, "<=", right)


def closest(condition: Any) -> JoinCondition:
    parsed = _condition(condition)
    if parsed.operator == "==":
        raise ValueError("closest() requires an inequality condition")
    return replace(parsed, rolling=True)


def _bounds(bounds: str) -> tuple[bool, bool]:
    if bounds not in {"[]", "[)", "(]", "()"}:
        raise ValueError("bounds must be one of '[]', '[)', '(]', or '()'")
    return bounds[0] == "[", bounds[1] == "]"


def between(
    left: str,
    right_lower: str,
    right_upper: str,
    *,
    bounds: str = "[]",
) -> JoinConditions:
    lower, upper = _bounds(bounds)
    return JoinConditions(
        (
            JoinCondition(left, ">=" if lower else ">", right_lower),
            JoinCondition(left, "<=" if upper else "<", right_upper),
        )
    )


def within(
    left_lower: str,
    left_upper: str,
    right_lower: str,
    right_upper: str,
) -> JoinConditions:
    return JoinConditions(
        (
            JoinCondition(left_lower, ">=", right_lower),
            JoinCondition(left_upper, "<=", right_upper),
        )
    )


def overlaps(
    left_lower: str,
    left_upper: str,
    right_lower: str,
    right_upper: str,
    *,
    bounds: str = "[]",
) -> JoinConditions:
    inclusive = bounds == "[]"
    _bounds(bounds)
    return JoinConditions(
        (
            JoinCondition(left_lower, "<=" if inclusive else "<", right_upper),
            JoinCondition(left_upper, ">=" if inclusive else ">", right_lower),
        )
    )


__all__ = [
    "JoinCondition",
    "JoinConditions",
    "JoinSpec",
    "between",
    "closest",
    "eq",
    "ge",
    "gt",
    "join_by",
    "le",
    "lt",
    "overlaps",
    "within",
]
