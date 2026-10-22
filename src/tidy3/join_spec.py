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
    left: Any,
    right_lower: Any,
    right_upper: Any,
    *,
    bounds: str = "[]",
) -> Any:
    if (
        not isinstance(left, str)
        or not isinstance(right_lower, str)
        or not isinstance(right_upper, str)
    ):
        from tidy3.expr import between_values

        value = left
        if isinstance(left, str):
            from tidy3.expr import col

            value = col(left)
        return between_values(
            value, right_lower, right_upper, bounds=bounds
        )
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


JOIN_SUFFIX = ("_x", "_y")
# Internal name for a right column that clashes, renamed by join_names().
RIGHT_TAG = "__tidy3_y"


def _add_suffixes(names: list[str], others: list[str], suffix: str) -> list[str]:
    """dplyr's add_suffixes(): suffix *names* until none repeats *others*."""
    if suffix == "":
        return list(names)
    combined = [*others, *names]
    while True:
        seen: set[str] = set()
        repeats = []
        for index, name in enumerate(combined):
            if name in seen:
                repeats.append(index)
            seen.add(name)
        if not repeats:
            break
        for index in repeats:
            combined[index] = combined[index] + suffix
    return combined[len(others):]


def join_names(
    left: list[str],
    right: list[str],
    left_keys: list[str],
    right_keys: list[str],
    keep: bool,
    suffix: tuple[str, str] = JOIN_SUFFIX,
) -> tuple[dict[str, str], dict[str, str]]:
    """Output names of a join's left and right columns, by dplyr's rule.

    Columns in both tables get ``suffix`` ("_x"/"_y" here, dplyr's ".x"/".y"
    in Python spelling). Join keys are left alone unless ``keep=True``.
    """
    left_suffix, right_suffix = suffix
    if keep:
        left_out = _add_suffixes(left, right, left_suffix)
    else:
        ignore = [name for name in left if name in left_keys]
        check = [name for name in left if name not in left_keys]
        aux = [name for name in right if name not in right_keys]
        renamed = dict(zip(check, _add_suffixes(check, [*ignore, *aux], left_suffix)))
        left_out = [renamed.get(name, name) for name in left]
    right_out = _add_suffixes(right, left, right_suffix)
    return dict(zip(left, left_out)), dict(zip(right, right_out))
