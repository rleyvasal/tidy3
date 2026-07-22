"""R-style bare-name / backtick column masking (Jupyter layer only).

Two-phase design (SolveIt / IPython)::

1. **Source preparser** turns backticks into a neutral sentinel::

       `hp new`  →  __tidy3_bt__("hp new")

2. **AST transformer** resolves the sentinel (and bare names) by verb context::

       filter(mpg > 20)           → filter(col("mpg") > 20)      # expr
       select(mpg, cyl)           → select("mpg", "cyl")         # selector
       mutate(x = `hp new` / cyl) → mutate(x = col("hp new") / col("cyl"))

Plain ``.py`` files are unchanged — use ``col("x")`` / strings there.
"""

from __future__ import annotations

import ast
import builtins
import re
from typing import Any, Iterable, Literal

# Sentinel / injected names (must match jupyter registration).
BT_NAME = "__tidy3_bt__"
COL_NAME = "__tidy3_col__"
# Spaced / odd *new* column names in mutate/select/rename:
#   `new hp` = expr  →  __tidy3_assign__("new hp", (expr))
ASSIGN_NAME = "__tidy3_assign__"

Mode = Literal["expr", "selector"]

# ── Verb classification (keep in sync with verbs.py / tidyr.py exports) ─────
# Positional args: data-mask *expressions* (bare name → col("name")).
_EXPR_ARG_VERBS = frozenset(
    {
        "filter",
        "filter_out",
        "arrange",
    }
)

# Keyword RHS (and **{"new col": expr}): new column *expressions*.
# Also supports `new col` = expr via source rewrite → **{...}.
_ASSIGN_VERBS = frozenset(
    {
        "mutate",
        "transmute",
        "summarise",
        "summarize",
        "reframe",
        "distinct",  # computed kwargs only; args handled separately
    }
)

# Positional args: column *selectors* (bare name → "name").
_SELECTOR_ARG_VERBS = frozenset(
    {
        "select",
        "drop",
        "relocate",
        "ungroup",
        "pull",
        "rename_with",
        "rowwise",
        "group_split",
        "group_nest",
        "group_map",
        "group_modify",
        # tidyr
        "drop_na",
        "fill",
        "expand",
        "complete",
        "nest",
        "pack",
        "unite",
        "separate",
        "separate_longer_delim",
        "separate_wider_delim",
        "pivot_longer",
        "pivot_wider",
        "unnest",
        "unnest_longer",
        "unnest_wider",
        "unpack",
        "hoist",
    }
)

_GROUP_VERBS = frozenset({"group_by"})
_COUNT_VERBS = frozenset({"count", "add_count", "tally", "add_tally"})
_RENAME_VERBS = frozenset({"rename"})  # values are old names (selectors)
_SLICE_ORDER_VERBS = frozenset({"slice_min", "slice_max"})
# by= selector; other args mostly positional ints / flags
_SLICE_BY_VERBS = frozenset(
    {
        "slice",
        "slice_head",
        "slice_tail",
        "slice_sample",
        "head",
        "sample_n",
        "sample_frac",
    }
)

# Keywords that always take selectors.
_SELECTOR_KW = frozenset(
    {
        "by",
        "before",
        "after",
        "name",  # often a *new* string name (passthrough) — see _NAME_KW
        "cols",
        "names_from",
        "values_from",
        "id_cols",
        "names_to",
        "values_to",
    }
)
# Keywords that are plain string labels (not column refs) — leave alone.
_NAME_KW = frozenset({"name", "names_sep", "names_prefix", "values_fn"})
# Keywords left alone (flags / enums / sizes).
_PASSTHROUGH_KW = frozenset(
    {
        "keep",
        "groups",
        "add",
        "drop",
        "sort",
        "na_rm",
        "with_ties",
        "n",
        "prop",
        "seed",
        "maintain_order",
        "keep_all",
        "replace",
        "remove",
        "names_sep",
        "names_prefix",
        "names_repair",
        "values_drop_na",
        "values_fill",
        "delim",
        "sep",
        "extra",
        "fill",
        "convert",
    }
)
# Keywords that are expressions (weights, order, …).
_EXPR_KW = frozenset({"wt", "order_by", "weight"})

_BT_RE = re.compile(r"`([^`\n]+)`")


def rewrite_backtick_keyword_assigns(source: str) -> str:
    """Rewrite R-style `` `new col` = expr `` to a positional assign sentinel.

    Python cannot use a call/spaced name as a keyword argument::

        mutate(`new hp` = hp)   # invalid after naive rewrite

    Becomes a *positional* call (safe next to other args)::

        mutate(__tidy3_assign__("new hp", (hp)))

    Runtime :class:`NamedAssign` objects are expanded inside mutate/select/rename.
    Uses ``=`` only (not ``==`` / ``!=`` / ``<=`` / ``>=``).
    """
    if "`" not in source or "=" not in source:
        return source
    out: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch != "`":
            out.append(ch)
            i += 1
            continue
        j = source.find("`", i + 1)
        if j < 0:
            out.append(source[i:])
            break
        name = source[i + 1 : j]
        k = j + 1
        while k < n and source[k] in " \t":
            k += 1
        is_assign = (
            k < n
            and source[k] == "="
            and not (k + 1 < n and source[k + 1] in "=~")
        )
        if is_assign:
            p = i - 1
            while p >= 0 and source[p] in " \t":
                p -= 1
            if p >= 0 and source[p] in "!<>":
                is_assign = False
        if not is_assign:
            out.append(source[i : j + 1])
            i = j + 1
            continue
        k += 1  # skip '='
        while k < n and source[k] in " \t":
            k += 1
        expr_start = k
        depth = 0
        in_str: str | None = None
        while k < n:
            c = source[k]
            if in_str is not None:
                if c == "\\" and k + 1 < n:
                    k += 2
                    continue
                if c == in_str:
                    in_str = None
                k += 1
                continue
            if c in ("'", '"'):
                in_str = c
                k += 1
                continue
            if c in "([{":
                depth += 1
            elif c in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif c == "," and depth == 0:
                break
            k += 1
        expr = source[expr_start:k].rstrip()
        # Positional sentinel — valid before/after other arguments.
        out.append(f"{ASSIGN_NAME}({name!r}, ({expr}))")
        i = k
    return "".join(out)


def rewrite_backticks(source: str) -> str:
    """Preparse backticks for Python.

    1. `` `new col` = expr `` → ``__tidy3_assign__("new col", (expr))``
    2. remaining `` `col` `` → ``__tidy3_bt__("col")``
    """
    source = rewrite_backtick_keyword_assigns(source)
    return _BT_RE.sub(lambda m: f"{BT_NAME}({m.group(1)!r})", source)


class NamedAssign:
    """Runtime carrier for `` `new col` = expr `` after source rewrite."""

    __slots__ = ("name", "value")

    def __init__(self, name: str, value: Any):
        self.name = str(name)
        self.value = value

    def __repr__(self) -> str:
        return f"NamedAssign({self.name!r}, {self.value!r})"


def make_named_assign(name: str, value: Any) -> NamedAssign:
    """Factory injected as ``__tidy3_assign__`` in the IPython user namespace."""
    return NamedAssign(name, value)


def _is_assign_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == ASSIGN_NAME
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    )


def tidy3_backtick_transform(lines: list[str]) -> list[str]:
    """IPython input transformer: backtick preparser."""
    if not lines:
        return lines
    src = "".join(lines)
    if "`" not in src:
        return lines
    out = rewrite_backticks(src)
    if out == src:
        return lines
    if out.endswith("\n"):
        return out.splitlines(keepends=True)
    parts = [ln + "\n" for ln in out.splitlines()]
    return parts or [out]


def _col_call(name: str) -> ast.Call:
    return ast.Call(
        func=ast.Name(id=COL_NAME, ctx=ast.Load()),
        args=[ast.Constant(value=name)],
        keywords=[],
    )


def _is_bt_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == BT_NAME
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    )


class MaskNames(ast.NodeTransformer):
    """Rewrite bare names / backtick sentinels inside one expression tree."""

    def __init__(self, mode: Mode, known: set[str]):
        self.mode = mode
        self.known = known

    def _replacement(self, name: str, old: ast.AST) -> ast.AST:
        if self.mode == "selector":
            new: ast.AST = ast.Constant(value=name)
        else:
            new = _col_call(name)
        return ast.copy_location(new, old)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id not in self.known:
            return self._replacement(node.id, node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if _is_bt_call(node):
            return self._replacement(str(node.args[0].value), node)

        # Keep function names (mean, if_else, n, …); rewrite receivers/args.
        if not isinstance(node.func, ast.Name):
            node.func = self.visit(node.func)
        node.args = [self.visit(arg) for arg in node.args]
        node.keywords = [
            ast.keyword(arg=kw.arg, value=self.visit(kw.value))
            for kw in node.keywords
        ]
        return node


def default_known_names(extra: Iterable[str] | None = None) -> set[str]:
    """Names that must not be rewritten to columns (funcs, builtins, sentinels)."""
    known = set(dir(builtins))
    known.update({BT_NAME, COL_NAME, "True", "False", "None"})
    # tidy3 helpers commonly used bare in expressions
    try:
        import tidy3 as t3

        for name in t3.__all__:
            if name.startswith("_"):
                continue
            known.add(name)
    except Exception:
        pass
    if extra:
        known.update(extra)
    return known


class Tidy3MaskTransformer(ast.NodeTransformer):
    """Top-level AST pass: choose expr vs selector masking per verb."""

    def __init__(self, known: set[str] | None = None):
        self._known_static = known

    def _known(self) -> set[str]:
        if self._known_static is not None:
            return set(self._known_static)
        extra: set[str] = set()
        try:
            from IPython import get_ipython

            ip = get_ipython()
            if ip is not None and getattr(ip, "user_ns", None) is not None:
                extra.update(ip.user_ns.keys())
        except Exception:
            pass
        return default_known_names(extra)

    def _mask(self, node: ast.AST, mode: Mode) -> ast.AST:
        return MaskNames(mode, self._known()).visit(node)

    def _mask_keyword(self, kw: ast.keyword, *, default: Mode) -> ast.keyword:
        """Apply masking rules for a single keyword by name."""
        if kw.arg is None:
            # bare **dict (rare); mask values with default mode
            if isinstance(kw.value, ast.Dict):
                values = [
                    self._mask(v, default) if v is not None else v
                    for v in kw.value.values
                ]
                return ast.keyword(
                    arg=None, value=ast.Dict(keys=kw.value.keys, values=values)
                )
            return kw
        if kw.arg in _PASSTHROUGH_KW or kw.arg in _NAME_KW:
            return kw
        if kw.arg in _EXPR_KW:
            return ast.keyword(arg=kw.arg, value=self._mask(kw.value, "expr"))
        if kw.arg in _SELECTOR_KW:
            return ast.keyword(arg=kw.arg, value=self._mask(kw.value, "selector"))
        return ast.keyword(arg=kw.arg, value=self._mask(kw.value, default))

    def _mask_assign_positional(self, node: ast.AST, *, value_mode: Mode) -> ast.AST:
        """Mask ``__tidy3_assign__("name", expr)`` value; leave name string."""
        if _is_assign_call(node):
            assert isinstance(node, ast.Call)
            node.args = [
                node.args[0],
                self._mask(node.args[1], value_mode),
                *node.args[2:],
            ]
            return node
        if value_mode == "selector":
            return self._mask(node, "selector")
        return self.visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if not isinstance(node.func, ast.Name):
            return self.generic_visit(node)

        verb = node.func.id

        if verb in _EXPR_ARG_VERBS:
            node.args = [self._mask(arg, "expr") for arg in node.args]
            node.keywords = [
                self._mask_keyword(kw, default="expr") for kw in node.keywords
            ]
            return node

        if verb in _ASSIGN_VERBS:
            # mutate/summarise: assign() value is expr; distinct pos = selectors
            if verb == "distinct":
                node.args = [
                    self._mask_assign_positional(a, value_mode="expr")
                    if _is_assign_call(a)
                    else self._mask(a, "selector")
                    for a in node.args
                ]
            else:
                node.args = [
                    self._mask_assign_positional(a, value_mode="expr")
                    for a in node.args
                ]
            node.keywords = [
                self._mask_keyword(kw, default="expr") for kw in node.keywords
            ]
            return node

        if verb == "select" or verb in _SELECTOR_ARG_VERBS:
            # select(`new` = old) → assign(value=selector); bare args = selectors
            node.args = [
                self._mask_assign_positional(a, value_mode="selector")
                if _is_assign_call(a)
                else self._mask(a, "selector")
                for a in node.args
            ]
            node.keywords = [
                self._mask_keyword(kw, default="selector") for kw in node.keywords
            ]
            return node

        if verb in _RENAME_VERBS:
            # rename(`new` = old) → assign; kwargs values are old names
            node.args = [
                self._mask_assign_positional(a, value_mode="selector")
                for a in node.args
            ]
            node.keywords = [
                self._mask_keyword(kw, default="selector") for kw in node.keywords
            ]
            return node

        if verb in _GROUP_VERBS:
            node.args = [
                self._mask_assign_positional(a, value_mode="expr")
                if _is_assign_call(a)
                else self._mask(a, "selector")
                for a in node.args
            ]
            node.keywords = [
                self._mask_keyword(kw, default="expr") for kw in node.keywords
            ]
            return node

        if verb in _COUNT_VERBS:
            node.args = [self._mask(arg, "selector") for arg in node.args]
            node.keywords = [
                self._mask_keyword(kw, default="expr") for kw in node.keywords
            ]
            return node

        if verb in _SLICE_ORDER_VERBS:
            node.args = [self._mask(arg, "expr") for arg in node.args]
            node.keywords = [
                self._mask_keyword(kw, default="expr") for kw in node.keywords
            ]
            return node

        if verb in _SLICE_BY_VERBS:
            node.args = [self.visit(a) for a in node.args]
            node.keywords = [
                self._mask_keyword(kw, default="selector") for kw in node.keywords
            ]
            return node

        return self.generic_visit(node)


def apply_masking(
    source: str,
    *,
    known: set[str] | None = None,
    backticks: bool = True,
) -> str:
    """Apply backtick rewrite + AST masking; return unparsed source (for tests)."""
    text = rewrite_backticks(source) if backticks else source
    tree = ast.parse(text)
    tree = Tidy3MaskTransformer(known=known or default_known_names()).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def is_mask_transformer(obj: Any) -> bool:
    return isinstance(obj, Tidy3MaskTransformer) or (
        type(obj).__name__ == "Tidy3MaskTransformer"
        and getattr(type(obj), "__module__", "").startswith("tidy3")
    )


def is_backtick_transformer(obj: Any) -> bool:
    return (
        getattr(obj, "__module__", "") in {"tidy3.masking", "tidy3.jupyter"}
        and getattr(obj, "__name__", "") == "tidy3_backtick_transform"
    )
