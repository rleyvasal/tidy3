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

Mode = Literal["expr", "selector"]

# Verbs whose *positional* args are data-mask expressions.
_EXPR_ARG_VERBS = frozenset({"filter", "filter_out", "arrange"})

# Verbs with keyword RHS expressions (new column assignments).
_ASSIGN_VERBS = frozenset(
    {"mutate", "transmute", "summarise", "summarize", "reframe"}
)

# Positional args are column selectors (strings).
_SELECTOR_ARG_VERBS = frozenset(
    {"select", "drop", "relocate", "ungroup", "pull", "rename_with"}
)

_GROUP_VERBS = frozenset({"group_by"})
_COUNT_VERBS = frozenset({"count", "add_count"})
_RENAME_VERBS = frozenset({"rename"})
_SLICE_ORDER_VERBS = frozenset({"slice_min", "slice_max"})

# Keywords that always take selectors even inside assign/filter verbs.
_SELECTOR_KW = frozenset({"by", "before", "after", "name"})
# Keywords left alone (flags / enums).
_PASSTHROUGH_KW = frozenset(
    {"keep", "groups", "add", "drop", "sort", "na_rm", "with_ties", "n", "prop"}
)

_BT_RE = re.compile(r"`([^`\n]+)`")


def rewrite_backtick_keyword_assigns(source: str) -> str:
    """Rewrite R-style `` `new col` = expr `` to valid Python ``**{"new col": (expr)}``.

    Python cannot use a call (or spaced name) as a keyword argument::

        mutate(`new hp` = hp)           # after naive backtick rewrite → SyntaxError
        mutate(**{"new hp": (hp)})      # valid; AST masking then rewrites *hp*

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
        # True assignment '=', not '==', '!=', '<=', '>=', ':='
        is_assign = (
            k < n
            and source[k] == "="
            and not (k + 1 < n and source[k + 1] in "=~")
            and not (k > 0 and source[k - 1] in "!<>")
        )
        # also reject if char before spaces-and-backtick chain is part of != 
        if is_assign:
            # scan back over whitespace from j's side already done; check char before `
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
        out.append(f"**{{{name!r}: ({expr})}}")
        i = k
    return "".join(out)


def rewrite_backticks(source: str) -> str:
    """Preparse backticks for Python.

    1. `` `new col` = expr `` → ``**{"new col": (expr)}`` (keyword assign)
    2. remaining `` `col` `` → ``__tidy3_bt__("col")`` (expression / selector)
    """
    source = rewrite_backtick_keyword_assigns(source)
    return _BT_RE.sub(lambda m: f"{BT_NAME}({m.group(1)!r})", source)


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

    def _mask_starstar(self, node: ast.AST) -> ast.AST:
        """Mask values inside ``**{...}`` dicts (keys stay string constants)."""
        if isinstance(node, ast.Dict):
            values = [
                self._mask(v, "expr") if v is not None else v for v in node.values
            ]
            return ast.Dict(keys=node.keys, values=values)
        return self._mask(node, "expr")

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Nested non-verb calls are handled inside MaskNames when we mask.
        if not isinstance(node.func, ast.Name):
            return self.generic_visit(node)

        verb = node.func.id

        if verb in _EXPR_ARG_VERBS:
            node.args = [self._mask(arg, "expr") for arg in node.args]
            node.keywords = [
                ast.keyword(
                    arg=kw.arg,
                    value=self._mask(kw.value, "selector")
                    if kw.arg in _SELECTOR_KW
                    else self._mask(kw.value, "expr")
                    if kw.arg not in _PASSTHROUGH_KW
                    else kw.value,
                )
                for kw in node.keywords
            ]
            return node

        if verb in _ASSIGN_VERBS:
            new_kw: list[ast.keyword] = []
            for kw in node.keywords:
                if kw.arg is None:
                    # **{"new hp": expr} from backtick keyword rewrite
                    value = self._mask_starstar(kw.value)
                elif kw.arg in _SELECTOR_KW:
                    value = self._mask(kw.value, "selector")
                elif kw.arg in _PASSTHROUGH_KW:
                    value = kw.value
                else:
                    value = self._mask(kw.value, "expr")
                new_kw.append(ast.keyword(arg=kw.arg, value=value))
            node.keywords = new_kw
            # rare positional assign specs
            node.args = [self.visit(a) for a in node.args]
            return node

        if verb in _SELECTOR_ARG_VERBS or verb == "select":
            node.args = [self._mask(arg, "selector") for arg in node.args]
            node.keywords = [
                ast.keyword(
                    arg=kw.arg,
                    value=self._mask(kw.value, "selector")
                    if kw.arg in _SELECTOR_KW or kw.arg is not None
                    else kw.value,
                )
                for kw in node.keywords
            ]
            return node

        if verb in _RENAME_VERBS:
            node.keywords = [
                ast.keyword(arg=kw.arg, value=self._mask(kw.value, "selector"))
                for kw in node.keywords
            ]
            node.args = [self.visit(a) for a in node.args]
            return node

        if verb in _GROUP_VERBS:
            node.args = [self._mask(arg, "selector") for arg in node.args]
            node.keywords = [
                ast.keyword(
                    arg=kw.arg,
                    value=kw.value
                    if kw.arg in {"add", "drop"}
                    else self._mask(kw.value, "expr"),
                )
                for kw in node.keywords
            ]
            return node

        if verb in _COUNT_VERBS:
            node.args = [self._mask(arg, "selector") for arg in node.args]
            node.keywords = [
                ast.keyword(
                    arg=kw.arg,
                    value=self._mask(kw.value, "expr")
                    if kw.arg == "wt"
                    else kw.value,
                )
                for kw in node.keywords
            ]
            return node

        if verb in _SLICE_ORDER_VERBS:
            node.args = [self._mask(arg, "expr") for arg in node.args]
            node.keywords = [
                ast.keyword(
                    arg=kw.arg,
                    value=self._mask(kw.value, "expr")
                    if kw.arg == "order_by"
                    else self._mask(kw.value, "selector")
                    if kw.arg in _SELECTOR_KW
                    else kw.value,
                )
                for kw in node.keywords
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
