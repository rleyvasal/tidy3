"""Export a tidy3 pipeline as plain Polars code.

:meth:`TidyFrame.to_polars_code` prints the lazy Polars query that tidy3
built, as a Python function that needs only Polars. The query is printed
as tidy3 compiled it, so the exported code returns the same result,
including dplyr's rules (a ``mean`` with a missing value is missing,
``arrange`` puts missing values last).

Two kinds of step cannot be exported, and raise :class:`ExportError`
naming the step:

- a column computed by a Python function (for example tidy3's text
  parsers and R-style ``round``), which has no plain Polars form;
- a step that read values from the data to build its query (factor
  levels, ``pivot_wider`` names, ``compute()``). Its query is right only
  for that data.

The printer reads Polars' serialized plan. Input frames become function
arguments: ``df``, then ``df2`` and so on, one per distinct input.
"""

from __future__ import annotations

import io
import json
import keyword
import warnings
from typing import Any

import polars as pl

__all__ = ["ExportError", "to_polars_code"]


class ExportError(ValueError):
    """The pipeline has a step with no plain-Polars form."""


def to_polars_code(frame: Any, *, name: str = "pipeline") -> str:
    """Python source of a function that runs *frame*'s pipeline with Polars only."""
    code, _inputs = _export(frame, name=name)
    return code


def _export(frame: Any, *, name: str = "pipeline") -> tuple[str, list[tuple[str, pl.DataFrame]]]:
    """The exported source and its inputs, as ``(argument name, frame)``."""
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"name must be a Python identifier, got {name!r}")
    reads = getattr(frame, "_data_reads", ())
    if reads:
        raise ExportError(
            "cannot export this pipeline: a step built its query from values in the data, "
            "so the query is right only for this data:\n  - " + "\n  - ".join(reads)
        )
    lf = frame._lf if hasattr(frame, "_lf") else frame
    with warnings.catch_warnings():
        # Polars marks JSON plan serialization deprecated; it is the only
        # readable form of the plan.
        warnings.simplefilter("ignore")
        try:
            plan = json.loads(lf.serialize(format="json"))
        except pl.exceptions.ComputeError as error:
            if "cloudpickle" not in str(error):
                raise
            raise ExportError(
                "cannot export this pipeline: Polars needs the cloudpickle package "
                "to read a plan with Python steps (pip install cloudpickle)"
            ) from None
    printer = _Printer()
    body = printer.plan(plan)
    args = ", ".join(f"{arg}: pl.DataFrame" for arg, _frame in printer.inputs)
    lines = ["import polars as pl"]
    if printer.uses_selectors:
        lines.append("import polars.selectors as cs")
    lines += ["", ""]
    lines.append(f"def {name}({args}) -> pl.DataFrame:")
    if printer.inputs:
        lines.append('    """Run the exported tidy3 pipeline.')
        lines.append("")
        for arg, data in printer.inputs:
            columns = ", ".join(f"{column} ({dtype})" for column, dtype in data.schema.items())
            lines.append(f"    {arg}: {columns or 'no columns'}")
        lines.append('    """')
    lines.append("    return (")
    lines.extend("        " + line for line in body.split("\n"))
    lines.append("        .collect()")
    lines.append("    )")
    return "\n".join(lines) + "\n", printer.inputs


# ── plans ───────────────────────────────────────────────────────────────


class _Printer:
    def __init__(self) -> None:
        self.inputs: list[tuple[str, pl.DataFrame]] = []
        self._by_bytes: dict[bytes, str] = {}
        self.uses_selectors = False
        # Schema of the frame the expressions being printed run against.
        self.current: dict[str, Any] = {}

    def unsupported(self, what: str) -> ExportError:
        return ExportError(f"cannot export this pipeline yet: {what} has no exporter")

    # A plan node is ``{"Kind": body}``; some are wrapped in ``{"IR": {"dsl": node}}``.
    def plan(self, node: Any) -> str:
        if isinstance(node, dict) and set(node) == {"IR"}:
            return self.plan(node["IR"]["dsl"])
        if not isinstance(node, dict) or len(node) != 1:
            raise self.unsupported(f"plan {str(node)[:60]}")
        (kind, body), = node.items()
        if kind == "Join" and "condition" in body:
            body = _legacy_join(body)
        method = getattr(self, "p_" + kind, None)
        if method is None:
            raise self.unsupported(f"the Polars plan step {kind}")
        return method(body)

    def chain(self, node: Any, call: Any) -> str:
        """*node*'s code, then ``.call``; *call* may be a function of the input schema."""
        source = self.plan(node)
        if callable(call):
            self.current = self.schema_of(source)
            call = call()
        return f"{source}\n.{call}"

    def schema_of(self, source: str) -> dict[str, Any]:
        """The schema of a printed plan, read lazily (no rows are computed)."""
        import polars.selectors as cs

        namespace = {"pl": pl, "cs": cs, **dict(self.inputs)}
        try:
            return dict(eval("(" + source + ")", namespace).collect_schema())
        except Exception:
            return {}

    def nested(self, node: Any) -> str:
        """A plan used as an argument (the right side of a join)."""
        source = self.plan(node)
        if "\n" not in source:
            return source
        return "(\n" + "\n".join("    " + line for line in source.split("\n")) + "\n)"

    def args(self, nodes: list[Any]) -> str:
        """Expressions as call arguments: no outer parentheses, one per line when long."""
        items = [_bare(self.expr(node)) for node in nodes]
        joined = ", ".join(items)
        if len(items) > 1 and (len(joined) > 80 or "\n" in joined):
            return "\n" + "".join(f"    {_indent(item)},\n" for item in items)
        return joined

    def names_or_exprs(self, nodes: list[Any]) -> str:
        """Plain columns by name, others as expressions, one per line when long."""
        items = [
            repr(name) if (name := _plain_column(node)) is not None else _bare(self.expr(node))
            for node in nodes
        ]
        joined = ", ".join(items)
        if len(items) > 1 and (len(joined) > 80 or "\n" in joined):
            return "\n" + "".join(f"    {_indent(item)},\n" for item in items)
        return joined

    def p_DataFrameScan(self, body: dict) -> str:
        raw = bytes(body["df"])
        data = pl.read_ipc_stream(io.BytesIO(raw))
        if data.width and all(column.startswith("__tidy3") for column in data.columns):
            # A lookup table tidy3 built from the call's arguments, not user data.
            return f"pl.LazyFrame({_frame_literal(data)})"
        name = self._by_bytes.get(raw)
        if name is None:
            name = "df" if not self.inputs else f"df{len(self.inputs) + 1}"
            self._by_bytes[raw] = name
            self.inputs.append((name, data))
        return f"{name}.lazy()"

    def p_Scan(self, body: dict) -> str:
        sources = body["sources"]
        paths = sources.get("Paths") if isinstance(sources, dict) else None
        if isinstance(paths, dict):
            paths = [paths]
        if isinstance(paths, list):
            paths = [path["inner"] if isinstance(path, dict) else path for path in paths]
        if not paths:
            raise self.unsupported("a scan of in-memory sources")
        target = paths[0] if len(paths) == 1 else paths
        scan_type = body["scan_type"]
        kind = next(iter(scan_type)) if isinstance(scan_type, dict) else scan_type
        if kind == "Parquet":
            return f"pl.scan_parquet({target!r})"
        if kind == "Ipc":
            return f"pl.scan_ipc({target!r})"
        if kind == "Csv":
            return f"pl.scan_csv({target!r}{self.csv_options(scan_type['Csv'], body)})"
        raise self.unsupported(f"a {kind} scan")

    def csv_options(self, csv: dict, body: dict) -> str:
        options = csv["options"]
        parse = options.get("parse_options", {})
        args: list[str] = []
        separator = chr(parse.get("separator", 44))
        if separator != ",":
            args.append(f"separator={separator!r}")
        if not options.get("has_header", True):
            args.append("has_header=False")
        if options.get("skip_rows"):
            args.append(f"skip_rows={options['skip_rows']}")
        comment = parse.get("comment_prefix")
        if comment:
            prefix = comment.get("Single") if isinstance(comment, dict) else comment
            prefix = chr(prefix) if isinstance(prefix, int) else prefix
            args.append(f"comment_prefix={prefix!r}")
        nulls = parse.get("null_values")
        if nulls:
            (kind, values), = nulls.items()
            args.append(f"null_values={values!r}")
        schema = options.get("schema")
        if schema:
            args.append(f"schema={self.schema(schema)}")
        overwrite = options.get("schema_overwrite")
        if overwrite:
            args.append(f"schema_overrides={self.schema(overwrite)}")
        if options.get("infer_schema_length") == 0:
            args.append("infer_schema=False")
        elif options.get("infer_schema_length") not in (None, 100):
            args.append(f"infer_schema_length={options['infer_schema_length']}")
        file_options = body.get("unified_scan_args") or body.get("file_options") or {}
        include = file_options.get("include_file_paths")
        if include:
            args.append(f"include_file_paths={include!r}")
        return "".join(", " + arg for arg in args)

    def schema(self, schema: dict) -> str:
        fields = schema.get("fields", schema)
        return "{" + ", ".join(f"{name!r}: {_dtype(dtype)}" for name, dtype in fields.items()) + "}"

    def p_Filter(self, body: dict) -> str:
        return self.chain(body["input"], lambda: f"filter({self.args([body['predicate']])})")

    def p_HStack(self, body: dict) -> str:
        return self.chain(body["input"], lambda: f"with_columns({self.args(body['exprs'])})")

    def p_Select(self, body: dict) -> str:
        source = body["input"]
        names = [_plain_column(node) for node in body["expr"]]
        inner = _unwrap(source)
        if all(names) and isinstance(inner, dict) and "Select" in inner:
            # select(a, b, c).select(a, b): the first projection adds nothing.
            earlier = [_plain_column(node) for node in inner["Select"]["expr"]]
            if all(earlier) and set(names) <= set(earlier):
                source = inner["Select"]["input"]
        if all(names):
            printed = self.plan(source)
            if list(self.schema_of(printed)) == names:
                return printed  # keeps every column in order: nothing to print
            self.current = self.schema_of(printed)
            return f"{printed}\n.select({self.names_or_exprs(body['expr'])})"
        return self.chain(source, lambda: f"select({self.names_or_exprs(body['expr'])})")

    def p_GroupBy(self, body: dict) -> str:
        if body.get("apply") is not None:
            raise self.unsupported("a group_by().map_groups() step")
        order = ", maintain_order=True" if body.get("maintain_order") else ""
        return self.chain(
            body["input"],
            lambda: f"group_by({self.names_or_exprs(body['keys'])}{order})\n.agg({self.args(body['aggs'])})",
        )

    def p_Sort(self, body: dict) -> str:
        return self.chain(body["input"], lambda: self.sort_call(body))

    def sort_call(self, body: dict) -> str:
        options = body["sort_options"]
        by = body["by_column"]
        keys = self.names_or_exprs(by)
        args = [keys if len(by) == 1 else f"[{keys}]"]
        for option in ("descending", "nulls_last"):
            values = options[option]
            if any(values):
                args.append(f"{option}={_flags(values)}")
        if options.get("maintain_order"):
            args.append("maintain_order=True")
        if options.get("multithreaded") is False:
            args.append("multithreaded=False")
        call = f"sort({', '.join(args)})"
        slice_ = body.get("slice")
        if slice_:
            offset, length = slice_[0], slice_[1]
            call += f"\n.slice({offset}, {length})"
        return call

    def p_Slice(self, body: dict) -> str:
        return self.chain(body["input"], f"slice({body['offset']}, {body['len']})")

    def p_Distinct(self, body: dict) -> str:
        options = body["options"]
        subset = options.get("subset")
        args = []
        if subset is not None:
            args.append(f"subset=[{self.selectors(subset)}]")
        keep = options.get("keep_strategy", "Any")
        args.append(f"keep={keep.lower()!r}")
        if options.get("maintain_order"):
            args.append("maintain_order=True")
        return self.chain(body["input"], f"unique({', '.join(args)})")

    def p_Join(self, body: dict) -> str:
        left = self.plan(body["input_left"])
        right = self.nested(body["input_right"])
        args = body["options"]["args"]
        how = args["how"]
        how_name = how if isinstance(how, str) else next(iter(how))
        extra: list[str] = []
        predicates = body.get("predicates") or []
        if predicates and not body.get("left_on"):
            if how_name != "Inner":
                raise self.unsupported(f"a {how_name} join with inequality conditions")
            suffix = args.get("suffix")
            suffix_text = f", suffix={suffix!r}" if suffix not in (None, "_right") else ""
            return f"{left}\n.join_where({right}, {self.exprs(predicates)}{suffix_text})"
        if how_name == "Cross":
            call = f"join({right}, how='cross'"
        elif how_name == "IEJoin":
            raise self.unsupported("an inequality join_where() step")
        else:
            mapping = {"Left": "left", "Right": "right", "Inner": "inner", "Full": "full", "Semi": "semi", "Anti": "anti"}
            if how_name not in mapping:
                if how_name == "AsOf":
                    return self.asof(left, right, body, how["AsOf"])
                raise self.unsupported(f"a {how_name} join")
            call = f"join({right}, {self.join_keys(body['left_on'], body['right_on'])}, how={mapping[how_name]!r}"
            if args.get("nulls_equal"):
                extra.append("nulls_equal=True")
            coalesce = args.get("coalesce")
            if coalesce == "CoalesceColumns":
                extra.append("coalesce=True")
            elif coalesce == "KeepColumns":
                extra.append("coalesce=False")
        if args.get("suffix") not in (None, "_right") and self.suffix_needed(body):
            extra.append(f"suffix={args['suffix']!r}")
        order = args.get("maintain_order")
        if order and order != "None":
            orders = {"Left": "left", "Right": "right", "LeftRight": "left_right", "RightLeft": "right_left"}
            extra.append(f"maintain_order={orders[order]!r}")
        validation = args.get("validation")
        if validation and validation != "ManyToMany":
            names = {"OneToOne": "1:1", "OneToMany": "1:m", "ManyToOne": "m:1"}
            extra.append(f"validate={names[validation]!r}")
        return f"{left}\n.{call}{''.join(', ' + item for item in extra)})"

    def suffix_needed(self, body: dict) -> bool:
        """Whether the two sides share a column the join would rename."""
        left = self.schema_of(self.plan(body["input_left"]))
        right = self.schema_of(self.plan(body["input_right"]))
        if not left or not right:
            return True
        keys = {_plain_column(node) for node in body.get("right_on", [])}
        return bool((set(left) & set(right)) - keys)

    def join_keys(self, left_on: list[Any], right_on: list[Any]) -> str:
        left_names = [_plain_column(node) for node in left_on]
        right_names = [_plain_column(node) for node in right_on]
        if all(left_names) and left_names == right_names:
            return f"on={_one_or_list(left_names)}"
        left = _one_or_list(left_names) if all(left_names) else f"[{self.args(left_on)}]"
        right = _one_or_list(right_names) if all(right_names) else f"[{self.args(right_on)}]"
        return f"left_on={left}, right_on={right}"

    def asof(self, left: str, right: str, body: dict, options: dict) -> str:
        strategy = options.get("strategy", "Backward").lower()
        args = [
            f"left_on={self.expr(body['left_on'][0])}",
            f"right_on={self.expr(body['right_on'][0])}",
            f"strategy={strategy!r}",
        ]
        left_by, right_by = options.get("left_by"), options.get("right_by")
        if left_by:
            args.append(f"by_left={list(left_by)!r}")
            args.append(f"by_right={list(right_by)!r}")
        if not options.get("allow_eq", True):
            args.append("allow_exact_matches=False")
        args.append("check_sortedness=False")
        coalesce = body["options"]["args"].get("coalesce")
        if coalesce == "KeepColumns":
            args.append("coalesce=False")
        return f"{left}\n.join_asof({right}, {', '.join(args)})"

    def p_Union(self, body: dict) -> str:
        inputs = ", ".join(self.nested(node) for node in body["inputs"])
        args = body.get("args", {})
        how = "vertical_relaxed" if args.get("to_supertypes") else "vertical"
        diagonal = args.get("diagonal")
        if diagonal:
            how = "diagonal_relaxed" if args.get("to_supertypes") else "diagonal"
        order = "" if args.get("maintain_order", True) else ", parallel=True"
        return f"pl.concat([{inputs}], how={how!r}{order})"

    def p_HConcat(self, body: dict) -> str:
        inputs = ", ".join(self.nested(node) for node in body["inputs"])
        # horizontal_extend pads shorter inputs; plain horizontal will require equal heights.
        strict = body.get("options", {}).get("strict", False)
        return f"pl.concat([{inputs}], how={'horizontal' if strict else 'horizontal_extend'!r})"

    def p_MapFunction(self, body: dict) -> str:
        function = body["function"]
        kind, args = (function, None) if isinstance(function, str) else next(iter(function.items()))
        if kind == "Rename":
            existing, new = args["existing"], args["new"]
            mapping = dict(zip(_names(existing), new))
            strict = "" if args.get("strict", True) else ", strict=False"
            return self.chain(body["input"], f"rename({mapping!r}{strict})")
        if kind == "RowIndex":
            offset = args.get("offset") or 0
            extra = f", offset={offset}" if offset else ""
            return self.chain(body["input"], f"with_row_index({args['name']!r}{extra})")
        if kind == "Explode":
            columns = self.selectors(args["columns"])
            options = args.get("options", args)
            # Always explicit: Polars warns that the empty_as_null default changes.
            extra = [
                f"empty_as_null={options.get('empty_as_null', True)}",
                f"keep_nulls={options.get('keep_nulls', True)}",
            ]
            return self.chain(body["input"], f"explode({columns}{''.join(', ' + e for e in extra)})")
        if kind == "Unnest":
            columns = self.selectors(args["columns"])
            separator = args.get("separator")
            extra = f", separator={separator!r}" if separator else ""
            return self.chain(body["input"], f"unnest({columns}{extra})")
        if kind == "Unpivot":
            return self.chain(body["input"], self.unpivot(args["args"]))
        if kind == "Drop":
            return self.chain(body["input"], f"drop({self.selectors(args['columns'])})")
        raise self.unsupported(f"the Polars plan step {kind}")

    def unpivot(self, args: dict) -> str:
        parts = [f"on=[{self.selectors(args['on'])}]", f"index=[{self.selectors(args['index'])}]"]
        if args.get("variable_name"):
            parts.append(f"variable_name={args['variable_name']!r}")
        if args.get("value_name"):
            parts.append(f"value_name={args['value_name']!r}")
        return f"unpivot({', '.join(parts)})"

    def p_Pivot(self, body: dict) -> str:
        on_columns = pl.read_ipc_stream(io.BytesIO(bytes(body["on_columns"])))
        args = [f"[{self.selectors(body['on'])}]", f"on_columns=pl.DataFrame({_frame_literal(on_columns)})"]
        if body.get("index") is not None:
            args.append(f"index=[{self.selectors(body['index'])}]")
        if body.get("values") is not None:
            args.append(f"values=[{self.selectors(body['values'])}]")
        if body.get("agg") is not None:
            args.append(f"aggregate_function={self.expr(body['agg'])}")
        if body.get("maintain_order"):
            args.append("maintain_order=True")
        if body.get("separator", "_") != "_":
            args.append(f"separator={body['separator']!r}")
        if str(body.get("column_naming", "Auto")).lower() != "auto":
            args.append(f"column_naming={str(body['column_naming']).lower()!r}")
        return self.chain(body["input"], f"pivot({', '.join(args)})")

    def p_ExtContext(self, body: dict) -> str:
        raise self.unsupported("an extended-context step")

    def p_Cache(self, body: dict) -> str:
        return self.plan(body["input"])

    def p_Sink(self, body: dict) -> str:
        raise self.unsupported("a sink step")

    # ── expressions ─────────────────────────────────────────────────────

    def exprs(self, nodes: list[Any]) -> str:
        return ", ".join(self.expr(node) for node in nodes)

    def selectors(self, node: Any) -> str:
        """Column names when the selector only names columns, else a cs selector."""
        names = _selector_names(node)
        if names is not None:
            return ", ".join(repr(name) for name in names)
        return self.selector(node)

    def selector(self, node: Any) -> str:
        """A ``polars.selectors`` expression for a serialized selector."""
        if isinstance(node, list):
            return " | ".join(self.selector(item) for item in node)
        if node == "Wildcard":
            self.uses_selectors = True
            return "cs.all()"
        if not isinstance(node, dict) or len(node) != 1:
            raise self.unsupported(f"the column selector {str(node)[:60]}")
        (kind, body), = node.items()
        if kind in ("Selector", "Expr"):
            return self.selector(body)
        names = _selector_names(node)
        if names is not None:
            self.uses_selectors = True
            strict = body.get("strict", True) if isinstance(body, dict) else True
            return f"cs.by_name({', '.join(repr(n) for n in names)}{'' if strict else ', require_all=False'})"
        operators = {"Union": "|", "Difference": "-", "Intersect": "&", "ExclusiveOr": "^"}
        if kind in operators:
            left, right = body if isinstance(body, list) else (body["left"], body["right"])
            return f"({self.selector(left)} {operators[kind]} {self.selector(right)})"
        raise self.unsupported(f"the column selector {kind}")

    def expr(self, node: Any) -> str:
        if isinstance(node, str):
            units = {"Len": "pl.len()", "Wildcard": "pl.all()", "Element": "pl.element()"}
            if node in units:
                return units[node]
            raise self.unsupported(f"the expression {node}")
        (kind, body), = node.items()
        method = getattr(self, "e_" + kind, None)
        if method is None:
            raise self.unsupported(f"the Polars expression {kind}")
        return method(body)

    def e_Column(self, body: str) -> str:
        return f"pl.col({body!r})"

    def e_Columns(self, body: list[str]) -> str:
        return "pl.col(" + ", ".join(repr(name) for name in body) + ")"

    def e_Selector(self, body: Any) -> str:
        names = _selector_names({"Selector": body})
        if names is None:
            return self.selector(body)
        return "pl.col(" + ", ".join(repr(name) for name in names) + ")"

    def e_Alias(self, body: list) -> str:
        inner, name = body
        if _plain_column(inner) == name:
            return f"pl.col({name!r})"
        try:
            return f"{self.expr(inner)}.alias({name!r})"
        except ExportError as error:
            message = str(error)
            if "a column is computed" in message:
                raise ExportError(message.replace("a column is computed", f"column {name!r} is computed", 1)) from None
            raise

    def e_KeepName(self, body: Any) -> str:
        return f"{self.expr(body)}.name.keep()"

    def e_Literal(self, body: dict) -> str:
        text = _literal(body)
        # A bare number cannot take a method call (0.alias(...)).
        return f"pl.lit({text})" if "Dyn" in body else text

    def operand(self, node: Any) -> str:
        """Bare numbers read better on either side of an operator."""
        if isinstance(node, dict) and "Literal" in node and "Dyn" in node["Literal"]:
            return _literal(node["Literal"])
        return self.expr(node)

    def e_BinaryExpr(self, body: dict) -> str:
        left, right = self.operand(body["left"]), self.operand(body["right"])
        if _is_dyn(body["left"]) and _is_dyn(body["right"]):
            left = f"pl.lit({left})"  # two bare numbers would fold in Python
        op = body["op"]
        methods = {"EqValidity": "eq_missing", "NotEqValidity": "ne_missing", "LogicalAnd": "and_", "LogicalOr": "or_"}
        if op in methods:
            return f"{left}.{methods[op]}({right})"
        symbol = _BINARY.get(op)
        if symbol is None:
            raise self.unsupported(f"the operator {op}")
        return f"({left} {symbol} {right})"

    def e_Cast(self, body: dict) -> str:
        dtype = body["dtype"]
        dtype = dtype["Literal"] if isinstance(dtype, dict) and "Literal" in dtype else dtype
        strict = "" if body.get("options") == "Strict" else ", strict=False"
        if body.get("options") == "Overflowing":
            strict = ", wrap_numerical=True"
        target = _dtype(dtype)
        column = _plain_column(body["expr"])
        if column is not None and column in self.current and _dtype_of(self.current[column]) == target:
            return self.expr(body["expr"])  # already that type
        return f"{self.expr(body['expr'])}.cast({target}{strict})"

    def e_Ternary(self, body: dict) -> str:
        branches = []
        node: Any = {"Ternary": body}
        while isinstance(node, dict) and "Ternary" in node:
            ternary = node["Ternary"]
            branches.append((_bare(self.expr(ternary["predicate"])), _bare(self.expr(ternary["truthy"]))))
            node = ternary["falsy"]
        first, *rest = branches
        text = f"pl.when({first[0]}).then({first[1]})"
        for predicate, value in rest:
            text += f".when({predicate}).then({value})"
        return text + f".otherwise({_bare(self.expr(node))})"

    def e_Agg(self, body: dict) -> str:
        (kind, inner), = body.items() if isinstance(body, dict) else ((body, None),)
        args = ""
        if isinstance(inner, list):
            inner, *rest = inner
            if kind in ("Std", "Var") and rest:
                args = f"ddof={rest[0]}"
        elif isinstance(inner, dict) and "input" in inner:
            if kind == "Item":
                allow = ", allow_empty=True" if inner.get("allow_empty") else ""
                return f"{self.expr(inner['input'])}.item({allow.lstrip(', ')})"
            if kind in ("Min", "Max", "Sum") and inner.get("propagate_nans"):
                return f"{self.expr(inner['input'])}.nan_{kind.lower()}()"
            if kind == "Count":
                return f"{self.expr(inner['input'])}.count()" if not inner.get("include_nulls") else f"{self.expr(inner['input'])}.len()"
            if kind == "Quantile":
                return f"{self.expr(inner['expr'])}.quantile({self.expr(inner['quantile'])}, interpolation={inner['method'].lower()!r})"
            inner = inner["input"]
        method = _AGG.get(kind)
        if method is None:
            raise self.unsupported(f"the aggregation {kind}")
        return f"{self.expr(inner)}.{method}({args})"

    def e_Len(self, body: Any) -> str:
        return "pl.len()"

    def e_Over(self, body: dict) -> str:
        partition = self.exprs(body["partition_by"])
        order = body.get("order_by")
        extra = ""
        if order:
            expr, options = order
            extra = f", order_by={self.expr(expr)}"
            if options.get("descending"):
                extra += ", descending=True"
            if options.get("nulls_last"):
                extra += ", nulls_last=True"
        mapping = body.get("mapping", "GroupsToRows")
        if mapping == "Explode":
            extra += ", mapping_strategy='explode'"
        elif mapping == "Join":
            extra += ", mapping_strategy='join'"
        return f"{self.expr(body['function'])}.over({partition}{extra})"

    def e_Sort(self, body: dict) -> str:
        options = body["options"]
        args = []
        if options.get("descending"):
            args.append("descending=True")
        if options.get("nulls_last"):
            args.append("nulls_last=True")
        if options.get("maintain_order"):
            args.append("maintain_order=True")
        return f"{self.expr(body['expr'])}.sort({', '.join(args)})"

    def e_SortBy(self, body: dict) -> str:
        options = body["sort_options"]
        args = [f"[{self.exprs(body['by'])}]"]
        if any(options["descending"]):
            args.append(f"descending={options['descending']}")
        if any(options["nulls_last"]):
            args.append(f"nulls_last={options['nulls_last']}")
        if options.get("maintain_order"):
            args.append("maintain_order=True")
        return f"{self.expr(body['expr'])}.sort_by({', '.join(args)})"

    def e_Filter(self, body: dict) -> str:
        return f"{self.expr(body['input'])}.filter({self.expr(body['by'])})"

    def e_Gather(self, body: dict) -> str:
        method = "get" if body.get("returns_scalar") else "gather"
        null_on_oob = ", null_on_oob=True" if body.get("null_on_oob") else ""
        return f"{self.expr(body['expr'])}.{method}({self.expr(body['idx'])}{null_on_oob})"

    def e_Slice(self, body: dict) -> str:
        return f"{self.expr(body['input'])}.slice({self.expr(body['offset'])}, {self.expr(body['length'])})"

    def e_Explode(self, body: dict) -> str:
        inner = body["input"] if isinstance(body, dict) and "input" in body else body
        options = body.get("options", {}) if isinstance(body, dict) else {}
        empty = options.get("empty_as_null", True)
        keep = options.get("keep_nulls", True)
        return f"{self.expr(inner)}.explode(empty_as_null={empty}, keep_nulls={keep})"

    def e_First(self, body: Any) -> str:
        return "pl.first()"

    def e_Last(self, body: Any) -> str:
        return "pl.last()"

    def e_Field(self, body: list) -> str:
        return "pl.field(" + ", ".join(repr(name) for name in body) + ")"

    def e_Eval(self, body: dict) -> str:
        variant = body.get("variant", "List")
        variant = variant if isinstance(variant, str) else next(iter(variant))
        if variant == "List":
            return f"{self.expr(body['expr'])}.list.eval({self.expr(body['evaluation'])})"
        if variant in ("Cumulative", "Array"):
            raise self.unsupported(f"a {variant} eval")
        raise self.unsupported(f"a {variant} eval")

    def e_AnonymousFunction(self, body: dict) -> str:
        text = bytes(body.get("function") or []).decode("latin-1")
        if any(guard in text for guard in _GUARDS):
            # tidy3's lazy input check: True whenever the input is valid. The
            # exported code skips it, as documented.
            return "pl.lit(True)"
        raise ExportError(
            "cannot export this pipeline: a column is computed by a Python function "
            f"({_udf_name(body)}), which has no plain Polars form"
        )

    def e_Function(self, body: dict) -> str:
        return _function(self, [self.expr(node) for node in body["input"]], body["function"], body["input"])


def _legacy_join(body: dict) -> dict:
    """A Polars 2 join, ``condition: {Equi | NonEqui}``, in the 1.x keys."""
    condition = body["condition"]
    equi = condition.get("Equi") or {}
    non_equi = condition.get("NonEqui") or {}
    if set(condition) - {"Equi", "NonEqui"}:
        raise ExportError(f"cannot export this pipeline: the join condition {sorted(condition)} has no exporter")
    rest = {key: value for key, value in body.items() if key != "condition"}
    return {
        **rest,
        "left_on": equi.get("left_on", []),
        "right_on": equi.get("right_on", []),
        "predicates": non_equi.get("predicates", []),
    }


def _indent(text: str) -> str:
    return text.replace("\n", "\n    ")


def _unwrap(node: Any) -> Any:
    while isinstance(node, dict) and set(node) == {"IR"}:
        node = node["IR"]["dsl"]
    return node


def _plain_column(node: Any) -> str | None:
    """``x`` for ``pl.col("x")`` or ``pl.col("x").alias("x")``, else None."""
    if isinstance(node, dict) and "Column" in node:
        return node["Column"]
    if isinstance(node, dict) and "Alias" in node:
        inner, name = node["Alias"]
        if isinstance(inner, dict) and inner.get("Column") == name:
            return name
    return None


def _one_or_list(names: list[str]) -> str:
    return repr(names[0]) if len(names) == 1 else repr(names)


def _flags(values: list[bool]) -> str:
    return repr(values[0]) if len(set(values)) == 1 else repr(values)


def _bare(text: str) -> str:
    """*text* without one pair of parentheses around all of it."""
    if not (text.startswith("(") and text.endswith(")")):
        return text
    depth, quote, escaped = 0, "", False
    for index, char in enumerate(text):
        if quote:
            # Parentheses inside a string literal do not count.
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in "'\"":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index < len(text) - 1:
                return text
    return text[1:-1]


def _is_dyn(node: Any) -> bool:
    return isinstance(node, dict) and "Literal" in node and "Dyn" in node["Literal"]


def _names(node: Any) -> list[str]:
    if isinstance(node, list):
        return [str(name) for name in node]
    names = _selector_names(node)
    if names is None:
        raise ExportError(f"cannot export this pipeline yet: rename of {str(node)[:60]}")
    return names


def _selector_names(node: Any) -> list[str] | None:
    """The column names of a selector that only names columns, else None."""
    if isinstance(node, list):
        out: list[str] = []
        for item in node:
            names = _selector_names(item)
            if names is None:
                return None
            out.extend(names)
        return out
    if isinstance(node, str):
        return [node]
    if not isinstance(node, dict) or len(node) != 1:
        return None
    (kind, body), = node.items()
    if kind == "Column":
        return [body]
    if kind == "Columns":
        return list(body)
    if kind == "Selector":
        return _selector_names(body)
    if kind == "ByName":
        return list(body["names"]) if isinstance(body, dict) else list(body)
    if kind == "Union":
        left, right = body if isinstance(body, list) else (body["left"], body["right"])
        a, b = _selector_names(left), _selector_names(right)
        return None if a is None or b is None else a + [name for name in b if name not in a]
    if kind == "Expr":
        return _selector_names(body)
    return None


# Lazy input checks: the function returns True unless the input is invalid.
_GUARDS = ("_pl_assert_rows.<locals>.validate", "_pl_guard_no_rows.<locals>.validate")

# tidy3's Python functions, by module and qualified name, as the user wrote them.
_UDF_NAMES = (
    ("tidy3.convert", "_failure_expr", "problems() tracking for a text conversion"),
    ("tidy3.convert", "compile_polars", "a text conversion (as_*, parse_*, ymd(), ...)"),
    ("tidy3.helpers", "_round_expr", "round(), which follows R's rounding"),
    ("tidy3.helpers", "_round_half_up_expr", "round_half_up(), which follows janitor's rounding"),
    ("tidy3.helpers", "_signif_half_up_expr", "signif_half_up(), which follows janitor's rounding"),
    ("tidy3.helpers", "_round_to_fraction_expr", "round_to_fraction(), which follows janitor's rounding"),
    ("tidy3.helpers", "_single_value_expr", "single_value(), which checks each group's values"),
    ("tidy3.dates", "_excel_polars", "excel_numeric_to_date()"),
    ("tidy3.dates", "compile_polars", "force_tz()"),
    ("tidy3.dates", "_clock_polars", "hour(), minute(), or second() on an unknown type"),
    ("tidy3.dates", "_ceiling_polars", "ceiling_date() on an unknown type"),
    ("tidy3.helpers", "_flag_expr", "is_na() and friends on an unknown type"),
    ("tidy3.stringr", "", "a stringr function that runs per value in Python"),
)


def _udf_name(body: dict) -> str:
    """The tidy3 function behind a Python callback, read from its pickled bytes."""
    raw = bytes(body.get("function") or [])
    text = raw.decode("latin-1")
    for module, qualname, label in _UDF_NAMES:
        if module in text and (not qualname or f"{qualname}.<locals>" in text):
            return label
    for marker in ("tidy3.", "polars."):
        start = text.find(marker)
        if start >= 0:
            end = start
            while end < len(text) and (text[end].isalnum() or text[end] in "._<>"):
                end += 1
            return text[start:end].rstrip(".")
    return "a Python function"


# ── literals and data types ─────────────────────────────────────────────

_BINARY = {
    "Plus": "+", "Minus": "-", "Multiply": "*", "TrueDivide": "/", "FloorDivide": "//",
    "Modulus": "%", "Gt": ">", "GtEq": ">=", "Lt": "<", "LtEq": "<=", "Eq": "==",
    "NotEq": "!=", "And": "&", "Or": "|", "Xor": "^",
}

_AGG = {
    "Sum": "sum", "Mean": "mean", "Min": "min", "Max": "max", "Median": "median",
    "First": "first", "Last": "last", "NUnique": "n_unique", "Std": "std", "Var": "var",
    "Count": "count", "Item": "item", "AggGroups": "agg_groups", "Implode": "implode",
}

def _zone(value: Any) -> Any:
    """A serialized time zone, ``{"inner": "UTC"}``, as its name."""
    return value.get("inner") if isinstance(value, dict) else value


_TIME_UNITS = {"Nanoseconds": "ns", "Microseconds": "us", "Milliseconds": "ms"}


def _dtype(node: Any) -> str:
    """Python source for a serialized Polars data type."""
    if isinstance(node, str):
        simple = {
            "Boolean", "Int8", "Int16", "Int32", "Int64", "Int128", "UInt8", "UInt16", "UInt32",
            "UInt64", "Float32", "Float64", "String", "Binary", "Date", "Time", "Null",
            "Categorical", "Object",
        }
        if node in simple:
            return f"pl.{node}"
        raise ExportError(f"cannot export this pipeline yet: data type {node}")
    (kind, body), = node.items()
    if kind == "Literal":
        return _dtype(body)
    if kind == "Datetime":
        unit, zone = body
        zone = _zone(zone)
        zone_text = f", {zone!r}" if zone else ""
        return f"pl.Datetime({_TIME_UNITS[unit]!r}{zone_text})"
    if kind == "Duration":
        return f"pl.Duration({_TIME_UNITS[body]!r})"
    if kind == "List":
        return f"pl.List({_dtype(body)})"
    if kind == "Array":
        inner, size = body
        return f"pl.Array({_dtype(inner)}, {size})"
    if kind == "Struct":
        fields = ", ".join(f"pl.Field({field['name']!r}, {_dtype(field['dtype'])})" for field in body)
        return f"pl.Struct([{fields}])"
    if kind == "Enum":
        categories = pl.read_ipc_stream(io.BytesIO(bytes(body["strings"]))).to_series().to_list()
        return f"pl.Enum({categories!r})"
    if kind == "Decimal":
        precision, scale = body
        return f"pl.Decimal({precision}, {scale})"
    if kind == "Categorical":
        return "pl.Categorical"
    raise ExportError(f"cannot export this pipeline yet: data type {kind}")


def _literal(body: dict) -> str:
    (kind, value), = body.items()
    if kind == "Dyn":
        (_type, inner), = value.items()
        return repr(inner)
    if kind == "Scalar":
        return _scalar(value)
    if kind == "Series":
        series = pl.read_ipc_stream(io.BytesIO(bytes(value))).to_series()
        return f"pl.Series({series.name!r}, {series.to_list()!r}, dtype={_dtype_of(series.dtype)})"
    if kind == "Range":
        return f"pl.int_range({value['low']}, {value['high']})"
    raise ExportError(f"cannot export this pipeline yet: literal {kind}")


def _dtype_of(dtype: Any) -> str:
    """Python source for a Polars data type object."""
    if isinstance(dtype, pl.List):
        return f"pl.List({_dtype_of(dtype.inner)})"
    if isinstance(dtype, pl.Array):
        return f"pl.Array({_dtype_of(dtype.inner)}, {dtype.size})"
    if isinstance(dtype, pl.Struct):
        fields = ", ".join(f"pl.Field({f.name!r}, {_dtype_of(f.dtype)})" for f in dtype.fields)
        return f"pl.Struct([{fields}])"
    if isinstance(dtype, pl.Enum):
        return f"pl.Enum({dtype.categories.to_list()!r})"
    if isinstance(dtype, pl.Datetime):
        zone = f", {dtype.time_zone!r}" if dtype.time_zone else ""
        return f"pl.Datetime({dtype.time_unit!r}{zone})"
    if isinstance(dtype, pl.Duration):
        return f"pl.Duration({dtype.time_unit!r})"
    if isinstance(dtype, pl.Decimal):
        return f"pl.Decimal({dtype.precision}, {dtype.scale})"
    return f"pl.{dtype.base_type().__name__}"


def _frame_literal(data: pl.DataFrame) -> str:
    columns = ", ".join(
        f"{name!r}: pl.Series({data[name].to_list()!r}, dtype={_dtype_of(data[name].dtype)})"
        for name in data.columns
    )
    return "{" + columns + "}"


def _scalar(value: Any) -> str:
    """A typed scalar, ``{"Int32": 5}``, as ``pl.lit(5, dtype=pl.Int32)``."""
    (dtype_name, raw), = value.items()
    if dtype_name == "Null":
        # An expression wherever it stands: a bare None has no .alias().
        return "pl.lit(None)"
    if dtype_name == "List":
        values = pl.read_ipc_stream(io.BytesIO(bytes(raw))).to_series()
        return f"pl.lit({values.to_list()!r}, dtype=pl.List({_dtype_of(values.dtype)}))"
    if dtype_name == "Duration":
        ticks, unit = raw
        return f"pl.lit({int(ticks)}).cast(pl.Duration({_TIME_UNITS[unit]!r}))"
    python = _scalar_python(dtype_name, raw)
    if dtype_name in ("String", "Boolean"):
        return f"pl.lit({python})"
    return f"pl.lit({python}, dtype=pl.{dtype_name})"


_NUMERIC = {"Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64"}


def _scalar_python(dtype_name: str, raw: Any) -> str:
    if dtype_name in _NUMERIC or dtype_name in ("String", "Boolean"):
        if isinstance(raw, float) and raw != raw:
            return "float('nan')"
        if raw == float("inf"):
            return "float('inf')"
        if raw == float("-inf"):
            return "float('-inf')"
        return repr(raw)
    raise ExportError(f"cannot export this pipeline yet: a {dtype_name} literal")


# ── functions ───────────────────────────────────────────────────────────


def _variant(function: Any) -> tuple[str, Any]:
    if isinstance(function, str):
        return function, None
    (name, detail), = function.items()
    return name, detail


def _function(printer: _Printer, inputs: list[str], function: Any, raw_inputs: list[Any]) -> str:
    name, detail = _variant(function)
    first = inputs[0] if inputs else ""
    rest = inputs[1:]
    handler = _FUNCTIONS.get(name)
    if handler is not None:
        return handler(first, rest, detail, printer)
    simple = _SIMPLE.get(name)
    if simple is not None:
        return f"{first}.{simple}({', '.join(rest)})"
    raise printer.unsupported(f"the Polars function {name}{'.' + _variant(detail)[0] if detail and not isinstance(detail, (bool, int)) else ''}")


_SIMPLE = {
    "Abs": "abs", "Floor": "floor", "Ceil": "ceil", "Negate": "neg", "Reverse": "reverse",
    "NullCount": "null_count", "DropNulls": "drop_nulls", "DropNans": "drop_nans", "Sign": "sign",
    "UpperBound": "upper_bound", "LowerBound": "lower_bound", "Sqrt": "sqrt", "Exp": "exp",
    "IsFirstDistinct": "is_first_distinct", "IsLastDistinct": "is_last_distinct",
    "IsUnique": "is_unique", "IsDuplicated": "is_duplicated", "Unique": "unique", "ArgMin": "arg_min",
    "ArgMax": "arg_max", "Product": "product", "Diff": "diff", "Interpolate": "interpolate",
    "Hash": "hash", "UniqueCounts": "unique_counts", "Len": "len",
}


def _args(*items: str) -> str:
    return ", ".join(item for item in items if item)


def _fill_null(first, rest, detail, printer):
    return f"{first}.fill_null({rest[0]})"


def _fill_strategy(first, rest, detail, printer):
    strategy = detail if isinstance(detail, str) else next(iter(detail))
    limit = detail[strategy] if isinstance(detail, dict) else None
    names = {"Forward": "forward", "Backward": "backward", "Min": "min", "Max": "max", "Mean": "mean", "Zero": "zero", "One": "one"}
    limit_text = f", limit={limit}" if limit else ""
    return f"{first}.fill_null(strategy={names[strategy]!r}{limit_text})"


def _shift(first, rest, detail, printer):
    return f"{first}.shift({rest[0]})"


def _shift_and_fill(first, rest, detail, printer):
    return f"{first}.shift({rest[0]}, fill_value={rest[1]})"


def _cumulative(method):
    def handler(first, rest, detail, printer):
        reverse = detail.get("reverse") if isinstance(detail, dict) else bool(detail)
        return f"{first}.{method}({'reverse=True' if reverse else ''})"

    return handler


def _horizontal(function):
    def handler(first, rest, detail, printer):
        args = ", ".join([first, *rest])
        if function == "pl.sum_horizontal" and isinstance(detail, dict) and detail.get("ignore_nulls") is False:
            return f"{function}({args}, ignore_nulls=False)"
        if function == "pl.mean_horizontal" and isinstance(detail, dict) and detail.get("ignore_nulls") is False:
            return f"{function}({args}, ignore_nulls=False)"
        return f"{function}({args})"

    return handler


def _coalesce(first, rest, detail, printer):
    return f"pl.coalesce({', '.join([first, *rest])})"


def _boolean(first, rest, detail, printer):
    sub, options = _variant(detail)
    simple = {
        "IsNull": "is_null", "IsNotNull": "is_not_null", "Not": "not_", "IsNan": "is_nan",
        "IsNotNan": "is_not_nan", "IsFinite": "is_finite", "IsInfinite": "is_infinite",
        "IsFirstDistinct": "is_first_distinct", "IsLastDistinct": "is_last_distinct",
        "IsUnique": "is_unique", "IsDuplicated": "is_duplicated",
    }
    if sub in simple:
        return f"{first}.{simple[sub]}()"
    if sub in ("Any", "All"):
        default = "" if options["ignore_nulls"] else "ignore_nulls=False"
        return f"{first}.{sub.lower()}({default})"
    if sub == "AllHorizontal":
        return f"pl.all_horizontal({', '.join([first, *rest])})"
    if sub == "AnyHorizontal":
        return f"pl.any_horizontal({', '.join([first, *rest])})"
    if sub == "IsIn":
        nulls = options.get("nulls_equal") if isinstance(options, dict) else False
        return f"{first}.is_in({rest[0]}{', nulls_equal=True' if nulls else ''})"
    if sub == "IsBetween":
        closed = options["closed"].lower() if isinstance(options, dict) else "both"
        return f"{first}.is_between({rest[0]}, {rest[1]}, closed={closed!r})"
    raise printer.unsupported(f"the Polars function Boolean.{sub}")


def _string(first, rest, detail, printer):
    sub, options = _variant(detail)
    if sub in _LITERAL_STRING_ARGS:
        # A literal pattern or replacement reads better as a plain string; these
        # methods take a str as a literal (concat_str would read a column).
        rest = [_plain_string(arg) for arg in rest]
    simple = {
        "Lowercase": "to_lowercase", "Uppercase": "to_uppercase", "Titlecase": "to_titlecase",
        "LenChars": "len_chars", "LenBytes": "len_bytes", "Reverse": "reverse",
    }
    if sub in simple:
        return f"{first}.str.{simple[sub]}()"
    if sub == "Contains":
        return f"{first}.str.contains({rest[0]}, literal={options['literal']}, strict={options['strict']})"
    if sub == "StartsWith":
        return f"{first}.str.starts_with({rest[0]})"
    if sub == "EndsWith":
        return f"{first}.str.ends_with({rest[0]})"
    if sub == "CountMatches":
        return f"{first}.str.count_matches({rest[0]}, literal={options})"
    if sub == "Replace":
        return _replace(first, rest, options)
    if sub == "Extract":
        return f"{first}.str.extract({rest[0]}, group_index={options})"
    if sub == "ExtractAll":
        return f"{first}.str.extract_all({rest[0]})"
    if sub == "ExtractGroups":
        return f"{first}.str.extract_groups({options['pat']!r})"
    if sub == "Slice":
        return f"{first}.str.slice({rest[0]}, {rest[1]})"
    if sub in ("StripChars", "StripCharsStart", "StripCharsEnd"):
        method = {"StripChars": "strip_chars", "StripCharsStart": "strip_chars_start", "StripCharsEnd": "strip_chars_end"}[sub]
        return f"{first}.str.{method}({rest[0] if rest else ''})"
    if sub == "Split":
        method = "split_inclusive" if options else "split"
        return f"{first}.str.{method}({rest[0]})" if method == "split" else f"{first}.str.split({rest[0]}, inclusive=True)"
    if sub == "SplitExact":
        return f"{first}.str.split_exact({rest[0]}, {options['n']}{', inclusive=True' if options.get('inclusive') else ''})"
    if sub == "SplitN":
        return f"{first}.str.splitn({rest[0]}, {options})"
    if sub == "ConcatHorizontal":
        delimiter = options.get("delimiter", "") if isinstance(options, dict) else ""
        ignore = options.get("ignore_nulls", False) if isinstance(options, dict) else False
        return f"pl.concat_str([{', '.join([first, *rest])}], separator={delimiter!r}{', ignore_nulls=True' if ignore else ''})"
    if sub == "ConcatVertical":
        delimiter = options.get("delimiter", "") if isinstance(options, dict) else ""
        ignore = options.get("ignore_nulls", True) if isinstance(options, dict) else True
        return f"{first}.str.join({delimiter!r}{'' if ignore else ', ignore_nulls=False'})"
    if sub in ("PadStart", "PadEnd"):
        length = options["length"] if isinstance(options, dict) and "length" in options else rest[0]
        fill = options.get("fill_char", " ") if isinstance(options, dict) else " "
        method = "pad_start" if sub == "PadStart" else "pad_end"
        return f"{first}.str.{method}({length}, {fill!r})"
    if sub == "ZFill":
        return f"{first}.str.zfill({rest[0]})"
    if sub == "Normalize":
        form = options.get("form", "NFC") if isinstance(options, dict) else options
        return f"{first}.str.normalize({form!r})"
    if sub == "Strptime":
        return _strptime(first, rest, options, printer)
    if sub == "ToInteger":
        return f"{first}.str.to_integer(base={rest[0]}, strict={options.get('strict', True) if isinstance(options, dict) else options})"
    if sub == "Find":
        return f"{first}.str.find({rest[0]}, literal={options['literal']}, strict={options['strict']})"
    raise printer.unsupported(f"the Polars function StringExpr.{sub}")


_LITERAL_STRING_ARGS = frozenset({
    "Contains", "StartsWith", "EndsWith", "CountMatches", "Replace", "Extract", "ExtractAll",
    "Split", "SplitExact", "SplitN", "StripChars", "StripCharsStart", "StripCharsEnd", "Find",
})


def _plain_string(text: str) -> str:
    """``'a'`` for ``pl.lit('a')``; other arguments unchanged."""
    import ast

    if text.startswith("pl.lit(") and text.endswith(")"):
        inner = text[len("pl.lit("):-1]
        try:
            if isinstance(ast.literal_eval(inner), str):
                return inner
        except (ValueError, SyntaxError):
            pass
    return text


def _replace(first, rest, options):
    literal = options.get("literal", False) if isinstance(options, dict) else bool(options)
    n = options.get("n", 1) if isinstance(options, dict) else 1
    if n == -1:
        return f"{first}.str.replace_all({rest[0]}, {rest[1]}, literal={literal})"
    return f"{first}.str.replace({rest[0]}, {rest[1]}, literal={literal}, n={n})"


def _strptime(first, rest, options, printer):
    dtype, settings = options if isinstance(options, list) else (options["dtype"], options["options"])
    fmt = settings.get("format")
    strict = settings.get("strict", True)
    exact = settings.get("exact", True)
    args = [f"{_dtype(dtype)}"]
    if fmt is not None:
        args.append(f"format={fmt!r}")
    if not strict:
        args.append("strict=False")
    if not exact:
        args.append("exact=False")
    return f"{first}.str.strptime({', '.join(args)})"


_TOTALS = {
    f"Total{unit}": f"total_{unit.lower()}"
    for unit in ("Days", "Hours", "Minutes", "Seconds", "Milliseconds", "Microseconds", "Nanoseconds")
}


def _temporal(first, rest, detail, printer):
    sub, options = _variant(detail)
    simple = {
        "Year": "year", "Month": "month", "Day": "day", "Hour": "hour", "Minute": "minute",
        "Second": "second", "Week": "week", "WeekDay": "weekday", "OrdinalDay": "ordinal_day",
        "Quarter": "quarter", "Date": "date", "Time": "time", "MonthStart": "month_start",
        "MonthEnd": "month_end", "DaysInMonth": "days_in_month", "IsoYear": "iso_year",
        "Nanosecond": "nanosecond", "Microsecond": "microsecond", "Millisecond": "millisecond",
        "IsLeapYear": "is_leap_year",
    }
    if sub in simple:
        return f"{first}.dt.{simple[sub]}()"
    if sub == "Truncate":
        return f"{first}.dt.truncate({rest[0]})"
    if sub == "OffsetBy":
        return f"{first}.dt.offset_by({rest[0]})"
    if sub == "ReplaceTimeZone":
        zone, non_existent = (options if isinstance(options, list) else (options, "Raise"))
        args = [repr(_zone(zone))]
        if len(rest):
            args.append(f"ambiguous={rest[0]}")
        if non_existent and str(non_existent).lower() != "raise":
            args.append(f"non_existent={str(non_existent).lower()!r}")
        return f"{first}.dt.replace_time_zone({', '.join(args)})"
    if sub == "ConvertTimeZone":
        return f"{first}.dt.convert_time_zone({_zone(options)!r})"
    if sub in _TOTALS:
        fractional = isinstance(options, dict) and options.get("fractional")
        return f"{first}.dt.{_TOTALS[sub]}({'fractional=True' if fractional else ''})"
    if sub == "CastTimeUnit":
        return f"{first}.dt.cast_time_unit({_TIME_UNITS[options]!r})"
    raise printer.unsupported(f"the Polars function TemporalExpr.{sub}")


def _list(first, rest, detail, printer):
    sub, options = _variant(detail)
    if sub == "Length":
        return f"{first}.list.len()"
    if sub == "Get":
        return f"{first}.list.get({rest[0]}, null_on_oob={options})"
    if sub == "Concat":
        return f"pl.concat_list([{', '.join([first, *rest])}])"
    if sub == "Join":
        return f"{first}.list.join({rest[0]}, ignore_nulls={options})"
    if sub == "Slice":
        return f"{first}.list.slice({rest[0]}, {rest[1]})"
    if sub == "Contains":
        return f"{first}.list.contains({rest[0]})"
    if sub == "Unique":
        return f"{first}.list.unique(maintain_order={bool(options)})"
    raise printer.unsupported(f"the Polars function ListExpr.{sub}")


def _struct(first, rest, detail, printer):
    sub, options = _variant(detail)
    if sub == "FieldByName":
        return f"{first}.struct.field({options!r})"
    if sub == "FieldByIndex":
        return f"{first}.struct[{options}]"
    if sub == "RenameFields":
        return f"{first}.struct.rename_fields({list(options)!r})"
    raise printer.unsupported(f"the Polars function StructExpr.{sub}")


def _as_struct(first, rest, detail, printer):
    return f"pl.struct({', '.join([first, *rest])})"


def _range(first, rest, detail, printer):
    sub, options = _variant(detail)
    if sub == "IntRange":
        step = options.get("step", 1) if isinstance(options, dict) else 1
        dtype = options.get("dtype") if isinstance(options, dict) else None
        dtype_text = f", dtype={_dtype(dtype)}" if dtype else ""
        return f"pl.int_range({first}, {rest[0]}, {step}{dtype_text})"
    if sub == "IntRanges":
        dtype = options.get("dtype") if isinstance(options, dict) else None
        dtype_text = f", dtype={_dtype(dtype)}" if dtype else ""
        return f"pl.int_ranges({first}, {rest[0]}, {rest[1] if len(rest) > 1 else 1}{dtype_text})"
    raise printer.unsupported(f"the Polars function Range.{sub}")


def _rank(first, rest, detail, printer):
    options = detail["options"] if isinstance(detail, dict) and "options" in detail else detail
    method = options.get("method", "Dense").lower()
    descending = options.get("descending", False)
    seed = detail.get("seed") if isinstance(detail, dict) else None
    args = [f"method={method!r}"]
    if descending:
        args.append("descending=True")
    if seed is not None:
        args.append(f"seed={seed}")
    return f"{first}.rank({', '.join(args)})"


def _arg_sort(first, rest, detail, printer):
    descending = detail.get("descending") if isinstance(detail, dict) else False
    nulls_last = detail.get("nulls_last") if isinstance(detail, dict) else False
    args = []
    if descending:
        args.append("descending=True")
    if nulls_last:
        args.append("nulls_last=True")
    return f"{first}.arg_sort({', '.join(args)})"


def _round(first, rest, detail, printer):
    decimals = detail.get("decimals", 0) if isinstance(detail, dict) else detail
    mode = detail.get("mode") if isinstance(detail, dict) else None
    mode_text = f", mode={_snake(mode)!r}" if mode else ""
    return f"{first}.round({decimals}{mode_text})"


def _clip(first, rest, detail, printer):
    has_min = detail.get("has_min") if isinstance(detail, dict) else True
    has_max = detail.get("has_max") if isinstance(detail, dict) else True
    args = []
    remaining = list(rest)
    if has_min:
        args.append(f"lower_bound={remaining.pop(0)}")
    if has_max:
        args.append(f"upper_bound={remaining.pop(0)}")
    return f"{first}.clip({', '.join(args)})"


def _snake(name: str) -> str:
    out = []
    for index, char in enumerate(name):
        if char.isupper() and index:
            out.append("_")
        out.append(char.lower())
    return "".join(out)


def _repeat(first, rest, detail, printer):
    return f"pl.repeat({first}, {rest[0]})"


def _replace_strict(first, rest, detail, printer):
    return f"{first}.replace_strict({rest[0]}, {rest[1]}, default={rest[2] if len(rest) > 2 else 'None'})"


def _replace_values(first, rest, detail, printer):
    return f"{first}.replace({rest[0]}, {rest[1]})"


def _null_count(first, rest, detail, printer):
    return f"{first}.null_count()"


def _pow(first, rest, detail, printer):
    return f"({first} ** {rest[0]})"


def _log(first, rest, detail, printer):
    base = detail if isinstance(detail, (int, float)) else "2.718281828459045"
    return f"{first}.log({base})"


def _cum_count(first, rest, detail, printer):
    reverse = detail.get("reverse") if isinstance(detail, dict) else bool(detail)
    return f"{first}.cum_count({'reverse=True' if reverse else ''})"


def _shrink(first, rest, detail, printer):
    return f"{first}.shrink_dtype()"


def _cat(first, rest, detail, printer):
    sub, _options = _variant(detail)
    if sub == "GetCategories":
        return f"{first}.cat.get_categories()"
    raise printer.unsupported(f"the Polars function CategoricalFunction.{sub}")


def _set_sorted(first, rest, detail, printer):
    return first


def _business(first, rest, detail, printer):
    raise printer.unsupported("a business-day function")


_FUNCTIONS = {
    "FillNull": _fill_null,
    "FillNullWithStrategy": _fill_strategy,
    "Shift": _shift,
    "ShiftAndFill": _shift_and_fill,
    "CumSum": _cumulative("cum_sum"),
    "CumMax": _cumulative("cum_max"),
    "CumMin": _cumulative("cum_min"),
    "CumProd": _cumulative("cum_prod"),
    "CumCount": _cum_count,
    "SumHorizontal": _horizontal("pl.sum_horizontal"),
    "MaxHorizontal": _horizontal("pl.max_horizontal"),
    "MinHorizontal": _horizontal("pl.min_horizontal"),
    "MeanHorizontal": _horizontal("pl.mean_horizontal"),
    "Coalesce": _coalesce,
    "Boolean": _boolean,
    "StringExpr": _string,
    "TemporalExpr": _temporal,
    "ListExpr": _list,
    "StructExpr": _struct,
    "AsStruct": _as_struct,
    "Range": _range,
    "Rank": _rank,
    "ArgSort": _arg_sort,
    "Round": _round,
    "Clip": _clip,
    "Repeat": _repeat,
    "ReplaceStrict": _replace_strict,
    "Replace": _replace_values,
    "NullCount": _null_count,
    "Pow": _pow,
    "Log": _log,
    "ShrinkType": _shrink,
    "Categorical": _cat,
    "SetSortedFlag": _set_sorted,
    "Business": _business,
}
