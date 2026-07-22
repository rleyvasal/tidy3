"""CLI: ``python -m tidy3 export notebook.ipynb -o script.py``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_export(args: argparse.Namespace) -> int:
    from tidy3.export import nb_export

    result = nb_export(
        args.notebook,
        args.output,
        only_export=args.only_export,
        with_plot3=not args.no_plot3,
        known_extra=args.known or None,
    )
    print(result)
    if result.warnings:
        for w in result.warnings:
            print(f"  warning: {w}", file=sys.stderr)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Run a ``.py`` file after applying the same R-style transforms as export."""
    from tidy3.export import collect_known_names, transform_source

    path = Path(args.script).expanduser().resolve()
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1
    src = path.read_text(encoding="utf-8")
    known = collect_known_names([src])
    if args.known:
        known.update(args.known)
    try:
        code = transform_source(src, known=known, with_plot3=not args.no_plot3)
    except SyntaxError as e:
        print(f"error: transform failed: {e}", file=sys.stderr)
        return 1
    ns: dict = {"__name__": "__main__", "__file__": str(path)}
    # Inject public APIs so short scripts work like notebooks
    if not args.no_inject:
        import tidy3 as t3
        from tidy3.masking import make_named_assign

        for name in t3.__all__:
            if name.startswith("_"):
                continue
            try:
                ns[name] = getattr(t3, name)
            except AttributeError:
                pass
        ns["tidy3"] = t3
        ns["__tidy3_assign__"] = make_named_assign
        try:
            import plot3 as p3

            for name in getattr(p3, "__all__", ()):
                if str(name).startswith("_"):
                    continue
                try:
                    ns[name] = getattr(p3, name)
                except AttributeError:
                    pass
            ns["plot3"] = p3
        except ImportError:
            pass
    exec(compile(code, str(path), "exec"), ns, ns)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tidy3",
        description="tidy3 utilities: export notebooks, run R-style scripts",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser(
        "export",
        help="export a Jupyter notebook to plain Python (nbdev-style)",
    )
    p_export.add_argument("notebook", type=str, help="path to .ipynb")
    p_export.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="output .py path (default: notebook stem + .py)",
    )
    p_export.add_argument(
        "--only-export",
        action="store_true",
        help="only include cells marked with #| export",
    )
    p_export.add_argument(
        "--no-plot3",
        action="store_true",
        help="do not apply plot3 aes/facet masking",
    )
    p_export.add_argument(
        "--known",
        action="append",
        default=[],
        help="extra non-column name (repeatable), e.g. --known cars",
    )
    p_export.set_defaults(func=_cmd_export)

    p_run = sub.add_parser(
        "run",
        help="run a .py file after R-style bare-name/backtick transforms",
    )
    p_run.add_argument("script", type=str, help="path to .py")
    p_run.add_argument(
        "--no-plot3",
        action="store_true",
        help="do not apply plot3 aes/facet masking",
    )
    p_run.add_argument(
        "--no-inject",
        action="store_true",
        help="do not inject tidy3/plot3 into the script namespace",
    )
    p_run.add_argument(
        "--known",
        action="append",
        default=[],
        help="extra non-column name (repeatable)",
    )
    p_run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
