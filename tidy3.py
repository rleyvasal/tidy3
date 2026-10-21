"""Compatibility notebook entrypoint; normal imports resolve to src/tidy3."""

from pathlib import Path

_module_run = getattr(globals().get("__spec__"), "name", None) == "tidy3" and __name__ == "__main__"
if __name__ == "tidy3" or _module_run:
    import importlib.util
    import sys

    _package_dir = Path(__file__).resolve().parent / "src" / "tidy3"
    _spec = importlib.util.spec_from_file_location(
        "tidy3", _package_dir / "__init__.py",
        submodule_search_locations=[str(_package_dir)],
    )
    _package = importlib.util.module_from_spec(_spec)
    sys.modules["tidy3"] = _package
    _spec.loader.exec_module(_package)
    if _module_run:
        import runpy

        runpy.run_module("tidy3", run_name="__main__")
else:
    _loader = Path(__file__).resolve().parent / "load.py"
    exec(compile(_loader.read_text(encoding="utf-8"), str(_loader), "exec"), globals())
