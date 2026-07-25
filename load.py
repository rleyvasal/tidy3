"""Alias entrypoint — same as ``tidy3.py``.

::

    %run /path/to/tidy3/load.py
    %run /path/to/tidy3/tidy3.py
"""

from pathlib import Path

# Execute the canonical loader in this namespace (works under %run).
_loader = Path(__file__).resolve().parent / "tidy3.py"
exec(compile(_loader.read_text(encoding="utf-8"), str(_loader), "exec"), globals())
