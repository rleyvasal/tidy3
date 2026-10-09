"""Optional plot3 handoff helpers."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from tidy3.frame import TidyFrame


def to_ggplot(tf: TidyFrame, mapping=None, **kwargs: Any):
    """Hand a TidyFrame to plot3's ggplot (plot3 reads it directly)."""
    try:
        from plot3 import ggplot
    except ImportError as e:
        raise ImportError(
            "plot3 is not installed: pip install plot3"
        ) from e
    return ggplot(tf, mapping, **kwargs)
