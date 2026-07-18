"""Optional plot3 handoff helpers."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from tidy3.frame import TidyFrame


def to_ggplot(tf: TidyFrame, mapping=None, **kwargs: Any):
    """Convert a TidyFrame to a plot3 ggplot (pandas materialization)."""
    try:
        from plot3 import ggplot
    except ImportError as e:
        raise ImportError(
            "plot3 is not installed. Clone https://github.com/rleyvasal/plot3 "
            "and ensure it is on PYTHONPATH."
        ) from e
    return ggplot(tf.to_pandas(), mapping, **kwargs)
