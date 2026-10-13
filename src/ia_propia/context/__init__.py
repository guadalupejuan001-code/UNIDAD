"""Independent, source-backed conversation context selection.

The package uses local artifacts and never calls a language service or tools.
"""

from .manager import ContextManager, PreparedContext

__all__ = ["ContextManager", "PreparedContext"]
