"""MCP-based memory server for llmbothub.

Exposes memory tools via Model Context Protocol (FastMCP).
Also provides a MemoryClient for unified access to memory operations.
"""

from typing import TYPE_CHECKING

# Client imports are always available (no mcp dependency)
from .client import MemoryClient, MemoryResult, MessageResult, get_memory_client

# Server imports require mcp package - lazy load
if TYPE_CHECKING:
    from .server import mcp, run_server

__all__ = [
    "mcp",
    "run_server",
    "MemoryClient",
    "MemoryResult",
    "MessageResult",
    "get_memory_client",
]


def __getattr__(name: str):
    """Lazy import server components that require mcp package."""
    if name in ("mcp", "run_server"):
        try:
            from .server import mcp, run_server
            return mcp if name == "mcp" else run_server
        except ImportError as e:
            raise ImportError(
                f"The '{name}' component requires the 'mcp' package. "
                "Install it with: pipx runpip llmbothub install 'mcp[cli]'"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
