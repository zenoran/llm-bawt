"""Allow running as: uv run python -m llmbothub.memory_server"""

from .server import run_server

if __name__ == "__main__":
    run_server()
