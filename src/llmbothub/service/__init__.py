"""
llmbothub Background Service

A separate async service for handling background tasks like:
- Memory extraction (with dedicated small model)
- Context compaction/summarization
- Embedding generation
- OpenAI-compatible API server

The main CLI works standalone but can offload tasks to this service
when available for improved performance and capability.

Run the service:
    python -m llmbothub.service
    # or
    llm-service --port 8642
    llm-service --verbose  # Show request/response details
    llm-service --debug    # Low-level DEBUG logging
"""

from .client import ServiceClient, ServiceStatus
from .tasks import TaskType, Task, TaskResult
from .logging import ServiceLogger, get_service_logger, setup_service_logging

__all__ = [
    "ServiceClient",
    "ServiceStatus", 
    "TaskType",
    "Task",
    "TaskResult",
    "ServiceLogger",
    "get_service_logger",
    "setup_service_logging",
]
