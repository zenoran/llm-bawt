from .base import ModelAdapter


class DefaultAdapter(ModelAdapter):
    """Default no-op adapter for well-behaved models."""
    name = "default"
