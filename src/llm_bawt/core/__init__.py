"""Core LLM processing components.

This package provides the main orchestration for llm-bawt:
- LLMBawt: Main client class for CLI usage
- PromptBuilder: Template-based system prompt assembly
- BaseLLMBawt: Shared logic for CLI and service modes
- ModelLifecycleManager: Singleton for model loading/unloading
"""

from .prompt_builder import PromptBuilder, PromptSection, SectionPosition
from .base import BaseLLMBawt
from .client import LLMBawt
from .model_lifecycle import ModelLifecycleManager, get_model_lifecycle

__all__ = [
    "LLMBawt",
    "PromptBuilder",
    "PromptSection",
    "SectionPosition",
    "BaseLLMBawt",
    "ModelLifecycleManager",
    "get_model_lifecycle",
]
