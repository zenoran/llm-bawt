"""Core LLM processing components.

This package provides the main orchestration for llmbothub:
- LLMBotHub: Main client class for CLI usage
- PromptBuilder: Template-based system prompt assembly
- RequestPipeline: Modular request processing with hooks
- BaseLLMBotHub: Shared logic for CLI and service modes
- ModelLifecycleManager: Singleton for model loading/unloading
"""

from .prompt_builder import PromptBuilder, PromptSection, SectionPosition
from .pipeline import RequestPipeline, PipelineStage, PipelineContext
from .base import BaseLLMBotHub
from .client import LLMBotHub
from .model_lifecycle import ModelLifecycleManager, get_model_lifecycle

__all__ = [
    "LLMBotHub",
    "PromptBuilder",
    "PromptSection",
    "SectionPosition",
    "RequestPipeline",
    "PipelineStage",
    "PipelineContext",
    "BaseLLMBotHub",
    "ModelLifecycleManager",
    "get_model_lifecycle",
]
