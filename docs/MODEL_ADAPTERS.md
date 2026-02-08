# Model Adapters

## Overview

Model adapters handle model-specific formatting quirks, stop sequences, and output cleaning. They sit between the core orchestration and LLM clients, providing a centralized place for per-model behavior without polluting format handlers or client code.

## Architecture

```
BaseLLMBawt.__init__()
       ↓
   ModelAdapter ←── get_adapter(model_alias, model_def)
       │                    ↓
       │              Auto-detected from model name/repo
       │              or explicit 'adapter' field in model def
       ↓
   LLM Query
       ↓
   Raw Response
       ↓
   adapter.clean_output()    ←── Per-model artifact removal
       ↓
   handler.sanitize_response()  ←── Format-specific cleaning
       ↓
   Return to user
```

## Adapter Interface

All adapters extend `ModelAdapter` (`adapters/base.py`):

```python
class ModelAdapter(ABC):
    name: str = "default"

    def get_stop_sequences(self) -> list[str]
        # Model-specific stop sequences (combined with format handler's)

    def clean_output(self, response: str) -> str
        # Remove model-specific artifacts from output

    def supports_system_role(self) -> bool
        # Whether model supports system role messages

    def transform_messages(self, messages: list[Message]) -> list[Message]
        # Apply model-specific message transformations
```

## Available Adapters

### DefaultAdapter (`adapters/default.py`)

No-op adapter for well-behaved models. All methods use base class defaults (passthrough).

### PygmalionAdapter (`adapters/pygmalion.py`)

For Pygmalion-family and character models (MythoMax, etc.) that output:
- Role markers: `[HUMAN]`, `[/HUMAN]`, `[INST]`, `[/INST]`
- BBCode formatting: `[FONT=Arial]`, `[/FONT]`, `[B]`, `[/B]`
- Hallucinated conversation turns

**Stop sequences:** `[HUMAN]`, `[/HUMAN]`, `[INST]`, `[/INST]`, `### Instruction:`, `### Human:`, `<|im_start|>user`

**Output cleaning:**
1. Removes content inside role blocks (`[HUMAN]...[/HUMAN]`, `[INST]...[/INST]`)
2. Removes standalone role markers
3. Strips BBCode formatting tags
4. Normalizes excessive whitespace

### DolphinAdapter (`adapters/dolphin.py`)

For Dolphin models (Dolphin3.0-Llama, Dolphin-Qwen, etc.) that sometimes hallucinate tool observations.

**Output cleaning:** Truncates output at hallucinated `Observation:` markers, providing a safety net when the model generates fake tool results instead of waiting.

## Registry

Adapters are managed through a registry (`adapters/registry.py`):

```python
from llm_bawt.adapters import get_adapter

adapter = get_adapter(model_alias="mythomax", model_def=model_definition)
```

**Resolution order:**
1. Explicit `adapter` field in model definition (if present)
2. Auto-detection based on model alias or `repo_id` (pattern matching for `pygmalion`, `mytho`, `mythomax`)
3. Falls back to `DefaultAdapter`

Adapters are auto-registered at module load via `_register_builtins()`.

## Integration Points

### Core (`core/base.py`)

The adapter is initialized in `BaseLLMBawt.__init__()` and stored as `self.adapter`. It's passed to `query_with_tools()` for use in the tool loop.

### Tool Loop (`tools/loop.py`)

Stop sequences from the format handler and model adapter are combined:

```python
stop_sequences = list(handler.get_stop_sequences())
if self.adapter:
    adapter_stops = self.adapter.get_stop_sequences()
    if adapter_stops:
        stop_sequences.extend(adapter_stops)
```

### Streaming (`tools/streaming.py`)

Same stop sequence combination plus `adapter.clean_output()` is called on the final response before format handler sanitization.

## File Locations

```
src/llm_bawt/adapters/
├── __init__.py     # Exports: ModelAdapter, DefaultAdapter, get_adapter, register_adapter
├── base.py         # ModelAdapter ABC
├── registry.py     # get_adapter(), register_adapter(), auto-detection
├── default.py      # DefaultAdapter (no-op)
├── pygmalion.py    # PygmalionAdapter (BBCode/role marker cleaning)
└── dolphin.py      # DolphinAdapter (observation hallucination cleanup)
```

## Adding a New Adapter

1. Create a new file in `adapters/` extending `ModelAdapter`
2. Override the methods relevant to your model's quirks
3. Register it in `registry.py`'s `_register_builtins()` function
4. Optionally add auto-detection patterns in `_auto_detect_adapter()`
