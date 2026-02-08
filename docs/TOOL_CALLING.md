# Tool Calling System

## Overview

llm-bawt supports tool calling through a dual-mode architecture: **native tool calling** for providers that support it (OpenAI) and **ReAct format** as a universal fallback for all other models (GGUF, Ollama). A legacy XML format is retained for backward compatibility.

## Architecture

```
┌─────────────────────────────────────────────────┐
│               Tool Format Router                │
├─────────────────────────────────────────────────┤
│  OpenAI API models                              │
│    └─ Native tool calling (tools parameter)     │
│                                                 │
│  GGUF / Ollama / HuggingFace models             │
│    └─ ReAct format with stop sequences          │
│                                                 │
│  Legacy fallback                                │
│    └─ XML <tool_call> format                    │
└─────────────────────────────────────────────────┘
```

Tool format is automatically determined by model type via `Config.get_tool_format()`:

| Model Type | Tool Format | Mechanism |
|-----------|-------------|-----------|
| OpenAI    | `native`    | OpenAI `tools` parameter, structured responses |
| GGUF      | `react`     | Text-based ReAct parsing with stop sequences |
| Ollama    | `react`     | Text-based ReAct parsing with stop sequences |
| HuggingFace | `react`   | Text-based ReAct parsing with stop sequences |

Models can override this with an explicit `tool_format` field in their definition.

## Format Handlers

All tool formats implement the `ToolFormatHandler` interface (`tools/formats/base.py`):

```python
class ToolFormatHandler(ABC):
    def get_system_prompt(self, tools: list) -> str
    def get_stop_sequences(self) -> list[str]
    def parse_response(self, response: str) -> tuple[list[ToolCallRequest], str]
    def format_result(self, tool_name: str, result: str, tool_call_id=None, error=False) -> str | dict
    def sanitize_response(self, response: str) -> str
```

### ReAct Format (`tools/formats/react.py`)

Used for local and non-OpenAI models. Follows the industry-standard ReAct (Reasoning + Acting) pattern.

**Prompt format:**
```
Thought: [explain what you need to do]
Action: [tool_name]
Action Input: {"param": "value"}
```

The system injects an `Observation:` with the tool result, then the model continues:
```
Thought: I now have the information needed
Final Answer: [response to the user]
```

**Key features:**
- Multi-format parser handles model variations:
  - Standard ReAct: `Action: / Action Input:`
  - Alternative: `# Tool: / # Arguments:` (common with Dolphin models)
  - Code block JSON: tool calls wrapped in markdown code blocks
- Tool name normalization with 30+ aliases mapping model-invented names to actual tools
- Argument normalization for common parameter variations
- Comprehensive stop sequences prevent models from hallucinating observations
- Stop sequences are skipped after tool execution so the model can generate a complete final answer

**Stop sequence behavior:**
```python
has_executed_tools = False

for iteration in range(1, max_iterations + 1):
    # After executing tools, don't use stop sequences
    current_stop_sequences = None if has_executed_tools else stop_sequences

    response = query_llm(messages, stop=current_stop_sequences)

    if tool_calls_found:
        execute_tools()
        has_executed_tools = True  # Next iteration completes naturally
```

### Native OpenAI Format (`tools/formats/native_openai.py`)

Used for OpenAI API models. Tools are passed via the `tools` parameter in the API call, and tool calls are returned as structured objects -- no text parsing needed.

**Key features:**
- Converts tool definitions to OpenAI function calling schema
- No stop sequences needed (API handles tool call boundaries)
- Handles `finish_reason == "tool_calls"` for structured tool call extraction
- Supports both streaming and non-streaming tool calls
- Tool results formatted as `{"role": "tool", "content": ..., "tool_call_id": ...}`

### Legacy XML Format (`tools/formats/xml_legacy.py`)

Retained for backward compatibility. Uses `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` format.

## Tool Loop (`tools/loop.py`)

The `ToolLoop` class orchestrates multi-turn tool calling:

1. Detects whether to use native or text-based tool calling
2. Queries the LLM with combined stop sequences (format handler + model adapter)
3. Parses response for tool calls using the format handler
4. Executes tools and formats results
5. Adds results to conversation and loops (up to `max_iterations`)
6. Sanitizes the final response to remove any unparsed format markers

The convenience function `query_with_tools()` wraps the loop and returns `(final_response, tool_context_summary)`.

## Streaming Support (`tools/streaming.py`)

`stream_with_tools()` provides streaming tool calling:

1. Buffers initial tokens (~80 chars) to detect tool call vs regular text
2. If a tool call is detected, buffers the full response, executes tools, and loops
3. If regular text is detected, yields tokens to the user with adapter/format cleaning
4. Integrates with model adapters for stop sequence combination and output cleaning

## GGUF Models and Native Tool Calling

Native tool calling via llama-cpp-python's `chatml-function-calling` chat format was tested but found to fail for multi-turn conversations -- after executing a tool and adding the result, the second LLM call returns empty content. For this reason, GGUF models use the ReAct text-based format exclusively. The `_should_use_native_tools()` method only returns `True` for the `NATIVE_OPENAI` format.

## File Locations

```
src/llm_bawt/tools/
├── formats/
│   ├── __init__.py          # ToolFormat enum, get_format_handler()
│   ├── base.py              # ToolFormatHandler ABC, ToolCallRequest
│   ├── react.py             # ReAct format handler (multi-format parser)
│   ├── native_openai.py     # OpenAI native tool calling handler
│   └── xml_legacy.py        # Legacy XML format handler
├── definitions.py           # Tool definitions, get_tools_prompt()
├── executor.py              # Tool execution
├── loop.py                  # ToolLoop class, query_with_tools()
├── parser.py                # Legacy XML parser (used by xml_legacy handler)
└── streaming.py             # stream_with_tools()
```

## Adding a New Tool

1. Add the tool definition in `tools/definitions.py`
2. Add the executor in `tools/executor.py`
3. The tool loop and format handlers handle the rest automatically

## Configuration

Tool format is determined automatically but can be overridden per model:

```yaml
models:
  my-model:
    type: gguf
    tool_format: react    # Explicit override (usually not needed)
```

Relevant config method: `Config.get_tool_format(model_alias, model_def)`
