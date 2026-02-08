# vLLM Integration Plan

## Overview

Add vLLM as a new model provider alongside `gguf` and `openai`. vLLM loads HuggingFace models (safetensors, AWQ, GPTQ, FP8) with better GPU utilization and native OpenAI-compatible tool calling. It runs in-process inside the llm-bawt service, same as llama-cpp loads GGUF models today.

### Why

- **Model support**: New architectures (e.g. `mistral3`) work immediately — no waiting for llama-cpp-python releases.
- **Performance**: ~25% faster generation than llama-cpp at equivalent quantization (benchmarked: 70 vs 56 tok/s on Qwen 32B 4-bit).
- **Native tool calling**: vLLM supports OpenAI-format structured tool calls. No ReAct text parsing needed.
- **Quantization**: Supports AWQ, GPTQ, FP8, BF16 — comparable VRAM to GGUF quants.
- **GGUF support**: vLLM can also load GGUF files, so existing downloaded models can optionally run through vLLM.

### Tradeoffs vs llama-cpp

| | llama-cpp | vLLM |
|---|---|---|
| Load time | ~2-3s | ~30-150s (CUDA graph compile, cached after first run) |
| Generation | ~56 tok/s (Q4_K_M 32B) | ~70 tok/s (AWQ 4-bit 32B) |
| Install size | Light (~50MB) | Heavy (~2GB+, pulls torch/triton) |
| CPU offload | Yes | No (GPU only) |
| Tool calling | Text-based (ReAct) | Native OpenAI format |
| New model archs | Waits for llama.cpp release | Immediate |

---

## 1. Model Config (`models.yaml`)

### New `type: vllm` entry

```yaml
models:
  qwen-32b-awq:
    type: vllm
    model_id: Qwen/Qwen2.5-32B-Instruct-AWQ     # HuggingFace model id (required)
    description: Qwen 2.5 32B AWQ 4-bit
    context_window: 32768
    max_tokens: 4096
    tool_support: native                          # native | react | none (see §5)

    # vLLM-specific (all optional, sensible defaults)
    quantization: awq                             # awq | gptq | fp8 | null (auto-detect)
    dtype: float16                                # float16 | bfloat16 | auto
    max_model_len: 32768                          # max sequence length
    gpu_memory_utilization: 0.85                  # fraction of GPU memory to use
    enforce_eager: false                          # disable CUDA graphs (debug)
    enable_prefix_caching: true                   # KV cache reuse across requests
```

### Running GGUF through vLLM

Existing GGUF models can optionally use vLLM by adding `backend: vllm`:

```yaml
models:
  cydonia:
    type: gguf
    repo_id: TheDrummer/Cydonia-24B-v4.3-GGUF
    filename: Cydonia-24B-v4zg-Q5_K_M.gguf
    backend: vllm                                 # optional: use vLLM instead of llama-cpp
```

When `backend: vllm` is set on a GGUF model, the service passes the local GGUF path to vLLM's `LLM(model=path)` instead of `LlamaCppClient`.

---

## 2. Provider Constant + Config

### `src/llm_bawt/utils/config.py`

```python
PROVIDER_VLLM = "vllm"
```

Add to `get_model_options()`:
```python
elif model_type == PROVIDER_VLLM:
    if is_vllm_available():
        available.append(alias)
```

Add availability check:
```python
def is_vllm_available() -> bool:
    try:
        import vllm
        return True
    except ImportError:
        return False
```

---

## 3. New Client: `VLLMClient`

### `src/llm_bawt/clients/vllm_client.py`

```python
class VLLMClient(LLMClient):
    SUPPORTS_STREAMING = True

    def __init__(self, model_id: str, config: Config, model_definition: dict | None = None):
        super().__init__(model_id, config, model_definition)
        self.engine = None  # vllm.LLM instance, loaded lazily or in __init__
        self._load_engine()

    def _load_engine(self):
        from vllm import LLM
        md = self.model_definition or {}
        self.engine = LLM(
            model=self.model,
            quantization=md.get("quantization"),
            dtype=md.get("dtype", "auto"),
            max_model_len=md.get("max_model_len", 4096),
            gpu_memory_utilization=md.get("gpu_memory_utilization", 0.85),
            enforce_eager=md.get("enforce_eager", False),
            enable_prefix_caching=md.get("enable_prefix_caching", True),
            trust_remote_code=True,
        )

    def query(self, messages, plaintext_output=False, **kwargs) -> str:
        # Apply chat template, call self.engine.generate(), return text

    def stream_raw(self, messages, **kwargs):
        # vLLM Python API doesn't stream per-token easily
        # Option A: yield full response as single chunk
        # Option B: use vllm.AsyncLLMEngine for token-by-token (complex)
        # Start with Option A, upgrade later

    def supports_native_tools(self) -> bool:
        return False  # In-process vLLM doesn't produce OpenAI tool_call objects
        # Native tool support requires the OpenAI-compatible server layer
        # Use react or native_json format instead

    def unload(self):
        # Delete engine, run torch.cuda.empty_cache()
        # Note: full VRAM release may require process restart (torch limitation)

    def get_styling(self) -> tuple[str | None, str]:
        return ("bright_cyan", f"vLLM: {self.model}")
```

### Key design note: streaming

vLLM's Python `LLM.generate()` API is batch-oriented, not streaming. For true token-by-token streaming:
- **Phase 1**: Return full response as single chunk (functional, not ideal UX).
- **Phase 2**: Use `AsyncLLMEngine` with `add_request()` + async iterator for real streaming. This is more complex but gives proper per-token output.

### Alternative: server-mode via OpenAI client

For users who want full streaming + native tool calls, they can run vLLM as a server (`vllm serve MODEL`) and configure it as `type: openai` with `base_url` pointing to the vLLM server. This already works today with zero code changes. Document this as the "power user" path.

---

## 4. Client Initialization

### `src/llm_bawt/service/core.py` — `ServiceLLMBawt._initialize_client()`

Add `elif` branch:

```python
elif model_type == PROVIDER_VLLM:
    from llm_bawt.clients.vllm_client import VLLMClient
    model_id = self.model_definition.get("model_id", self.resolved_model_alias)
    client = VLLMClient(model_id, config=self.config, model_definition=self.model_definition)

elif model_type == PROVIDER_GGUF and self.model_definition.get("backend") == "vllm":
    # GGUF-through-vLLM path
    from llm_bawt.clients.vllm_client import VLLMClient
    model_path = get_or_download_gguf_model(repo_id, filename, self.config)
    client = VLLMClient(str(model_path), config=self.config, model_definition=self.model_definition)
```

### `src/llm_bawt/core/client.py` — `LLMBawt._initialize_client()`

Optionally support vLLM in CLI mode too (currently restricted to `openai` only). If not, the existing service-mode routing handles it.

---

## 5. Tool Support Config

### Problem

Today, tool format is auto-selected by provider type:
- `openai` → `native` (structured tool calls)
- `gguf` / `ollama` / `hf` → `react` (text parsing)

This doesn't account for whether the model actually supports tool calling, and there's no way to disable tools for a specific model.

### New field: `tool_support`

Add an explicit `tool_support` field to model definitions:

```yaml
tool_support: native | react | none
```

| Value | Behavior |
|-------|----------|
| `native` | Use OpenAI-format structured tool calls (requires a client that supports it) |
| `react` | Use text-based ReAct parsing (works with any model) |
| `none` | Tools disabled for this model. If bot has `uses_tools: true`, tools are silently skipped and the model runs without them. Log a warning. |

If `tool_support` is not set, fall back to current auto-detection by provider type.

### Updated `get_tool_format()` in `config.py`

```python
def get_tool_format(self, model_alias=None, model_def=None) -> str:
    model_def = model_def or {}

    # 1. Explicit tool_support (new)
    tool_support = model_def.get("tool_support")
    if tool_support == "none":
        return "none"
    if tool_support in ("native", "react", "xml"):
        return tool_support

    # 2. Legacy tool_format override (existing)
    tool_format = model_def.get("tool_format")
    if tool_format:
        return str(tool_format)

    # 3. Auto-detect by provider type (existing)
    model_type = model_def.get("type")
    if model_type == PROVIDER_OPENAI:
        return "native"
    if model_type == PROVIDER_VLLM:
        return "react"  # in-process vLLM doesn't produce tool_call objects
    if model_type in (PROVIDER_GGUF, PROVIDER_OLLAMA, PROVIDER_HF):
        return "react"
    return "xml"
```

### Tool loop changes

In `BaseLLMBawt.query()` and the service streaming path, check for `tool_format == "none"`:

```python
if self.tool_format == "none":
    use_tools = False
```

This cleanly disables tools without breaking anything. The bot still loads, memory still works for context augmentation — the model just doesn't get tool instructions or enter the tool loop.

---

## 6. `llm --add-model vllm`

### CLI flow

```
$ llm --add-model vllm

? How would you like to add a model?
  1. Paste a HuggingFace model ID
  2. Search HuggingFace

> 2

? Search query: qwen 32b instruct awq

  # | Model                                    | Quant | Size   | Downloads
  1 | Qwen/Qwen2.5-32B-Instruct-AWQ            | AWQ   | 18 GB  | 45.2k
  2 | Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4      | GPTQ  | 18 GB  | 12.1k
  3 | Qwen/Qwen2.5-32B-Instruct                 | BF16  | 65 GB  | 89.3k

> 1

? Alias [qwen-32b-awq]:
✓ Added 'qwen-32b-awq' (type: vllm)
```

### Implementation: `src/llm_bawt/cli/vllm_handler.py`

```python
def handle_add_vllm(config: Config):
    """Interactive flow for adding a vLLM model."""
    # 1. Choose mode: paste or search
    # 2. If search:
    #    - Use huggingface_hub.HfApi().list_models(search=query, sort="downloads")
    #    - Filter: has safetensors or GGUF files
    #    - Detect quantization from model name/config (AWQ, GPTQ, FP8)
    #    - Show table with Rich
    # 3. If paste: accept model_id directly
    # 4. Auto-detect quantization from model config.json if possible
    # 5. Generate alias, prompt for confirmation
    # 6. Write to models.yaml
```

### Search filters (applied automatically)

- Must have safetensors or GGUF files
- Sort by downloads (descending)
- Show top 10 results
- Auto-detect quant type from model name patterns: `AWQ`, `GPTQ`, `FP8`, `GGUF`
- Show estimated model size from `safetensors_params` metadata

### Parser update: `src/llm_bawt/cli/parser.py`

Add `vllm` to `--add-model` choices:

```python
parser.add_argument("--add-model", choices=["ollama", "openai", "gguf", "vllm"])
```

---

## 7. Dependencies

### `pyproject.toml`

```toml
[project.optional-dependencies]
vllm = ["vllm>=0.15.0"]
```

### Install

```bash
# Via pipx
pipx runpip llm-bawt install vllm

# Via install.sh
./install.sh --with-vllm

# Docker
# Add build arg or separate compose service
```

### Availability check

```python
# src/llm_bawt/utils/config.py
def is_vllm_available() -> bool:
    try:
        import vllm
        return True
    except ImportError:
        return False
```

---

## 8. File Changes Summary

| File | Change |
|------|--------|
| `src/llm_bawt/utils/config.py` | Add `PROVIDER_VLLM`, `is_vllm_available()`, update `get_tool_format()`, `get_model_options()` |
| `src/llm_bawt/clients/vllm_client.py` | **New file** — `VLLMClient(LLMClient)` |
| `src/llm_bawt/service/core.py` | Add `elif model_type == "vllm"` in `_initialize_client()` |
| `src/llm_bawt/cli/parser.py` | Add `vllm` to `--add-model` choices |
| `src/llm_bawt/cli/app.py` | Add `elif args.add_model == "vllm"` handler |
| `src/llm_bawt/cli/vllm_handler.py` | **New file** — `handle_add_vllm()` with paste + search |
| `src/llm_bawt/core/base.py` | Handle `tool_format == "none"` (skip tool loop) |
| `pyproject.toml` | Add `[vllm]` optional dependency |
| `install.sh` | Add `--with-vllm` flag |
| `Dockerfile` | Add optional vLLM install layer |

---

## 9. Implementation Order

1. **Config**: Add `PROVIDER_VLLM`, `is_vllm_available()`, `tool_support` field handling
2. **Client**: Create `VLLMClient` with basic `query()` + `unload()`
3. **Service hookup**: Wire `_initialize_client()` for `type: vllm`
4. **Test manually**: Add a model to `models.yaml` by hand, verify it loads and responds
5. **CLI**: Add `--add-model vllm` with paste mode first, search second
6. **Tool support config**: Add `tool_support: none` handling
7. **GGUF backend override**: Add `backend: vllm` support for existing GGUF models
8. **Streaming**: Upgrade from single-chunk to real streaming (Phase 2)

---

## 10. Power User: vLLM as External Server

For users who want the best experience (real streaming, native tool calls, separate process lifecycle), document this existing capability:

```bash
# Start vLLM server
vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ --port 8000

# Add to models.yaml as type: openai
models:
  qwen-vllm:
    type: openai
    model_id: Qwen/Qwen2.5-32B-Instruct-AWQ
    base_url: http://localhost:8000/v1
    tool_support: native
```

This works today with zero code changes. Native tool calling, streaming, everything.
