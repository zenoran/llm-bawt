# llm-bawt — Agent Guide

A model-agnostic LLM platform providing a unified, OpenAI-compatible API for configurable chatbots and multi-agent systems across cloud and local models.

**Python 3.12+** | **uv** for dependency management | **ruff** for linting/formatting | **mypy** for type checking

---

## Project Overview

llm-bawt normalizes providers (OpenAI, Ollama, GGUF), augments conversations with persistent semantic memory, and integrates MCP tools and web search — enabling consistent behavior, shared context, and extensible tooling through a single interface.

### Key Features
- **Multi-Provider Support** — OpenAI, Ollama-compatible APIs, and local GGUF models
- **Bot Personalities** — Configurable chatbot identities with isolated memory, tools, and default models (YAML-driven)
- **Persistent Memory** — PostgreSQL + pgvector semantic memory with temporal decay and contradiction detection
- **Tool System** — Dual-mode tool calling: native OpenAI function calling and ReAct format for local/open models
- **Web Search** — Pluggable search providers (DuckDuckGo, Tavily, Brave)
- **MCP Server** — Model Context Protocol memory server for cross-service memory operations
- **Background Service** — FastAPI-based OpenAI-compatible API with async task processing

---

## Project Structure

```
src/llm_bawt/
├── cli/                    # CLI package
│   ├── app.py              # Main CLI logic, argument parsing, query handling
│   ├── config_setup.py     # Interactive .env config walkthrough (llm --setup)
│   ├── main.py             # Entry point
│   └── commands/           # Subcommands (models, profile, status)
├── core/                   # Core orchestration
│   ├── base.py             # BaseLLMBawt - shared logic for CLI/service
│   ├── client.py           # Client initialization
│   ├── pipeline.py         # RequestPipeline (7-stage processing)
│   ├── prompt_builder.py   # PromptBuilder, SectionPosition
│   └── model_lifecycle.py  # Model lifecycle management
├── clients/                # LLM clients
│   ├── base.py             # LLMClient ABC
│   ├── openai_client.py    # OpenAI/Ollama-compatible
│   └── llama_cpp_client.py # Local GGUF models
├── adapters/               # Model-specific output adapters
│   ├── base.py             # ModelAdapter ABC
│   ├── registry.py         # Adapter registration/discovery
│   ├── default.py          # DefaultAdapter (passthrough)
│   ├── pygmalion.py        # BBCode/role marker cleanup
│   └── dolphin.py          # Observation hallucination cleanup
├── tools/                  # Tool/function calling system
│   ├── definitions.py      # Tool definitions
│   ├── executor.py         # Tool execution
│   ├── loop.py             # Tool call loop handling
│   ├── parser.py           # Tool call parsing
│   ├── streaming.py        # Streaming with tools
│   └── formats/            # Tool format handlers
│       ├── base.py         # ToolFormatHandler ABC
│       ├── native_openai.py # Native OpenAI format
│       ├── react.py        # ReAct format
│       └── xml_legacy.py   # Legacy XML format
├── search/                 # Web search providers
│   ├── base.py             # SearchClient ABC
│   ├── ddgs_client.py      # DuckDuckGo (free, no API key)
│   ├── tavily_client.py    # Tavily (production, API key)
│   ├── brave_client.py     # Brave Search
│   └── factory.py          # get_search_client()
├── memory/                 # Persistent semantic memory
│   ├── base.py             # MemoryBackend ABC
│   ├── postgresql.py       # pgvector storage, decay, search
│   ├── embeddings.py       # sentence-transformers (MiniLM)
│   ├── context_builder.py  # Memory context assembly
│   ├── consolidation.py    # Memory consolidation
│   ├── summarization.py    # Conversation summarization
│   ├── maintenance.py      # Memory maintenance tasks
│   ├── profile_maintenance.py # Profile memory maintenance
│   ├── migrations.py       # Database migrations
│   └── extraction/         # LLM-based fact extraction
│       ├── prompts.py      # Extraction prompts
│       └── service.py      # Extraction service
├── memory_server/          # MCP memory server
│   ├── __main__.py         # Direct execution entry
│   ├── server.py           # FastMCP server
│   ├── client.py           # MemoryClient
│   ├── storage.py          # Storage backend
│   └── extraction.py       # Extraction service
├── service/                # Background FastAPI service
│   ├── server.py           # FastAPI server
│   ├── api.py              # API routes
│   ├── core.py             # ServiceLLMBawt
│   ├── client.py           # ServiceClient
│   ├── logging.py          # Request/response logging
│   ├── scheduler.py        # JobScheduler
│   └── tasks.py            # Async task processing
├── integrations/           # External service integrations
│   └── nextcloud/          # Nextcloud Talk bot routing
├── models/
│   └── message.py          # Message model
├── shared/
│   └── logging.py          # Shared logging setup
├── bots.py                 # Bot, BotManager (YAML-driven)
├── bots.yaml               # Bot personality definitions
├── profiles.py             # ProfileManager
├── model_manager.py        # Model definitions, aliases
├── gguf_handler.py         # GGUF model handling
├── memory_debug.py         # Memory debugging utility
└── utils/
    ├── config.py           # Config (pydantic-settings)
    ├── env.py              # Environment handling
    ├── history.py          # HistoryManager
    ├── input_handler.py    # MultilineInputHandler
    ├── paths.py            # Log directory resolution
    ├── streaming.py        # render_streaming_response()
    └── vram.py             # VRAMInfo, GPU detection
```

### Other Important Files

```
├── pyproject.toml          # Project config, dependencies, entry points
├── Dockerfile              # Multi-stage NVIDIA CUDA build
├── docker-compose.yml      # Production Docker compose
├── docker-compose.dev.yml  # Dev mode override (mounts ./src)
├── .env.docker             # Docker environment template
├── Makefile                # Cross-platform build targets
├── install.sh              # Installer script (pipx/uv)
├── server.sh               # MCP + LLM service management
├── run.sh                  # Docker wrapper script
├── tests/                  # Test suite (pytest)
├── docs/                   # Extended documentation
└── scripts/                # Utility scripts (cleanup_profile, rebuild_profile)
```

---

## Build and Development Commands

### Docker Development (Recommended)

```bash
./run.sh dev                # Start dev mode (live source mounting from ./src)
./run.sh restart            # Restart after code changes (no rebuild needed)
./run.sh logs               # Follow container logs
./run.sh exec llm --status  # Run commands in container
./run.sh shell              # Open bash in container
./run.sh down               # Stop everything
./run.sh rebuild            # Full rebuild (only for dependency changes)
```

### Makefile Targets (Cross-Platform)

```bash
make dev                    # Sync .venv with all extras (uv-based)
make docker-dev             # Docker compose dev mode
make up                     # Docker compose production
make down                   # Stop Docker containers
make test                   # Run tests
make test-cov               # Run tests with coverage
make lint                   # Lint with ruff
make format                 # Format with ruff
make typecheck              # Type check with mypy
make server-start           # Start local MCP + LLM services
make server-stop            # Stop local services
make clean                  # Remove build artifacts and caches
make clean-all              # Deep clean including .venv
```

### Non-Docker Development

```bash
./install.sh --dev          # Sync .venv with all deps
./server.sh start --dev     # Start MCP + LLM service with auto-reload
./server.sh stop            # Stop services
uv run llm --status         # Run CLI through uv
```

### CLI Commands

```bash
llm "question"              # Ask a question
llm                         # Interactive mode
llm -m gpt4 "question"     # Specific model alias
llm -b nova "question"     # Specific bot personality
llm --setup                 # Interactive .env config walkthrough
llm --status                # System status
llm --list-models           # Available models
llm --list-bots             # Available bots
```

---

## Testing Instructions

```bash
uv run pytest                               # Run all tests
uv run pytest tests/test_adapters.py        # Run specific test file
uv run pytest --cov=src --cov-report=term-missing  # With coverage
```

Test files in `tests/`:
- `test_adapters.py` - Model adapter tests
- `test_brave_search.py` - Brave search tests
- `test_integration.py` - Integration tests
- `test_scheduler.py` - Scheduler tests
- `test_tool_formats.py` - Tool format tests
- `test_tool_loop_integration.py` - Tool loop tests

---

## Code Style Guidelines

### Type Annotations
- **Required** on all public methods, properties, and dataclass fields
- Use Python 3.10+ union syntax: `str | None` (not `Optional[str]`)
- Use `TYPE_CHECKING` imports to avoid circular dependencies:
  ```python
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from ..other_module import SomeClass
  ```

### Naming Conventions
- **Classes**: PascalCase (`RequestPipeline`, `PromptBuilder`)
- **Functions/methods**: snake_case (`get_max_tokens()`, `format_message()`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_BOT`, `PRE_PROCESS`)
- **Private**: leading underscore (`_stage_pre_process()`, `_text_similarity()`)
- **Bot slugs**: lowercase, no spaces

### Logging
```python
logger = logging.getLogger(__name__)  # Module-level
```
Use `logger.debug()` for internal detail, `logger.info()` for user-facing events, `logger.warning()` for recoverable issues, `logger.exception()` for caught exceptions with traceback.

### Terminal Output
All user-facing output via **Rich** (`Console`, `Panel`, `Markdown`). Respect `PLAIN_OUTPUT` config for non-terminal contexts. Escaped brackets in Panel titles: `\[model\]`.

### Error Handling
- Try/except at system boundaries (file I/O, network, database)
- Log exceptions with `logger.exception()`
- Graceful fallbacks: memory/search failures log warnings but don't crash queries
- Validation errors in `__init__` with explicit `ValueError`

### Dataclasses
Used extensively for value objects: `Message`, `SearchResult`, `PromptSection`, `VRAMInfo`, `PipelineContext`, `ToolCallRequest`, `Bot`. Use `field(default_factory=list)` for mutable defaults. `__post_init__` for validation.

### Import Style
- Standard library → third-party → local imports
- Relative imports within the package (`from ..utils.config import Config`)
- Lazy imports in functions when optional dependencies may be missing

---

## Configuration

**Config location:** `~/.config/llm-bawt/.env`

**Discovery order:** `LLM_BAWT_ENV_FILE` env var → `cwd/.env` → `~/.config/llm-bawt/.env`

All settings use `LLM_BAWT_` prefix (pydantic-settings `env_prefix`):

| Setting | Purpose |
|---------|---------|
| `LLM_BAWT_DEFAULT_MODEL_ALIAS` | Default model alias |
| `LLM_BAWT_DEFAULT_BOT` | Default bot personality |
| `LLM_BAWT_DEFAULT_USER` | Default user identity |
| `LLM_BAWT_MEMORY_SERVER_URL` | MCP server URL for service communication |
| `LLM_BAWT_POSTGRES_HOST/PORT/USER/PASSWORD/DATABASE` | PostgreSQL connection |
| `LLM_BAWT_SEARCH_PROVIDER` | Search backend (duckduckgo/tavily/brave) |
| `LLM_BAWT_MAX_TOKENS` | Default max tokens for generation |
| `LLM_BAWT_HISTORY_DURATION` | History window in seconds |
| `LLM_BAWT_MEMORY_EXTRACTION_ENABLED` | Enable LLM-based fact extraction |
| `LLM_BAWT_MEMORY_DECAY_ENABLED` | Enable memory relevance decay |
| `LLM_BAWT_SCHEDULER_ENABLED` | Enable background task scheduler |
| `LLM_BAWT_DEBUG_TURN_LOG` | Enable detailed turn logging |

See `.env.docker` for a complete template with all available settings.

---

## Service Architecture

Two-service stack:

1. **MCP Memory Server** (port 8001) - `llm-mcp-server`
   - Model Context Protocol server for memory operations and fact extraction
   - Started first; LLM service waits for it to be ready

2. **LLM Service** (port 8642) - `llm-service`
   - OpenAI-compatible REST API
   - Async task processing with JobScheduler
   - Health check at `GET /health`

Managed by `server.sh` (local) or Docker compose.

---

## Entry Points

| Command | Module | Purpose |
|---------|--------|---------|
| `llm` / `llm-bawt` | `llm_bawt.main:main` | CLI interface |
| `llm-service` / `llm-bawt-service` | `llm_bawt.service.server:main` | FastAPI service |
| `llm-mcp-server` | `llm_bawt.memory_server:run_server` | MCP memory server |
| `llm-memory` | `llm_bawt.memory_debug:main` | Memory debug utility |
| `llm-nextcloud` | `llm_bawt.integrations.nextcloud.cli:nextcloud_cli` | Nextcloud CLI |

---

## Architecture Patterns

### Pipeline Pattern (`core/pipeline.py`)

Request processing uses a 7-stage pipeline: `PRE_PROCESS` → `CONTEXT_BUILD` → `MEMORY_RETRIEVAL` → `HISTORY_FILTER` → `MESSAGE_ASSEMBLY` → `EXECUTE` → `POST_PROCESS`. Each stage is a discrete method (`_stage_*`). `PipelineContext` dataclass carries state through all stages. Hooks can be registered at any stage via `add_hook()`.

### ABC + Registry Pattern

Extensible components use abstract base classes with registry/factory patterns:

| ABC | Location | Implementations |
|-----|----------|-----------------|
| `LLMClient` | `clients/base.py` | OpenAIClient, LlamaCppClient |
| `ModelAdapter` | `adapters/base.py` | DefaultAdapter, PygmalionAdapter, DolphinAdapter |
| `SearchClient` | `search/base.py` | DDGSClient, TavilyClient, BraveClient |
| `MemoryBackend` | `memory/base.py` | PostgreSQLMemoryBackend |
| `ToolFormatHandler` | `tools/formats/base.py` | NativeOpenAIHandler, ReActHandler, XMLLegacyHandler |

### Prompt Builder (`core/prompt_builder.py`)

Composable prompt assembly with ordered sections. Each `PromptSection` has a name, content, and position. `SectionPosition` defines standard orderings (`USER_CONTEXT=0`, `BOT_TRAITS=1`, etc.). Supports method chaining.

### Bot System (`bots.py` + `bots.yaml`)

Bot personalities defined in YAML with slugs, system prompts, and capability flags (`requires_memory`, `uses_tools`, `uses_search`). User overrides via `~/.config/llm-bawt/bots.yaml` are deep-merged with repo defaults.

### Config Pattern (`utils/config.py`)

Uses pydantic-settings `BaseSettings` with `LLM_BAWT_` prefix. All settings declared as `Field()` with descriptions and defaults. Dependency availability checked via `is_huggingface_available()`, `is_llama_cpp_available()` helpers.

---

## Common Tasks

### Adding a New Tool
1. Add definition in `tools/definitions.py`
2. Add executor in `tools/executor.py`
3. Tool loop in `tools/loop.py` handles iteration automatically

### Adding a New Search Provider
1. Create client in `search/` inheriting from `SearchClient`
2. Add `PROVIDER` class variable and `REQUIRES_API_KEY` flag
3. Implement `search()` method (and optionally `search_news()`)
4. Register in `search/factory.py`

### Adding a New LLM Client
1. Create client in `clients/` inheriting from `LLMClient`
2. Implement abstract methods: `query()`, `get_styling()`
3. Optionally override `stream_raw()`, `query_with_tools()`
4. Register in `core/client.py`

### Adding a New Model Adapter
1. Create adapter in `adapters/` inheriting from `ModelAdapter`
2. Override methods as needed: `clean_output()`, `get_stop_sequences()`, `supports_system_role()`, `transform_messages()`
3. Register in `adapters/registry.py`

### Adding a New Bot
1. Add entry to `src/llm_bawt/bots.yaml` with slug, name, description, system_prompt, and capability flags
2. Or add to user override file at `~/.config/llm-bawt/bots.yaml`

### Adding a Configuration Setting
1. Add `Field()` to `Config` class in `utils/config.py` with `LLM_BAWT_` prefix
2. Add to `.env.docker` template with documentation comment

---

## Dependencies (Extras)

| Extra | Purpose | Key packages |
|-------|---------|-------------|
| `dev` | Development tools | ruff, mypy, pytest |
| `memory` | Embeddings | sentence-transformers |
| `service` | API service | fastapi, uvicorn, httpx |
| `search` | Web search | ddgs, tavily-python |
| `mcp` | Memory server | mcp[cli] |
| `huggingface` | Local HF models | transformers, torch, accelerate |
| `llamacpp` | Local GGUF models | llama-cpp-python |

---

## Documentation

Extended docs in `docs/`:
- `TOOL_CALLING.md` - Tool calling system
- `MODEL_ADAPTERS.md` - Model adapter pattern
- `SEARCH_PROVIDERS.md` - Search providers
- `BACKGROUND_SCHEDULER.md` - Job scheduler
- `CONTEXT_AND_MEMORY_REDESIGN.md` - Memory system design
- `NEXTCLOUD_INTEGRATION.md` - Nextcloud Talk integration

---

## Security Considerations

- API keys stored in `~/.config/llm-bawt/.env` (user-only readable)
- PostgreSQL password is the key indicator for database availability
- Nextcloud bot secrets should be kept secure
- Tavily/Brave API keys are optional; DuckDuckGo is free and requires no key
- Docker containers run with GPU access when available

---

## Branching Strategy

This project uses a Gitflow-style branching model:

| Branch | Purpose |
|--------|---------|
| `main` | Production-ready releases. Protected — merge via PR only. |
| `release/*` | Release candidates. Branch from `develop`, merge to `main` and back to `develop`. |
| `develop` | Integration branch. All feature work merges here. |
| `feature/*` | Feature branches. Branch from `develop`, PR back to `develop`. |
| `hotfix/*` | Urgent fixes. Branch from `main`, merge to both `main` and `develop`. |

**Workflow:**
1. Create a feature branch from `develop`: `git checkout -b feature/my-feature develop`
2. Do your work, commit, push
3. Open a PR targeting `develop`
4. After review and merge, `develop` accumulates features for the next release
5. When ready to release, create `release/x.y.z` from `develop` for final testing
6. Merge `release/x.y.z` into `main` and tag the release
