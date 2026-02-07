# LLMBotHub - AI Coding Agent Instructions

> ⚠️ **IMPORTANT**: When running any tests through the LLM CLI, always use `nova` as the bot (`--bot nova` or default). **Do NOT use `mira`** for testing - Mira is a conversational companion with an unfiltered personality designed for personal use, not suitable for development/testing workflows.

## Architecture Overview

LLMBotHub is a model-agnostic LLM platform providing a unified API for configurable chatbots with persistent memory. Key components:

- **Entry Point**: `src/llmbothub/main.py` → `cli.py` (argparse-based CLI)
- **Core Engine**: `src/llmbothub/core.py` - `LLMBotHub` class orchestrates clients, bots, memory, and history
- **LLM Clients**: `src/llmbothub/clients/` - Abstract `LLMClient` base with implementations for OpenAI, Ollama, GGUF (llama-cpp), HuggingFace
- **Bot System**: `src/llmbothub/bots.py` + `bots.yaml` - Personality-based AI with isolated memory per bot
- **Memory Backend**: `src/llmbothub/memory/postgresql.py` - PostgreSQL + pgvector for semantic search
- **Memory Extraction**: `src/llmbothub/memory/extraction/` - LLM-based fact extraction from conversations
- **User Profiles**: `src/llmbothub/user_profile.py` - SQLModel ORM for user preferences injected into system prompts
- **Background Service**: `src/llmbothub/service/` - Optional FastAPI service for async/background LLM queries

### Data Flow
```
CLI args → Config (pydantic-settings) → LLMBotHub → LLMClient → Response
                                          ↓
                               HistoryManager ←→ Memory Backend
                                          ↓
                               MemoryExtractionService (fact distillation)
```

## Key Patterns

### Adding a New LLM Client
1. Create `src/llmbothub/clients/your_client.py` extending `LLMClient` from `base.py`
2. Implement required methods: `query()`, `get_styling()`, optionally `stream_raw()` for streaming
3. Set `SUPPORTS_STREAMING = True/False`
4. Register in `core.py:initialize_client()` method

### Bot Configuration
Bots are defined in `src/llmbothub/bots.yaml`. Each bot has:
- `slug`: Unique identifier (lowercase)
- `name`: Display name for UI
- `description`: Short description for `--list-bots`
- `system_prompt`: Personality/instructions
- `requires_memory`: Whether it needs PostgreSQL
- `voice_optimized`: Whether output is optimized for TTS
- `default_model`: Optional model override

**Available bots:**
- `nova` (default): Full-featured assistant with persistent memory
- `spark`: Lightweight local assistant (no database)
- `mira`: Conversational companion (personal use only)

### Memory System
- **Entry point plugin**: `pyproject.toml` registers `llmbothub.memory` via `discover_memory_backends()` in `core.py`
- **Two-tier storage**: `{bot_id}_messages` (raw history) + `{bot_id}_memories` (extracted facts with embeddings)
- **Memory extraction**: LLM-based fact distillation via `MemoryExtractionService`
- **Embeddings**: Local sentence-transformers (default: `all-MiniLM-L6-v2`) for semantic search
- **Local mode** (`--local`): Bypasses database, uses filesystem-based history

### Configuration
Config uses `pydantic-settings` with `LLMBOTHUB_` env prefix. Key files:
- `~/.config/llmbothub/.env` - API keys and database credentials
- `~/.config/llmbothub/models.yaml` - Model definitions

**Key config sections:**
- Memory settings: `MEMORY_N_RESULTS`, `MEMORY_MIN_RELEVANCE`, `MEMORY_DECAY_HALF_LIFE_DAYS`
- PostgreSQL: `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DATABASE`
- LLM generation: `MAX_TOKENS`, `TEMPERATURE`, `TOP_P`
- Service: `SERVICE_HOST`, `SERVICE_PORT`

### Background Service
Optional FastAPI-based background service for async LLM queries:
- Install with: `./install.sh --with-service`
- Run with: `llm-service` or `llmbothub-service`
- Client: `ServiceClient` in `src/llmbothub/service/client.py`

## Installation & Development Setup

The project uses **pipx** for global installation (no venv needed other than IDE). The `install.sh` script manages installation:

### Fresh Install (from GitHub)
```bash
curl -fsSL https://raw.githubusercontent.com/zenoran/llmbothub/main/install.sh | bash
```

### Development Install (editable mode)
```bash
# From the project directory - installs globally via pipx in editable mode
./install.sh --local . --with-service

# With all optional dependencies
./install.sh --local . --all
```

### Reinstalling After Code Changes
For most changes, editable mode means changes take effect immediately. If entry points or dependencies change:
```bash
./install.sh --local .
```

### Install Script Options
- `--with-llama`: Install llama-cpp-python for local GGUF models
- `--with-hf`: Install HuggingFace transformers + torch
- `--with-service`: Install FastAPI service for background tasks & API
- `--all`: Install all optional dependencies
- `--no-cuda`: Skip CUDA support for llama-cpp-python
- `--local <path>`: Install from local path in editable mode
- `--uninstall`: Remove llmbothub

### Inject Additional Dependencies
```bash
# Add dev dependencies to the pipx environment
pipx runpip llmbothub install ruff mypy pytest pytest-cov pytest-mock
```

## Development Commands

```bash
# Linting and formatting (run from project root)
ruff check src/                        # Lint
ruff format src/                       # Format
mypy src/                              # Type check
pytest                                 # Run tests

# Search codebase (grep is aliased to rg/ripgrep)
rg "pattern" src/                      # Search for pattern
rg -t py "function_name" src/          # Search only Python files
rg -l "import something" src/          # List files containing pattern
```

## Important Conventions

- **Installation**: Uses `pipx` for global install via `install.sh` (not venv)
- **Python command**: Always use `uv run python` (not `python` or `python3`) for running Python scripts
- **Search tool**: `grep` is aliased to `rg` (ripgrep) - use ripgrep syntax
- **Source layout**: All code in `src/llmbothub/` (pyproject.toml `package-dir`)
- **Logging**: Use `logging.getLogger(__name__)` - verbosity controlled by `--verbose` flag
- **Rich output**: Use `Console` from rich for all terminal output; respect `PLAIN_OUTPUT` config
- **Type hints**: Required throughout; checked with mypy
- **Model definitions**: External YAML, not hardcoded - use `config.defined_models.get("models", {})`
- **Optional dependencies**: Check availability with `is_huggingface_available()`, `is_llama_cpp_available()`

## Common Tasks

### Test a model change
```bash
llm --status                    # Check current config
llm -m gpt4 "test"              # Test specific model
llm --local "test"              # Test without database
llm --bot nova "test"           # Explicitly use nova bot (recommended for testing)
```

### Debug memory issues
- Check `POSTGRES_*` env vars in `~/.config/llmbothub/.env`
- Memory tables are per-bot: `nova_messages`, `nova_memories`, etc.
- Use `--verbose` to see memory retrieval logs

### Add a new config setting
1. Add `Field()` to `Config` class in `src/llmbothub/utils/config.py`
2. Use `LLMBOTHUB_` prefix for environment variable
3. Access via `config.YOUR_SETTING` in components

### Add a new bot
1. Add bot definition to `src/llmbothub/bots.yaml`
2. Bot automatically becomes available via `--bot <slug>`
3. Memory is automatically isolated per bot

### Troubleshooting Installation
If `llm` command is not found after code changes:
```bash
# Reinstall in editable mode
./install.sh --local .

# Verify installation
which llm                              # Should be in ~/.local/bin/
llm --status                           # Check it works
```

If dependencies are missing:
```bash
# Inject missing packages into the pipx environment
pipx runpip llmbothub install <package>
```

## Optional Dependencies
- `[dev]`: ruff, mypy, pytest for development (inject via `pipx runpip`)
- `[memory]`: sentence-transformers for local embeddings
- `[huggingface]`: transformers, torch for HuggingFace models
- `[llamacpp]`: llama-cpp-python for local GGUF models
- `[service]`: fastapi, uvicorn for background service
