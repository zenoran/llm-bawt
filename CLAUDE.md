# Claude Code Instructions for llm-bawt

## Project Overview

Model-agnostic LLM platform providing a unified, OpenAI-compatible API for configurable chatbots and multi-agent systems. Supports OpenAI, Ollama-compatible APIs, and local GGUF models with persistent semantic memory, MCP tools, and web search.

## Project Structure

```
src/llm_bawt/
├── cli/                    # CLI package
│   ├── app.py              # Main CLI logic, query handling
│   ├── parser.py           # Argument parsing
│   ├── main.py             # Entry point
│   ├── config_wizard.py    # Interactive config setup
│   └── commands/           # Subcommands (status, models, profile)
├── core/                   # Core orchestration
│   ├── base.py             # BaseLLMBawt - shared logic for CLI/service
│   ├── client.py           # Client initialization
│   ├── pipeline.py         # RequestPipeline, PipelineContext
│   ├── prompt_builder.py   # PromptBuilder, SectionPosition
│   └── model_lifecycle.py  # Model lifecycle management
├── clients/                # LLM clients
│   ├── base.py             # LLMClient base class
│   ├── openai_client.py    # OpenAI/Ollama-compatible
│   └── llama_cpp_client.py # Local GGUF models
├── adapters/               # Model-specific adapters
│   ├── base.py             # ModelAdapter ABC
│   ├── registry.py         # Adapter registration/discovery
│   ├── default.py          # DefaultAdapter (no-op)
│   ├── pygmalion.py        # BBCode/role marker cleanup
│   └── dolphin.py          # Observation hallucination cleanup
├── tools/                  # Tool system
│   ├── definitions.py      # Tool definitions
│   ├── executor.py         # Tool execution
│   ├── loop.py             # Tool loop handling
│   ├── parser.py           # Tool call parsing
│   ├── streaming.py        # Streaming with tools
│   └── formats/            # Tool format handlers (native/ReAct/XML)
├── search/                 # Web search
│   ├── base.py             # SearchClient base
│   ├── ddgs_client.py      # DuckDuckGo
│   ├── tavily_client.py    # Tavily
│   ├── brave_client.py     # Brave Search
│   └── factory.py          # get_search_client()
├── memory/                 # Memory backend
│   ├── postgresql.py       # pgvector storage, decay, search
│   ├── embeddings.py       # sentence-transformers (MiniLM)
│   ├── extraction/         # LLM-based fact extraction
│   ├── context_builder.py  # Memory context assembly
│   ├── consolidation.py    # Memory consolidation
│   └── summarization.py    # Conversation summarization
├── memory_server/          # MCP memory server
│   ├── server.py           # FastMCP server
│   ├── client.py           # MemoryClient, get_memory_client()
│   ├── storage.py          # Storage backend
│   └── extraction.py       # Extraction service
├── service/                # Background API service
│   ├── server.py           # FastAPI server
│   ├── api.py              # API routes
│   ├── core.py             # ServiceLLMBawt
│   ├── client.py           # ServiceClient
│   ├── scheduler.py        # JobScheduler, background tasks
│   └── tasks.py            # Async task processing
├── integrations/           # External service integrations
│   └── nextcloud/          # Nextcloud Talk bot routing
├── bots.py                 # Bot personalities, BotManager
├── bots.yaml               # Bot personality definitions
├── profiles.py             # ProfileManager, EntityType, AttributeCategory
├── model_manager.py        # Model definitions, aliases
├── gguf_handler.py         # GGUF model handling
└── utils/
    ├── config.py           # Config class, has_database_credentials()
    ├── env.py              # Environment handling
    ├── history.py          # HistoryManager, Message
    ├── input_handler.py    # MultilineInputHandler
    └── streaming.py        # render_streaming_response()
```

## Development Workflow

**Primary development is Docker-based.** Use `./start.sh` for most development tasks.

```bash
# Start dev environment (live source mounting from ./src)
./start.sh dev

# After code changes - just restart (no rebuild needed)
./start.sh restart

# View logs
./start.sh logs

# Run commands in container
./start.sh exec llm --status
./start.sh shell            # Open bash in container

# Stop everything
./start.sh down

# Full rebuild (only needed for dependency changes)
./start.sh rebuild
```

### Non-Docker Development (alternative)

```bash
./install.sh --dev          # Sync .venv with all deps
./server.sh start --dev     # Start MCP + LLM service with auto-reload
./server.sh stop            # Stop services
```

## Key Commands

```bash
# CLI
llm "question"              # Ask a question
llm                         # Interactive mode
llm -m gpt4 "question"      # Specific model
llm -b nova "question"      # Specific bot
llm --status                # System status
llm --list-models           # Available models
llm --list-bots             # Available bots
```

## Service Architecture

Two-service stack managed by `server.sh`:

1. **MCP Memory Server** (port 8001) - `llm-mcp-server`
   - Model Context Protocol server
   - Memory operations, fact extraction

2. **LLM Service** (port 8642) - `llm-service`
   - OpenAI-compatible API
   - Async task processing

## Key Patterns

### Client Initialization
```python
from llm_bawt.clients import LLMClient
from llm_bawt.core import LLMBawt
```

### Memory Client
```python
from llm_bawt.memory_server.client import MemoryClient, get_memory_client
```

### Search Client
```python
from llm_bawt.search import get_search_client, SearchClient
```

### Config Access
```python
from llm_bawt.utils.config import Config, has_database_credentials
config = Config()
```

### Bot/Profile Management
```python
from llm_bawt.bots import Bot, BotManager
from llm_bawt.profiles import ProfileManager, EntityType, AttributeCategory
```

## Configuration

Config location: `~/.config/llm-bawt/.env`

All settings use `LLM_BAWT_` prefix:
- `LLM_BAWT_DEFAULT_MODEL_ALIAS`
- `LLM_BAWT_DEFAULT_BOT`
- `LLM_BAWT_DEFAULT_USER`
- `LLM_BAWT_MEMORY_SERVER_URL` (for service communication)

## Dependencies (extras)

- `memory` - sentence-transformers for embeddings
- `service` - FastAPI, uvicorn, httpx
- `search` - ddgs, tavily-python
- `mcp` - FastMCP for memory server
- `huggingface` - transformers, torch

## Testing

```bash
uv run pytest
uv run llm --status         # Quick sanity check
```

## Common Tasks

### Adding a new tool
1. Add definition in `tools/definitions.py`
2. Add executor in `tools/executor.py`
3. Tool loop in `tools/loop.py` handles iteration

### Adding a new search provider
1. Create client in `search/` inheriting from `SearchClient`
2. Register in `search/factory.py`

### Adding a new LLM client
1. Create client in `clients/` inheriting from `LLMClient`
2. Implement `chat()` and `chat_stream()` methods
