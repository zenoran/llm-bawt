# llm-bawt

A model-agnostic LLM platform that provides a unified, OpenAI-compatible API for running configurable chatbots and multi-agent systems across cloud and local models.

llm-bawt normalizes providers (OpenAI, Ollama, GGUF), augments conversations with persistent semantic memory, and integrates MCP tools and web search — enabling consistent behavior, shared context, and extensible tooling through a single interface.

## Features

- **Multi-Provider Support** — OpenAI, Ollama-compatible APIs, and local GGUF models through a unified client interface
- **Bot Personalities** — Configurable chatbot identities with isolated memory, tools, and default models (defined in YAML)
- **Persistent Memory** — PostgreSQL + pgvector semantic memory with temporal decay, access reinforcement, and contradiction detection
- **Tool System** — Dual-mode tool calling: native OpenAI function calling and ReAct format for local/open models
- **Web Search** — Pluggable search providers (DuckDuckGo, Tavily, Brave) with factory pattern
- **MCP Server** — Model Context Protocol memory server for cross-service memory operations
- **Background Service** — FastAPI-based OpenAI-compatible API with async task processing and job scheduling
- **Model Adapters** — Per-model output formatting and cleanup (role markers, BBCode, hallucination truncation)
- **Streaming** — Rich terminal output with streaming responses and tool call detection
- **Docker-First** — CUDA-enabled multi-stage Docker build with live source mounting for development

## Quick Start

### Install

```bash
# From GitHub (basic)
curl -fsSL https://raw.githubusercontent.com/zenoran/llm-bawt/main/install.sh | bash

# With all optional features
curl -fsSL https://raw.githubusercontent.com/zenoran/llm-bawt/main/install.sh | bash -s -- --all

# Local development
git clone https://github.com/zenoran/llm-bawt.git
cd llm-bawt
./install.sh --dev
```

### Configure

```bash
mkdir -p ~/.config/llm-bawt

# API key (required for cloud models)
echo "OPENAI_API_KEY=sk-..." >> ~/.config/llm-bawt/.env

# PostgreSQL for persistent memory (optional)
echo "POSTGRES_HOST=localhost" >> ~/.config/llm-bawt/.env
echo "POSTGRES_USER=llm_bawt" >> ~/.config/llm-bawt/.env
echo "POSTGRES_PASSWORD=yourpassword" >> ~/.config/llm-bawt/.env
echo "POSTGRES_DATABASE=llm_bawt" >> ~/.config/llm-bawt/.env
```

### Use

```bash
llm "what is the meaning of life"     # Ask a question
llm                                   # Interactive mode
llm -m gpt4 "explain quantum physics" # Use a specific model
llm --local "hello"                   # Use a local GGUF model
llm -b nova "help me code"            # Use a specific bot personality
llm --status                          # System status
llm --list-models                     # Available models
llm --list-bots                       # Available bot personalities
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                            CLI (llm)                                 │
│  Input → Bot/User Context → Memory Retrieval → LLM Client → Output  │
│                                  ↕                  ↕                │
│                           Tools / Search                             │
└──────────────────────────────────────────────────────────────────────┘
                                   │
┌──────────────────────────────────────────────────────────────────────┐
│                     Service Stack (server.sh)                        │
│  ┌──────────────────────────┐  ┌──────────────────────────────────┐  │
│  │ MCP Memory Server :8001  │←→│     LLM Service :8642            │  │
│  │  Memory ops, extraction  │  │  OpenAI-compat API, async tasks  │  │
│  └──────────────────────────┘  └──────────────────────────────────┘  │
│                          ↕                                           │
│                PostgreSQL + pgvector                                  │
│              (messages, memories, embeddings)                         │
└──────────────────────────────────────────────────────────────────────┘
```

### Query Pipeline

1. **Input** — Parse prompt, load bot personality, inject user profile
2. **Memory** — Semantic search for relevant past context (pgvector)
3. **Context** — Assemble system prompt + memories + conversation history
4. **Query** — Stream to configured model via unified client
5. **Tools** — Execute tool calls (search, memory recall) in a loop if needed
6. **Extract** — Background extraction of facts from the conversation into memory

### Project Structure

```
src/llm_bawt/
├── cli/                 # CLI entry point, argument parsing, subcommands
├── core/                # Orchestration: pipeline, prompt builder, model lifecycle
├── clients/             # LLM clients: OpenAI, llama.cpp (GGUF)
├── adapters/            # Per-model output formatting and cleanup
├── tools/               # Tool definitions, execution, format handlers (native/ReAct/XML)
├── search/              # Web search: DuckDuckGo, Tavily, Brave
├── memory/              # PostgreSQL backend, embeddings, extraction, consolidation
├── memory_server/       # MCP (Model Context Protocol) memory server
├── service/             # FastAPI background service, scheduler, API routes
├── integrations/        # External integrations (Nextcloud Talk)
├── utils/               # Config, history, streaming, input handling
├── bots.py              # Bot manager and personality loading
├── profiles.py          # User/bot profile management
├── model_manager.py     # Model definitions and aliases
└── bots.yaml            # Bot personality definitions
```

## Bots

Bot personalities are defined in `bots.yaml`. Each bot has its own system prompt, memory isolation, tool access, and optional default model.

| Bot | Description | Memory | Tools | Search |
|-----|-------------|--------|-------|--------|
| **nova** | Full-featured technical assistant (default) | Yes | Yes | Yes |
| **spark** | Lightweight local assistant — no database required | No | No | No |
| **proto** | Testing bot with separate memory for development | Yes | Yes | Yes |

Custom bots can be added by editing `src/llm_bawt/bots.yaml`.

## Memory System

The memory system uses PostgreSQL with pgvector for semantic storage. It's designed to evolve naturally rather than fossilize into static facts.

- **Fact Extraction** — LLM-based extraction of facts from conversations
- **Local Embeddings** — sentence-transformers (MiniLM) for vector similarity, no external API calls
- **Temporal Decay** — Memories fade over time unless reinforced by access
- **Contradiction Detection** — New facts supersede old ones (history preserved)
- **Diversity Sampling** — Retrieval includes varied time periods and memory types
- **Background Consolidation** — Scheduled jobs merge duplicates and maintain profiles

Memory types have different decay rates: core facts persist longest (~180 days half-life), while plans and events decay faster (~27-45 days).

## Tool System

Dual-mode tool calling with automatic format detection:

- **Native** — OpenAI function calling for compatible models (structured tool calls)
- **ReAct** — Thought/Action/Observation format for local and open models (30+ model aliases supported)
- **Available Tools** — Web search, memory recall, and extensible via `tools/definitions.py`

## Service Stack

Two-service architecture for background operation:

| Service | Port | Purpose |
|---------|------|---------|
| MCP Memory Server | 8001 | Memory operations and fact extraction via Model Context Protocol |
| LLM Service | 8642 | OpenAI-compatible API, async tasks, job scheduling |

```bash
# Development (non-Docker)
./server.sh start --dev     # Start both services with auto-reload
./server.sh stop            # Stop both
./server.sh status          # Show status

# Docker
./start.sh dev              # Dev mode with live source mounting
./start.sh up               # Production mode
./start.sh logs             # Follow logs
./start.sh shell            # Shell into container
```

## Integration Runbooks

- OpenClaw integration: [docs/OPENCLAW_INTEGRATION.md](docs/OPENCLAW_INTEGRATION.md)
- Nextcloud integration: [docs/NEXTCLOUD_INTEGRATION.md](docs/NEXTCLOUD_INTEGRATION.md)

## Development

### Setup

```bash
git clone https://github.com/zenoran/llm-bawt.git
cd llm-bawt
./install.sh --dev          # Sync .venv with all dependencies

# Or with Docker (recommended)
./start.sh dev              # Live source mounting, debug logging
./start.sh restart          # After code changes (no rebuild needed)
./start.sh rebuild          # Only for dependency changes
```

### Branching Strategy

This project uses a Gitflow-style branching model for multi-agent/team contribution:

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

### Testing

```bash
uv run pytest               # Run test suite
llm --status                # Quick sanity check
```

### Install Options

```bash
./install.sh --dev          # All deps for local development
./install.sh --with-llama   # Local GGUF model support (CUDA)
./install.sh --with-search  # Web search providers
./install.sh --with-service # FastAPI background service
./install.sh --all          # Everything
```

## Configuration

Config file: `~/.config/llm-bawt/.env` — all settings use the `LLM_BAWT_` prefix.

```bash
# Core
LLM_BAWT_DEFAULT_MODEL_ALIAS=gpt4     # Default model alias
LLM_BAWT_DEFAULT_BOT=nova              # Default bot personality
LLM_BAWT_DEFAULT_USER=your-user-id    # User identifier for memory

# Memory
LLM_BAWT_MEMORY_DECAY_ENABLED=true
LLM_BAWT_MEMORY_DECAY_HALF_LIFE_DAYS=90
LLM_BAWT_MEMORY_EMBEDDING_MODEL=all-MiniLM-L6-v2

# PostgreSQL
LLM_BAWT_POSTGRES_HOST=localhost
LLM_BAWT_POSTGRES_USER=llm_bawt
LLM_BAWT_POSTGRES_PASSWORD=yourpassword
LLM_BAWT_POSTGRES_DATABASE=llm_bawt

# Search
LLM_BAWT_SEARCH_PROVIDER=ddgs          # ddgs, tavily, or brave
LLM_BAWT_TAVILY_API_KEY=tvly-...       # If using Tavily
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `llm` | Main CLI — query LLMs, interactive mode |
| `llm-service` | Start the background LLM service (port 8642) |
| `llm-mcp-server` | Start the MCP memory server (port 8001) |
| `llm-memory` | Memory debugging utilities |
| `llm-nextcloud` | Nextcloud Talk bot management |
| `./server.sh` | Manage the full service stack (development) |
| `./start.sh` | Docker container management |

## License

MIT
