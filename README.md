# llm-bawt

A self-hosted AI assistant platform that gives you a single, consistent interface to talk to any LLM — cloud or local — with persistent memory, tool use, web search, and configurable bot personalities. You wire it up once, and every conversation your bots have is enriched with semantic memory that grows and fades naturally over time.

The system exposes an OpenAI-compatible API, so any frontend (web UI, voice interface, CLI) can connect to it. The companion project **[unmute](https://github.com/zenoran/unmute)** provides a real-time voice + chat web UI that connects to llm-bawt as its backend.

## What This System Does

llm-bawt is the brain behind an AI assistant that:

- **Remembers you** — facts from past conversations persist in semantic memory, decay over time unless reinforced, and are recalled when relevant
- **Uses tools** — the LLM can search the web, store/recall memories, control smart home devices, and fetch web pages mid-conversation
- **Works with any model** — swap between OpenAI, Grok (xAI), Claude (via Agent SDK), local GGUF models, Ollama, or vLLM without changing how you interact
- **Has personalities** — different bots with their own system prompts, memory spaces, tool access, and default models
- **Enables bot collaboration** — bots can communicate with each other via MCP tools, enabling delegation, specialist workflows, and multi-bot interactions
- **Streams everything** — responses stream to the terminal or web UI in real time with rich formatting

## Prerequisites

### Required

| Dependency | Purpose | Install |
|------------|---------|---------|
| **Python 3.12+** | Runtime | `apt install python3` or [python.org](https://www.python.org/) |
| **uv** | Package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **PostgreSQL 15+** with **pgvector** | Memory storage + vector search | [pgvector install guide](https://github.com/pgvector/pgvector#installation) |
| **At least one LLM API key** | Model access | OpenAI, xAI (Grok), or a local model |

### Optional

| Dependency | Purpose | When Needed |
|------------|---------|-------------|
| **Docker + Docker Compose** | Containerized deployment (recommended) | Production / service mode |
| **NVIDIA GPU + CUDA** | Local model inference (GGUF, vLLM) | Running models locally |
| **Redis** | Event transport for OpenClaw agent bridge | Agent backend integration |
| **pipx** | Isolated CLI install without Docker | CLI-only usage |

### PostgreSQL Setup

llm-bawt requires PostgreSQL with the pgvector extension. This is **not included** in the Docker Compose stack — you need to provide your own PostgreSQL instance.

```bash
# Option 1: Install locally
sudo apt install postgresql postgresql-contrib
# Install pgvector (Ubuntu/Debian)
sudo apt install postgresql-16-pgvector

# Option 2: Use an existing PostgreSQL server on your network

# Create the database and enable pgvector
sudo -u postgres psql <<EOF
CREATE USER llm_bawt WITH PASSWORD 'your-secure-password';
CREATE DATABASE llm_bawt OWNER llm_bawt;
\c llm_bawt
CREATE EXTENSION IF NOT EXISTS vector;
EOF
```

Tables are created automatically on first run via SQLAlchemy migrations.

## Installation

### Option A: Docker (Recommended for Full System)

This runs llm-bawt as a background service with the MCP memory server, Redis, web scraping, and GPU inference — all containerized.

```bash
git clone https://github.com/zenoran/llm-bawt.git
cd llm-bawt

# Create your environment config
cp .env.example .env   # or create from scratch (see Configuration below)

# Build and start (requires NVIDIA GPU + Docker with nvidia-container-toolkit)
make up                # Production mode
# or
make docker-dev        # Dev mode — mounts ./src for live code changes
```

The Docker build is a multi-stage CUDA-enabled image (~12 GB) that compiles llama-cpp-python and vLLM with GPU support. First build takes a while; subsequent builds use cached layers.

**What starts:**

| Container | Port | Purpose |
|-----------|------|---------|
| `llm-bawt-app` | 8001, 8642 | MCP memory server + OpenAI-compatible LLM API |
| `llm-bawt-redis` | 6379 | Event transport (for agent bridges) |
| `llm-bawt-crawl4ai` | 11235 | Web page extraction for the `web_fetch` tool |
| `llm-bawt-openclaw-bridge` | — | WebSocket bridge for OpenClaw agent backend (optional) |
| `llm-bawt-claude-code-bridge` | — | Claude Agent SDK bridge for Claude Code (optional) |

### Option B: CLI-Only Install (pipx)

If you just want the `llm` command without running the full service stack. Still requires PostgreSQL for memory features.

```bash
# Install from GitHub
pipx install "git+https://github.com/zenoran/llm-bawt.git"

# Or from a local clone (editable — code changes apply immediately)
git clone https://github.com/zenoran/llm-bawt.git
cd llm-bawt
pipx install --editable .

# Verify
llm --status
```

### Option C: Local Development [MOST TESTED]

```bash
git clone https://github.com/zenoran/llm-bawt.git
cd llm-bawt

# Sync .venv with all dependencies
./install.sh --dev

# Optional: add local model support
make dev-llama        # llama-cpp-python with CUDA
make dev-vllm         # vLLM for HuggingFace models

# Run from the venv
uv run llm --status
uv run llm "hello"

# Start the service stack locally (both MCP + LLM service)
./server.sh start --dev --stdout
```

## Configuration

All settings use the `LLM_BAWT_` prefix and are read from (in priority order):
1. Environment variables
2. `.env` file in the project root (for Docker) or `~/.config/llm-bawt/.env` (for CLI)

### Minimum Viable Config

```bash
# ~/.config/llm-bawt/.env  (CLI)
# or .env in repo root     (Docker)

# === REQUIRED ===
LLM_BAWT_DEFAULT_USER=your-name          # User ID for memory isolation
LLM_BAWT_POSTGRES_HOST=your-postgres-host
LLM_BAWT_POSTGRES_USER=llm_bawt
LLM_BAWT_POSTGRES_PASSWORD=your-secure-password
LLM_BAWT_POSTGRES_DATABASE=llm_bawt

# At least one model provider API key:
OPENAI_API_KEY=sk-...                     # OpenAI models
# and/or
LLM_BAWT_XAI_API_KEY=xai-...             # Grok models (also used for memory extraction)

# === RECOMMENDED ===
LLM_BAWT_DEFAULT_BOT=nova                # Default bot personality

# Grok highly recommended for a mostly unrestricted chat experience
LLM_BAWT_DEFAULT_MODEL_ALIAS=grok-4-fast # Default model to use
```

### Full Configuration Reference

```bash
# --- Service Mode ---
LLM_BAWT_USE_SERVICE=true                 # CLI delegates to running service
LLM_BAWT_SERVICE_HOST=0.0.0.0             # Service bind address
LLM_BAWT_SERVICE_PORT=8642                # LLM API port
LLM_BAWT_MEMORY_SERVER_URL=http://127.0.0.1:8001  # MCP memory server

# --- Memory ---
LLM_BAWT_MEMORY_DECAY_ENABLED=true
LLM_BAWT_MEMORY_N_RESULTS=10              # Semantic search results per query
LLM_BAWT_MEMORY_MIN_RELEVANCE=0.01        # Min confidence (0.0–1.0)
LLM_BAWT_MAINTENANCE_MODEL=grok-4-fast    # Model used for memory extraction jobs

# --- Generation ---
LLM_BAWT_MAX_CONTEXT_TOKENS=12000         # Total prompt token budget
LLM_BAWT_MAX_OUTPUT_TOKENS=4096           # Max response length
LLM_BAWT_TEMPERATURE=0.8                  # Sampling temperature

# --- Search Providers ---
LLM_BAWT_SEARCH_PROVIDER=brave            # brave, tavily, ddgs, or reddit
LLM_BAWT_BRAVE_API_KEY=...                # Brave Search API key
LLM_BAWT_TAVILY_API_KEY=tvly-...          # Tavily API key
LLM_BAWT_REDDIT_CLIENT_ID=...             # Reddit OAuth2
LLM_BAWT_REDDIT_CLIENT_SECRET=...
LLM_BAWT_NEWSAPI_API_KEY=...              # NewsAPI for current events

# --- Local Models ---
LLM_BAWT_LLAMA_CPP_N_CTX=4096            # Context window for GGUF models
LLM_BAWT_LLAMA_CPP_N_GPU_LAYERS=-1       # GPU layers (-1 = all)
LLM_BAWT_LLAMA_CPP_FLASH_ATTN=true       # Flash attention

# --- Scheduler ---
LLM_BAWT_SCHEDULER_ENABLED=true           # Background jobs (extraction, consolidation)
LLM_BAWT_SCHEDULER_CHECK_INTERVAL_SECONDS=30

# --- Integrations (optional) ---
LLM_BAWT_HA_NATIVE_MCP_URL=http://...    # Home Assistant MCP endpoint
LLM_BAWT_HA_NATIVE_MCP_TOKEN=...          # HA long-lived access token
LLM_BAWT_CRAWL4AI_URL=http://localhost:11235  # Crawl4AI for web_fetch tool
LLM_BAWT_REDIS_URL=redis://localhost:6379/0   # Redis for agent bridge

# --- Docker-specific (in .env at repo root) ---
COMPOSE_FILE=docker-compose.yml:docker-compose.dev.yml
WITH_CUDA=true
CUDA_ARCHS=120                             # Your GPU architecture (e.g., 89 for RTX 4090)
```

## Usage

### CLI

```bash
llm "what is the meaning of life"         # One-shot query
llm                                       # Interactive mode (multi-turn)
llm -m grok-4 "explain quantum physics"   # Specific model
llm -b proto "help me test this"          # Specific bot personality
llm --local "summarize this"              # Force local model (no service)
llm --service "look this up"              # Force service mode
llm -c "ls -al" "whats in my directory?"  # Send stdout as prompt

# Discovery
llm --status                              # Full system diagnostics
llm --list-models                         # Available models
llm --list-bots                           # Available bot personalities
llm --list-config                         # Current configuration

# Bot management
llm --add-bot                             # Interactive bot creation wizard
llm --add-model                           # Add a new model alias

# Debug
llm -v "test"                             # Verbose logging
llm --debug "test"                        # Full debug output with I/O inspection
```

### Service API

When running as a service (Docker or `./server.sh`), llm-bawt exposes an OpenAI-compatible REST API:

```bash
# Chat completion (streaming)
curl http://localhost:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "grok-4-fast", "messages": [{"role": "user", "content": "hello"}], "stream": true}'

# List models
curl http://localhost:8642/v1/models

# Health check
curl http://localhost:8642/health
```

**Additional service endpoints:**

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | OpenAI-compatible chat (streaming SSE) |
| `POST /v1/chat/session/reset` | Reset agent backend session for a bot |
| `POST /v1/bots/{bot_id}/chat` | Bot-scoped chat with isolated memory |
| `GET /v1/models` | List available models |
| `GET/POST /v1/bots` | Bot CRUD and management |
| `PUT /v1/bots/{slug}/profile` | Full bot profile update |
| `PATCH /v1/bots/{slug}/profile` | Partial bot profile update |
| `GET/POST /v1/memory/*` | Memory operations |
| `GET /v1/history/*` | Conversation history |
| `GET /v1/settings/*` | Runtime settings |
| `GET /health` | Service health diagnostics |

### Connecting a Frontend

llm-bawt is designed as a backend. The **[unmute](https://github.com/zenoran/unmute)** project provides a full web UI (Next.js) with:
- Text chat with streaming, markdown rendering, and tool call visibility
- Real-time voice conversations (STT → LLM → TTS pipeline)
- Bot personality switching
- Conversation history browser
- Home server dashboard

unmute connects to llm-bawt via `UNMUTE_LLM_URL=http://host.docker.internal:8642` and proxies requests through Next.js API routes to the `/v1/*` endpoints.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  User Interface                                                         │
│  ┌─────────────┐  ┌──────────────────────────────────────┐              │
│  │  CLI (llm)  │  │  unmute Web UI (Next.js, port 80)    │              │
│  └──────┬──────┘  └──────────────────┬───────────────────┘              │
│         │ local or service mode      │ HTTP proxy                       │
└─────────┼────────────────────────────┼──────────────────────────────────┘
          │                            │
┌─────────▼────────────────────────────▼──────────────────────────────────┐
│  llm-bawt Service Stack (Docker)                                        │
│                                                                         │
│  ┌──────────────────────┐  ┌──────────────────────────────────────────┐ │
│  │ MCP Memory Server    │  │  LLM Service (FastAPI, port 8642)        │ │
│  │ (port 8001)          │←→│  OpenAI-compatible API                   │ │
│  │ Memory read/write    │  │  Streaming SSE, tool calling             │ │
│  │ Fact extraction      │  │  Bot management, job scheduler           │ │
│  └──────────────────────┘  └───────────┬──────────────────────────────┘ │
│                                        │                                │
│  ┌────────────────┐  ┌────────────────┐│ ┌──────────────────────────┐   │
│  │ Redis          │  │ Crawl4AI       ││ │ Agent Bridges (optional) │   │
│  │ Event transport│  │ Web scraping   ││ │ OpenClaw + Claude Code   │   │
│  └────────────────┘  └────────────────┘│ └──────────────────────────┘   │
└────────────────────────────────────────┼────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼────────────────────────────────┐
│  External Dependencies (you provide)                                    │
│                                                                         │
│  ┌─────────────────────────┐  ┌──────────────────────────────────────┐  │
│  │ PostgreSQL + pgvector   │  │  LLM Providers                       │  │
│  │ Semantic memory storage │  │  OpenAI, Grok, Claude, Ollama, GGUF  │  │
│  └─────────────────────────┘  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Request Pipeline

Every query flows through a 7-stage pipeline:

1. **Pre-process** — Validate input, normalize message roles
2. **Context Build** — Assemble system prompt from bot personality, user profile, tool definitions
3. **Memory Retrieval** — Semantic search against pgvector for relevant past context
4. **History Filter** — Select conversation history within token budget
5. **Message Assembly** — Build final message list (system + memories + history + user input)
6. **Execute** — Stream to LLM, run tool loop if the model invokes tools
7. **Post-process** — Extract facts into memory, persist conversation history

### Memory System

Persistent semantic memory using PostgreSQL + pgvector with sentence-transformers (MiniLM) for local embeddings — no external API calls for vector search.

- **Temporal decay** — memories fade unless reinforced by access (~180 day half-life for core facts, ~30 days for plans/events)
- **Contradiction detection** — new facts supersede old ones (history preserved)
- **Background extraction** — a scheduler job uses Grok to extract facts from conversations
- **Consolidation** — duplicate memories are merged automatically
- **Bot isolation** — each bot has its own memory space

### Tool System

Dual-mode tool calling with automatic format detection:

| Format | Used By | How It Works |
|--------|---------|--------------|
| **Native** | OpenAI, Grok | Structured function calling via API |
| **ReAct** | GGUF, Ollama, vLLM | Text-based Thought/Action/Observation loop |

**Available tools:** `memory` (search/store/update), `history` (search past conversations), `profile` (get/set user attributes), `search` (web search), `web_fetch` (extract page content), `home` (Home Assistant device control), `model` (list/switch models), `time` (current time)

## Bots

Bot personalities define system prompts, memory access, tool availability, and default models. They are seeded from `bots.yaml` but stored in the database for runtime updates.

| Bot | Description | Memory | Tools | Use Case |
|-----|-------------|--------|-------|----------|
| **nova** | Full-featured technical assistant (default) | Yes | All | General-purpose usage |
| **spark** | Lightweight — no database required | No | None | Quick queries, testing without DB |
| **proto** | Development bot with isolated memory | Yes | All | Testing and development |

Create custom bots via `llm --add-bot` or the `/v1/bots` API.

## Project Structure

```
src/llm_bawt/
├── cli/                 # CLI entry point, argument parsing, interactive mode
├── core/                # Request pipeline, prompt builder, model lifecycle
│   ├── pipeline.py      # 7-stage request processing with hooks
│   ├── prompt_builder.py # System prompt assembly with token budgeting
│   └── base.py          # BaseLLMBawt (shared CLI + service logic)
├── clients/             # LLM provider clients
│   ├── openai_client.py # OpenAI + compatible APIs
│   ├── grok_client.py   # Grok (xAI) — also used for memory extraction
│   ├── agent_backend_client.py # Agent backend wrapper (OpenClaw, Claude Code)
│   ├── llamacpp_client.py # Local GGUF models via llama-cpp-python
│   └── vllm_client.py   # vLLM GPU inference server
├── agent_backends/      # Pluggable agent backend system
│   ├── openclaw.py      # OpenClaw gateway backend (Redis → WS bridge)
│   └── claude_code.py   # Claude Code backend (Redis → Agent SDK bridge)
├── adapters/            # Per-model output cleanup (role markers, BBCode, etc.)
├── tools/               # Tool definitions, executor, format handlers
│   ├── loop.py          # Multi-turn tool calling orchestration
│   ├── executor.py      # Tool dispatch and result formatting
│   └── definitions.py   # Tool schemas (OpenAI function format)
├── search/              # Pluggable search providers
│   ├── brave_client.py  # Brave Search (recommended)
│   ├── tavily_client.py # Tavily
│   ├── ddgs_client.py   # DuckDuckGo (free, no API key)
│   └── reddit_client.py # Reddit search
├── memory/              # Persistent semantic memory
│   ├── postgresql.py    # PostgreSQL + pgvector backend
│   ├── embeddings.py    # sentence-transformers (MiniLM) embeddings
│   ├── extraction/      # LLM-based fact extraction from conversations
│   ├── consolidation.py # Duplicate memory merging
│   └── summarization.py # Session summarization
├── memory_server/       # MCP memory server (port 8001)
├── service/             # FastAPI service (port 8642)
│   ├── routes/          # API endpoint implementations
│   ├── scheduler.py     # Background job scheduler
│   └── background_service.py # Service orchestrator
├── integrations/        # Home Assistant, Nextcloud, web fetch
├── utils/               # Config, history, streaming, input handling
├── bots.py              # Bot manager and personality loading
├── bots.yaml            # Bot seed definitions
└── model_manager.py     # Model definitions and aliases
```

## Docker Management

```bash
make up                 # Start production containers
make docker-dev         # Start dev mode (live src/ mounting)
make run                # Rebuild + restart app + bridge, tail logs
make down               # Stop all containers
make restart            # Restart containers
make rebuild            # Full rebuild + restart
make logs               # Follow container logs
make docker-shell       # Interactive bash inside the container
make docker-exec CMD="llm --status"   # Run a command inside the container
```

## CLI Commands Reference

| Command | Description |
|---------|-------------|
| `llm` | Main CLI — query LLMs, interactive mode, bot/model management |
| `llm-service` | Start the FastAPI LLM service (port 8642) |
| `llm-mcp-server` | Start the MCP memory server (port 8001) |
| `llm-memory` | Memory debugging utilities |
| `llm-nextcloud` | Nextcloud Talk bot provisioning |
| `openclaw-bridge` | OpenClaw WebSocket agent bridge |
| `claude-code-bridge` | Claude Code Agent SDK bridge |
| `./server.sh start` | Start both services (MCP + LLM) for local development |

## Testing

```bash
# Inside Docker (recommended)
make docker-exec CMD="uv run pytest"
make docker-exec CMD="uv run pytest tests/test_adapters.py"

# Local development
uv run pytest
uv run pytest -m "not llm_call"    # Skip tests that make real API calls

# Quick sanity check
llm --status
```

## Development

### Branching Strategy

Gitflow model: `main` (releases) ← `develop` (integration) ← `feature/*` (work)

### Key Patterns

- **Type hints** everywhere, use `str | None` (not `Optional[str]`)
- **Logging:** `logging.getLogger(__name__)` module-level
- **Config:** pydantic-settings with `LLM_BAWT_` prefix, layered precedence
- **Imports:** relative within the package (`from ..utils.config import Config`)
- **Output:** Rich `Console` for terminal formatting

## Further Documentation

| Document | Description |
|----------|-------------|
| [docs/MEMORY.md](docs/MEMORY.md) | Memory system design — extraction, decay, consolidation |
| [docs/TOOL_CALLING.md](docs/TOOL_CALLING.md) | Tool system — native + ReAct formats, tool loop |
| [docs/SEARCH_PROVIDERS.md](docs/SEARCH_PROVIDERS.md) | Search provider architecture and configuration |
| [docs/MODEL_ADAPTERS.md](docs/MODEL_ADAPTERS.md) | Per-model output formatting adapters |
| [docs/VLLM_INTEGRATION.md](docs/VLLM_INTEGRATION.md) | vLLM GPU inference setup |
| [docs/WEB_FETCH_INTEGRATION.md](docs/WEB_FETCH_INTEGRATION.md) | Crawl4AI web page extraction |
| [docs/BACKGROUND_SCHEDULER.md](docs/BACKGROUND_SCHEDULER.md) | Background job scheduler |
| [docs/OPENCLAW_INTEGRATION.md](docs/OPENCLAW_INTEGRATION.md) | OpenClaw agent backend architecture |
| [docs/CLAUDE_CODE_BRIDGE.md](docs/CLAUDE_CODE_BRIDGE.md) | Claude Code Agent SDK bridge — setup, auth, session management |
| [docs/HA_NATIVE_MCP_INTEGRATION.md](docs/HA_NATIVE_MCP_INTEGRATION.md) | Home Assistant integration |
| [docs/NEXTCLOUD_INTEGRATION.md](docs/NEXTCLOUD_INTEGRATION.md) | Nextcloud Talk bot provisioning |

## License

MIT
