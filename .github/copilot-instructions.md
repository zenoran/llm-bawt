# llm-bawt - AI Coding Agent Instructions

See [AGENTS.md](../AGENTS.md) for complete project documentation, architecture, and conventions.

> **Testing**: Always use `nova` (default) or `proto` as the bot (`--bot proto`). **Do NOT use `mira`** — Mira is a personal conversational companion, not suitable for dev/testing.

## Quick Reference

### Everything Runs in Docker

The llm-bawt service, MCP server, and all dependencies run inside a Docker container (`llm-bawt-app`). Local `.venv` exists only for IDE tooling. All testing and debugging should use `docker exec`:

```bash
make docker-exec CMD="llm --status"           # Check config
make docker-exec CMD="llm --bot proto 'test'"   # Test a query
make docker-exec CMD="uv run pytest"           # Run tests
make docker-exec CMD="uv run pytest tests/test_adapters.py"  # Specific test
make docker-shell                              # Interactive bash
make run                                       # Stop, rebuild dev, tail logs
```

### Architecture

- **Core**: `src/llm_bawt/core/` — `BaseLLMBawt`, `RequestPipeline` (7-stage), `PromptBuilder`
- **Clients**: `src/llm_bawt/clients/` — OpenAI, Grok (xAI), vLLM, LlamaCpp
- **Bots**: `src/llm_bawt/bots.yaml` — YAML-driven personalities with isolated memory
- **Memory**: `src/llm_bawt/memory/` — PostgreSQL + pgvector, LLM-based extraction
- **Tools**: `src/llm_bawt/tools/` — memory, history, profile, search, news, home, model
- **Service**: `src/llm_bawt/service/` — FastAPI OpenAI-compatible API (port 8642)
- **Config**: `src/llm_bawt/utils/config.py` — pydantic-settings with `LLM_BAWT_` prefix

### Cross-Repository: unmute

llm-bawt is the backend for **unmute** (`/home/nick/dev/unmute`), a real-time voice + chat UI:
- Frontend proxies API calls via Next.js routes (`/api/chat/*`) → llm-bawt (`/v1/*`)
- API changes in llm-bawt may require updating unmute proxy routes and frontend components
- See `Cross-Repository: unmute` section in AGENTS.md for full mapping

### Key Patterns

- **Type hints**: Required everywhere, use `str | None` (not `Optional[str]`)
- **Logging**: `logging.getLogger(__name__)` module-level
- **Output**: Rich `Console` for all terminal output; respect `PLAIN_OUTPUT`
- **Config**: `LLM_BAWT_` prefix, pydantic-settings with layered precedence
- **Imports**: Relative within package (`from ..utils.config import Config`)
