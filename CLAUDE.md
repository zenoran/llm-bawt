# Claude Code Instructions

See [AGENTS.md](AGENTS.md) for all project documentation, conventions, and architecture details.

## Cross-Repository Development

This workspace frequently involves coordinated changes across two sibling repos:

| Repo | Path | Purpose |
|------|------|---------|
| **llm-bawt** | `/home/nick/dev/llm-bawt` | Python backend — LLM service, API, memory, tools |
| **unmute** | `/home/nick/dev/unmute` | Next.js frontend + Docker services for voice/chat/tools UI |

### unmute Frontend Location
- Frontend source: `/home/nick/dev/unmute/frontend/src/`
- App routes: `/home/nick/dev/unmute/frontend/src/app/`
- API proxy routes: `/home/nick/dev/unmute/frontend/src/app/api/`
- Shared libs: `/home/nick/dev/unmute/frontend/src/lib/`
- unmute project docs: `/home/nick/dev/unmute/AGENTS.md`

### How They Connect
- The unmute frontend proxies API calls through Next.js API routes (`/api/chat/*`) to the llm-bawt service (default `http://host.docker.internal:8642`).
- llm-bawt exposes an OpenAI-compatible REST API at port 8642 (`/v1/chat/completions`, `/v1/bots`, `/v1/tool-calls`, `/v1/settings/*`, etc.).
- Changes to llm-bawt API endpoints often require corresponding changes in unmute's proxy routes and frontend components.

### Common Cross-Repo Workflows
- **API endpoint changes**: Update llm-bawt route → update unmute proxy route (`/api/chat/...`) → update frontend component that calls it.
- **New query params or response fields**: Add to llm-bawt endpoint → pass through unmute proxy → consume in frontend.
- **UI bugs revealing backend issues**: Diagnose in frontend, trace through proxy, fix in llm-bawt.

### Key Mapping: Frontend → Backend
| Frontend File | Proxy Route | Backend Endpoint |
|---------------|-------------|------------------|
| `app/chat/ChatUI.tsx` | `app/api/chat/tool-calls/route.ts` | `/v1/tool-calls` |
| `app/chat/ChatUI.tsx` | `app/api/chat/route.ts` | `/v1/chat/completions` |
| `AppMainNav.tsx` | `app/api/chat/bots/route.ts` | `/v1/bots` |
| `AppMainNav.tsx` | `app/api/chat/proxy/*/route.ts` | `/v1/settings/*` |
| `app/tools/ToolMenu.tsx` | `app/api/chat/proxy/*/route.ts` | `/v1/tools/*` |
| `MemoryPanel.tsx` | `app/api/chat/proxy/*/route.ts` | `/v1/memory/*` |

### Build Commands (unmute frontend)
```bash
pnpm -C /home/nick/dev/unmute/frontend exec next build   # Production build
pnpm -C /home/nick/dev/unmute/frontend dev                # Dev server
```
