"""Anthropic-compatible proxy that mounts inside the claude_code_bridge daemon.

Lets the Claude Agent SDK target non-Anthropic providers (OpenAI ChatGPT
subscription via OAuth, future Grok/Kimi/etc.) by translating Anthropic
Messages API requests to the OpenAI Responses API and back.

Architecture:

  Claude SDK subprocess
        │  (ANTHROPIC_BASE_URL = http://127.0.0.1:<ephemeral-port>)
        ▼
  POST /v1/messages   (this proxy, FastAPI, bound 127.0.0.1)
        │  model="<provider>/<upstream_model>"   ← LiteLLM/OpenRouter pattern
        ▼
  ProviderAdapter (registry lookup by provider prefix)
        │  authorize() → bearer + base_url
        ▼
  Upstream (api.openai.com/v1/responses, etc.)
        │  SSE
        ▼
  Translated SSE → back to Claude SDK as Anthropic events
"""

from .app import create_app, ProxyServer  # noqa: F401
from .adapters import REGISTRY, register, lookup  # noqa: F401

__all__ = ["create_app", "ProxyServer", "REGISTRY", "register", "lookup"]
