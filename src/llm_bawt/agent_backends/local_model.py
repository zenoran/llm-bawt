"""Local model agent backend (TASK-276).

Routes local GPU inference (gguf via llama-cpp + vLLM) to the standalone
``local_model_bridge`` process over the shared agent-bridge Redis protocol,
instead of loading the model in the main app process.

Motivation: a CUDA abort() in local inference used to kill the whole app
(and with it the MCP server + every live agent session).  By running the
model in a separate bridge process, an abort can only take down the bridge.

Session keys are scoped by bot + user (``f"{bot_id}:{user_id}"``), mirroring
the claude-code backend, so each user gets an independent local-model session
and chat.abort RPCs reach the right active stream.
"""

from __future__ import annotations

from .agent_bridge import AgentBridgeBackend


class LocalModelBackend(AgentBridgeBackend):
    """Agent backend for local GPU models via the local-model-bridge.

    Inherits the full Redis command/event protocol from AgentBridgeBackend.
    The chat.send command carries the model *alias* (model_id); the bridge
    resolves it against the main app's /v1/models catalog and loads the
    appropriate client (LlamaCppClient for gguf, VLLMClient for vllm).
    """

    name = "local"

    def _resolve_session_key(self, config: dict) -> str:
        # Route by bot + user so each user gets an independent local-model
        # session for a given bot — matches chat_streaming's abort routing.
        bot_id = str(config.get("bot_id") or "main").strip() or "main"
        user_id = str(config.get("user_id") or "default").strip() or "default"
        return f"{bot_id}:{user_id}"
