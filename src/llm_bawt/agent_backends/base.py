"""Abstract base class for agent backend plugins.

Agent backends handle chat requests by delegating to external agent systems
(e.g. OpenClaw, Claude Code, custom agents) instead of using llm-bawt's
internal LLM pipeline.

Third-party packages can implement this interface and register via entry points:

    [project.entry-points."llm_bawt.agent_backends"]
    myagent = "mypackage.agents:MyAgentBackend"
"""

from abc import ABC, abstractmethod


class AgentBackend(ABC):
    """Plugin interface for external agent chat systems."""

    name: str = "base"

    @abstractmethod
    async def chat(
        self,
        prompt: str,
        config: dict,
        stream: bool = False,
    ) -> str:
        """Handle a chat request and return the response text.

        Args:
            prompt: The user's message.
            config: Backend-specific configuration from bot's
                ``agent_backend_config`` in bots.yaml.
            stream: Whether streaming is requested (reserved for future use).

        Returns:
            The agent's response as a string.
        """
        ...

    async def health_check(self, config: dict) -> bool:
        """Check if the backend is reachable.

        Args:
            config: Backend-specific configuration.

        Returns:
            True if the backend is healthy and reachable.
        """
        return True
