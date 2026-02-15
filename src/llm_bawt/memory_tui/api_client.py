"""API client for llm-service integration."""

from __future__ import annotations

import httpx
from typing import Any
from datetime import datetime

from llm_bawt.utils.config import config


class MemoryAPIClient:
    """HTTP client for llm-service memory API."""
    
    DEFAULT_TIMEOUT = 30.0
    LONG_TIMEOUT = 120.0
    
    def __init__(self, base_url: str | None = None):
        """Initialize API client.
        
        Args:
            base_url: Service URL. If None, reads from config.
        """
        if base_url is None:
            host = getattr(config, "SERVICE_HOST", "localhost")
            port = getattr(config, "SERVICE_PORT", 8642)
            base_url = f"http://{host}:{port}"
        
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT)
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def health_check(self) -> dict[str, Any]:
        """Check service health."""
        response = await self._client.get(f"{self.base_url}/health", timeout=5.0)
        response.raise_for_status()
        return response.json()
    
    async def get_status(self) -> dict[str, Any]:
        """Get service status."""
        response = await self._client.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()
    
    # ==================================================================
    # Memory Operations
    # ==================================================================
    
    async def search_memories(
        self,
        query: str,
        bot_id: str | None = None,
        method: str = "all",
        limit: int = 50,
        min_importance: float = 0.0,
    ) -> dict[str, Any]:
        """Search memories."""
        payload = {
            "query": query,
            "method": method,
            "limit": limit,
            "min_importance": min_importance,
        }
        if bot_id:
            payload["bot_id"] = bot_id
        
        response = await self._client.post(
            f"{self.base_url}/v1/memory/search",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def list_memories(
        self,
        bot_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List memories."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if bot_id:
            params["bot_id"] = bot_id
        
        response = await self._client.get(
            f"{self.base_url}/v1/memory",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def get_memory(self, memory_id: str, bot_id: str | None = None) -> dict[str, Any]:
        """Get a specific memory by ID."""
        # Memory search with empty query and filter by ID
        result = await self.search_memories("", bot_id=bot_id, limit=100)
        for mem in result.get("results", []):
            if mem.get("id", "").startswith(memory_id):
                return mem
        raise ValueError(f"Memory {memory_id} not found")
    
    async def delete_memory(self, memory_id: str, bot_id: str | None = None) -> dict[str, Any]:
        """Delete a memory."""
        params: dict[str, Any] = {}
        if bot_id:
            params["bot_id"] = bot_id
        
        response = await self._client.delete(
            f"{self.base_url}/v1/memory/{memory_id}",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        bot_id: str | None = None,
    ) -> dict[str, Any]:
        """Update a memory."""
        # Note: The API may not support direct updates - may need to delete+recreate
        # For now, this is a placeholder
        payload: dict[str, Any] = {}
        if content is not None:
            payload["content"] = content
        if importance is not None:
            payload["importance"] = importance
        if tags is not None:
            payload["tags"] = tags
        
        params: dict[str, Any] = {}
        if bot_id:
            params["bot_id"] = bot_id
        
        response = await self._client.patch(
            f"{self.base_url}/v1/memory/{memory_id}",
            json=payload,
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def consolidate_memories(
        self,
        bot_id: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Consolidate redundant memories."""
        payload = {"dry_run": dry_run}
        params: dict[str, Any] = {}
        if bot_id:
            params["bot_id"] = bot_id
        
        response = await self._client.post(
            f"{self.base_url}/v1/memory/consolidate",
            json=payload,
            params=params,
            timeout=self.LONG_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    
    async def regenerate_embeddings(self, bot_id: str | None = None) -> dict[str, Any]:
        """Regenerate all embeddings."""
        params: dict[str, Any] = {}
        if bot_id:
            params["bot_id"] = bot_id
        
        response = await self._client.post(
            f"{self.base_url}/v1/memory/regenerate-embeddings",
            params=params,
            timeout=self.LONG_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    
    async def get_memory_stats(self, bot_id: str | None = None) -> dict[str, Any]:
        """Get memory statistics."""
        params: dict[str, Any] = {}
        if bot_id:
            params["bot_id"] = bot_id
        
        response = await self._client.get(
            f"{self.base_url}/v1/memory/stats",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    # ==================================================================
    # Message Operations
    # ==================================================================
    
    async def get_messages(
        self,
        bot_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        before: str | None = None,
    ) -> dict[str, Any]:
        """Get messages, optionally paginating older history with a before cursor."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if bot_id:
            params["bot_id"] = bot_id
        if before:
            params["before"] = before
        
        response = await self._client.get(
            f"{self.base_url}/v1/history",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def search_messages(
        self,
        query: str,
        bot_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Search messages."""
        params: dict[str, Any] = {"query": query, "limit": limit}
        if bot_id:
            params["bot_id"] = bot_id
        
        response = await self._client.post(
            f"{self.base_url}/v1/history/search",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def get_message(self, message_id: str, bot_id: str | None = None) -> dict[str, Any]:
        """Get a specific message by ID."""
        params: dict[str, Any] = {"message_id": message_id}
        if bot_id:
            params["bot_id"] = bot_id
        
        response = await self._client.get(
            f"{self.base_url}/v1/memory/message",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def forget_messages(
        self,
        count: int | None = None,
        minutes: int | None = None,
        message_id: str | None = None,
        bot_id: str | None = None,
    ) -> dict[str, Any]:
        """Soft-delete messages."""
        payload: dict[str, Any] = {}
        if count is not None:
            payload["count"] = count
        if minutes is not None:
            payload["minutes"] = minutes
        if message_id is not None:
            payload["message_id"] = message_id
        
        params: dict[str, Any] = {}
        if bot_id:
            params["bot_id"] = bot_id
        
        response = await self._client.post(
            f"{self.base_url}/v1/memory/forget",
            json=payload,
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def restore_messages(self, bot_id: str | None = None) -> dict[str, Any]:
        """Restore forgotten messages."""
        params: dict[str, Any] = {}
        if bot_id:
            params["bot_id"] = bot_id
        
        response = await self._client.post(
            f"{self.base_url}/v1/memory/restore",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def preview_recent_messages(
        self,
        count: int,
        bot_id: str | None = None,
    ) -> dict[str, Any]:
        """Preview recent messages (for confirmation)."""
        params: dict[str, Any] = {"count": count}
        if bot_id:
            params["bot_id"] = bot_id
        
        response = await self._client.get(
            f"{self.base_url}/v1/memory/preview/recent",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def preview_messages_since(
        self,
        minutes: int,
        bot_id: str | None = None,
    ) -> dict[str, Any]:
        """Preview messages from last N minutes."""
        params: dict[str, Any] = {"minutes": minutes}
        if bot_id:
            params["bot_id"] = bot_id
        
        response = await self._client.get(
            f"{self.base_url}/v1/memory/preview/minutes",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def preview_forgotten_messages(self, bot_id: str | None = None) -> dict[str, Any]:
        """Preview forgotten messages."""
        params: dict[str, Any] = {}
        if bot_id:
            params["bot_id"] = bot_id
        
        response = await self._client.get(
            f"{self.base_url}/v1/memory/preview/ignored",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    # ==================================================================
    # Summaries
    # ==================================================================
    
    async def list_summaries(self, bot_id: str | None = None) -> dict[str, Any]:
        """List message summaries."""
        params: dict[str, Any] = {}
        if bot_id:
            params["bot_id"] = bot_id
        
        response = await self._client.get(
            f"{self.base_url}/v1/history/summaries",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def preview_summarizable_sessions(self, bot_id: str | None = None) -> dict[str, Any]:
        """Preview sessions eligible for summarization."""
        params: dict[str, Any] = {}
        if bot_id:
            params["bot_id"] = bot_id
        
        response = await self._client.get(
            f"{self.base_url}/v1/history/summarize/preview",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def summarize_sessions(
        self,
        bot_id: str | None = None,
        use_heuristic: bool = False,
    ) -> dict[str, Any]:
        """Summarize old sessions."""
        params: dict[str, Any] = {"use_heuristic": use_heuristic}
        if bot_id:
            params["bot_id"] = bot_id
        
        response = await self._client.post(
            f"{self.base_url}/v1/history/summarize",
            params=params,
            timeout=self.LONG_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    
    async def delete_summary(self, summary_id: str, bot_id: str | None = None) -> dict[str, Any]:
        """Delete a summary."""
        params: dict[str, Any] = {}
        if bot_id:
            params["bot_id"] = bot_id
        
        response = await self._client.delete(
            f"{self.base_url}/v1/history/summary/{summary_id}",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    # ==================================================================
    # Profile Operations
    # ==================================================================
    
    async def list_profiles(self, entity_type: str = "user") -> dict[str, Any]:
        """List profiles of a given type."""
        response = await self._client.get(
            f"{self.base_url}/v1/profiles/list/{entity_type}"
        )
        response.raise_for_status()
        return response.json()
    
    async def get_profile(self, entity_id: str) -> dict[str, Any]:
        """Get profile for any entity (auto-detects type)."""
        response = await self._client.get(
            f"{self.base_url}/v1/profiles/{entity_id}"
        )
        response.raise_for_status()
        return response.json()
    
    async def get_user_profile(self, user_id: str) -> dict[str, Any]:
        """Get user profile."""
        response = await self._client.get(
            f"{self.base_url}/v1/users/{user_id}"
        )
        response.raise_for_status()
        return response.json()
    
    async def delete_attribute(self, attribute_id: int) -> dict[str, Any]:
        """Delete a profile attribute."""
        response = await self._client.delete(
            f"{self.base_url}/v1/users/attribute/{attribute_id}"
        )
        response.raise_for_status()
        return response.json()
    
    # ==================================================================
    # Bot Operations
    # ==================================================================
    
    async def list_bots(self) -> dict[str, Any]:
        """List available bots."""
        response = await self._client.get(f"{self.base_url}/v1/bots")
        response.raise_for_status()
        return response.json()
