"""Upstream model-catalog discovery providers.

The Add Model dialog consumes one normalized contract regardless of whether the
source is a local bridge, an SDK, or a provider HTTP endpoint.  Provider-specific
credential and response handling belongs here rather than in the models route.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

import httpx


class ModelDiscoveryError(RuntimeError):
    """Expected discovery failure with an HTTP status suitable for the API."""

    def __init__(self, message: str, *, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


class ModelDiscoveryProvider(ABC):
    """Provider-specific source for normalized upstream model metadata."""

    aliases: tuple[str, ...] = ()

    @abstractmethod
    def fetch(self) -> list[dict[str, Any]]:
        """Return normalized ``id`` / ``description`` model entries."""


def _normalize(models: Iterable[object]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id") or item.get("model_id")
        if not model_id:
            continue
        row: dict[str, Any] = {
            "id": str(model_id),
            "description": str(
                item.get("description")
                or item.get("summary")
                or item.get("display_name")
                or item.get("name")
                or ""
            ),
        }
        context_length = item.get("context_length")
        if isinstance(context_length, int) and not isinstance(context_length, bool):
            row["context_length"] = context_length
        normalized.append(row)
    return normalized


class ExistingFetcherProvider(ModelDiscoveryProvider):
    """Adapter for the established model-manager fetch functions."""

    def __init__(
        self,
        *,
        aliases: tuple[str, ...],
        label: str,
        fetcher: Callable[..., tuple[bool, list[dict[str, Any]]]],
        key_envs: tuple[str, ...] = (),
        missing_key_message: str | None = None,
        pass_key: bool = False,
    ):
        self.aliases = aliases
        self.label = label
        self._fetcher = fetcher
        self._key_envs = key_envs
        self._missing_key_message = missing_key_message
        self._pass_key = pass_key

    def fetch(self) -> list[dict[str, Any]]:
        key = next((os.getenv(name) for name in self._key_envs if os.getenv(name)), None)
        if self._key_envs and not key:
            raise ModelDiscoveryError(
                self._missing_key_message or f"{self.label} discovery requires an API key",
                status_code=503,
            )
        ok, models = self._fetcher(key) if self._pass_key else self._fetcher()
        if not ok:
            raise ModelDiscoveryError(
                f"Failed to fetch {self.label} model catalog from upstream"
            )
        return _normalize(models)


class KimiCodingDiscoveryProvider(ModelDiscoveryProvider):
    """Kimi For Coding subscription catalog from its OpenAI-compatible API."""

    aliases = ("kimi", "kimi_coding", "kimi-code")
    _url = "https://api.kimi.com/coding/v1/models"
    _key_envs = ("KIMI_CODING_API_KEY", "KIMI_API_KEY")

    def fetch(self) -> list[dict[str, Any]]:
        key = next((os.getenv(name) for name in self._key_envs if os.getenv(name)), None)
        if not key:
            raise ModelDiscoveryError(
                "Kimi discovery requires KIMI_CODING_API_KEY in env",
                status_code=503,
            )
        headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        try:
            response = httpx.get(self._url, headers=headers, timeout=20.0)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401:
                raise ModelDiscoveryError(
                    "Kimi rejected KIMI_CODING_API_KEY while listing models",
                    status_code=503,
                ) from exc
            raise ModelDiscoveryError(
                f"Kimi model catalog returned HTTP {status}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelDiscoveryError(
                f"Network error reaching Kimi model catalog: {exc}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ModelDiscoveryError("Kimi model catalog returned non-JSON data") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ModelDiscoveryError("Kimi model catalog returned an invalid payload")

        # Kimi currently lists Highspeed even for BASIC subscriptions, then
        # rejects inference with 401. Hide that known-unusable choice; upgraded
        # plans keep the full catalog. A usage-read failure is non-fatal and
        # falls back to the provider's model list.
        try:
            usage = httpx.get(
                "https://api.kimi.com/coding/v1/usages",
                headers=headers,
                timeout=20.0,
            )
            usage.raise_for_status()
            usage_payload = usage.json()
            membership = (
                usage_payload.get("user", {}).get("membership", {}).get("level")
                if isinstance(usage_payload, dict)
                else None
            )
            if membership == "LEVEL_BASIC":
                data = [
                    item
                    for item in data
                    if not isinstance(item, dict)
                    or item.get("id") != "kimi-for-coding-highspeed"
                ]
        except (httpx.HTTPError, ValueError, AttributeError):
            pass

        return _normalize(data)


def _providers() -> tuple[ModelDiscoveryProvider, ...]:
    # Imports stay lazy so the model-manager's optional provider SDKs remain off
    # the service import path until discovery is actually requested.
    from ..model_manager import (
        fetch_anthropic_api_models,
        fetch_codex_models,
        fetch_grok_api_models,
        fetch_openai_api_models,
    )

    return (
        ExistingFetcherProvider(
            aliases=("codex",),
            label="codex",
            fetcher=fetch_codex_models,
        ),
        ExistingFetcherProvider(
            aliases=("openai",),
            label="openai",
            fetcher=fetch_openai_api_models,
        ),
        ExistingFetcherProvider(
            aliases=("grok", "xai"),
            label="Grok",
            fetcher=fetch_grok_api_models,
            key_envs=("LLM_BAWT_XAI_API_KEY", "XAI_API_KEY"),
            missing_key_message="Grok discovery requires LLM_BAWT_XAI_API_KEY in env",
            pass_key=True,
        ),
        ExistingFetcherProvider(
            aliases=("anthropic", "claude-code"),
            label="Anthropic",
            fetcher=fetch_anthropic_api_models,
            key_envs=("LLM_BAWT_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
            missing_key_message=(
                "Anthropic discovery requires ANTHROPIC_API_KEY in env "
                "(the Claude Code OAuth token does not work for /v1/models)."
            ),
        ),
        KimiCodingDiscoveryProvider(),
    )


def discover_models(provider: str) -> list[dict[str, Any]]:
    """Resolve a provider alias and return its normalized live catalog."""

    provider_key = provider.strip().lower()
    for adapter in _providers():
        if provider_key in adapter.aliases:
            return adapter.fetch()
    supported = sorted(alias for adapter in _providers() for alias in adapter.aliases)
    raise ModelDiscoveryError(
        f"Unknown provider '{provider}'. Use {', '.join(supported)}.",
        status_code=400,
    )
