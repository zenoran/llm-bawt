"""Normalized model endpoint catalog and harness-aware dispatch resolver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from sqlalchemy import Engine, text


class ModelResolutionError(ValueError):
    """Base error for normalized model resolution."""


class ModelNotFoundError(ModelResolutionError):
    """Raised when a model/endpoint reference is unknown."""


class AmbiguousModelError(ModelResolutionError):
    """Raised when a model has multiple viable endpoints without enough context."""


class IncompatibleModelError(ModelResolutionError):
    """Raised when an endpoint cannot be driven by the requested harness."""


def resolve_model_config(
    config: Any,
    ref: str | int,
    harness: str | None = None,
    default: Any = None,
) -> Any:
    """Use Config.resolve_model with a compatibility path for test/legacy configs."""
    resolver = getattr(config, "resolve_model", None)
    if callable(resolver):
        return resolver(ref, harness=harness, default=default)
    return (
        getattr(config, "defined_models", {}).get("models", {}).get(str(ref), default)
    )


@dataclass(frozen=True)
class ModelIdentity:
    id: int
    key: str
    vendor: str
    display_name: str
    description: str | None = None
    default_context_window: int | None = None
    default_tool_support: str | None = None


@dataclass(frozen=True)
class AccessPath:
    id: int
    key: str
    vendor: str
    protocol: str
    base_url: str | None
    auth_mechanism: str
    engine_kind: str | None = None


@dataclass(frozen=True)
class ModelEndpoint:
    id: int
    model: ModelIdentity
    access_path: AccessPath
    upstream_model_id: str | None
    serving_config: Mapping[str, Any] = field(default_factory=dict)
    context_window_override: int | None = None
    tool_support_override: str | None = None
    pricing: Mapping[str, Any] | None = None
    legacy_type: str | None = None

    @property
    def ref(self) -> str:
        return f"{self.model.key}@{self.access_path.key}"


class ProtocolCompatibility:
    """Single source of truth for harness/access-path compatibility."""

    HARNESS_PROTOCOLS: Mapping[str, str | None] = {
        "chat": "chat-completions",
        "codex": "responses",
        "claude-code": "anthropic-messages",
        "claude-proxy": "anthropic-messages",
        "openclaw": None,
    }
    PROXY_TARGETS = frozenset({"chat-completions", "responses"})

    @classmethod
    def normalize_harness(cls, harness: str | None) -> str | None:
        value = (harness or "").strip().lower()
        return value or None

    @classmethod
    def is_compatible(cls, harness: str | None, access_path: AccessPath) -> bool:
        normalized = cls.normalize_harness(harness)
        if normalized is None:
            return True
        if normalized not in cls.HARNESS_PROTOCOLS:
            return False
        if normalized == "openclaw":
            return True
        if normalized == "claude-proxy":
            return (
                access_path.vendor != "anthropic"
                and access_path.protocol in cls.PROXY_TARGETS
            )
        return cls.HARNESS_PROTOCOLS[normalized] == access_path.protocol


class ModelCatalog:
    """Immutable, indexed endpoint catalog with harness-aware resolution."""

    def __init__(self, endpoints: Iterable[ModelEndpoint] = ()) -> None:
        endpoint_list = tuple(endpoints)
        self._by_id = {endpoint.id: endpoint for endpoint in endpoint_list}
        self._by_ref = {endpoint.ref: endpoint for endpoint in endpoint_list}
        by_model: dict[str, list[ModelEndpoint]] = {}
        for endpoint in endpoint_list:
            by_model.setdefault(endpoint.model.key, []).append(endpoint)
        self._by_model = {key: tuple(rows) for key, rows in by_model.items()}
        self._endpoints = endpoint_list

    def __len__(self) -> int:
        return len(self._endpoints)

    def model_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_model))

    def endpoints(self, harness: str | None = None) -> tuple[ModelEndpoint, ...]:
        if harness is None:
            return self._endpoints
        return tuple(
            endpoint
            for endpoint in self._endpoints
            if ProtocolCompatibility.is_compatible(harness, endpoint.access_path)
        )

    def resolve_endpoint(
        self,
        ref: str | int | ModelEndpoint,
        harness: str | None = None,
    ) -> ModelEndpoint:
        if isinstance(ref, ModelEndpoint):
            candidates = (ref,)
        elif isinstance(ref, int):
            endpoint = self._by_id.get(ref)
            if endpoint is None:
                raise ModelNotFoundError(f"Unknown model endpoint id: {ref}")
            candidates = (endpoint,)
        else:
            normalized = str(ref or "").strip()
            if not normalized:
                raise ModelNotFoundError("Model reference is empty")
            endpoint = self._by_ref.get(normalized)
            if endpoint is None and normalized.isdigit():
                endpoint = self._by_id.get(int(normalized))
            candidates = (
                (endpoint,)
                if endpoint is not None
                else self._by_model.get(normalized, ())
            )
            if not candidates:
                raise ModelNotFoundError(f"Unknown model reference: {normalized}")

        compatible = tuple(
            endpoint
            for endpoint in candidates
            if ProtocolCompatibility.is_compatible(harness, endpoint.access_path)
        )
        if not compatible:
            requested = (
                ProtocolCompatibility.normalize_harness(harness) or "unspecified"
            )
            raise IncompatibleModelError(
                f"Model reference {ref!r} has no endpoint compatible with harness={requested}"
            )
        if len(compatible) > 1:
            refs = ", ".join(endpoint.ref for endpoint in compatible)
            raise AmbiguousModelError(
                f"Model reference {ref!r} is ambiguous ({refs}); pass harness or endpoint ref"
            )
        return compatible[0]

    def resolve(
        self,
        ref: str | int | ModelEndpoint,
        harness: str | None = None,
    ) -> dict[str, Any]:
        endpoint = self.resolve_endpoint(ref, harness=harness)
        return self._dispatch_config(endpoint, harness=harness)

    def compatibility_mapping(self) -> dict[str, dict[str, Any]]:
        """Legacy alias map for iteration-only callers during the cutover."""
        result: dict[str, dict[str, Any]] = {}
        for key, endpoints in self._by_model.items():
            if len(endpoints) == 1:
                result[key] = self._dispatch_config(endpoints[0], harness=None)
            else:
                for endpoint in endpoints:
                    result[endpoint.ref] = self._dispatch_config(endpoint, harness=None)
        return result

    @staticmethod
    def _provider_prefix(access_path: AccessPath) -> str | None:
        if access_path.key == "openai-oauth":
            return "openai_chatgpt"
        if access_path.vendor == "xai":
            return "xai"
        if access_path.vendor == "zai":
            return "zai"
        return None

    @classmethod
    def _dispatch_config(
        cls,
        endpoint: ModelEndpoint,
        harness: str | None,
    ) -> dict[str, Any]:
        normalized_harness = ProtocolCompatibility.normalize_harness(harness)
        compat_extra = endpoint.serving_config.get("compat_extra")
        config = dict(compat_extra) if isinstance(compat_extra, dict) else {}
        for key, value in endpoint.serving_config.items():
            if key != "compat_extra":
                config[key] = value

        model_type = endpoint.legacy_type or "openai-compatible"
        upstream = endpoint.upstream_model_id
        concrete_model = upstream
        backend: str | None = None

        if normalized_harness is None and endpoint.legacy_type == "claude-code":
            prefix = cls._provider_prefix(endpoint.access_path)
            if prefix and upstream:
                concrete_model = f"{prefix}/{upstream}"
        elif normalized_harness == "codex":
            model_type = "agent_backend"
            backend = "codex"
        elif normalized_harness in {"claude-code", "claude-proxy"}:
            model_type = "claude-code"
            backend = "claude-code"
            prefix = cls._provider_prefix(endpoint.access_path)
            if prefix and upstream:
                concrete_model = f"{prefix}/{upstream}"
        elif normalized_harness == "openclaw":
            model_type = "agent_backend"
            backend = "openclaw"
        elif normalized_harness == "chat":
            if endpoint.access_path.engine_kind == "llama-cpp":
                model_type = "gguf"
            elif endpoint.access_path.engine_kind in {"vllm", "ollama"}:
                model_type = endpoint.access_path.engine_kind
            elif endpoint.access_path.vendor == "xai":
                model_type = "grok"
            elif endpoint.access_path.vendor == "openai":
                model_type = "openai"

        config.update(
            {
                "type": model_type,
                "model_id": concrete_model,
                "endpoint_id": endpoint.id,
                "model_key": endpoint.model.key,
                "access_path": endpoint.access_path.key,
                "protocol": endpoint.access_path.protocol,
                "base_url": endpoint.access_path.base_url,
                "auth_mechanism": endpoint.access_path.auth_mechanism,
                "engine_kind": endpoint.access_path.engine_kind,
                "harness": normalized_harness,
                "upstream_model_id": upstream,
            }
        )
        if backend:
            config["backend"] = backend
        if endpoint.model.description:
            config["description"] = endpoint.model.description
        context_window = (
            endpoint.context_window_override or endpoint.model.default_context_window
        )
        if context_window is not None:
            config["context_window"] = context_window
        tool_support = (
            endpoint.tool_support_override or endpoint.model.default_tool_support
        )
        if tool_support is not None:
            config["tool_support"] = tool_support
        if endpoint.pricing is not None:
            config["pricing"] = dict(endpoint.pricing)
        return config


class ModelCatalogStore:
    """Load the normalized catalog from PostgreSQL in one query."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def load(self) -> ModelCatalog:
        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    text("""
                SELECT
                    e.id AS endpoint_id,
                    e.upstream_model_id,
                    e.serving_config,
                    e.context_window_override,
                    e.tool_support_override,
                    e.pricing,
                    e.legacy_type,
                    m.id AS model_id,
                    m.key AS model_key,
                    m.vendor AS model_vendor,
                    m.display_name,
                    m.description,
                    m.default_context_window,
                    m.default_tool_support,
                    a.id AS access_path_id,
                    a.key AS access_path_key,
                    a.vendor AS access_vendor,
                    a.protocol,
                    a.base_url,
                    a.auth_mechanism,
                    a.engine_kind
                FROM model_endpoints e
                JOIN models m ON m.id = e.model_id
                JOIN access_paths a ON a.id = e.access_path_id
                ORDER BY m.key, a.key
            """)
                )
                .mappings()
                .all()
            )

        endpoints = []
        for row in rows:
            model = ModelIdentity(
                id=int(row["model_id"]),
                key=row["model_key"],
                vendor=row["model_vendor"],
                display_name=row["display_name"],
                description=row["description"],
                default_context_window=row["default_context_window"],
                default_tool_support=row["default_tool_support"],
            )
            access_path = AccessPath(
                id=int(row["access_path_id"]),
                key=row["access_path_key"],
                vendor=row["access_vendor"],
                protocol=row["protocol"],
                base_url=row["base_url"],
                auth_mechanism=row["auth_mechanism"],
                engine_kind=row["engine_kind"],
            )
            endpoints.append(
                ModelEndpoint(
                    id=int(row["endpoint_id"]),
                    model=model,
                    access_path=access_path,
                    upstream_model_id=row["upstream_model_id"],
                    serving_config=row["serving_config"] or {},
                    context_window_override=row["context_window_override"],
                    tool_support_override=row["tool_support_override"],
                    pricing=row["pricing"],
                    legacy_type=row["legacy_type"],
                )
            )
        return ModelCatalog(endpoints)
