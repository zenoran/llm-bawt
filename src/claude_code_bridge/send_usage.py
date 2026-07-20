"""Usage / context-window accounting for the Claude send path (TASK-623).

Extracted from ``ClaudeSendMixin._handle_send``'s ResultMessage branch. The
computation is a verbatim move: given the terminal ``ResultMessage`` plus the
per-iteration / stream usage snapshots captured during the turn, produce the
``token_usage_payload`` dict and the resolved ``ctx_window`` / ``max_output``
scalars the caller stamps onto ASSISTANT_DONE.

Composed onto ``ClaudeSendMixin`` so ``self._model_provider_prefix`` and the
assembled-instance state resolve normally.
"""

from __future__ import annotations

import logging

from ._bridge_helpers import (
    _estimate_proxy_cost_usd,
    _pick_iteration_usage,
    _usage_input_total,
)

logger = logging.getLogger("claude_code_bridge.bridge")


class ClaudeUsageMixin:
    """Token-usage / context-window extraction for a completed turn."""

    def _compute_result_usage(
        self,
        msg,
        *,
        actual_model: str,
        model: str,
        bot_context_window: int | None,
        latest_assistant_usage: dict | None,
        latest_stream_usage: dict | None,
    ) -> tuple[dict | None, object, object]:
        """Return ``(token_usage_payload, ctx_window, max_output)``.

        Mirrors the monolith exactly: a failure mid-extraction is swallowed and
        whatever partial ``ctx_window`` / ``max_output`` were resolved (plus a
        ``None`` payload) are returned, so the caller's downstream compact
        override sees the same values it did before the split.
        """
        # Extract token usage + context window for UI surfacing.
        #
        # IMPORTANT: ResultMessage.usage is CUMULATIVE across all
        # internal API iterations in the turn — for a multi-tool-use
        # turn that re-reads cached context on each call, the summed
        # cache_read_input_tokens can exceed the context_window itself
        # and produce nonsense >100% counters in the UI. We instead
        # use the LAST AssistantMessage's per-iteration usage, which
        # represents the model's final view of the context (what the
        # user actually wants to see as "context fullness"). Cumulative
        # output_tokens and total_cost_usd still come from ResultMessage
        # since those genuinely accumulate across the turn.
        #
        # ResultMessage.model_usage is keyed by model id and exposes
        # the model's contextWindow + maxOutputTokens.
        token_usage_payload: dict | None = None
        ctx_window = None
        max_output = None
        try:
            cumulative_usage = getattr(msg, "usage", None) or {}
            proxy_model = self._model_provider_prefix(
                actual_model or model
            ) is not None
            # Prefer stream message_delta for proxy providers
            # (xAI/ChatGPT/z.ai) — AssistantMessage often keeps
            # message_start zeros or a partial merge.
            iter_usage = _pick_iteration_usage(
                latest_assistant_usage,
                latest_stream_usage,
                cumulative_usage,
                proxy_model=proxy_model,
            )
            model_usage = getattr(msg, "model_usage", None) or {}
            ctx_window = None
            max_output = None
            if isinstance(model_usage, dict):
                # Prefer the actual model we ran on; fall back to any entry.
                mu_entry = (
                    model_usage.get(actual_model)
                    if actual_model
                    else None
                )
                if mu_entry is None and model_usage:
                    mu_entry = next(iter(model_usage.values()), None)
                if isinstance(mu_entry, dict):
                    ctx_window = mu_entry.get("contextWindow")
                    max_output = mu_entry.get("maxOutputTokens")
            # TASK-609: Claude Code defaults unknown (proxy-routed)
            # models to 200k. Override with the app-resolved catalog
            # window for proxy providers only — the CLI never knows
            # xAI/OpenAI/z.ai windows. Direct-Anthropic models keep the
            # SDK's own contextWindow (the app value would agree, and
            # the SDK is authoritative for its native models).
            if bot_context_window and proxy_model:
                if not ctx_window or int(ctx_window) in (200_000, 0):
                    ctx_window = bot_context_window
                elif int(ctx_window) < bot_context_window:
                    # e.g. CLI said 200k for a 500k/1M model
                    ctx_window = bot_context_window
            if iter_usage or ctx_window:
                # z.ai reports input_tokens only in message_delta, so the
                # per-iteration AssistantMessage.usage (iter_usage) carries
                # the message_start value (0). Fall back to the cumulative
                # ResultMessage.usage, which via the SDK's last-non-zero merge
                # holds the real final-context input. No-op for Anthropic,
                # where iter_usage.input_tokens is already >0 (its
                # message_delta sends explicit 0s that updateUsage ignores).
                _input_tokens = int(iter_usage.get("input_tokens", 0) or 0)
                if _input_tokens == 0:
                    _input_tokens = int(
                        cumulative_usage.get("input_tokens", 0) or 0
                    )
                _cache_read = int(
                    iter_usage.get("cache_read_input_tokens", 0) or 0
                )
                _cache_create = int(
                    iter_usage.get("cache_creation_input_tokens", 0) or 0
                )
                # If the chosen snapshot still has zero total input but
                # cumulative does not, take cache fields from cumulative too.
                if (
                    _input_tokens + _cache_read + _cache_create
                ) == 0 and isinstance(cumulative_usage, dict):
                    _cache_read = int(
                        cumulative_usage.get(
                            "cache_read_input_tokens", 0
                        )
                        or 0
                    )
                    _cache_create = int(
                        cumulative_usage.get(
                            "cache_creation_input_tokens", 0
                        )
                        or 0
                    )
                _out_tokens = int(
                    cumulative_usage.get("output_tokens", 0) or 0
                )
                if _out_tokens == 0:
                    _out_tokens = int(
                        iter_usage.get("output_tokens", 0) or 0
                    )
                # Cost must be CUMULATIVE across the turn
                # (every internal API iteration billed). Context
                # fullness above is last-iteration only.
                if isinstance(cumulative_usage, dict) and (
                    _usage_input_total(cumulative_usage) > 0
                    or int(cumulative_usage.get("output_tokens", 0) or 0) > 0
                ):
                    usage_for_cost = {
                        "input_tokens": int(
                            cumulative_usage.get("input_tokens", 0) or 0
                        ),
                        "cache_read_input_tokens": int(
                            cumulative_usage.get(
                                "cache_read_input_tokens", 0
                            )
                            or 0
                        ),
                        "cache_creation_input_tokens": int(
                            cumulative_usage.get(
                                "cache_creation_input_tokens", 0
                            )
                            or 0
                        ),
                        "output_tokens": int(
                            cumulative_usage.get("output_tokens", 0)
                            or 0
                        ),
                    }
                else:
                    usage_for_cost = {
                        "input_tokens": _input_tokens,
                        "cache_read_input_tokens": _cache_read,
                        "cache_creation_input_tokens": _cache_create,
                        "output_tokens": _out_tokens,
                    }
                # Prefer provider-accurate cost for proxy models;
                # CLI prices unknowns as Opus-tier ($5/$25).
                cost = _estimate_proxy_cost_usd(
                    actual_model or model, usage_for_cost
                )
                if cost is None:
                    cost = getattr(msg, "total_cost_usd", None)
                token_usage_payload = {
                    "input_tokens": _input_tokens,
                    "cache_read_tokens": _cache_read,
                    "cache_creation_tokens": _cache_create,
                    # Output is still the cumulative turn total — that's
                    # what the user generated overall, regardless of how
                    # many internal iterations produced it.
                    "output_tokens": _out_tokens,
                    "context_window": ctx_window,
                    "max_output_tokens": max_output,
                    "total_cost_usd": cost,
                }
                if actual_model and (
                    str(actual_model).startswith("zai/")
                    or str(actual_model).startswith("xai/")
                ):
                    logger.info(
                        "proxy usage: model=%s iter_in=%s stream_in=%s "
                        "cum_in=%s out=%s ctx=%s cost=%s",
                        actual_model,
                        (latest_assistant_usage or {}).get(
                            "input_tokens"
                        )
                        if isinstance(
                            latest_assistant_usage, dict
                        )
                        else None,
                        (latest_stream_usage or {}).get(
                            "input_tokens"
                        )
                        if isinstance(latest_stream_usage, dict)
                        else None,
                        cumulative_usage.get("input_tokens")
                        if isinstance(cumulative_usage, dict)
                        else None,
                        _out_tokens,
                        ctx_window,
                        cost,
                    )
        except Exception as _usage_err:
            logger.debug("Failed to extract token usage: %s", _usage_err)

        return token_usage_payload, ctx_window, max_output
