"""TASK-609: the single Tier-2 budget authority — Config.resolve_context_budget.

Proves the semantics the audit (TASK-607) locked in:
- prompt_budget subtracts the centrally configured REQUEST RESERVE, not the raw
  model max-output capability;
- the reserve is bounded by capability (min);
- a non-positive window yields a 0 budget (never negative);
- request_output_reserve is a GLOBAL-ONLY setting (bot overrides impossible).
"""

import llm_bawt.runtime_setting_resolution as rsr
from llm_bawt.setting_definitions import SETTING_DEFINITIONS, applies_to_bot_type
from llm_bawt.utils.config import Config, config


def _patch(monkeypatch, *, window, capability, reserve):
    # Config is a pydantic BaseSettings — instance attrs are locked, so patch the
    # methods at the class level.
    monkeypatch.setattr(Config, "get_model_context_window", lambda self, alias=None: window)
    monkeypatch.setattr(Config, "get_model_max_tokens", lambda self, alias=None: capability)
    # resolve_context_budget does a local `from ..runtime_setting_resolution import
    # resolve_global_runtime_setting`, so patch it on the module it's imported from.
    monkeypatch.setattr(
        rsr, "resolve_global_runtime_setting", lambda cfg, key, *a, **k: reserve
    )


def test_budget_subtracts_reserve_not_capability(monkeypatch):
    # Reserve (4096) is SMALLER than capability (16384): the old code subtracted
    # capability and wasted 12288 tokens. The authority must subtract the reserve.
    _patch(monkeypatch, window=500_000, capability=16_384, reserve=4_096)
    cw, effective_reserve, budget = config.resolve_context_budget("any")
    assert cw == 500_000
    assert effective_reserve == 4_096
    assert budget == 495_904  # 500000 - 4096, NOT 500000 - 16384


def test_reserve_is_bounded_by_capability(monkeypatch):
    # If someone sets a reserve larger than the model can emit, cap it.
    _patch(monkeypatch, window=128_000, capability=4_096, reserve=32_000)
    cw, effective_reserve, budget = config.resolve_context_budget("any")
    assert effective_reserve == 4_096  # min(32000, 4096)
    assert budget == 123_904


def test_nonpositive_window_yields_zero_budget(monkeypatch):
    _patch(monkeypatch, window=0, capability=4_096, reserve=4_096)
    cw, effective_reserve, budget = config.resolve_context_budget("any")
    assert cw == 0
    assert budget == 0  # never negative


def test_null_catalog_uses_global_default_window(monkeypatch):
    # A model with no catalog window falls to the global default (128000) — the
    # exact case that used to starve claude-code bots at 8192 -> 4096 budget.
    _patch(monkeypatch, window=128_000, capability=4_096, reserve=4_096)
    _, _, budget = config.resolve_context_budget("__null_catalog__")
    assert budget == 123_904


def test_request_output_reserve_is_global_only():
    d = SETTING_DEFINITIONS["request_output_reserve"]
    assert d.default == 4096
    assert d.applies_to == ()  # global-only: rendered in no per-bot UI
    assert applies_to_bot_type("request_output_reserve", "chat") is False
    assert applies_to_bot_type("request_output_reserve", "agent") is False
