"""TASK-610: the single Tier-1 job authority — Config.resolve_summarization_job.

Proves the semantics TASK-602/608 locked in:
- ONE global dict with every key guaranteed present (session_gap_seconds,
  min_messages_per_session, protected_recent_turns, trigger_tokens, model);
- canonical CODE defaults when no row exists (no legacy env read);
- a stored PARTIAL dict merges OVER defaults (a missing key can't drop a field);
- a non-dict stored value falls back to the full defaults;
- summarization_job is GLOBAL-ONLY (applies_to=() -> no per-bot override).
"""

import llm_bawt.runtime_setting_resolution as rsr
from llm_bawt.setting_definitions import (
    SETTING_DEFINITIONS,
    SUMMARIZATION_JOB_DEFAULTS,
    definitions_for_bot_type,
)
from llm_bawt.utils.config import config

_KEYS = {
    "session_gap_seconds",
    "min_messages_per_session",
    "protected_recent_turns",
    "trigger_tokens",
    "model",
}


def _patch_stored(monkeypatch, value):
    # resolve_summarization_job does a local import of resolve_global_runtime_setting
    # from runtime_setting_resolution — patch it on that module.
    monkeypatch.setattr(
        rsr,
        "resolve_global_runtime_setting",
        lambda cfg, key, *a, **k: value if key == "summarization_job" else None,
    )


def test_registration_is_global_only_json():
    d = SETTING_DEFINITIONS["summarization_job"]
    assert d.applies_to == ()          # global-only: bot overrides impossible
    assert d.type == "json"
    # Invisible in every per-bot settings surface.
    assert d not in definitions_for_bot_type("chat")
    assert d not in definitions_for_bot_type("agent")


def test_defaults_are_the_code_baseline():
    assert set(SUMMARIZATION_JOB_DEFAULTS) == _KEYS
    assert SUMMARIZATION_JOB_DEFAULTS["session_gap_seconds"] == 3600
    assert SUMMARIZATION_JOB_DEFAULTS["min_messages_per_session"] == 2
    assert SUMMARIZATION_JOB_DEFAULTS["protected_recent_turns"] == 3
    assert SUMMARIZATION_JOB_DEFAULTS["trigger_tokens"] == 12000
    assert SUMMARIZATION_JOB_DEFAULTS["model"] is None


def test_no_row_yields_full_default_dict(monkeypatch):
    _patch_stored(monkeypatch, None)
    job = config.resolve_summarization_job()
    assert set(job) == _KEYS
    assert job["trigger_tokens"] == 12000
    assert job["model"] is None


def test_partial_dict_merges_over_defaults(monkeypatch):
    _patch_stored(monkeypatch, {"trigger_tokens": 9000, "model": "grok-4"})
    job = config.resolve_summarization_job()
    assert job["trigger_tokens"] == 9000        # overridden
    assert job["model"] == "grok-4"             # overridden
    assert job["session_gap_seconds"] == 3600   # default preserved
    assert job["min_messages_per_session"] == 2  # default preserved
    assert job["protected_recent_turns"] == 3   # default preserved
    assert set(job) == _KEYS


def test_unknown_keys_in_stored_dict_are_ignored(monkeypatch):
    _patch_stored(monkeypatch, {"trigger_tokens": 5000, "bogus": 1})
    job = config.resolve_summarization_job()
    assert job["trigger_tokens"] == 5000
    assert "bogus" not in job
    assert set(job) == _KEYS


def test_non_dict_stored_value_falls_back_to_defaults(monkeypatch):
    _patch_stored(monkeypatch, "garbage")
    job = config.resolve_summarization_job()
    assert job == SUMMARIZATION_JOB_DEFAULTS
    # And the returned dict is a copy — mutating it must not poison the defaults.
    job["trigger_tokens"] = -1
    assert SUMMARIZATION_JOB_DEFAULTS["trigger_tokens"] == 12000
