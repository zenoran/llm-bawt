from __future__ import annotations

from types import SimpleNamespace

from claude_code_bridge.send_errors import (
    AuthRetryPolicy,
    CLAUDE_CREDENTIAL_ERROR_MARKER,
    classify_terminal_error,
    result_message_error,
)
from claude_code_bridge.send_stream import ClaudeStreamMixin


def _result(**overrides):
    values = {
        "is_error": True,
        "api_error_status": None,
        "errors": None,
        "result": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_snark_401_result_is_terminal_credential_failure():
    error = result_message_error(
        _result(
            api_error_status=401,
            errors=[
                "Failed to authenticate. API Error: 401 OAuth access token has been revoked."
            ],
        ),
        fallback="HTTP 401: authentication_failed",
    )

    assert error is not None
    assert error.credential_error is True
    text, raw = classify_terminal_error(error)
    assert text.startswith(CLAUDE_CREDENTIAL_ERROR_MARKER)
    assert raw == {"error_code": "credential_expired", "provider": "claude"}


def test_proxy_auth_failure_does_not_emit_claude_reconnect_marker():
    error = result_message_error(
        _result(api_error_status=401, errors=["authentication_failed"])
    )

    assert error is not None
    text, raw = classify_terminal_error(error, direct_anthropic=False)
    assert CLAUDE_CREDENTIAL_ERROR_MARKER not in text
    assert raw is None


def test_generic_upstream_exhaustion_is_error_but_not_credential_failure():
    error = result_message_error(
        _result(
            api_error_status=529,
            errors=["overloaded_error"],
        ),
        fallback="HTTP 529: overloaded_error",
    )

    assert error is not None
    assert error.credential_error is False
    text, raw = classify_terminal_error(error)
    assert "529" in text
    assert CLAUDE_CREDENTIAL_ERROR_MARKER not in text
    assert raw is None


def test_success_result_is_not_classified_as_error():
    assert result_message_error(
        _result(is_error=False, api_error_status=None, errors=None, result="ok")
    ) is None


def test_auth_retry_policy_allows_exactly_one_side_effect_free_direct_retry():
    policy = AuthRetryPolicy()

    assert policy.claim(
        is_auth_failure=True,
        direct_anthropic=True,
        model_side_effects=False,
    ) is True
    assert policy.claim(
        is_auth_failure=True,
        direct_anthropic=True,
        model_side_effects=False,
    ) is False


def test_auth_retry_policy_rejects_proxy_and_side_effectful_attempts():
    assert AuthRetryPolicy().claim(
        is_auth_failure=True,
        direct_anthropic=False,
        model_side_effects=False,
    ) is False
    assert AuthRetryPolicy().claim(
        is_auth_failure=True,
        direct_anthropic=True,
        model_side_effects=True,
    ) is False


class _RetryStatusHarness(ClaudeStreamMixin):
    def __init__(self):
        self.events: list[dict] = []

    def _publish_event(self, *_args, **kwargs):
        self.events.append(kwargs)


def test_auth_retry_status_is_silent_before_self_heal():
    harness = _RetryStatusHarness()
    text_parts: list[str] = []

    seq, attempt, error, surfaced = harness._on_api_retry_status(
        {
            "attempt": 1,
            "max_retries": 10,
            "error_status": 401,
            "error": "authentication_failed",
        },
        request_id="req",
        session_key="snark:nick",
        seq=0,
        text_parts=text_parts,
        already_surfaced=False,
    )

    assert (seq, attempt, error, surfaced) == (
        0,
        1,
        "HTTP 401: authentication_failed",
        False,
    )
    assert text_parts == []
    assert harness.events == []


def test_generic_retry_status_remains_visible():
    harness = _RetryStatusHarness()
    text_parts: list[str] = []

    seq, _attempt, _error, surfaced = harness._on_api_retry_status(
        {
            "attempt": 1,
            "max_retries": 10,
            "error_status": 529,
            "error": "overloaded_error",
        },
        request_id="req",
        session_key="snark:nick",
        seq=0,
        text_parts=text_parts,
        already_surfaced=False,
    )

    assert seq == 1
    assert surfaced is True
    assert text_parts == ["⏳ Upstream unavailable (overloaded_error), retrying…"]
    assert len(harness.events) == 1
