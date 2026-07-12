import pytest
from fastapi import HTTPException
from unittest.mock import MagicMock

from llm_bawt.service.routes.model_catalog import (
    _check_endpoint_cas,
    _delete_model_sources,
    _normalized_harness,
    list_harnesses,
)


def test_catalog_harnesses_are_driven_by_protocol_compatibility():
    rows = {row["key"]: row for row in list_harnesses()["harnesses"]}

    assert rows["chat"]["protocol"] == "chat-completions"
    assert rows["codex"]["protocol"] == "responses"
    assert rows["claude-proxy"]["proxies"] is True


def test_unknown_harness_is_rejected_before_querying_catalog():
    with pytest.raises(HTTPException) as exc:
        _normalized_harness("not-a-harness")

    assert exc.value.status_code == 400


def test_endpoint_compare_and_set_rejects_stale_or_missing_identity():
    _check_endpoint_cas(20, None)
    _check_endpoint_cas(20, 20)

    with pytest.raises(HTTPException, match="expected 19, found 20"):
        _check_endpoint_cas(20, 19)
    with pytest.raises(HTTPException, match="no longer exists"):
        _check_endpoint_cas(None, 20)


def test_delete_model_removes_legacy_source_before_normalized_row():
    conn = MagicMock()
    conn.execute.return_value.first.return_value = (42,)

    assert _delete_model_sources(conn, "grok-old") == (42,)

    statements = [str(call.args[0]) for call in conn.execute.call_args_list]
    assert "DELETE FROM model_definitions" in statements[0]
    assert "DELETE FROM models" in statements[1]
    assert all(call.args[1] == {"key": "grok-old"} for call in conn.execute.call_args_list)
