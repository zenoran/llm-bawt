"""Ignore is user intent, not Git evidence or a permanent path exclusion."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_changed_files_store import store, _conversation, _trigger, _mod  # noqa: F401
from llm_bawt.service.changed_file_decisions import (
    ChangedFileDecisionConflict, ChangedFileDecisionsStore,
)


def capture(store, turn="t1", after="v1", session="s"):
    _trigger(store, message_id=turn, session_id=session)
    store.save_turn_files(turn_id=turn, bot_id="b", user_id="u", trigger_message_id=turn,
                          files=[_mod("same.py", "base", after), _mod("other.py", "x", "y")])
    return store.summary_for_turn(turn)["files"][1]  # sorted paths: other, same


def decide(store, files, ignored=True, **kwargs):
    return ChangedFileDecisionsStore(store.engine).set_decision(
        session_id=kwargs.pop("session_id", "s"), bot_id=kwargs.pop("bot_id", "b"),
        user_id=kwargs.pop("user_id", "u"), files=files, ignored=ignored,
    )


def test_ignore_restore_durable_history_and_git_independent(store):
    _conversation(store, session_id="s")
    file = capture(store)
    assert decide(store, [file]) == ("s", 1)
    # Fresh store simulates reload, without any in-memory decision cache.
    fresh = ChangedFileDecisionsStore(store.engine)
    _, summary = fresh.uncommitted_summary_for_session(session_id="s", bot_id="b", user_id="u")
    assert [f["path"] for f in summary["files"]] == ["other.py"]
    ignored = next(f for f in summary["history_files"] if f["path"] == "same.py")
    assert ignored["ignored"] is True
    assert ignored["commit_state"] == "unknown"
    assert ignored["commit_requested"] is False
    assert decide(store, [file], False) == ("s", 1)
    _, summary = fresh.uncommitted_summary_for_session(session_id="s", bot_id="b", user_id="u")
    assert summary["total_files"] == 2


def test_new_turn_and_in_place_edits_become_pending_again(store):
    _conversation(store, session_id="s")
    file = capture(store)
    decide(store, [file])
    # Same payload retry keeps intent.
    store.save_turn_files(turn_id="t1", bot_id="b", user_id="u", trigger_message_id="t1",
                          files=[_mod("same.py", "base", "v1")])
    assert store.summary_for_turn("t1")["files"][1]["ignored"]
    # Editing in the same active turn invalidates the decision.
    store.save_turn_files(turn_id="t1", bot_id="b", user_id="u", trigger_message_id="t1",
                          files=[_mod("same.py", "base", "v2")])
    assert not store.summary_for_turn("t1")["files"][1]["ignored"]
    # Returning to old bytes must not revive a cleared Ignore decision.
    store.save_turn_files(turn_id="t1", bot_id="b", user_id="u", trigger_message_id="t1",
                          files=[_mod("same.py", "base", "v1")])
    assert not store.summary_for_turn("t1")["files"][1]["ignored"]
    store.save_turn_files(turn_id="t1", bot_id="b", user_id="u", trigger_message_id="t1",
                          files=[_mod("same.py", "base", "v2")])
    with pytest.raises(ChangedFileDecisionConflict):
        decide(store, [file])
    latest = store.summary_for_turn("t1")["files"][1]
    decide(store, [latest])
    capture(store, turn="t2", after="v3")
    _, summary = store.uncommitted_summary_for_session(session_id="s", bot_id="b", user_id="u")
    assert any(f["path"] == "same.py" and f["turn_id"] == "t2" for f in summary["files"])


def test_bulk_atomic_foreign_session_rejected_and_owner_scoped(store):
    _conversation(store, session_id="s")
    _conversation(store, session_id="foreign", status="archived")
    file = capture(store)
    foreign = capture(store, turn="foreign-turn", session="foreign")
    with pytest.raises(LookupError):
        decide(store, [file, foreign])
    assert not store.summary_for_turn("t1")["files"][1]["ignored"]
    assert decide(store, [file], user_id="someone-else") == (None, 0)
    assert decide(store, [file], bot_id="someone-else") == (None, 0)
    assert decide(store, [file], session_id=None) == ("s", 1)


def test_ignore_latest_does_not_resurrect_older_capture(store):
    _conversation(store, session_id="s")
    old = capture(store)
    new = capture(store, turn="t2", after="v2")
    decide(store, [old, new])
    _, summary, other = store.uncommitted_summaries_for_session(
        session_id="s", anchor_turn_id="t2", bot_id="b", user_id="u")
    assert all(f["path"] != "same.py" for f in summary["files"] + other["files"])
    assert len([f for f in summary["history_files"] if f["ignored"]]) == 2


def test_decision_route_validation_errors_and_restore(store, monkeypatch):
    from llm_bawt.service.routes import turn_logs
    _conversation(store, session_id="s")
    file = capture(store)
    monkeypatch.setattr(turn_logs, "get_turn_log_store", lambda: store)
    app = FastAPI()
    app.include_router(turn_logs.router)
    client = TestClient(app)
    payload = {"session_id": "s", "bot_id": "b", "user_id": "u", "files": [file], "ignored": True}
    url = "/v1/turn-changed-files/decision"
    assert client.post(url, json=payload).json()["marked_files"] == 1
    assert client.post(url, json={**payload, "ignored": False}).status_code == 200
    assert client.post(url, json={**payload, "user_id": "foreign"}).status_code == 404
    assert client.post(url, json={**payload, "files": []}).status_code == 422
    assert client.post(url, json={**payload, "files": [{**file, "snapshot_version": "0" * 64}]}).status_code == 409
    assert client.post(url, json={**payload, "files": [{**file, "path": "missing"}]}).status_code == 404
