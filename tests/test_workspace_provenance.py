from datetime import datetime, timezone

from llm_bawt.service.workspace_provenance import (
    CurrentWorkspaceFile,
    LineChange,
    TurnFileSnapshot,
    attribute_workspace,
    sha256_text,
)


def current(before: str, after: str) -> CurrentWorkspaceFile:
    return CurrentWorkspaceFile(
        repo_id="repo-1",
        repo_key="bawthub",
        repo_aliases=("bawthub", "/repos/bawthub"),
        path="src/example.ts",
        staged=False,
        status="M",
        old_text=before,
        new_text=after,
    )


def snapshot(
    turn_id: str,
    before: str,
    after: str,
    *,
    created_at: datetime,
    path: str = "src/example.ts",
) -> TurnFileSnapshot:
    return TurnFileSnapshot(
        turn_id=turn_id,
        trigger_message_id=f"msg-{turn_id}",
        bot_id="caid",
        user_id="nick",
        created_at=created_at,
        repo_key="workspace",
        repo_label="bawthub",
        path=path,
        old_path=None,
        change_kind="modified",
        before_sha256=sha256_text(before),
        after_sha256=sha256_text(after),
        before_text=before,
        after_text=after,
        binary=False,
        truncated=False,
    )


def test_exact_transition_is_owned() -> None:
    before = "one\ntwo\n"
    after = "one\nchanged\n"
    result = attribute_workspace([
        current(before, after),
    ], [
        snapshot("turn-a", before, after, created_at=datetime(2026, 8, 4, tzinfo=timezone.utc)),
    ])[0]

    assert result.ownership == "owned"
    assert result.unmatched_changes == 0
    assert [candidate.confidence for candidate in result.candidates] == ["exact"]
    assert result.candidates[0].exact_transition is True


def test_disjoint_turn_changes_are_shared() -> None:
    original = "alpha\nbeta\ngamma\ndelta\n"
    after_first = "ALPHA\nbeta\ngamma\ndelta\n"
    final = "ALPHA\nbeta\ngamma\nDELTA\n"
    result = attribute_workspace([
        current(original, final),
    ], [
        snapshot("turn-a", original, after_first, created_at=datetime(2026, 8, 4, 1, tzinfo=timezone.utc)),
        snapshot("turn-b", after_first, final, created_at=datetime(2026, 8, 4, 2, tzinfo=timezone.utc)),
    ])[0]

    assert result.ownership == "shared"
    assert result.unmatched_changes == 0
    assert {candidate.turn_id for candidate in result.candidates} == {"turn-a", "turn-b"}
    assert all(candidate.dirty_hunks for candidate in result.candidates)


def test_path_only_candidate_does_not_claim_ownership() -> None:
    result = attribute_workspace([
        current("before\n", "manual\n"),
    ], [
        snapshot(
            "turn-a",
            "different\n",
            "unrelated\n",
            created_at=datetime(2026, 8, 4, tzinfo=timezone.utc),
        ),
    ])[0]

    assert result.ownership == "unattributed"
    assert result.candidates[0].confidence == "probable"
    assert result.unmatched_changes > 0


def test_supplied_hunk_ids_are_preserved() -> None:
    before = "alpha\nbeta\n"
    after = "ALPHA\nbeta\n"
    live = current(before, after)
    live = CurrentWorkspaceFile(**{
        **live.__dict__,
        "changes": (
            LineChange("-", "alpha\n", 7, 1),
            LineChange("+", "ALPHA\n", 7, 1),
        ),
    })
    result = attribute_workspace([live], [
        snapshot("turn-a", before, after, created_at=datetime(2026, 8, 4, tzinfo=timezone.utc)),
    ])[0]

    assert result.candidates[0].dirty_hunks == [7]


def test_same_path_in_different_repo_does_not_match() -> None:
    candidate = snapshot(
        "turn-a",
        "before\n",
        "after\n",
        created_at=datetime(2026, 8, 4, tzinfo=timezone.utc),
    )
    candidate = TurnFileSnapshot(**{
        **candidate.__dict__,
        "repo_label": "llm-bawt",
        "repo_key": "llm-bawt",
    })

    result = attribute_workspace([current("before\n", "after\n")], [candidate])[0]

    assert result.ownership == "unattributed"
    assert result.candidates == ()
