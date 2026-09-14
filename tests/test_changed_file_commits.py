import httpx

from llm_bawt.service.changed_file_commits import reconcile_files


def test_unavailable_evidence_never_reports_success(monkeypatch):
    def unavailable(*args, **kwargs):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(httpx.Client, "post", unavailable)
    files = [{"path": "~/dev/repo/file", "in_repo": True, "commit_requested": True}]
    reconcile_files(files)
    assert files[0]["commit_state"] == "unknown"
    assert files[0]["commit_requested"] is True


def test_incomplete_success_response_is_rejected(monkeypatch):
    def incomplete(*args, **kwargs):
        return httpx.Response(200, json={"evidence": {"0": {"state": "committed", "commit_hash": "abc"}}},
                              request=httpx.Request("POST", "http://test"))
    monkeypatch.setattr(httpx.Client, "post", incomplete)
    files = [{"path": "~/dev/repo/file", "in_repo": True}]
    reconcile_files(files)
    assert files[0]["commit_state"] == "unknown"


def test_scratch_does_not_contact_evidence_service(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("must not request Git for scratch files")
    monkeypatch.setattr(httpx.Client, "post", unexpected)
    files = [{"path": "/tmp/scratch", "in_repo": False}]
    reconcile_files(files)
    assert files[0]["commit_state"] == "not_applicable"
