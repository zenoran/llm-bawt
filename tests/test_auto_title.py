"""TASK-256: auto-title heuristic tests (hermetic — no DB)."""

import pytest

from llm_bawt.sessions.auto_title import (
    MAX_TITLE_LEN,
    _settled_sessions,
    derive_thread_title,
    maybe_auto_title,
)


# ---------------------------------------------------------------------------
# derive_thread_title (pure)
# ---------------------------------------------------------------------------

class TestDeriveThreadTitle:
    def test_plain_message_passes_through(self):
        assert (
            derive_thread_title("Hey loopy can you help me refactor the auth flow?")
            == "Hey loopy can you help me refactor the auth flow?"
        )

    def test_slash_command_rejected(self):
        assert derive_thread_title("/new") is None
        assert derive_thread_title("  /help args here") is None

    def test_empty_and_whitespace_rejected(self):
        assert derive_thread_title("") is None
        assert derive_thread_title("   \n\t ") is None

    def test_truncates_on_word_boundary_with_ellipsis(self):
        long = "word " * 40
        title = derive_thread_title(long)
        assert title is not None
        assert title.endswith("…")
        assert len(title) <= MAX_TITLE_LEN + 1  # +1 for the ellipsis
        assert not title[:-1].endswith(" ")

    def test_unbreakable_run_hard_truncates(self):
        title = derive_thread_title("x" * 200)
        assert title is not None
        assert len(title) == MAX_TITLE_LEN + 1
        assert title.endswith("…")

    def test_code_fences_and_md_prefixes_stripped(self):
        assert derive_thread_title("```python\nfix this\n```") == "fix this"
        assert derive_thread_title("# Heading text") == "Heading text"
        assert derive_thread_title("> quoted opener") == "quoted opener"

    def test_whitespace_collapses(self):
        assert derive_thread_title("a  b\n\nc\t d") == "a b"  # first line wins

    def test_first_nonempty_line_wins(self):
        assert derive_thread_title("\n\n  \nreal question here\nmore") == "real question here"


# ---------------------------------------------------------------------------
# maybe_auto_title (fake manager)
# ---------------------------------------------------------------------------

class FakeManager:
    def __init__(self, meta):
        self.meta = meta
        self.patches = []

    def get_session(self, session_id):
        return {"id": session_id, "session_metadata": self.meta}

    def update_session_metadata(self, session_id, patch):
        self.patches.append((session_id, patch))
        return True


@pytest.fixture(autouse=True)
def _clear_memo():
    _settled_sessions.clear()
    yield
    _settled_sessions.clear()


class TestMaybeAutoTitle:
    def test_default_thread_gets_titled(self):
        mgr = FakeManager({"title": None, "title_source": "default"})
        maybe_auto_title(mgr, "s1", "help me refactor")
        assert mgr.patches == [("s1", {"title": "help me refactor", "title_source": "auto"})]
        assert "s1" in _settled_sessions

    def test_null_metadata_treated_as_default(self):
        mgr = FakeManager(None)
        maybe_auto_title(mgr, "s2", "first message")
        assert mgr.patches and mgr.patches[0][1]["title_source"] == "auto"

    def test_user_titled_thread_untouched(self):
        mgr = FakeManager({"title": "My name", "title_source": "user"})
        maybe_auto_title(mgr, "s3", "should not retitle")
        assert mgr.patches == []
        assert "s3" in _settled_sessions  # settled — future calls skip the lookup

    def test_auto_titled_thread_untouched(self):
        mgr = FakeManager({"title": "Existing", "title_source": "auto"})
        maybe_auto_title(mgr, "s4", "second message")
        assert mgr.patches == []

    def test_slash_command_leaves_thread_open_for_real_message(self):
        mgr = FakeManager({"title": None, "title_source": "default"})
        maybe_auto_title(mgr, "s5", "/new")
        assert mgr.patches == []
        assert "s5" not in _settled_sessions  # NOT settled — next message titles
        maybe_auto_title(mgr, "s5", "the real first message")
        assert mgr.patches[0][1]["title"] == "the real first message"

    def test_memo_short_circuits(self):
        mgr = FakeManager({"title": None, "title_source": "default"})
        maybe_auto_title(mgr, "s6", "one")
        calls_after_first = len(mgr.patches)
        maybe_auto_title(mgr, "s6", "two")
        assert len(mgr.patches) == calls_after_first

    def test_never_raises_on_broken_manager(self):
        class Broken:
            def get_session(self, _):
                raise RuntimeError("db down")

        maybe_auto_title(Broken(), "s7", "hello")  # must not raise

    def test_missing_session_noop(self):
        class Gone:
            def get_session(self, _):
                return None

        maybe_auto_title(Gone(), "s8", "hello")  # no patch, no raise
