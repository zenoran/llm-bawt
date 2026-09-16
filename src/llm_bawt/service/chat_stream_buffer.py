"""Sentence-boundary SSE content buffering, independent of turn coordination."""
import json


def content_buffer(response_id, created, model_alias):
    _content_buf: list[str] = []
    _CONTENT_MAX_HOLD = 120  # force a word-boundary release past this
    _HARD_BOUNDARY = frozenset(".!?…\n")       # sentence / hard stops
    _SOFT_BOUNDARY = frozenset(",;:)]}\"'’”")   # clause / closing punct

    def _content_release_cut(s: str) -> int:
        """Slice index: release s[:cut], hold s[cut:]. 0 ⇒ hold all."""
        # Release through the last hard boundary (sentence end / newline).
        for i in range(len(s) - 1, -1, -1):
            if s[i] in _HARD_BOUNDARY:
                return i + 1
        # Else through the last clause-closing punctuation.
        for i in range(len(s) - 1, -1, -1):
            if s[i] in _SOFT_BOUNDARY:
                return i + 1
        # No punctuation yet: keep holding until we've buffered enough
        # that a long, punctuation-free run (code, URLs, lists) would
        # stall — then release up to the last whitespace so we never
        # leave a mid-word tail on the wire.
        if len(s) >= _CONTENT_MAX_HOLD:
            ws = s.rfind(" ")
            return ws + 1 if ws > 0 else len(s)
        return 0

    def _drain_content_buf(flush_all: bool):
        """SSE line for releasable buffered content, or None. Mutates buf."""
        if not _content_buf:
            return None
        s = "".join(_content_buf)
        cut = len(s) if flush_all else _content_release_cut(s)
        if cut <= 0:
            return None
        _content_buf.clear()
        if cut < len(s):
            _content_buf.append(s[cut:])
        data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_alias,
            "choices": [{"index": 0, "delta": {"content": s[:cut]}, "finish_reason": None}],
        }
        return f"data: {json.dumps(data)}\n\n"

    return _content_buf, _drain_content_buf
