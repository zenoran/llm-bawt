from claude_code_bridge.send_boundaries import (
    publish_run_done_once,
    separator_before_new_block,
    should_read_native_context_usage,
)


def test_new_reasoning_block_gets_markdown_paragraph_boundary() -> None:
    assert separator_before_new_block("**first thought**") == "\n\n"


def test_first_or_already_separated_reasoning_block_gets_no_extra_boundary() -> None:
    assert separator_before_new_block("") == ""
    assert separator_before_new_block("**first thought**\n") == ""
    assert separator_before_new_block("**first thought**\n\n") == ""


def test_proxy_turn_skips_native_context_control_request() -> None:
    assert should_read_native_context_usage(use_proxy=True) is False


def test_direct_turn_keeps_native_context_control_request() -> None:
    assert should_read_native_context_usage(use_proxy=False) is True


def test_run_done_is_published_exactly_once() -> None:
    class Publisher:
        def __init__(self) -> None:
            self.request_ids: list[str] = []

        def publish_run_done(self, request_id: str) -> None:
            self.request_ids.append(request_id)

    publisher = Publisher()
    published = publish_run_done_once(
        publisher,
        "req-terminal",
        already_published=False,
    )
    published = publish_run_done_once(
        publisher,
        "req-terminal",
        already_published=published,
    )

    assert published is True
    assert publisher.request_ids == ["req-terminal"]
