from claude_code_bridge.send_boundaries import separator_before_new_block


def test_new_reasoning_block_gets_markdown_paragraph_boundary() -> None:
    assert separator_before_new_block("**first thought**") == "\n\n"


def test_first_or_already_separated_reasoning_block_gets_no_extra_boundary() -> None:
    assert separator_before_new_block("") == ""
    assert separator_before_new_block("**first thought**\n") == ""
    assert separator_before_new_block("**first thought**\n\n") == ""
