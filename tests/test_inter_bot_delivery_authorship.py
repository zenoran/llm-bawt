"""TASK-792 UI-originated delivery authorship regressions."""

from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.routes.inter_bot_deliveries import _format_delivery_message


def test_user_authored_delivery_is_not_wrapped_as_a_bot_message() -> None:
    message = "Repository workspace decision request: TASK-792"

    assert _format_delivery_message(message, AuthorReference.user("nick")) == message


def test_bot_authored_delivery_keeps_the_bot_wrapper() -> None:
    assert _format_delivery_message(
        "Review this change",
        AuthorReference.bot("loopy"),
    ) == "Message from bot 'loopy': Review this change"
