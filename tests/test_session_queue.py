from __future__ import annotations

import asyncio

from agent_bridge.session_queue import SessionQueue


def test_completed_task_cannot_clear_newer_active_task() -> None:
    async def run() -> None:
        queue = SessionQueue()
        release_first = asyncio.Event()
        release_second = asyncio.Event()

        async def wait_for(event: asyncio.Event) -> None:
            await event.wait()

        first = asyncio.create_task(wait_for(release_first))
        second = asyncio.create_task(wait_for(release_second))
        queue.set_active_task("codex:nick", first)
        queue.set_active_task("codex:nick", second)

        release_first.set()
        await first
        await asyncio.sleep(0)

        assert queue.has_active_task("codex:nick")
        assert queue.cancel_active("codex:nick") is True
        try:
            await second
        except asyncio.CancelledError:
            pass

    asyncio.run(run())


def test_queued_task_never_replaces_lock_holder_as_active() -> None:
    async def run() -> None:
        queue = SessionQueue()
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        second_entered = asyncio.Event()

        async def first_send() -> None:
            async with queue.active("codex:nick"):
                first_entered.set()
                await release_first.wait()

        async def second_send() -> None:
            async with queue.active("codex:nick"):
                second_entered.set()

        first = asyncio.create_task(first_send())
        await first_entered.wait()
        second = asyncio.create_task(second_send())
        await asyncio.sleep(0)

        assert not second_entered.is_set()
        assert queue.cancel_active("codex:nick") is True
        try:
            await first
        except asyncio.CancelledError:
            pass

        await second
        assert second_entered.is_set()
        assert not queue.has_active_task("codex:nick")

    asyncio.run(run())


def test_queued_task_emits_heartbeats_without_losing_queue_position() -> None:
    async def run() -> None:
        queue = SessionQueue()
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        second_entered = asyncio.Event()
        heartbeats: list[str] = []

        async def first_send() -> None:
            async with queue.active("snark:nick"):
                first_entered.set()
                await release_first.wait()

        async def second_send() -> None:
            async with queue.active(
                "snark:nick",
                on_wait=lambda: heartbeats.append("queued"),
                wait_interval=0.01,
            ):
                second_entered.set()

        first = asyncio.create_task(first_send())
        await first_entered.wait()
        second = asyncio.create_task(second_send())
        await asyncio.sleep(0.035)

        assert len(heartbeats) >= 2
        assert not second_entered.is_set()
        assert queue.has_active_task("snark:nick")

        release_first.set()
        await first
        await second

        assert second_entered.is_set()
        assert not queue.has_active_task("snark:nick")

    asyncio.run(run())


def test_uncontended_task_does_not_emit_queue_heartbeat() -> None:
    async def run() -> None:
        queue = SessionQueue()
        heartbeats: list[str] = []

        async with queue.active(
            "loopy:nick",
            on_wait=lambda: heartbeats.append("queued"),
            wait_interval=0.01,
        ):
            assert queue.has_active_task("loopy:nick")

        assert heartbeats == []

    asyncio.run(run())
