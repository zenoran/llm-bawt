"""SSE endpoint to subscribe to in-progress turn events."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..dependencies import get_service

router = APIRouter()


@router.get("/v1/turns/{turn_id}/events", tags=["Debug"])
async def stream_turn_events(turn_id: str):
    """Subscribe to SSE events for a turn.

    Returns buffered events first (replay), then live events as they arrive.
    The stream ends when the turn completes (``[DONE]`` sentinel).
    """
    service = get_service()

    # Check the turn buffer exists (turn is in-progress or recently completed)
    if turn_id not in service._turn_event_buffers:
        raise HTTPException(status_code=404, detail="Turn not found or already expired")

    return StreamingResponse(
        service.subscribe_turn_events(turn_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
