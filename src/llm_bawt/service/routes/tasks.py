"""Background task submission and status routes."""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from ..dependencies import get_service
from ..schemas import (
    TaskListItem,
    TaskListResponse,
    TaskStatusResponse,
    TaskSubmitRequest,
    TaskSubmitResponse,
)
from ..tasks import Task, TaskType

router = APIRouter()

@router.post("/v1/tasks", response_model=TaskSubmitResponse, tags=["Tasks"])
async def submit_task(request: TaskSubmitRequest):
    """Submit a background task for processing."""
    service = get_service()
    
    try:
        task_type = TaskType(request.task_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task type: {request.task_type}"
        )
    
    task = Task(
        task_type=task_type,
        payload=request.payload,
        bot_id=request.bot_id or service._default_bot,
        user_id=request.user_id,
        priority=request.priority,
    )
    
    task_id = service.submit_task(task)
    return TaskSubmitResponse(task_id=task_id)

@router.get("/v1/tasks/{task_id}", response_model=TaskStatusResponse, tags=["Tasks"])
async def get_task_status(
    task_id: str,
    wait: bool = Query(False, description="Wait for task completion"),
    timeout: float = Query(30.0, description="Wait timeout in seconds"),
):
    """Get the status of a submitted task."""
    service = get_service()
    
    if wait:
        result = await service.wait_for_result(task_id, timeout)
    else:
        result = service.get_result(task_id)
    
    if result:
        return TaskStatusResponse(
            task_id=task_id,
            status=result.status.value,
            result=result.result,
            error=result.error,
            processing_time_ms=result.processing_time_ms,
        )
    else:
        return TaskStatusResponse(task_id=task_id, status="pending")


@router.get("/v1/tasks", response_model=TaskListResponse, tags=["Tasks"])
async def list_tasks(
    status: str | None = Query(
        None,
        description="Filter by status: pending, completed, failed, running",
    ),
    task_type: str | None = Query(None, description="Filter by task type"),
    bot_id: str | None = Query(None, description="Filter by bot ID"),
    user_id: str | None = Query(None, description="Filter by user ID"),
    limit: int = Query(100, ge=1, le=1000, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """List submitted task records with filtering and pagination."""
    service = get_service()

    normalized_status = status.strip().lower() if status else None
    if normalized_status and normalized_status not in {"pending", "running", "completed", "failed"}:
        raise HTTPException(status_code=400, detail="Invalid status filter")

    normalized_task_type = task_type.strip().lower() if task_type else None
    normalized_bot_id = bot_id.strip().lower() if bot_id else None
    normalized_user_id = user_id.strip().lower() if user_id else None

    items: list[TaskListItem] = []

    # Pending tasks (queue)
    if normalized_status in (None, "pending", "running"):
        with service._task_queue.mutex:
            queued = list(service._task_queue.queue)
        for queued_item in queued:
            task_obj = getattr(queued_item, "task", None)
            if not task_obj:
                continue
            if normalized_task_type and task_obj.task_type.value != normalized_task_type:
                continue
            if normalized_bot_id and (task_obj.bot_id or "").strip().lower() != normalized_bot_id:
                continue
            if normalized_user_id and (task_obj.user_id or "").strip().lower() != normalized_user_id:
                continue

            items.append(
                TaskListItem(
                    task_id=task_obj.task_id,
                    status="pending",
                    task_type=task_obj.task_type.value,
                    bot_id=task_obj.bot_id,
                    user_id=task_obj.user_id,
                    priority=task_obj.priority,
                    created_at=task_obj.created_at.isoformat() if isinstance(task_obj.created_at, datetime) else None,
                )
            )

    # Completed/failed tasks (results)
    if normalized_status in (None, "completed", "failed"):
        for task_id, result in service._results.items():
            result_status = result.status.value
            if normalized_status and result_status != normalized_status:
                continue

            submitted = service._submitted_tasks.get(task_id)
            submitted_type = submitted.task_type.value if submitted else None
            submitted_bot = submitted.bot_id if submitted else None
            submitted_user = submitted.user_id if submitted else None

            if normalized_task_type and submitted_type != normalized_task_type:
                continue
            if normalized_bot_id and (submitted_bot or "").strip().lower() != normalized_bot_id:
                continue
            if normalized_user_id and (submitted_user or "").strip().lower() != normalized_user_id:
                continue

            items.append(
                TaskListItem(
                    task_id=task_id,
                    status=result_status,
                    task_type=submitted_type,
                    bot_id=submitted_bot,
                    user_id=submitted_user,
                    priority=submitted.priority if submitted else None,
                    created_at=submitted.created_at.isoformat() if submitted else None,
                    completed_at=result.completed_at.isoformat() if result.completed_at else None,
                    processing_time_ms=result.processing_time_ms,
                    error=result.error,
                )
            )

    items.sort(
        key=lambda row: row.created_at or row.completed_at or "",
        reverse=True,
    )
    total_count = len(items)
    page = items[offset : offset + limit]

    return TaskListResponse(
        tasks=page,
        total_count=total_count,
        filters={
            "status": normalized_status,
            "task_type": normalized_task_type,
            "bot_id": normalized_bot_id,
            "user_id": normalized_user_id,
            "limit": limit,
            "offset": offset,
        },
    )

# -------------------------------------------------------------------------
# History Endpoints
# -------------------------------------------------------------------------
