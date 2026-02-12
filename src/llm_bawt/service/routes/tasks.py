"""Background task submission and status routes."""

from fastapi import APIRouter, HTTPException, Query

from ..dependencies import get_service
from ..schemas import TaskStatusResponse, TaskSubmitRequest, TaskSubmitResponse
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

# -------------------------------------------------------------------------
# History Endpoints
# -------------------------------------------------------------------------
