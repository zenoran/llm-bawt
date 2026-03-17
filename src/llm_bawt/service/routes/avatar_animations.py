"""Avatar animation CRUD endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_service
from ..animation_tool import AvatarAnimationStore
from ..schemas import (
    AvatarAnimationCreate,
    AvatarAnimationResponse,
    AvatarAnimationUpdate,
)

router = APIRouter()


def _get_store(service=Depends(get_service)) -> AvatarAnimationStore:
    return AvatarAnimationStore(service.config)


@router.get("/v1/avatar/animations", response_model=list[AvatarAnimationResponse])
def list_animations(store: AvatarAnimationStore = Depends(_get_store)):
    return store.list_all()


@router.post("/v1/avatar/animations", response_model=AvatarAnimationResponse, status_code=201)
def create_animation(
    body: AvatarAnimationCreate,
    store: AvatarAnimationStore = Depends(_get_store),
):
    try:
        return store.create(body.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/v1/avatar/animations/{animation_id}", response_model=AvatarAnimationResponse)
def update_animation(
    animation_id: int,
    body: AvatarAnimationUpdate,
    store: AvatarAnimationStore = Depends(_get_store),
):
    row = store.update(animation_id, body.model_dump(exclude_unset=True))
    if row is None:
        raise HTTPException(status_code=404, detail="Animation not found")
    return row


@router.delete("/v1/avatar/animations/{animation_id}", status_code=204)
def delete_animation(
    animation_id: int,
    store: AvatarAnimationStore = Depends(_get_store),
):
    if not store.delete(animation_id):
        raise HTTPException(status_code=404, detail="Animation not found")
