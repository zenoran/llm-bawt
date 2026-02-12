"""User and unified profile routes."""

from fastapi import APIRouter, HTTPException

from ..dependencies import attribute_to_response, get_service
from ..logging import get_service_logger
from ..schemas import (
    ProfileDetail,
    ProfileListResponse,
    UserListResponse,
    UserProfileDetail,
    UserProfileSummary,
)

router = APIRouter()
log = get_service_logger(__name__)


def _get_profile_detail(manager, entity_type_enum, entity_type: str, entity_id: str) -> ProfileDetail:
    profile = manager.get_profile(entity_type_enum, entity_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"{entity_type.capitalize()} '{entity_id}' not found")

    attributes = manager.get_all_attributes(entity_type_enum, entity_id)
    return ProfileDetail(
        entity_type=entity_type,
        entity_id=entity_id,
        display_name=profile.display_name,
        description=profile.description,
        summary=profile.summary,
        attributes=[attribute_to_response(attr) for attr in attributes],
        created_at=profile.created_at.isoformat() if profile.created_at else None,
    )


@router.get("/v1/users", response_model=UserListResponse, tags=["Users"])
async def list_users():
    """List all user profiles."""
    service = get_service()

    try:
        from ..profiles import EntityType, ProfileManager

        manager = ProfileManager(service.config)
        profiles = manager.list_profiles(EntityType.USER)

        users = []
        for profile in profiles:
            attr_count = len(manager.get_all_attributes(EntityType.USER, profile.entity_id))
            users.append(
                UserProfileSummary(
                    user_id=profile.entity_id,
                    display_name=profile.display_name,
                    description=profile.description,
                    attribute_count=attr_count,
                    created_at=profile.created_at.isoformat() if profile.created_at else None,
                )
            )

        return UserListResponse(users=users, total_count=len(users))
    except Exception as e:
        log.error(f"Failed to list users: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/users/{user_id}", response_model=UserProfileDetail, tags=["Users"])
async def get_user_profile(user_id: str):
    """Get detailed user profile with attributes."""
    service = get_service()

    try:
        from ..profiles import EntityType, ProfileManager

        manager = ProfileManager(service.config)
        profile = manager.get_profile(EntityType.USER, user_id)

        if not profile:
            raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")

        attributes = manager.get_all_attributes(EntityType.USER, user_id)

        return UserProfileDetail(
            user_id=user_id,
            display_name=profile.display_name,
            description=profile.description,
            attributes=[attribute_to_response(attr) for attr in attributes],
            created_at=profile.created_at.isoformat() if profile.created_at else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to get user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/v1/users/attribute/{attribute_id}", tags=["Users"])
async def delete_user_attribute(attribute_id: int):
    """Delete a user profile attribute by its ID."""
    service = get_service()

    try:
        from ..profiles import ProfileManager

        manager = ProfileManager(service.config)

        attr = manager.get_attribute_by_id(attribute_id)
        if not attr:
            raise HTTPException(status_code=404, detail=f"Attribute with ID {attribute_id} not found")

        success = manager.delete_attribute_by_id(attribute_id)
        if success:
            return {
                "success": True,
                "message": f"Deleted attribute {attr.category}.{attr.key} from {attr.entity_type}/{attr.entity_id}",
                "deleted": {
                    "id": attribute_id,
                    "entity_type": str(attr.entity_type),
                    "entity_id": attr.entity_id,
                    "category": attr.category,
                    "key": attr.key,
                    "value": attr.value,
                },
            }
        raise HTTPException(status_code=500, detail="Failed to delete attribute")
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to delete user attribute: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/profiles/{entity_id}", response_model=ProfileDetail, tags=["Profiles"])
async def get_profile_auto(entity_id: str):
    """Get profile with attributes - auto-detects entity type (user or bot)."""
    if entity_id in ("user", "bot"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{entity_id}' is a reserved word. To list all {entity_id} profiles, "
                f"use GET /v1/profiles/list/{entity_id}"
            ),
        )

    service = get_service()

    try:
        from ..profiles import EntityType, ProfileManager

        manager = ProfileManager(service.config)

        profile = manager.get_profile(EntityType.USER, entity_id)
        entity_type_str = "user"
        entity_type_enum = EntityType.USER

        if not profile:
            profile = manager.get_profile(EntityType.BOT, entity_id)
            entity_type_str = "bot"
            entity_type_enum = EntityType.BOT

        if not profile:
            raise HTTPException(status_code=404, detail=f"No profile found for entity '{entity_id}'")

        return _get_profile_detail(manager, entity_type_enum, entity_type_str, entity_id)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to get profile for '{entity_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/profiles/list/{entity_type}", response_model=ProfileListResponse, tags=["Profiles"])
async def list_profiles(entity_type: str):
    """List all profiles of a given type (user or bot)."""
    service = get_service()

    if entity_type not in ("user", "bot"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid entity_type '{entity_type}'. Must be 'user' or 'bot'.",
        )

    try:
        from ..profiles import EntityType, ProfileManager

        entity_type_enum = EntityType.USER if entity_type == "user" else EntityType.BOT

        manager = ProfileManager(service.config)
        profiles = manager.list_profiles(entity_type_enum)

        result_profiles = [
            _get_profile_detail(manager, entity_type_enum, entity_type, profile.entity_id)
            for profile in profiles
        ]

        return ProfileListResponse(profiles=result_profiles, total_count=len(result_profiles))
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to list {entity_type} profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/profiles/{entity_type}/{entity_id}", response_model=ProfileDetail, tags=["Profiles"])
async def get_profile(entity_type: str, entity_id: str):
    """Get profile with attributes for any entity type (user or bot)."""
    service = get_service()

    if entity_type not in ("user", "bot"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid entity_type '{entity_type}'. Must be 'user' or 'bot'.",
        )

    try:
        from ..profiles import EntityType, ProfileManager

        entity_type_enum = EntityType.USER if entity_type == "user" else EntityType.BOT
        manager = ProfileManager(service.config)
        return _get_profile_detail(manager, entity_type_enum, entity_type, entity_id)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to get {entity_type} profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))
