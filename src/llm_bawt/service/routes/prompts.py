"""Prompt template management routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ...prompt_registry import PromptResolver, PromptTemplate, extract_placeholders
from ..dependencies import get_service
from ..schemas import (
    PromptTemplateListResponse,
    PromptTemplateResponse,
    PromptTemplateSeedResponse,
    PromptTemplateUpsertRequest,
    PromptTemplateValidateRequest,
    PromptTemplateValidateResponse,
    PromptTemplateVersionResponse,
    PromptTemplateVersionsResponse,
)

router = APIRouter()


def _row_to_response(row: PromptTemplate) -> PromptTemplateResponse:
    from ...prompt_registry import _parse_json_dict, _parse_json_list

    return PromptTemplateResponse(
        key=row.key,
        title=row.title or row.key,
        category=row.category or "custom",
        format=row.format or "plain_text",
        body=row.body,
        scope_type=row.scope_type,
        scope_id=row.scope_id,
        source="db_override",
        required_vars=_parse_json_list(row.required_vars_json),
        placeholders=extract_placeholders(row.body),
        metadata=_parse_json_dict(row.metadata_json),
        updated_at=row.updated_at,
    )


def _resolved_to_response(resolved) -> PromptTemplateResponse:
    return PromptTemplateResponse(
        key=resolved.key,
        title=resolved.title,
        category=resolved.category,
        format=resolved.format,
        body=resolved.body,
        scope_type=resolved.scope_type,
        scope_id=resolved.scope_id,
        source=resolved.source,
        required_vars=list(resolved.required_vars),
        placeholders=extract_placeholders(resolved.body),
        metadata=dict(resolved.metadata),
        updated_at=resolved.updated_at,
    )


@router.get("/v1/prompts", response_model=PromptTemplateListResponse, tags=["Admin"])
async def list_prompts(
    category: str | None = Query(None, description="Filter by prompt category"),
    scope_type: str = Query("global", description="Resolution scope: global or bot"),
    scope_id: str | None = Query(None, description="Scope ID for bot-scoped resolution"),
    include_defaults: bool = Query(True, description="Include code-default prompts in the listing"),
):
    """List prompt templates, resolving DB overrides against code defaults."""
    service = get_service()
    resolver = PromptResolver(service.config)

    prompts: list[PromptTemplateResponse] = []
    seen_keys: set[tuple[str, str, str]] = set()
    normalized_scope_type = scope_type.strip().lower() if scope_type else "global"
    normalized_scope_id = scope_id

    try:
        if include_defaults:
            for key, definition in sorted(resolver.definitions().items()):
                if category and definition.category != category.strip().lower():
                    continue
                resolved = resolver.resolve(key, normalized_scope_type, normalized_scope_id)
                if resolved is None:
                    continue
                item = _resolved_to_response(resolved)
                prompts.append(item)
                seen_keys.add((item.key, item.scope_type, item.scope_id))
        rows: list[PromptTemplate] = []
        if include_defaults:
            rows.extend(
                resolver.store.list_rows(
                    category=category,
                    scope_type="global",
                    scope_id="*",
                )
            )
            if normalized_scope_type == "bot":
                rows.extend(
                    resolver.store.list_rows(
                        category=category,
                        scope_type="bot",
                        scope_id=normalized_scope_id,
                    )
                )
        else:
            rows = resolver.store.list_rows(
                category=category,
                scope_type=normalized_scope_type,
                scope_id=normalized_scope_id,
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    for row in rows:
        marker = (row.key, row.scope_type, row.scope_id)
        if marker in seen_keys:
            continue
        prompts.append(_row_to_response(row))

    return PromptTemplateListResponse(
        prompts=prompts,
        total_count=len(prompts),
        filters={
            "category": category,
            "scope_type": normalized_scope_type,
            "scope_id": normalized_scope_id,
            "include_defaults": include_defaults,
        },
    )


@router.post("/v1/prompts/seed-defaults", response_model=PromptTemplateSeedResponse, tags=["Admin"])
async def seed_prompt_defaults():
    """Insert any missing built-in prompt templates into the DB."""
    service = get_service()
    resolver = PromptResolver(service.config)
    if resolver.store.engine is None:
        raise HTTPException(status_code=503, detail="Prompt template DB unavailable")

    result = resolver.store.seed_defaults()
    return PromptTemplateSeedResponse(**result)


@router.get("/v1/prompts/{key}/versions", response_model=PromptTemplateVersionsResponse, tags=["Admin"])
async def list_prompt_versions(
    key: str,
    scope_type: str = Query("global", description="Scope type: global or bot"),
    scope_id: str | None = Query(None, description="Scope ID for bot-scoped prompts"),
):
    """List saved versions for one prompt key/scope."""
    service = get_service()
    resolver = PromptResolver(service.config)
    if resolver.store.engine is None:
        raise HTTPException(status_code=503, detail="Prompt template DB unavailable")

    try:
        versions = resolver.store.list_versions(key, scope_type, scope_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PromptTemplateVersionsResponse(
        key=key,
        scope_type=scope_type,
        scope_id=scope_id or ("*" if scope_type == "global" else ""),
        versions=[
            PromptTemplateVersionResponse(
                version=row.version,
                body=row.body,
                change_note=row.change_note,
                created_by=row.created_by,
                created_at=row.created_at,
            )
            for row in versions
        ],
        total_count=len(versions),
    )


@router.get("/v1/prompts/{key}", response_model=PromptTemplateResponse, tags=["Admin"])
async def get_prompt(
    key: str,
    scope_type: str = Query("global", description="Resolution scope: global or bot"),
    scope_id: str | None = Query(None, description="Scope ID for bot-scoped prompts"),
    exact: bool = Query(False, description="If true, fetch the exact DB row only (no fallback)"),
):
    """Get one prompt template, optionally with code-default fallback resolution."""
    service = get_service()
    resolver = PromptResolver(service.config)

    if exact:
        if resolver.store.engine is None:
            raise HTTPException(status_code=503, detail="Prompt template DB unavailable")
        try:
            row = resolver.store.get_exact(key, scope_type, scope_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if row is None:
            raise HTTPException(status_code=404, detail="Prompt template not found")
        return _row_to_response(row)

    try:
        resolved = resolver.resolve(key, scope_type, scope_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if resolved is None:
        raise HTTPException(status_code=404, detail="Prompt template not found")
    return _resolved_to_response(resolved)


@router.put("/v1/prompts/{key}", response_model=PromptTemplateResponse, tags=["Admin"])
@router.patch("/v1/prompts/{key}", response_model=PromptTemplateResponse, tags=["Admin"])
async def upsert_prompt(key: str, request: PromptTemplateUpsertRequest):
    """Create or update one prompt template override."""
    service = get_service()
    resolver = PromptResolver(service.config)
    if resolver.store.engine is None:
        raise HTTPException(status_code=503, detail="Prompt template DB unavailable")

    validation = resolver.validate(key=key, body=request.body, required_vars=request.required_vars or None)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail={"errors": validation["errors"], **validation})

    definition = resolver.definition_for(key)
    try:
        row = resolver.store.upsert(
            key=key,
            body=request.body,
            scope_type=request.scope_type,
            scope_id=request.scope_id,
            title=request.title or (definition.title if definition else key),
            category=(request.category or (definition.category if definition else "custom")).strip().lower(),
            format=request.format or (definition.format if definition else "plain_text"),
            required_vars=validation["required_vars"],
            metadata=request.metadata,
            updated_by=request.updated_by,
            change_note=request.change_note,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return _row_to_response(row)


@router.post("/v1/prompts/{key}/validate", response_model=PromptTemplateValidateResponse, tags=["Admin"])
async def validate_prompt(key: str, request: PromptTemplateValidateRequest):
    """Validate a prompt template body and optionally render a preview."""
    service = get_service()
    resolver = PromptResolver(service.config)

    body = request.body
    if body is None:
        resolved = resolver.resolve(key)
        if resolved is None:
            raise HTTPException(status_code=404, detail="Prompt template not found")
        body = resolved.body

    validation = resolver.validate(key=key, body=body, required_vars=request.required_vars)
    rendered_preview = None
    if validation["valid"] and request.variables:
        try:
            rendered_preview = resolver.render(key=key, variables=request.variables, body_override=body)
        except KeyError as e:
            validation["valid"] = False
            validation["errors"].append(f"Missing preview variable: {e.args[0]}")

    return PromptTemplateValidateResponse(
        valid=validation["valid"],
        required_vars=validation["required_vars"],
        placeholders=validation["placeholders"],
        missing_required=validation["missing_required"],
        unknown_placeholders=validation["unknown_placeholders"],
        rendered_preview=rendered_preview,
        errors=validation["errors"],
    )


@router.post("/v1/prompts/{key}/preview", response_model=PromptTemplateValidateResponse, tags=["Admin"])
async def preview_prompt(
    key: str,
    request: PromptTemplateValidateRequest,
    scope_type: str = Query("global", description="Resolution scope: global or bot"),
    scope_id: str | None = Query(None, description="Scope ID for bot-scoped prompts"),
):
    """Render a prompt template with variables to preview the final prompt text."""
    service = get_service()
    resolver = PromptResolver(service.config)

    body = request.body
    if body is None:
        try:
            resolved = resolver.resolve(key, scope_type, scope_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if resolved is None:
            raise HTTPException(status_code=404, detail="Prompt template not found")
        body = resolved.body

    validation = resolver.validate(key=key, body=body, required_vars=request.required_vars)
    if not validation["valid"]:
        return PromptTemplateValidateResponse(
            valid=False,
            required_vars=validation["required_vars"],
            placeholders=validation["placeholders"],
            missing_required=validation["missing_required"],
            unknown_placeholders=validation["unknown_placeholders"],
            rendered_preview=None,
            errors=validation["errors"],
        )

    try:
        preview = resolver.render(key=key, variables=request.variables, body_override=body)
    except KeyError as e:
        return PromptTemplateValidateResponse(
            valid=False,
            required_vars=validation["required_vars"],
            placeholders=validation["placeholders"],
            missing_required=validation["missing_required"],
            unknown_placeholders=validation["unknown_placeholders"],
            rendered_preview=None,
            errors=[f"Missing preview variable: {e.args[0]}"],
        )

    return PromptTemplateValidateResponse(
        valid=True,
        required_vars=validation["required_vars"],
        placeholders=validation["placeholders"],
        missing_required=validation["missing_required"],
        unknown_placeholders=validation["unknown_placeholders"],
        rendered_preview=preview,
        errors=[],
    )


@router.post("/v1/prompts/{key}/reset", response_model=PromptTemplateResponse, tags=["Admin"])
async def reset_prompt(
    key: str,
    scope_type: str = Query("global", description="Scope type: global or bot"),
    scope_id: str | None = Query(None, description="Scope ID for bot-scoped prompts"),
):
    """Delete a DB override and fall back to the code-default prompt."""
    service = get_service()
    resolver = PromptResolver(service.config)
    if resolver.store.engine is None:
        raise HTTPException(status_code=503, detail="Prompt template DB unavailable")

    try:
        deleted = resolver.store.reset(key, scope_type, scope_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    resolved = resolver.resolve(key, scope_type, scope_id)
    if resolved is None:
        if not deleted:
            raise HTTPException(status_code=404, detail="Prompt template not found")
        raise HTTPException(status_code=404, detail="Prompt template reset but no code default exists")
    return _resolved_to_response(resolved)
