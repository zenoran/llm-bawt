"""Focused tests for unified profile route behavior."""

from types import SimpleNamespace

from llm_bawt.service.routes import profiles as profile_routes
from llm_bawt.service.schemas import ProfileUpdateRequest


class _ProfileManager:
    def __init__(self) -> None:
        self.avatar: str | None = "🐕"
        self.avatar_render: str | None = "data:image/svg+xml,old"
        self.update_calls: list[dict[str, object]] = []

    def update_profile(self, _entity_type, _entity_id, **fields):
        self.update_calls.append(fields)
        if fields.get("avatar") is not None:
            self.avatar = str(fields["avatar"]).strip() or None
        if fields.get("avatar_render") is not None:
            self.avatar_render = str(fields["avatar_render"]) or None
        return self.get_profile(_entity_type, _entity_id)

    def get_profile(self, _entity_type, _entity_id):
        return SimpleNamespace(
            email="nick@example.com",
            display_name="Nick",
            description=None,
            color="#012f7b",
            avatar=self.avatar,
            avatar_render=self.avatar_render,
            summary=None,
            summary_updated_at=None,
            created_at=None,
        )

    def get_all_attributes(self, _entity_type, _entity_id):
        return []


def test_update_typed_profile_clears_cached_avatar_render(monkeypatch) -> None:
    manager = _ProfileManager()
    monkeypatch.setattr(profile_routes, "get_service", lambda: SimpleNamespace(config=object()))
    monkeypatch.setattr(profile_routes, "get_profile_manager", lambda _config: manager)
    monkeypatch.setattr("llm_bawt.media.avatar.resolve_avatar_render", lambda _avatar: None)

    result = profile_routes.update_typed_profile(
        "user",
        "nick",
        ProfileUpdateRequest(avatar=""),
    )

    assert manager.update_calls[0]["avatar"] == ""
    assert manager.update_calls[0]["avatar_render"] == ""
    assert result.avatar is None
    assert result.avatar_render is None
