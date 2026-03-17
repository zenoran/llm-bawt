"""Avatar animation tool injection for tts_mode requests.

When tts_mode=True, a virtual `trigger_animation` tool is injected into the
LLM request so the model picks the right gesture for its response. The tool
call is intercepted and never executed — only the chosen name is captured.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import Column, DateTime, Integer, String, Text, Boolean
from sqlmodel import Field, Session, SQLModel, create_engine, select

from ..utils.config import Config, has_database_credentials
from ..utils.db import set_utc_on_connect

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQLModel table
# ---------------------------------------------------------------------------

class AvatarAnimation(SQLModel, table=True):
    """DB-backed avatar animation definition."""

    __tablename__ = "avatar_animations"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(sa_column=Column(String(255), nullable=False, unique=True))
    description: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    enabled: bool = Field(
        default=True,
        sa_column=Column(Boolean, nullable=False, server_default="true"),
    )
    sort_order: int = Field(
        default=0,
        sa_column=Column(Integer, nullable=False, server_default="0"),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


# ---------------------------------------------------------------------------
# Default seed data
# ---------------------------------------------------------------------------

_DEFAULT_ANIMATIONS: list[dict[str, Any]] = [
    {"name": "Head Nod Yes",        "description": "Use when agreeing, confirming, or saying yes",              "sort_order": 1},
    {"name": "Hard Head Nod",       "description": "Use when strongly agreeing or emphasizing a point",         "sort_order": 2},
    {"name": "Lengthy Head Nod",    "description": "Use when patiently listening or showing extended agreement", "sort_order": 3},
    {"name": "Acknowledging",       "description": "Use when showing you understand or are listening",           "sort_order": 4},
    {"name": "Shaking Head No",     "description": "Use when disagreeing, correcting, or saying no",            "sort_order": 5},
    {"name": "Thoughtful Head Shake","description": "Use when uncertain, hedging, or saying 'it depends'",      "sort_order": 6},
    {"name": "Happy Hand Gesture",  "description": "Use when excited, enthusiastic, or sharing good news",      "sort_order": 7},
    {"name": "Relieved Sigh",       "description": "Use when something is resolved, fixed, or a relief",        "sort_order": 8},
    {"name": "Being Cocky",         "description": "Use when confident or self-assured",                        "sort_order": 9},
    {"name": "Dismissing Gesture",  "description": "Use when brushing off something unimportant",               "sort_order": 10},
    {"name": "Sarcastic Head Nod",  "description": "Use when being sarcastic or ironic",                        "sort_order": 11},
    {"name": "Annoyed Head Shake",  "description": "Use when mildly frustrated or exasperated",                 "sort_order": 12},
    {"name": "Angry Gesture",       "description": "Use when strongly objecting or expressing frustration",     "sort_order": 13},
    {"name": "Look Away Gesture",   "description": "Use when thinking, hesitating, or considering something",   "sort_order": 14},
    {"name": "Weight Shift",        "description": "Use as a neutral default when no other animation fits",     "sort_order": 15},
]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class AvatarAnimationStore:
    """DB access for avatar animation definitions."""

    def __init__(self, config: Config):
        self.config = config
        self.engine = None
        if not has_database_credentials(config):
            return

        try:
            host = getattr(config, "POSTGRES_HOST", "localhost")
            port = int(getattr(config, "POSTGRES_PORT", 5432))
            user = getattr(config, "POSTGRES_USER", "llm_bawt")
            password = getattr(config, "POSTGRES_PASSWORD", "")
            database = getattr(config, "POSTGRES_DATABASE", "llm_bawt")
            encoded_password = quote_plus(password)
            connection_url = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{database}"
            self.engine = create_engine(connection_url, echo=False)
            set_utc_on_connect(self.engine)
            self._ensure_tables_exist()
        except Exception as e:
            self.engine = None
            logger.warning("Avatar animations DB unavailable: %s", e)

    def _ensure_tables_exist(self) -> None:
        if self.engine is None:
            return
        SQLModel.metadata.create_all(self.engine, tables=[AvatarAnimation.__table__])
        # Add sort_order column if missing (upgrade from old schema that had weight)
        from sqlalchemy import text as sa_text
        with self.engine.connect() as conn:
            try:
                conn.execute(sa_text(
                    "ALTER TABLE avatar_animations ADD COLUMN IF NOT EXISTS sort_order INTEGER NOT NULL DEFAULT 0"
                ))
                conn.commit()
            except Exception:
                pass
        self._seed_defaults_if_empty()

    def _seed_defaults_if_empty(self) -> None:
        if self.engine is None:
            return
        with Session(self.engine) as session:
            if session.exec(select(AvatarAnimation)).first() is not None:
                return
            now = datetime.now(timezone.utc)
            for entry in _DEFAULT_ANIMATIONS:
                session.add(AvatarAnimation(
                    name=entry["name"],
                    description=entry["description"],
                    sort_order=entry["sort_order"],
                    enabled=True,
                    created_at=now,
                    updated_at=now,
                ))
            session.commit()
            logger.info("Seeded %d default avatar animations", len(_DEFAULT_ANIMATIONS))

    def list_all(self) -> list[AvatarAnimation]:
        if self.engine is None:
            return []
        with Session(self.engine) as session:
            return list(session.exec(
                select(AvatarAnimation).order_by(AvatarAnimation.sort_order, AvatarAnimation.id)
            ).all())

    def list_enabled(self) -> list[AvatarAnimation]:
        if self.engine is None:
            return []
        with Session(self.engine) as session:
            return list(session.exec(
                select(AvatarAnimation)
                .where(AvatarAnimation.enabled == True)  # noqa: E712
                .order_by(AvatarAnimation.sort_order, AvatarAnimation.id)
            ).all())

    def get(self, animation_id: int) -> AvatarAnimation | None:
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            return session.exec(
                select(AvatarAnimation).where(AvatarAnimation.id == animation_id)
            ).first()

    def create(self, data: dict[str, Any]) -> AvatarAnimation:
        if self.engine is None:
            raise RuntimeError("Avatar animations DB unavailable")
        now = datetime.now(timezone.utc)
        row = AvatarAnimation(
            name=str(data["name"]),
            description=data.get("description"),
            enabled=bool(data.get("enabled", True)),
            sort_order=int(data.get("sort_order", 0)),
            created_at=now,
            updated_at=now,
        )
        with Session(self.engine) as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def update(self, animation_id: int, data: dict[str, Any]) -> AvatarAnimation | None:
        if self.engine is None:
            raise RuntimeError("Avatar animations DB unavailable")
        now = datetime.now(timezone.utc)
        with Session(self.engine) as session:
            row = session.exec(
                select(AvatarAnimation).where(AvatarAnimation.id == animation_id)
            ).first()
            if row is None:
                return None
            if "name" in data and data["name"] is not None:
                row.name = str(data["name"])
            if "description" in data:
                row.description = data["description"]
            if "enabled" in data and data["enabled"] is not None:
                row.enabled = bool(data["enabled"])
            if "sort_order" in data and data["sort_order"] is not None:
                row.sort_order = int(data["sort_order"])
            row.updated_at = now
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def delete(self, animation_id: int) -> bool:
        if self.engine is None:
            raise RuntimeError("Avatar animations DB unavailable")
        with Session(self.engine) as session:
            row = session.exec(
                select(AvatarAnimation).where(AvatarAnimation.id == animation_id)
            ).first()
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True


# ---------------------------------------------------------------------------
# Tool builder
# ---------------------------------------------------------------------------

def build_trigger_animation_tool(animations: list[AvatarAnimation]) -> dict:
    """Build the trigger_animation tool definition from enabled animations."""
    animation_names = [a.name for a in animations]
    descriptions = "\n".join(
        f'- "{a.name}": {a.description or a.name}' for a in animations
    )
    return {
        "type": "function",
        "function": {
            "name": "trigger_animation",
            "description": (
                "Choose an animation for the avatar. "
                "Write your COMPLETE text response first, then call this function once at the end. "
                "Pick the animation that best fits the emotional tone of your response. "
                "Use 'Weight Shift' as the neutral default if nothing else fits.\n\n"
                f"Available animations:\n{descriptions}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": animation_names,
                        "description": "The animation to play",
                    }
                },
                "required": ["name"],
            },
        },
    }
