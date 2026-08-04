"""Tool-attributed changed-file persistence and lazy diff storage.

TASK-664. Completed file-modifying tool calls upsert one row per
``(turn_id, repo_key, path)`` immediately, making the file list durable before
the turn finishes. Optional bridge-captured before/after bytes enrich that same
row and are stored content-addressed for the existing lazy diff viewer.

The app never infers attribution from a shared workspace diff. One serializer
feeds incremental ``file_changed`` events, terminal ``turn_complete`` metadata,
and DB-backed history so live and refreshed views converge on the same shape.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    inspect,
    text as sa_text,
)
from sqlmodel import Field, SQLModel, Session, select
from ..utils.schema import SchemaBootstrapGuard

if TYPE_CHECKING:
    from ..media.object_store import BlobBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bounds — keep payloads and storage sane. The bridge is expected to enforce
# the same caps at capture time; these are the app-side backstop so a
# misbehaving/older bridge can't blow up Postgres or the object store.
# ---------------------------------------------------------------------------
MAX_FILES_PER_TURN = 200
MAX_BYTES_PER_SIDE = 1_048_576  # 1 MiB per side; larger is truncated + flagged

_VALID_KINDS = {"added", "modified", "deleted", "renamed"}

DEFAULT_DIFFBLOBS_FS_ROOT = "/var/lib/llm-bawt/turn-diffs"

_DIFF_BLOB_BACKEND: "BlobBackend | None" = None
_DIFF_BLOB_RESOLVED = False


def _get_diff_blob_backend() -> "BlobBackend | None":
    """Return the (cached) blob backend for diff bytes, or None.

    None means "no object store available" — callers fall back to the bounded
    Postgres payload table. Resolution never raises; a misconfigured backend is
    treated as absent so a turn never dies over storage.
    """
    global _DIFF_BLOB_BACKEND, _DIFF_BLOB_RESOLVED
    if _DIFF_BLOB_RESOLVED:
        return _DIFF_BLOB_BACKEND
    _DIFF_BLOB_RESOLVED = True
    try:
        from ..media.object_store import get_blob_backend, s3_config_from_env

        fs_root = Path(
            os.environ.get("LLM_BAWT_DIFFBLOBS_FS_ROOT", DEFAULT_DIFFBLOBS_FS_ROOT)
        )
        _DIFF_BLOB_BACKEND = get_blob_backend(
            "turn_diffs", fs_root=fs_root, s3_cfg=s3_config_from_env()
        )
    except Exception:
        logger.exception(
            "diff blob backend unavailable; diff bytes will use Postgres fallback"
        )
        _DIFF_BLOB_BACKEND = None
    return _DIFF_BLOB_BACKEND


def reset_diff_blob_backend() -> None:
    """Drop the cached backend (tests / config reload)."""
    global _DIFF_BLOB_BACKEND, _DIFF_BLOB_RESOLVED
    _DIFF_BLOB_BACKEND = None
    _DIFF_BLOB_RESOLVED = False


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class TurnChangedFile(SQLModel, table=True):
    __tablename__ = "turn_changed_files"

    id: int | None = Field(
        default=None,
        sa_column=Column(Integer, primary_key=True, autoincrement=True),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )
    turn_id: str = Field(sa_column=Column(String(128), nullable=False, index=True))
    bot_id: str | None = Field(default=None, index=True)
    user_id: str | None = Field(default=None, index=True)
    # The user message this turn answered — lets the chat UI key changed files
    # the same way it keys tool cards (by trigger_message_id) on a refresh.
    trigger_message_id: str | None = Field(default=None, index=True)

    # Stable workspace identity + human display label.
    repo_key: str = Field(sa_column=Column(String(128), nullable=False))
    repo_label: str | None = Field(default=None, sa_column=Column(String(255), nullable=True))

    path: str = Field(sa_column=Column(Text, nullable=False))
    old_path: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    change_kind: str = Field(default="modified", sa_column=Column(String(16), nullable=False))
    additions: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))
    deletions: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))
    binary: bool = Field(default=False, sa_column=Column(Boolean, nullable=False))
    truncated: bool = Field(default=False, sa_column=Column(Boolean, nullable=False))
    # Content-addressed object-store keys (sha256) for each side; NULL when that
    # side is absent (added file has no before; deleted file has no after) or
    # the file is binary (no content stored).
    before_sha256: str | None = Field(default=None, sa_column=Column(String(64), nullable=True))
    before_bytes: int | None = Field(default=None, sa_column=Column(BigInteger, nullable=True))
    before_blob_key: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    after_sha256: str | None = Field(default=None, sa_column=Column(String(64), nullable=True))
    after_bytes: int | None = Field(default=None, sa_column=Column(BigInteger, nullable=True))
    after_blob_key: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    content_type: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))

    # Optional provenance: tool_use ids that produced this change.
    source_tool_call_ids: str | None = Field(default=None, sa_column=Column(Text, nullable=True))


class TurnDiffPayloadRecord(SQLModel, table=True):
    """Bounded Postgres fallback for diff bytes when no object store exists.

    Content-addressed by sha256 (primary key) so identical content across files
    and turns stores one row.
    """

    __tablename__ = "turn_diff_payloads"

    sha256: str = Field(sa_column=Column(String(64), primary_key=True))
    content: str = Field(sa_column=Column(Text, nullable=False))
    total_bytes: int = Field(sa_column=Column(BigInteger, nullable=False))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


# ---------------------------------------------------------------------------
# Input contract (produced by the bridge tracker; consumed here)
# ---------------------------------------------------------------------------


@dataclass
class ChangedFileInput:
    """One changed file for a turn, as reported by the capture tracker.

    ``before``/``after`` are raw bytes (None = that side absent). For text files
    the store persists them content-addressed; binary files carry no content.
    """

    repo_key: str
    path: str
    change_kind: str = "modified"
    repo_label: str | None = None
    old_path: str | None = None
    additions: int | None = None
    deletions: int | None = None
    binary: bool = False
    truncated: bool = False
    content_type: str | None = None
    before: bytes | None = None
    after: bytes | None = None
    source_tool_call_ids: list[str] = field(default_factory=list)


def decode_file_changes(files: list[Any]) -> list[ChangedFileInput]:
    """Decode tool-attributed file events into store inputs.

    Before/after bytes ride as base64 because the bridge and app communicate via
    JSON. Malformed entries are skipped rather than failing the owning turn.
    """
    out: list[ChangedFileInput] = []
    for raw in files or []:
        if not isinstance(raw, dict):
            continue
        repo_key = raw.get("repo_key")
        path = raw.get("path")
        if not repo_key or not path:
            continue
        kind = raw.get("change_kind") or "modified"
        out.append(ChangedFileInput(
            repo_key=str(repo_key),
            path=str(path),
            change_kind=kind if kind in _VALID_KINDS else "modified",
            repo_label=raw.get("repo_label"),
            old_path=raw.get("old_path"),
            additions=raw.get("additions"),
            deletions=raw.get("deletions"),
            binary=bool(raw.get("binary")),
            truncated=bool(raw.get("truncated")),
            content_type=raw.get("content_type"),
            before=_decode_b64(raw.get("before_b64")),
            after=_decode_b64(raw.get("after_b64")),
        ))
    return out


def _decode_b64(value: Any) -> bytes | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        logger.debug("changed-file event carried undecodable base64")
        return None


# ---------------------------------------------------------------------------
# Serializer — the single read model (event + API share this)
# ---------------------------------------------------------------------------


def serialize_changed_file(row: TurnChangedFile) -> dict[str, Any]:
    """Metadata-only shape for one changed file. No bytes; the client fetches
    content lazily by (turn_id, repo_key, path)."""
    has_before = bool(row.before_blob_key or row.before_sha256)
    has_after = bool(row.after_blob_key or row.after_sha256)
    return {
        # Real turn id so the client can lazily fetch this file's content by
        # (turn_id, repo_key, path) regardless of how the summary is keyed.
        "turn_id": row.turn_id,
        "repo_key": row.repo_key,
        "repo_label": row.repo_label,
        "path": row.path,
        "old_path": row.old_path,
        "change_kind": row.change_kind,
        "additions": row.additions,
        "deletions": row.deletions,
        "binary": row.binary,
        "truncated": row.truncated,
        # Whether a text diff is available to open. Binary and content-less
        # rows render as a non-clickable entry.
        "has_content": (not row.binary) and (has_before or has_after),
        "content_type": row.content_type,
    }


def build_turn_summary(turn_id: str, rows: list[TurnChangedFile]) -> dict[str, Any]:
    """Aggregate a turn's changed-file rows into one summary payload.

    Deduplicated already (unique per (turn_id, repo_key, path) in the table).
    Used verbatim as ``turn_complete.changed_files`` AND the read-API body.
    """
    files = [serialize_changed_file(r) for r in rows]
    total_add = sum(f["additions"] or 0 for f in files)
    total_del = sum(f["deletions"] or 0 for f in files)
    return {
        "turn_id": turn_id,
        "files": files,
        "total_files": len(files),
        "total_additions": total_add,
        "total_deletions": total_del,
    }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileContent:
    """Resolved text content for one side of a diff."""

    text: str
    binary: bool
    truncated: bool


class ChangedFilesStore:
    """DB access + blob offload for per-turn changed files."""

    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, engine: Any):
        self.engine = engine

    # -- schema -----------------------------------------------------------

    def ensure_schema(self) -> None:
        if self.engine is None:
            return

        def bootstrap(conn) -> None:
            SQLModel.metadata.create_all(
                bind=conn,
                tables=[TurnChangedFile.__table__, TurnDiffPayloadRecord.__table__],
            )
            legacy_columns = {
                column["name"]
                for column in inspect(conn).get_columns("turn_changed_files")
            }
            for obsolete in ("overlapping", "manifest_truncated"):
                if obsolete in legacy_columns:
                    conn.execute(sa_text(
                        f"ALTER TABLE turn_changed_files DROP COLUMN {obsolete}"
                    ))
            conn.execute(sa_text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_turn_changed_files_turn_repo_path"
                " ON turn_changed_files (turn_id, repo_key, path)"
            ))
            conn.execute(sa_text(
                "CREATE INDEX IF NOT EXISTS ix_turn_changed_files_owner_created"
                " ON turn_changed_files (bot_id, user_id, created_at)"
            ))

        self._schema_guard.run(self.engine, "changed-files-store", bootstrap)

    # -- write ------------------------------------------------------------

    def save_turn_files(
        self,
        *,
        turn_id: str,
        bot_id: str | None,
        user_id: str | None,
        trigger_message_id: str | None,
        files: list[ChangedFileInput],
    ) -> int:
        """Upsert the changed files for a turn. Returns the number persisted.

        Idempotent per (turn_id, repo_key, path): re-persisting a turn (e.g. a
        retry) overwrites the prior rows rather than duplicating. Never raises —
        a storage failure returns the count persisted so far and logs.
        """
        if self.engine is None or not turn_id:
            return 0
        capped = files[:MAX_FILES_PER_TURN]
        if len(files) > MAX_FILES_PER_TURN:
            logger.warning(
                "turn %s reported %d changed files; capping at %d",
                turn_id, len(files), MAX_FILES_PER_TURN,
            )
        saved = 0
        try:
            with Session(self.engine) as session:
                for f in capped:
                    kind = f.change_kind if f.change_kind in _VALID_KINDS else "modified"
                    before_key, before_sha, before_n, trunc_b = self._offload_side(
                        session, f.before, f.binary
                    )
                    after_key, after_sha, after_n, trunc_a = self._offload_side(
                        session, f.after, f.binary
                    )
                    row = session.exec(
                        select(TurnChangedFile)
                        .where(TurnChangedFile.turn_id == turn_id)
                        .where(TurnChangedFile.repo_key == f.repo_key)
                        .where(TurnChangedFile.path == f.path)
                    ).first()
                    is_new = row is None
                    if row is None:
                        row = TurnChangedFile(turn_id=turn_id, repo_key=f.repo_key, path=f.path)
                        session.add(row)
                    row.bot_id = bot_id
                    row.user_id = user_id
                    row.trigger_message_id = trigger_message_id
                    row.repo_label = f.repo_label
                    row.old_path = f.old_path
                    has_detail = bool(
                        f.binary
                        or f.before is not None
                        or f.after is not None
                        or f.additions is not None
                        or f.deletions is not None
                        or f.content_type is not None
                        or kind != "modified"
                    )
                    if is_new or kind in {"added", "deleted"}:
                        row.change_kind = kind
                    elif has_detail and row.change_kind not in {"added", "deleted"}:
                        row.change_kind = "modified"
                    if is_new or f.additions is not None:
                        row.additions = f.additions
                    if is_new or f.deletions is not None:
                        row.deletions = f.deletions
                    if is_new or has_detail:
                        row.binary = bool(f.binary)
                        row.truncated = bool(f.truncated or trunc_b or trunc_a)
                    if is_new or f.content_type is not None:
                        row.content_type = f.content_type
                    # Multiple Edit/Write calls may target the same path in one
                    # turn. Keep the FIRST before-side and the LATEST after-side
                    # so the diff represents the whole turn, not only its last
                    # tool call.
                    if is_new or (row.before_sha256 is None and before_sha is not None):
                        row.before_blob_key = before_key
                        row.before_sha256 = before_sha
                        row.before_bytes = before_n
                    if is_new or after_sha is not None or kind == "deleted":
                        row.after_blob_key = after_key
                        row.after_sha256 = after_sha
                        row.after_bytes = after_n
                    existing_sources: list[str] = []
                    if row.source_tool_call_ids:
                        try:
                            decoded_sources = json.loads(row.source_tool_call_ids)
                            if isinstance(decoded_sources, list):
                                existing_sources = [str(value) for value in decoded_sources]
                        except (TypeError, ValueError):
                            existing_sources = []
                    combined_sources = list(dict.fromkeys([
                        *existing_sources,
                        *(str(value) for value in f.source_tool_call_ids),
                    ]))
                    row.source_tool_call_ids = (
                        json.dumps(combined_sources) if combined_sources else None
                    )
                    session.add(row)
                    saved += 1
                session.commit()
        except Exception:
            logger.exception("Failed to persist changed files for turn=%s", turn_id)
        return saved

    def _offload_side(
        self, session: Session, data: bytes | None, binary: bool
    ) -> tuple[str | None, str | None, int | None, bool]:
        """Persist one side's bytes content-addressed. Returns
        (blob_key, sha256, total_bytes, truncated). Binary/None => no content."""
        if data is None or binary:
            return None, None, None, False
        truncated = False
        if len(data) > MAX_BYTES_PER_SIDE:
            data = data[:MAX_BYTES_PER_SIDE]
            truncated = True
        sha = hashlib.sha256(data).hexdigest()
        total = len(data)
        backend = _get_diff_blob_backend()
        if backend is not None:
            try:
                if not backend.exists(sha):
                    backend.put(sha, data, "text/plain; charset=utf-8")
                return sha, sha, total, truncated
            except Exception:
                logger.exception("diff blob offload failed sha=%s; using Postgres", sha)
        # Fallback: bounded Postgres payload (content-addressed, dedup by sha).
        try:
            existing = session.get(TurnDiffPayloadRecord, sha)
            if existing is None:
                session.add(TurnDiffPayloadRecord(
                    sha256=sha,
                    content=data.decode("utf-8", errors="replace"),
                    total_bytes=total,
                ))
                session.flush()
        except Exception:
            logger.exception("diff Postgres fallback failed sha=%s", sha)
            return None, sha, total, truncated
        # blob_key None signals "read from Postgres fallback by sha256".
        return None, sha, total, truncated

    # -- read -------------------------------------------------------------

    def recent_rows(
        self, *, user_id: str, since_hours: int, limit: int = 2000
    ) -> tuple[list[TurnChangedFile], bool]:
        """Return bounded recent provenance rows for one user.

        The extra row detects truncation without a separate count query. Callers
        still resolve diff bytes through this store so object/Postgres backends
        remain an implementation detail.
        """
        if self.engine is None:
            raise RuntimeError("changed-file provenance DB unavailable")
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, since_hours))
        with Session(self.engine) as session:
            rows = list(session.exec(
                select(TurnChangedFile)
                .where(TurnChangedFile.user_id == user_id)
                .where(TurnChangedFile.created_at >= cutoff)
                .order_by(TurnChangedFile.created_at.desc(), TurnChangedFile.id.desc())
                .limit(limit + 1)
            ).all())
        return rows[:limit], len(rows) > limit

    def summaries_for_turns(self, turn_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return {turn_id: summary} for the given turns (empty if none)."""
        if self.engine is None or not turn_ids:
            return {}
        out: dict[str, list[TurnChangedFile]] = {}
        try:
            with Session(self.engine) as session:
                rows = session.exec(
                    select(TurnChangedFile)
                    .where(TurnChangedFile.turn_id.in_(turn_ids))  # type: ignore[attr-defined]
                    .order_by(TurnChangedFile.repo_key, TurnChangedFile.path)
                ).all()
            for r in rows:
                out.setdefault(r.turn_id, []).append(r)
        except Exception:
            logger.exception("changed-file summary query failed turns=%s", turn_ids)
            return {}
        return {tid: build_turn_summary(tid, rws) for tid, rws in out.items()}

    def summaries_for_triggers(self, message_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return {trigger_message_id: summary} keyed by user message id."""
        if self.engine is None or not message_ids:
            return {}
        out: dict[str, list[TurnChangedFile]] = {}
        try:
            with Session(self.engine) as session:
                rows = session.exec(
                    select(TurnChangedFile)
                    .where(TurnChangedFile.trigger_message_id.in_(message_ids))  # type: ignore[attr-defined]
                    .order_by(TurnChangedFile.repo_key, TurnChangedFile.path)
                ).all()
            for r in rows:
                if r.trigger_message_id:
                    out.setdefault(r.trigger_message_id, []).append(r)
        except Exception:
            logger.exception("changed-file trigger query failed msgs=%s", message_ids)
            return {}
        # One trigger message maps to one turn, so use that turn's real id for
        # the summary; per-file entries carry it too.
        return {
            mid: build_turn_summary(rws[0].turn_id, rws)
            for mid, rws in out.items()
        }

    def summary_for_turn(self, turn_id: str) -> dict[str, Any]:
        return self.summaries_for_turns([turn_id]).get(
            turn_id, build_turn_summary(turn_id, [])
        )

    def content_for_rows(
        self, rows: list[TurnChangedFile]
    ) -> dict[int, tuple[FileContent, FileContent]]:
        """Resolve content for already-loaded rows in one database session."""
        if self.engine is None or not rows:
            return {}
        output: dict[int, tuple[FileContent, FileContent]] = {}
        with Session(self.engine) as session:
            for row in rows:
                if row.id is None:
                    continue
                output[row.id] = (
                    self._resolve_side(
                        session, row.before_blob_key, row.before_sha256, row.binary, row.truncated
                    ),
                    self._resolve_side(
                        session, row.after_blob_key, row.after_sha256, row.binary, row.truncated
                    ),
                )
        return output

    def get_file_content(
        self, *, turn_id: str, repo_key: str, path: str
    ) -> tuple[FileContent, FileContent] | None:
        """Resolve (before, after) text content for one changed file.

        Returns None if the row doesn't exist. Binary files return empty text
        with binary=True on both sides.
        """
        if self.engine is None:
            return None
        try:
            with Session(self.engine) as session:
                row = session.exec(
                    select(TurnChangedFile)
                    .where(TurnChangedFile.turn_id == turn_id)
                    .where(TurnChangedFile.repo_key == repo_key)
                    .where(TurnChangedFile.path == path)
                ).first()
                if row is None:
                    return None
                before = self._resolve_side(
                    session, row.before_blob_key, row.before_sha256, row.binary, row.truncated
                )
                after = self._resolve_side(
                    session, row.after_blob_key, row.after_sha256, row.binary, row.truncated
                )
                return before, after
        except Exception:
            logger.exception(
                "changed-file content read failed turn=%s path=%s", turn_id, path
            )
            return None

    def _resolve_side(
        self,
        session: Session,
        blob_key: str | None,
        sha: str | None,
        binary: bool,
        truncated: bool,
    ) -> FileContent:
        if binary:
            return FileContent(text="", binary=True, truncated=truncated)
        if sha is None:
            # Side absent (added/deleted) — empty content is correct.
            return FileContent(text="", binary=False, truncated=truncated)
        if blob_key is not None:
            backend = _get_diff_blob_backend()
            if backend is not None:
                try:
                    data = backend.get(blob_key)
                    if data is not None:
                        return FileContent(
                            text=data.decode("utf-8", errors="replace"),
                            binary=False,
                            truncated=truncated,
                        )
                except Exception:
                    logger.exception("diff blob read failed key=%s", blob_key)
        # Postgres fallback by sha.
        rec = session.get(TurnDiffPayloadRecord, sha)
        if rec is not None:
            return FileContent(text=rec.content, binary=False, truncated=truncated)
        return FileContent(text="", binary=False, truncated=truncated)
