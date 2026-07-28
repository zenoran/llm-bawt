"""Provider-neutral per-turn changed-file capture (TASK-661, slice 3).

The app container does **not** mount the agent workspace — only the bridges do
— so canonical capture has to live here, on the bridge side, and ship a
manifest to the app over the existing Redis event transport.

How a turn is captured
----------------------

At turn start we snapshot every configured git workspace, and at turn end we
snapshot again and diff the two snapshots. A snapshot is a real git tree object
written through a **temporary index**::

    cp .git/index $TMP                      # keep the stat cache -> fast add
    GIT_INDEX_FILE=$TMP git add -A          # honors .gitignore
    GIT_INDEX_FILE=$TMP git write-tree      # -> tree sha

``GIT_INDEX_FILE`` means the repo's real index is never touched, so a snapshot
cannot stage, unstage, or otherwise disturb whatever Nick (or a sibling bot)
has in flight. ``git add -A`` respects ``.gitignore``, which is what keeps
``node_modules``/``.venv``/build output out of the manifest for free.

Diffing two trees with ``git diff-tree -r -M`` gives rename detection and the
exact before/after blob shas, so the captured bytes are the real content for
this turn rather than a re-read of a file that may have moved on.

Concurrency
-----------

Several turns can share one bind-mounted worktree, and git trees alone cannot
attribute a change to a specific turn. Serializing turns per workspace would
stall parallel agents to fix a *reporting* problem, so instead this module
reports the union and sets ``overlapping`` on any repo that had more than one
tracker live during the window. The diff is still correct about what changed;
the flag is what stops it from lying about *who* changed it.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Bounds. The app-side store (`changed_files_store`) enforces the same caps as
# a backstop; these keep the Redis event itself from becoming a payload bomb.
MAX_FILES_PER_TURN = 200
MAX_BYTES_PER_SIDE = 1_048_576  # 1 MiB, matches changed_files_store
MAX_TOTAL_CONTENT_BYTES = 4 * 1_048_576  # whole-manifest content budget

# A tree sha of all zeros means "this side does not exist" in diff-tree output.
_NULL_SHA = "0" * 40

_STATUS_TO_KIND = {
    "A": "added",
    "M": "modified",
    "D": "deleted",
    "R": "renamed",
    "C": "added",
    "T": "modified",
}

DEFAULT_WORKSPACE_ROOT = "~/dev"
_GIT_TIMEOUT = 60.0


# ---------------------------------------------------------------------------
# Overlap registry — see the module docstring's "Concurrency" note.
# ---------------------------------------------------------------------------

_ACTIVE: dict[str, set[str]] = {}
_ACTIVE_LOCK = asyncio.Lock()

# Live trackers, so a newly-started turn can retro-flag the ones already
# running on the same workspace (see _mark_peers_overlapping).
_LIVE_TRACKERS: dict[str, "ChangedFileTracker"] = {}


async def _register_active(repo_path: str, tracker_id: str) -> set[str]:
    """Mark a tracker live on a workspace; return the ids it now overlaps."""
    async with _ACTIVE_LOCK:
        live = _ACTIVE.setdefault(repo_path, set())
        others = set(live)
        live.add(tracker_id)
        return others


async def _unregister_active(repo_path: str, tracker_id: str) -> None:
    async with _ACTIVE_LOCK:
        live = _ACTIVE.get(repo_path)
        if not live:
            return
        live.discard(tracker_id)
        if not live:
            _ACTIVE.pop(repo_path, None)


# ---------------------------------------------------------------------------
# Wire shape
# ---------------------------------------------------------------------------


@dataclass
class ChangedFile:
    """One changed file, ready for the Redis manifest.

    Field names match ``changed_files_store.ChangedFileInput`` so the app-side
    consumer is a straight mapping — content rides as base64 because the
    transport is JSON and the bytes may be any encoding.
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
    before_b64: str | None = None
    after_b64: str | None = None

    def to_dict(self) -> dict:
        return {
            "repo_key": self.repo_key,
            "path": self.path,
            "change_kind": self.change_kind,
            "repo_label": self.repo_label,
            "old_path": self.old_path,
            "additions": self.additions,
            "deletions": self.deletions,
            "binary": self.binary,
            "truncated": self.truncated,
            "content_type": self.content_type,
            "before_b64": self.before_b64,
            "after_b64": self.after_b64,
        }


@dataclass
class _Workspace:
    """A git workspace being tracked across one turn."""

    key: str
    label: str
    path: Path
    tree_before: str | None = None
    overlapping: bool = False
    error: str | None = None


@dataclass
class _RawEntry:
    """One parsed ``git diff-tree --raw`` record."""

    status: str
    before_sha: str
    after_sha: str
    path: str
    old_path: str | None = None
    additions: int | None = None
    deletions: int | None = None
    binary: bool = False


# ---------------------------------------------------------------------------
# git plumbing
# ---------------------------------------------------------------------------


async def _git(
    repo: Path, *args: str, env_extra: dict[str, str] | None = None
) -> tuple[int, bytes, bytes]:
    """Run one git command in ``repo``. Never raises on non-zero exit."""
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(repo), *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=_GIT_TIMEOUT)
    except asyncio.TimeoutError:
        with_kill = getattr(proc, "kill", None)
        if with_kill:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        raise
    return proc.returncode or 0, out, err


async def _write_tree(repo: Path) -> str | None:
    """Snapshot the full working tree (tracked + untracked, gitignore-honoring)
    as a git tree object, WITHOUT touching the repo's real index."""
    # Cheap liveness probe first. `~/dev` collects stale linked worktrees whose
    # admin dir is gone (or points at a host path that doesn't exist inside the
    # container); those are a permanent condition, so they get a debug line
    # rather than a WARNING on every single turn.
    rc, _, err = await _git(repo, "rev-parse", "--git-dir")
    if rc != 0:
        logger.debug(
            "skipping unusable workspace %s: %s",
            repo, err.decode(errors="replace").strip()[:200],
        )
        return None

    tmpdir = tempfile.mkdtemp(prefix="llmbawt-idx-")
    index_path = os.path.join(tmpdir, "index")
    try:
        # Seed from the real index when present: it carries the stat cache, so
        # `add -A` only re-hashes files that actually changed. Falling back to
        # an empty index is correct, just slower on the first snapshot.
        real_index = repo / ".git" / "index"
        if real_index.is_file():
            try:
                shutil.copyfile(real_index, index_path)
            except OSError:
                logger.debug("index copy failed for %s; using empty index", repo)

        env = {"GIT_INDEX_FILE": index_path}
        rc, _, err = await _git(repo, "add", "-A", env_extra=env)
        if rc != 0:
            logger.warning("git add -A failed in %s: %s", repo, err.decode(errors="replace")[:300])
            return None
        rc, out, err = await _git(repo, "write-tree", env_extra=env)
        if rc != 0:
            logger.warning("git write-tree failed in %s: %s", repo, err.decode(errors="replace")[:300])
            return None
        return out.decode().strip() or None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _parse_raw_z(payload: bytes) -> list[_RawEntry]:
    """Parse ``git diff-tree -r -M -z --raw`` output.

    Each record is ``:<m1> <m2> <sha1> <sha2> <status>\\0<path>\\0`` — and for
    rename/copy, ``\\0<src>\\0<dst>\\0``.
    """
    tokens = payload.split(b"\0")
    entries: list[_RawEntry] = []
    i = 0
    while i < len(tokens):
        head = tokens[i]
        i += 1
        if not head.startswith(b":"):
            continue
        try:
            fields = head[1:].decode("utf-8", errors="replace").split()
            _mode_before, _mode_after, sha_before, sha_after, status = fields[:5]
        except (ValueError, IndexError):
            continue
        letter = status[0].upper()
        if letter in ("R", "C"):
            if i + 1 >= len(tokens):
                break
            old_path = tokens[i].decode("utf-8", errors="replace")
            path = tokens[i + 1].decode("utf-8", errors="replace")
            i += 2
        else:
            if i >= len(tokens):
                break
            path = tokens[i].decode("utf-8", errors="replace")
            old_path = None
            i += 1
        entries.append(_RawEntry(
            status=letter,
            before_sha=sha_before,
            after_sha=sha_after,
            path=path,
            old_path=old_path,
        ))
    return entries


def _parse_numstat_z(payload: bytes) -> dict[str, tuple[int | None, int | None, bool]]:
    """Parse ``git diff-tree -r -M -z --numstat`` into {path: (add, del, binary)}.

    Records are ``adds\\tdels\\tpath\\0``; renames drop the path from the first
    token and follow it with ``src\\0dst\\0``. Binary files report ``-`` counts.
    """
    tokens = payload.split(b"\0")
    out: dict[str, tuple[int | None, int | None, bool]] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i].decode("utf-8", errors="replace")
        i += 1
        if "\t" not in token:
            continue
        parts = token.split("\t")
        if len(parts) < 3:
            continue
        adds_s, dels_s, rest = parts[0], parts[1], parts[2]
        if rest == "":
            # Rename/copy: the next two tokens are src and dst; key on dst.
            if i + 1 >= len(tokens):
                break
            path = tokens[i + 1].decode("utf-8", errors="replace")
            i += 2
        else:
            path = rest
        binary = adds_s == "-" or dels_s == "-"
        adds = None if binary else _safe_int(adds_s)
        dels = None if binary else _safe_int(dels_s)
        out[path] = (adds, dels, binary)
    return out


def _safe_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class ChangedFileTracker:
    """Snapshots git workspaces around one turn and reports what changed.

    Every public method is failure-tolerant on purpose: a broken workspace, a
    missing git binary, or a timeout degrades to "no changed files reported"
    and never propagates into the turn. Losing the file list is a cosmetic
    regression; killing the turn over it is not.
    """

    def __init__(
        self,
        roots: list[Path] | None = None,
        *,
        max_files: int = MAX_FILES_PER_TURN,
        max_bytes_per_side: int = MAX_BYTES_PER_SIDE,
        max_total_content: int = MAX_TOTAL_CONTENT_BYTES,
    ):
        self._roots = roots if roots is not None else self._default_roots()
        self._max_files = max_files
        self._max_bytes_per_side = max_bytes_per_side
        self._max_total_content = max_total_content
        self._workspaces: list[_Workspace] = []
        self._id = uuid.uuid4().hex
        self._started = False
        self._finalized = False

    # -- discovery --------------------------------------------------------

    @staticmethod
    def _default_roots() -> list[Path]:
        raw = os.environ.get("AGENT_BRIDGE_CHANGED_FILES_ROOTS", DEFAULT_WORKSPACE_ROOT)
        roots: list[Path] = []
        for chunk in raw.split(":"):
            chunk = chunk.strip()
            if not chunk:
                continue
            roots.append(Path(chunk).expanduser())
        return roots

    @classmethod
    def disabled(cls) -> bool:
        flag = os.environ.get("AGENT_BRIDGE_CHANGED_FILES_DISABLED", "")
        return flag.strip().lower() in ("1", "true", "yes")

    def _discover(self) -> list[_Workspace]:
        """Find git workspaces one level under each configured root.

        Human labels remain the directory basename. ``key`` is upgraded with a
        short canonical-path hash only when two configured workspaces share that
        basename, preventing ``(turn_id, repo_key, path)`` persistence collisions
        without changing the common single-``~/dev`` URL shape.
        """
        discovered: list[tuple[str, Path]] = []
        seen: set[str] = set()
        for root in self._roots:
            try:
                if not root.is_dir():
                    continue
                # A root that is itself a repo counts, otherwise scan children.
                candidates = [root] if (root / ".git").exists() else sorted(
                    p for p in root.iterdir() if p.is_dir()
                )
                for candidate in candidates:
                    if not (candidate / ".git").exists():
                        continue
                    resolved = str(candidate.resolve())
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    discovered.append((resolved, candidate))
            except OSError:
                logger.debug("workspace scan failed for root=%s", root, exc_info=True)

        name_counts: dict[str, int] = {}
        for _resolved, candidate in discovered:
            name_counts[candidate.name] = name_counts.get(candidate.name, 0) + 1

        found: list[_Workspace] = []
        for resolved, candidate in discovered:
            key = candidate.name
            if name_counts[key] > 1:
                suffix = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
                key = f"{key}-{suffix}"
            found.append(_Workspace(key=key, label=candidate.name, path=candidate))
        return found

    # -- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Snapshot every workspace. Safe to call once per turn."""
        if self._started or self.disabled():
            return
        self._started = True
        _LIVE_TRACKERS[self._id] = self
        try:
            self._workspaces = self._discover()
        except Exception:
            logger.debug("changed-file discovery failed", exc_info=True)
            self._workspaces = []
            return

        async def snapshot(ws: _Workspace) -> None:
            try:
                others = await _register_active(str(ws.path.resolve()), self._id)
                if others:
                    ws.overlapping = True
                    await self._mark_peers_overlapping(ws)
                ws.tree_before = await _write_tree(ws.path)
            except Exception as exc:
                ws.error = str(exc)
                logger.debug("snapshot failed for %s", ws.path, exc_info=True)

        await asyncio.gather(*(snapshot(ws) for ws in self._workspaces))
        live = [w for w in self._workspaces if w.tree_before]
        logger.info(
            "Changed-file tracker armed: %d/%d workspaces (tracker=%s)",
            len(live), len(self._workspaces), self._id[:8],
        )

    async def _mark_peers_overlapping(self, ws: _Workspace) -> None:
        """Tell already-live trackers on this workspace that they now overlap."""
        for tracker in list(_LIVE_TRACKERS.values()):
            if tracker is self:
                continue
            for other in tracker._workspaces:
                if other.path == ws.path:
                    other.overlapping = True

    async def finalize(self) -> dict | None:
        """Diff each workspace and return the manifest, or None if empty.

        Shape::

            {"files": [ChangedFile.to_dict(), ...],
             "overlapping_repos": ["llm-bawt", ...],
             "truncated": bool}
        """
        # Idempotent: a turn can reach its finalize point more than once (auth
        # retry, fresh-session retry, cancel-then-teardown). Only the first
        # call diffs and reports; the rest are no-ops.
        if not self._started or self._finalized:
            return None
        self._finalized = True
        files: list[ChangedFile] = []
        overlapping: list[str] = []
        budget = _ContentBudget(self._max_total_content)
        try:
            for ws in self._workspaces:
                try:
                    if ws.overlapping:
                        overlapping.append(ws.key)
                    if not ws.tree_before:
                        continue
                    files.extend(await self._diff_workspace(ws, budget))
                except Exception:
                    logger.debug("diff failed for %s", ws.path, exc_info=True)
        finally:
            for ws in self._workspaces:
                await _unregister_active(str(ws.path.resolve()), self._id)
            _LIVE_TRACKERS.pop(self._id, None)

        capped = len(files) > self._max_files
        if capped:
            logger.warning(
                "Turn changed %d files; capping manifest at %d",
                len(files), self._max_files,
            )
            files = files[: self._max_files]
        if not files:
            return None
        return {
            "files": [f.to_dict() for f in files],
            "overlapping_repos": overlapping,
            "truncated": capped or budget.exhausted,
        }

    # -- diff -------------------------------------------------------------

    async def _diff_workspace(
        self, ws: _Workspace, budget: "_ContentBudget"
    ) -> list[ChangedFile]:
        tree_after = await _write_tree(ws.path)
        if not tree_after or tree_after == ws.tree_before:
            return []
        rc, raw_out, _ = await _git(
            ws.path, "diff-tree", "-r", "-M", "-z", "--raw",
            ws.tree_before or "", tree_after,
        )
        if rc != 0:
            return []
        entries = _parse_raw_z(raw_out)
        if not entries:
            return []
        rc, num_out, _ = await _git(
            ws.path, "diff-tree", "-r", "-M", "-z", "--numstat",
            ws.tree_before or "", tree_after,
        )
        numstat = _parse_numstat_z(num_out) if rc == 0 else {}

        out: list[ChangedFile] = []
        for entry in entries:
            adds, dels, binary = numstat.get(entry.path, (None, None, False))
            before = await self._blob(ws.path, entry.before_sha, binary, budget)
            after = await self._blob(ws.path, entry.after_sha, binary, budget)
            out.append(ChangedFile(
                repo_key=ws.key,
                repo_label=ws.label,
                path=entry.path,
                old_path=entry.old_path,
                change_kind=_STATUS_TO_KIND.get(entry.status, "modified"),
                additions=adds,
                deletions=dels,
                binary=binary,
                truncated=(before.truncated or after.truncated),
                before_b64=before.b64,
                after_b64=after.b64,
            ))
        return out

    async def _blob(
        self, repo: Path, sha: str, binary: bool, budget: "_ContentBudget"
    ) -> "_BlobResult":
        """Read one side's bytes, honoring the per-side and total budgets."""
        if binary or not sha or sha.strip("0") == "":
            return _BlobResult(None, False)
        rc, data, _ = await _git(repo, "cat-file", "blob", sha)
        if rc != 0:
            return _BlobResult(None, False)
        truncated = False
        if len(data) > self._max_bytes_per_side:
            data = data[: self._max_bytes_per_side]
            truncated = True
        if b"\0" in data[:8192]:
            # NUL in the head means git's numstat missed a binary file.
            return _BlobResult(None, False)
        if not budget.take(len(data)):
            return _BlobResult(None, True)
        return _BlobResult(base64.b64encode(data).decode("ascii"), truncated)


@dataclass
class _BlobResult:
    b64: str | None
    truncated: bool


@dataclass
class _ContentBudget:
    """Whole-manifest content allowance. Metadata is always kept; only bytes
    are dropped once the budget runs out."""

    remaining: int
    exhausted: bool = field(default=False)

    def take(self, size: int) -> bool:
        if size > self.remaining:
            self.exhausted = True
            return False
        self.remaining -= size
        return True

