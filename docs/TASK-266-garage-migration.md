# TASK-266 — Garage S3 migration runbook (llm-bawt side)

Status: **Code shipped, cutover not yet performed.**

This doc covers the operational steps to move existing on-disk blobs into
Garage and flip llm-bawt over to the S3 backend. The bawthub side has a
parallel runbook — see `bawthub/docs/TASK-266-garage-migration.md`.

## Architecture recap

Two on-disk stores in llm-bawt share one storage abstraction (`BlobBackend`,
in `src/llm_bawt/media/object_store.py`):

| Store | Subsystem | FS root | S3 key prefix |
|---|---|---|---|
| `MediaStore` | chat upload variants | `LLM_BAWT_MEDIA_ROOT` (`/var/lib/llm-bawt/media/blobs`) | `blobs/` |
| `MediaStorage` | generated video / images | `MEDIA_STORAGE_PATH` (`/app/storage/media`) | `media/` |

Both write into one bucket per app: **`llmbawt-media`** for this repo.
Object keys are forward-slash relative paths identical to today's on-disk
layout — see the TASK-266 spec for the full namespace table.

Backend selection is env-driven:

```
LLM_BAWT_STORAGE_BACKEND=fs       # default — local filesystem
LLM_BAWT_STORAGE_BACKEND=s3       # Garage / any S3-compatible
LLM_BAWT_S3_FALLBACK_FS=true      # during cutover: read-through FS on S3 miss
```

The boto3 client is built lazily — the app boots cleanly even when Garage
is down. Failing operations surface as `BlobBackendUnavailable` and route
handlers map them to **HTTP 503 JSON**, never a stack trace.

## Phase 0 — Garage already provisioned (done 2026-06-13)

These are pre-conditions; they don't need to be repeated for cutover.

- Container `garage-s3-simple` running on Unraid at `10.0.2.38:9000`
- Bucket `llmbawt-media` exists with key `llmbawt-app` (RWO)
- Access key in Vaultwarden as `garage/llmbawt-app-s3-key`
  - `bw-secret garage/llmbawt-app-s3-key --user` → access key ID
  - `bw-secret garage/llmbawt-app-s3-key --pass` → secret key
- `zpool scrub cache` clean (3 stale tombstones cleared, no fresh errors)

If you ever need to re-create Garage from scratch, the spec doc on TASK-266
has the full provisioning recipe.

## Pre-cutover checklist

- [ ] `curl -sS -o /dev/null -w "%{http_code}\n" http://10.0.2.38:9000/` returns
      `403` (proves S3 API up; anonymous reads denied).
- [ ] `LLM_BAWT_S3_ACCESS_KEY` and `LLM_BAWT_S3_SECRET_KEY` are populated in
      `~/dev/llm-bawt/.env` on echo (the env block was appended during
      Phase 0; values are stored).
- [ ] Existing FS blobs are backed up — they're the migration source.
      The current NFS mount targets are `/mnt/user/home/echo/llmbawt-media/`
      (MediaStore) and `~/dev/llm-bawt/storage/media/` on echo
      (MediaStorage). Confirm both still exist and have non-zero bytes.

## Migration runbook

**rclone remote config** (write once to `~/.config/rclone/rclone.conf` on
whichever host runs the copy):

```ini
[garage]
type = s3
provider = Other
access_key_id = <from bw-secret garage/llmbawt-app-s3-key --user>
secret_access_key = <from bw-secret garage/llmbawt-app-s3-key --pass>
endpoint = http://10.0.2.38:9000
region = garage
force_path_style = true
```

### Copy steps

```bash
# MediaStore blobs (chat upload variants) — run from Unraid where the
# NFS share is just a local mount.
rclone copy /mnt/user/home/echo/llmbawt-media/ garage:llmbawt-media/blobs/

# MediaStorage outputs (generated video / images) — these live on echo,
# not on Unraid. Two options:
#   a) rclone from echo (set up rclone on echo first), OR
#   b) rsync to Unraid first, then rclone copy
# Option (a):
rclone copy ~/dev/llm-bawt/storage/media/ garage:llmbawt-media/media/
```

### Verify

```bash
rclone check /mnt/user/home/echo/llmbawt-media/   garage:llmbawt-media/blobs/
rclone check ~/dev/llm-bawt/storage/media/        garage:llmbawt-media/media/
# Expect both: "X differences found: 0" — sha-checksummed sample of
# every object in both directions.
```

Total bytes are ~18 MB combined as of the audit; copy + check finishes in
seconds. Save the check output in the task response when you mark the
step COMPLETED.

## Cutover

### 1. Flip to S3 with fallback armed

```bash
# In ~/dev/llm-bawt/.env on echo
LLM_BAWT_STORAGE_BACKEND=s3
LLM_BAWT_S3_FALLBACK_FS=true   # safety net during the cutover window
```

```bash
cd ~/dev/llm-bawt
docker compose up -d --build app
# (NOT make restart — that touches the bridges.)
```

### 2. Smoke-test

- New chat upload via bawthub UI → image renders thumb/preview/original
- Pre-existing chat history → old attachments still render (proves the
  rclone copy mapped the keys correctly; if these 404, that's the
  fallback covering for a key-prefix mismatch — check the `garage:` bucket
  layout against the namespace table in the spec before turning fallback
  off).
- Video generation → the content route serves with seek (Range support)
- `/v1/uploads/<id>` returns 200 with `Content-Type: image/webp`

### 3. Disable fallback once verified

```bash
# In ~/dev/llm-bawt/.env on echo
LLM_BAWT_S3_FALLBACK_FS=false
```

```bash
docker compose up -d --build app
```

If the smoke tests still pass, you're cut over.

## Rollback

Any of these is a clean revert:

- Flip env back to `LLM_BAWT_STORAGE_BACKEND=fs` and rebuild `app`. The
  FS bind mounts are unchanged; reads/writes go back to disk
  immediately. **Anything written to S3 between cutover and rollback
  stays in S3 but is no longer referenced.** That's harmless — it costs
  a small amount of storage and the next sweep of `media_gc.py` (if/when
  re-cut to S3) will tidy it up.
- Or: leave `LLM_BAWT_STORAGE_BACKEND=s3` and turn fallback back on. New
  writes go to S3, reads fall back to FS for anything missing.

## Failure drill (worth running once)

Confirm graceful degradation without a real outage:

```bash
ssh root@unraid 'docker stop garage-s3-simple'
# In another terminal, hit the upload endpoint:
curl -X POST -H "X-Entity-Id: nick" \
     -F "file=@some.png;type=image/png" \
     http://echo:8642/v1/uploads
# Expect: HTTP 503 JSON, not a 500 stack trace.
# Check the running app: chat itself still responds, history still loads.
ssh root@unraid 'docker start garage-s3-simple'
# Hit upload again — should succeed without an app restart.
```

The media_gc job will skip its pass after three consecutive
backend-unavailable errors (logged as `backend unavailable for N
consecutive deletes — aborting pass, next sweep will retry`) and the
nightly run picks up where it left off once Garage is back.

## Cleanup (after a week or two of verified-clean S3 operation)

- Drop the NFS bind mounts from `docker-compose.yml`:
  - `/mnt/user/home/echo/llmbawt-media:/var/lib/llm-bawt/media`
  - `${MEDIA_STORAGE_PATH:-./storage}:/app/storage`
- Optionally `rm -rf` the legacy FS roots once you're confident.
- Add `/mnt/user/appdata/garage-s3-simple/` to the Unraid backup rotation
  (infra-backups skill). The pool is mirrored but you still want
  off-host copies of the canonical attachment store.

## Source map

| Concern | File |
|---|---|
| Backend abstraction | `src/llm_bawt/media/object_store.py` |
| Chat upload pipeline | `src/llm_bawt/media/store.py` |
| Generated media pipeline | `src/llm_bawt/media/storage.py` |
| Range-aware content route | `src/llm_bawt/service/routes/media.py` |
| Upload routes (error mapping) | `src/llm_bawt/service/routes/uploads.py` |
| GC pass (skip-on-wedge) | `src/llm_bawt/service/jobs/media_gc.py` |
| Unit tests | `tests/test_object_store.py` (60 tests, moto-backed) |
| Env contract | `.env.docker` |
