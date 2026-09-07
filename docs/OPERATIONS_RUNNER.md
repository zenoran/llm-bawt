# Durable operations runner (TASK-861)

## Execution and recovery contract

Operations remain DB-configured Docker JSON action specs, not arbitrary shell.
There is no SSH, systemd, host script, app daemon-thread executor, or implicit
package installation. The LAN trust boundary is unchanged. The Docker socket is
privileged: network isolation/capability dropping do not make a socket-owning
worker safe for untrusted code. Only the operator-controlled worker image and
catalog are executable.

1. The service persists the full immutable invocation in `ops_jobs` (`queued`).
2. A conditional SQL claim reserves a per-operation concurrency slot and changes
   `queued -> dispatching` **before** any side-effecting Docker call.
3. It publishes/fsyncs an immutable `request.json` on a dedicated Docker volume,
   then creates `llm-bawt-ops-<full job UUID>` with an immutable image ID/digest.
   The worker has its own process/container lifetime, independent of app/bridges.
4. Docker accepting the worker means `accepted`, **not succeeded**. The worker
   persists `accepted`, owns the start delay, persists `running`, executes one
   Docker action, and atomically fsyncs `receipt.json` with the final result.
5. Status reads and the independent reconciler import receipts into canonical DB
   metadata. No success is inferred from submission, container existence, a
   stopped worker, or an app-owned timer.

A missing/abnormally exited worker without a terminal receipt is `lost`: the
side effect is unknown, and recovery never creates/replays a replacement.
A deterministic container still in `created` can be started if no prior start
attempt was persisted. Per-job volume locks and a durable start-attempt marker
serialize the inspect/start race; an uncertain prior start is not blindly retried.
A submission lock plus abandonment marker prevents a paused submitter dispatching
after recovery has declared its missing worker lost. A worker restarted manually
checks its receipt/started marker and will not repeat an action already begun.

The stdlib worker enforces a wall-clock execution deadline with a process alarm;
start delay is excluded. Docker `stop_grace_seconds` (spec field, default 10) is
separate from `timeout_seconds` (job execution deadline). **A timed-out Docker
request can continue inside the daemon**: `timed_out` does not mean the target
was untouched or rollback occurred. Neither timed-out nor lost jobs auto-retry.
An operator must inspect the actual target before deliberately requesting a new
invocation. Success means Docker completed its API action, not application
health/readiness. Pull downloads an image; it does not recreate its container.

Read-only prerequisite failures leave a queued job with an error. Fixing a
missing image/volume allows the same snapshotted request to proceed. Changing
the configured image cannot rewrite queued jobs: historical image IDs and volume
mounts must remain available for their lifetime. `max_concurrent` counts claimed,
accepted, and running jobs; the smallest limit among overlapping snapshots wins.

## Deployment prerequisites (not applied by this change)

Repository inspection verified the base compose app mounts Docker's socket and
`./.logs:/app/.logs`; the dev overlay mounts `./src:/app/src`. Those host bind
sources are not safe to guess from a bridge container. This change instead adds
`docker-compose.ops.yml`, a separate override with a dedicated **named volume**.
It does not alter existing compose hunks. The implementation session had no
Docker CLI/socket, so actual deployed mounts, daemon behavior and app-restart
survival have **not** been live-tested.

An operator deploying this must:

1. Inspect the current app image ID, current compose file list, Docker daemon,
   app mounts and networks. Verify the daemon is the one owning the target
   containers. Do not infer host bind paths from `/app/src` in a bridge.
2. Build `docker/Dockerfile.ops-worker` with `OPS_PYTHON_BASE` set to an
   operator-verified local Python image ID/digest. It only COPYs `worker.py`;
   no pip/uv/apt install is required. Record the resulting immutable image ID
   (or published digest) as `LLM_BAWT_OPS_WORKER_IMAGE`. Mutable tags are rejected.
3. Choose a dedicated per-stack volume name and set
   `LLM_BAWT_OPS_RECEIPT_VOLUME`. Set `LLM_BAWT_OPS_RECONCILER_IMAGE` to the
   verified existing CPU app image ID/digest. These are topology configuration,
   not catalog metadata. Preserve historical image/volume availability while
   jobs are active.
4. Append `docker-compose.ops.yml` **last** to the existing compose configuration
   (preserve any dev/prod overlays). Review merged config before applying. The
   override mounts the volume at `/var/lib/llm-bawt-ops` in app and reconciler,
   and starts a separate `python -m llm_bawt.ops.reconciler` service. It uses
   the existing app image, `.env` DB configuration, default stack network and
   read-only dev source mount. Adapt that mount/network explicitly for a baked
   production image or alternate database network; do not guess.
5. With explicit deployment permission, recreate **only** app and the new
   ops-reconciler service. A restart alone cannot add mounts/environment. Never
   restart bridges/Redis as part of this setup. Applying the override/building
   images/creating volumes is an operator action, not performed by tests.
6. Verify both submitter containers' inspected `Mounts` contain the configured
   writable named volume at the exact configured root. The executor performs
   this check itself using `HOSTNAME` (or `LLM_BAWT_OPS_APP_CONTAINER` if the
   container has a custom hostname). The worker image must already exist;
   dispatch never implicitly pulls it or creates a misspelled volume.
7. Run staged non-self-affecting smoke operations, then explicitly approved
   self-restart tests. Confirm accepted/running state, app-independent worker,
   receipt survival and DB reconciliation. The SQLite/fake-Docker tests are
   strong regression evidence, **not proof of live Docker survival**.

The worker is launched with no network, read-only rootfs, dropped capabilities,
no-new-privileges, bounded memory/PIDs and restart policy `no`. It receives only
the receipt volume and Docker socket, not DB credentials, app source mounts, or
app environment. The reconciler has DB access and needs the app's Docker SDK;
the standalone worker uses only Python's standard library. No Python dependency
changes are required.

Retain receipts/requests and exited worker containers until the job is terminal
and audit retention allows removal. No automatic Docker/volume garbage collector
is included. `compose down -v`, daemon-wide pruning, volume loss, host power loss,
manual receipt edits and disk-full conditions are not covered by an exactly-once
claim. Receipts contain resolved arguments (possibly secrets); restrict volume
access and backups. Public job JSON uses schema-sensitive argument redaction;
operator revision endpoints intentionally expose operator configuration.

## Approval integration

Trusted interception calls:

```python
snapshot = ops.prepare_invocation(operation_slug, args)
# Persist snapshot with approval request BEFORE presenting approval.
ops.dispatch_job(operation_slug=operation_slug, args=args,
                 idempotency_key=stable_key, approved_snapshot=snapshot, ...)
```

`prepare_invocation` returns detached JSON with `snapshot_version`, full
`operation` (`version`, `script_hash`, exact `command_script`, schema/default JSON,
all settings and attribution), parsed `spec`, `schema`, `defaults`, `input_args`,
`resolved_args`, `execution` (including immutable worker image and receipt
configuration), and `snapshot_hash` (canonical-JSON SHA256). Hash checking detects
corruption; it is **not authorization**. The snapshot must come from persisted
trusted approval context, never a caller-supplied tool field.

`ApprovedCallerContext.approved_snapshot` is the integration field read by
`ops_tools`; the public `ops_run` signature does not accept snapshots. An approved
context missing this snapshot is rejected: pre-migration approvals need fresh
approval, never silent execution of today's catalog. Contextvars
are captured/propagated before offloading blocking DB/Docker work with
`asyncio.to_thread`. A valid approved snapshot executes the old revision despite
later edits, but current disabled/soft-deleted state vetoes new dispatch. Existing
idempotent replay remains a read even if the operation was subsequently disabled.

No explicit key means a new UUID on every direct invocation. The same explicit
key + identical original operation/input returns the original job (including
queued jobs), without re-dispatch. A different payload or approved snapshot with
that key raises `idempotency_conflict`. An approved call with no explicit key uses
its trusted approval request ID. This is global key uniqueness, so integrations
should namespace their own explicit keys.

## API and migrations

- `GET /v1/ops/operations` and `/v1/ops/jobs` accept `limit` (1..200) and
  nonnegative `offset`; responses are `{operations|jobs, total, limit, offset}`.
  `total` is the count for the filters, not the current page length.
- Dispatch responses include both `id` and compatibility alias `job_id`.
- Direct HTTP dispatch accepts `actor`, `caller_user_id`, `caller_bot_id`,
  `caller_turn_id`, `caller_session_key` under the existing trusted LAN operator
  model. Backend is `http-operator`; no new global authorization layer is added.
- CRUD accepts operator `actor`; enable/delete are revisioned. Slugs are immutable
  to preserve invocation/history identity. Nullable fields can be cleared with
  explicit JSON null. Invalid schema/default/limits/spec updates are atomic errors.
- `GET /v1/ops/operations/{slug}/revisions` returns full historical configuration,
  actor and timestamp in `{revisions, total, limit, offset}`.
- Bootstrap safely adds nullable `invocation_snapshot_json`, `request_payload_json`,
  `caller_actor` to existing jobs and creates `ops_operation_revisions`; captures
  only the existing current revision as baseline. Earlier historical revisions
  cannot be reconstructed and are not fabricated. Legacy active jobs without
  snapshots reconcile to lost; never execute them using today's catalog.
- The supported schema subset is explicit; malformed/unsupported keywords are
  rejected rather than ignored. Empty `{}` means no args, all nested objects are
  closed, array items/types and numeric/string constraints are checked. Defaults
  must themselves be valid declared values; required caller fields can remain
  absent in the defaults object.
