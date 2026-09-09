# Skill plugin migration — TASK-844

## Implementation state (2026-09-09)

TASK-844 completed the content-preserving package migration and opt-in runtime
plumbing. Existing bots remain on legacy loading unless
`agent_backend_config.skill_bundle` is explicitly selected. No production
publication, bot-profile change, or bridge restart was performed.

TASK-871 owns live-runtime acceptance and controlled activation: Codex helper
execution under the current namespace restrictions, prepared-home credential and
session behavior, tenant filesystem isolation, workspace/session-boundary tests,
and install/update/rollback/disable/remove lifecycle receipts. Authentication in
the real Codex bridge works; its remaining probe failure is command execution,
not credential recovery.

## Components and ownership

- `agent_bridge.skill_packages`: validate dual manifests, SKILL.md metadata,
  contained resources, no symlinks/special files or executable plugin components.
- `agent_bridge.skill_registry`: validated immutable snapshots, source/hash
  receipts, atomic package and bundle indexes, rollback and logical removal.
- `agent_bridge.skill_selection`: Claude SDK plugin paths and exact qualified
  skill names; explicit settings sources; resume guard on bundle changes.
- `agent_bridge.skill_codex`: explicit native Codex marketplace/install commands
  in an isolated prepared home. Never writes a native cache by hand and never
  installs on the chat hot path. The SDK receives per-instance environment.
- `agent_bridge.skill_export`: private additive-overlay snapshots of the existing
  authoring tree. Public-owned product skills are excluded so personal + public
  bundles compose without duplicate names or drifting architecture copies.
- `bawthub-skills/plugins/bawthub`: canonical portable public product knowledge,
  including full routers plus bundled architectural references, one tree and two
  native manifests. Root-level old files/history are NOT publication-safe.

The existing Codex legacy resolver was corrected: packaged/explicitly linked
skills win over same-name private repo fallbacks. Broken explicit links fail
missing, not by silently substituting a private skill. Empty legacy mapping
entries retain prior repo/system lookup behavior.

## Install and manage (explicit operator CLI)

Run from the llm-bawt environment. Registry paths are deployment configuration,
not a hardcoded host. Provision/mount `AGENT_SKILL_REGISTRY` consistently in both
bridges. Install the new `tomlkit` dependency before using Codex preparation.
Production image/dependency activation still needs the authorized release flow.

```sh
python -m agent_bridge.skill_registry --root /path/to/registry install \
  /path/to/bawthub-skills/plugins/bawthub --visibility public \
  --provenance 'approved-repository@resolved-commit'
python -m agent_bridge.skill_registry --root /path/to/registry select \
  tenant bawthub --audience public
python -m agent_bridge.skill_registry --root /path/to/registry prepare-codex \
  tenant --binary /absolute/path/to/codex --base-home /path/to/existing/codex-home
python -m agent_bridge.skill_registry --root /path/to/registry inspect tenant
```

Use the existing bot profile API/config surface to set:

```json
{"agent_backend_config": {"skill_bundle": "tenant"}}
```

Merge with existing configuration; do not overwrite unrelated profile fields.
The selected name is validated on profile save and forwarded through Redis to
both bridges. A missing registry/bundle/prepared Codex home fails rather than
silently reverting to legacy private skill loading. Start a NEW conversation:
resumed conversations reject a changed selected-bundle hash. To disable packages,
assign a named explicit empty bundle for the new conversation; removing the
`skill_bundle` field restores legacy behavior and deliberately performs no registry
reads or writes.

Update: fetch an approved repo/ref outside chat, validate/install its new package,
select a new named candidate bundle, prepare Codex, then assign it for new
conversations. Never change package files under active sessions. A failed
install/preparation leaves previous immutable releases available.

```sh
# Roll back the package pointer, then select/prepare a fresh candidate bundle.
python -m agent_bridge.skill_registry --root /path/to/registry rollback bawthub
# Or restore the previous exact bundle selection.
python -m agent_bridge.skill_registry --root /path/to/registry rollback-bundle tenant
# Disable package loading with an explicit empty bundle.
python -m agent_bridge.skill_registry --root /path/to/registry select none --audience public
# Remove from future selection, preserving releases referenced by active sessions.
python -m agent_bridge.skill_registry --root /path/to/registry remove bawthub
```

Native caches/release directories are not garbage-collected automatically: old
session-safe files deliberately remain. These commands require no editing inside
a running container and no network on ordinary turns. Repository fetch trust,
automatic polling and public directory submission are not implemented here.

## Private compatibility export

```sh
python -m agent_bridge.skill_export /path/to/private-authoring-repo \
  /path/to/new-personal-package --version 1.0.0
python -m agent_bridge.skill_registry --root /path/to/private-registry install \
  /path/to/new-personal-package --visibility private --provenance 'private-repo@commit'
```

Never publish that artifact or mount it into tenants. Export refuses existing
destinations and symlinked source resources; it normalizes a legacy unquoted
one-line description representation and adjusts shared-script relative paths.
Absolute host references remain intentional PRIVATE compatibility content.
Existing mounted helper paths and skills-sync remain unchanged until live
migration is authorized. No unrelated pending ops/approval skill edits were
modified. Public skill creation/maintenance guidance points at the canonical
public source, not vendor caches or duplicated copies.

## Security and known limitations

- Public/private classification is operator asserted, not proof of sanitization.
  Scan and review the complete artifact. Public package identifiers were scanned
  for private domains, addresses/accounts and infrastructure names; no matches.
- A selected skill list is NOT filesystem security. Current shared dev mounts
  expose private files; do not claim tenant isolation from filtering. Tenants need
  separate mounts/registry/credentials with NO private package files delivered.
- Claude bundled/system skills still exist. Codex bundled and repository-scoped
  skills can still be discovered. Prepared homes remove ambient user plugins,
  but do not sandbox arbitrary workspace files or repository skills.
- Codex preparation retains the operator's non-plugin configuration and links
  auth/session storage from the base home. Use a separate registry per auth/home
  boundary. Native auth refresh behavior must be reconciled with the deployment's
  credential owner before activating prepared homes; symlinks alone are not a
  proven atomic credential-refresh contract.
- Base config is snapshotted during preparation. Re-run preparation after an
  MCP/config update: generation identity includes the config digest and base-home
  identity, so the new generation activates atomically and old homes stay intact.
  Resume guards reject a changed Codex generation until a new conversation.
- Legacy root content remains in bawthub-skills for recovery, outside the native
  package. Its Git history contains private material: publish ONLY the audited
  package artifact, not the repository/history. Repo visibility was not changed.

## Evidence

Versions: Claude Agent SDK 0.2.152 / Claude Code 2.1.259; Codex Python SDK
0.1.11; this Claude container's separate Codex CLI 0.147.0; actual Codex bridge
and host CLI 0.153.4. The legacy binary path missing in the Claude container
DOES exist and runs correctly in the actual Codex service; no path fix needed.

- Dual-manifest fixture: Claude native validation passed; Codex native marketplace
  add, plugin add and list passed, including paths with spaces.
- Codex 0.147.0 app-server `skills/list` discovered `compat-probe:probe`, exact
  installed file path, enabled=true, errors=[]; this is discovery, not exec.
- Claude real SDK initialization listed `compat-probe:probe`. An actual isolated
  inference invoked the Skill tool and Read on bundled reference/helper files,
  returning `TASK844_REFERENCE_OK` and `TASK844_HELPER_OK`. Helper was read,
  not executed, in the read-only Claude probe.
- New Codex preparation adapter ran native install successfully on 0.153.4 in
  an isolated host directory, producing `example@bawt-managed` version 1.0.0.
- Correction: the first Codex SDK probe used the RETIRED HOST credential path,
  not the deployed bridge path. Its refresh_token_reused error was not evidence
  of a bridge/provider failure. Live mounts confirm no host auth.json mount;
  the bridge materializes a container-local bundle from the app broker.
- Rerunning inside the real Codex bridge authenticated successfully and selected
  the packaged skill. Command execution then failed with bwrap namespace
  permissions under the probe's read-only sandbox. No sandbox policy was changed
  to force success. Receipt/helper execution remains unverified, NOT auth.
- Probe now requires matching bridge CODEX_HOME, CODEX_AUTH_PATH, CODEX_BIN and
  LLM_BAWT_API_URL before any inference, and fails unless successful commands
  produce both receipts. Runtime errors and bridge/proxy guidance no longer
  direct agents to a host login or retired auth mount. Probe script:
  `scripts/smoke/skill_plugin_probe.py`.
- Private additive overlay: 19 private-only skills validated successfully;
  public package: 9 skills with 21 bundled architectural references. The eight
  public-owned source skills are excluded from `personal` to prevent duplicate
  names and content drift.
- Unit/dispatch/profile suite: 90 passed, 7 skipped (existing integration skips).
- Recovered tenant helpers: 4 subprocess tests passed with fake psql, proving
  password/value absent from argv, stdin-only vault set, missing-credential error.
  Shell syntax checks passed. No live database secrets were queried or changed.
- Vault schema owner confirmed: BawtHub `deploy/bootstrap.sh` function
  `provision_vault_schema`, not the llm-bawt backend.

## Deferred live acceptance — TASK-871

TASK-871 owns these post-migration checks and any controlled activation:

1. Complete the Codex fixture/helper execution check through an appropriately
   authorized test surface; read-only sandbox commands currently fail namespace
   creation. Authentication is verified working; no credential recovery needed.
2. Validate credential/session sharing semantics for prepared Codex homes before
   activating them; config-aware generation identity is implemented.
3. Verify both harnesses list the 28 non-duplicated composed skills in nested
   workspaces and across fresh/resumed sessions: 19 private-only `personal:*`
   skills plus 9 canonical `bawthub:*` public skills.
4. Verify a clean tenant runtime with private mounts and credentials absent.
5. Exercise install, update, offline failure, rollback, named-empty disable,
   removal, and persistence across container recreation.

The public package-only exporter is implemented in bawthub-skills
`scripts/export-plugin.py`; its deterministic archive and full content inventory
are verified. Publication and activation remain explicit release actions.

## Standards consulted

- https://agentskills.io/specification
- https://code.claude.com/docs/en/plugins-reference
- https://developers.openai.com/codex/skills
- https://developers.openai.com/plugins/build/plugins
- https://developers.openai.com/plugins/guides/submit-claude-plugin
- Version-matched Codex config schema for rust-v0.147.0.

Current docs are not proof of installed headless support. Keep these receipts
separate from implementation claims and production activation.
