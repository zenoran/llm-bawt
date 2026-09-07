"""Data-minimal approval audit shared by producers and the app store.

Matching still uses the original subject. Audit deliberately does not: shell
strings and arbitrary MCP JSON cannot be reliably scrubbed with secret regexes.
"""
from __future__ import annotations


def audit_subject(subject: str) -> str:
    """Bounded, fully redacted display value; exact identity lives in the hash."""
    return "[subject redacted]" if subject else ""


def decision_metadata(decision, bundle, *, outcome: str | None = None) -> dict:
    policy = decision.policy
    return {
        "action": decision.action.value,
        "outcome": outcome or decision.action.value,
        "severity": decision.severity.value,
        "subject": audit_subject(decision.subject),
        "policy_id": policy.id if policy else None,
        "policy_version": policy.version if policy else None,
        "bundle_etag": bundle.etag if bundle else "",
        "invocation_hash": decision.grant_key,
    }
