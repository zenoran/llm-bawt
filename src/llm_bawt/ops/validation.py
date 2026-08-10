"""Minimal JSON Schema validator for the ops catalog (TASK-639).

Purpose-built to validate ``ops_run`` args against the operation's stored
``args_schema_json``. Deliberately does NOT depend on the ``jsonschema``
package — adding a runtime dep would require rebuilding the app image, and
this subset is small enough to inline safely.

Supports the JSON Schema constructs the seed catalog actually uses:

* ``type`` — ``"object"``, ``"string"``, ``"integer"``, ``"number"``,
  ``"boolean"``, ``"null"``, ``"array"``
* ``additionalProperties: false`` — ALWAYS enforced regardless of the schema,
  so unknown args are rejected even if the operator omits the flag
* ``required`` — list of required property names
* ``properties`` — per-key nested schemas
* ``enum`` — string enum on a scalar
* ``minLength`` / ``maxLength`` — for strings
* ``pattern`` — regex on strings

Any construct the schema uses that isn't in this list is passed through
without validation (with a debug log). That's acceptable for TASK-639: the
threat model is "agent supplied wrong args", not "operator supplied malicious
schema". The operator writes both the script and the schema; a schema they
wrote to be permissive is their call.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class ArgValidationError(ValueError):
    """Raised when the caller-supplied args don't satisfy the operation schema.

    Carries a list of specific violations so the MCP tool response can point
    the agent at exactly what to fix.
    """

    def __init__(self, violations: list[str]):
        self.violations = list(violations)
        joined = "; ".join(violations)
        super().__init__(f"args validation failed: {joined}")


_PRIMITIVE_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
    "array": list,
    "object": dict,
}


def _check_type(value: Any, expected: str) -> str | None:
    if expected not in _PRIMITIVE_TYPES:
        return None  # unknown → skip
    # bool is a subclass of int; JSON Schema treats them separately.
    if expected == "integer" and isinstance(value, bool):
        return "expected integer, got boolean"
    if expected == "number" and isinstance(value, bool):
        return "expected number, got boolean"
    expected_types = _PRIMITIVE_TYPES[expected]
    if not isinstance(value, expected_types):
        return f"expected {expected}, got {type(value).__name__}"
    return None


def _validate_scalar(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    problems: list[str] = []
    if "type" in schema:
        err = _check_type(value, schema["type"])
        if err:
            problems.append(f"{path}: {err}")
            return problems  # bail — further checks assume the type is right
    if "enum" in schema:
        if value not in schema["enum"]:
            problems.append(
                f"{path}: value {value!r} not in enum {list(schema['enum'])}"
            )
    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            problems.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            problems.append(f"{path}: longer than maxLength {schema['maxLength']}")
        if "pattern" in schema:
            try:
                if not re.search(schema["pattern"], value):
                    problems.append(f"{path}: does not match pattern")
            except re.error:
                # Bad schema regex — operator problem, log and continue.
                logger.debug("Invalid regex in schema at %s: %r", path, schema["pattern"])
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: above maximum {schema['maximum']}")
    return problems


def _validate_object(
    value: Any,
    schema: dict[str, Any],
    path: str,
) -> list[str]:
    problems: list[str] = []
    if not isinstance(value, dict):
        return [f"{path}: expected object, got {type(value).__name__}"]
    properties: dict[str, Any] = schema.get("properties", {}) or {}
    required: list[str] = list(schema.get("required", []) or [])

    # TASK-639 invariant: unknown keys are ALWAYS rejected. The schema-level
    # ``additionalProperties`` flag is enforced regardless of its value —
    # operators can't accidentally leave the door open.
    extras = [k for k in value.keys() if k not in properties]
    for k in extras:
        problems.append(f"{path}: unknown property {k!r}")

    for req in required:
        if req not in value:
            problems.append(f"{path}: required property {req!r} missing")

    for k, subschema in properties.items():
        if k not in value:
            continue
        sub_path = f"{path}.{k}" if path else k
        if isinstance(subschema, dict) and subschema.get("type") == "object":
            problems.extend(_validate_object(value[k], subschema, sub_path))
        else:
            problems.extend(_validate_scalar(value[k], subschema or {}, sub_path))

    return problems


def validate_args(
    args: dict[str, Any],
    schema_json: str,
    defaults_json: str | None = None,
) -> dict[str, Any]:
    """Validate ``args`` against ``schema_json``. Returns the merged args
    (defaults applied where the caller left a key out).

    Raises :class:`ArgValidationError` on any schema violation, listing every
    specific problem so the MCP tool response can be actionable.
    """
    try:
        schema = json.loads(schema_json or "{}")
    except (json.JSONDecodeError, TypeError) as exc:
        raise ArgValidationError([f"operation schema is not valid JSON: {exc}"]) from exc

    defaults: dict[str, Any] = {}
    if defaults_json:
        try:
            defaults = json.loads(defaults_json) or {}
        except (json.JSONDecodeError, TypeError):
            defaults = {}
        if not isinstance(defaults, dict):
            defaults = {}

    if not isinstance(args, dict):
        raise ArgValidationError(["args must be an object"])

    # Merge: caller-supplied wins; defaults only fill absent keys AND only
    # if the key is declared in the schema (defaults for undeclared keys are
    # a footgun that would defeat additionalProperties=false).
    properties: dict[str, Any] = schema.get("properties", {}) or {}
    merged: dict[str, Any] = {}
    for k, v in defaults.items():
        if k in properties and k not in args:
            merged[k] = v
    merged.update(args)

    problems = _validate_object(merged, schema, path="args") if schema else []

    if problems:
        raise ArgValidationError(problems)

    return merged


__all__ = ["validate_args", "ArgValidationError"]
