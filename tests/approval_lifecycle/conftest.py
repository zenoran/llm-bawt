"""TASK-861 tests cannot contact inherited databases, including during imports."""
import os

for _key in list(os.environ):
    if (_key.upper().startswith("PG") or any(part in _key.upper() for part in
            ("POSTGRES", "DATABASE", "DB_HOST", "DB_PORT", "DB_USER", "DB_PASS", "DB_NAME", "DB_URL"))):
        os.environ.pop(_key, None)
os.environ["LLM_BAWT_STORAGE_BACKEND"] = "fs"

import psycopg2  # noqa: E402


def _forbid_connect(*args, **kwargs):
    raise RuntimeError("Live PostgreSQL access forbidden by TASK-861 isolated tests")


# Import-time guard, intentionally installed BEFORE application module collection.
psycopg2.connect = _forbid_connect
