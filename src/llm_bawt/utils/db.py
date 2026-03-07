"""Database utilities shared across the application."""

from sqlalchemy import event


def set_utc_on_connect(engine) -> None:
    """Register a connect listener that forces UTC on every new connection.

    This prevents naive-datetime misinterpretation when the container's
    ``TZ`` env var is not UTC (e.g. ``America/New_York``).
    """

    @event.listens_for(engine, "connect")
    def _set_timezone(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("SET timezone = 'UTC'")
        cursor.close()
