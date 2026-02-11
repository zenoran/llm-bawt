"""Direct PostgreSQL database client for DB mode."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text, MetaData, Table, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from llm_bawt.utils.config import config


class MemoryDBClient:
    """Direct PostgreSQL client for raw database access."""
    
    def __init__(self):
        """Initialize database connection from config."""
        self.engine: Engine | None = None
        self.metadata = MetaData()
        self._inspector = None
        self._connected = False
        
    def connect(self) -> bool:
        """Establish database connection.
        
        Returns:
            True if connected successfully.
        """
        try:
            host = getattr(config, "POSTGRES_HOST", "localhost")
            port = int(getattr(config, "POSTGRES_PORT", 5432))
            user = getattr(config, "POSTGRES_USER", "llm_bawt")
            password = getattr(config, "POSTGRES_PASSWORD", "")
            database = getattr(config, "POSTGRES_DATABASE", "llm_bawt")
            
            if not password:
                return False
            
            encoded_password = quote_plus(password)
            connection_url = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{database}"
            
            self.engine = create_engine(
                connection_url,
                poolclass=NullPool,
                future=True,
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self._inspector = inspect(self.engine)
            self._connected = True
            return True
            
        except Exception as e:
            print(f"DB connection failed: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connected and self.engine is not None
    
    def get_tables(self) -> list[dict[str, Any]]:
        """Get list of all tables in database.
        
        Returns:
            List of table info dicts with name, schema, type.
        """
        if not self.is_connected():
            return []
        
        tables = []
        try:
            table_names = self._inspector.get_table_names()
            for name in table_names:
                # Categorize tables
                category = "other"
                if name.endswith("_memories"):
                    category = "memories"
                elif name.endswith("_messages"):
                    category = "messages"
                elif name.endswith("_forgotten_messages"):
                    category = "forgotten"
                elif name.endswith("_summaries"):
                    category = "summaries"
                elif name in ("user_profiles", "bot_profiles"):
                    category = "profiles"
                elif name in ("user_attributes", "bot_attributes"):
                    category = "attributes"
                
                tables.append({
                    "name": name,
                    "schema": "public",
                    "category": category,
                    "columns": [],
                })
            
            # Sort by category then name
            tables.sort(key=lambda x: (x["category"], x["name"]))
            return tables
            
        except Exception as e:
            print(f"Error getting tables: {e}")
            return []
    
    def get_table_columns(self, table_name: str) -> list[dict[str, Any]]:
        """Get columns for a table.
        
        Args:
            table_name: Name of the table.
            
        Returns:
            List of column info dicts.
        """
        if not self.is_connected():
            return []
        
        try:
            columns = self._inspector.get_columns(table_name)
            return [
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "default": col.get("default"),
                }
                for col in columns
            ]
        except Exception as e:
            print(f"Error getting columns for {table_name}: {e}")
            return []
    
    def execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        """Execute a SQL query.
        
        Args:
            query: SQL query string.
            params: Query parameters.
            limit: Maximum rows to return.
            
        Returns:
            Dict with columns, rows, rowcount, and error info.
        """
        if not self.is_connected():
            return {
                "success": False,
                "error": "Not connected to database",
                "columns": [],
                "rows": [],
                "rowcount": 0,
            }
        
        result = {
            "success": True,
            "error": None,
            "columns": [],
            "rows": [],
            "rowcount": 0,
        }
        
        try:
            with self.engine.connect() as conn:
                # Add LIMIT if SELECT and no LIMIT present
                query_stripped = query.strip().upper()
                if query_stripped.startswith("SELECT") and "LIMIT" not in query_stripped:
                    query = f"{query} LIMIT {limit}"
                
                stmt = text(query)
                cursor = conn.execute(stmt, params or {})
                
                # Get column names
                if cursor.cursor is not None:
                    result["columns"] = [desc[0] for desc in cursor.cursor.description]
                
                # Fetch rows
                rows = cursor.fetchall()
                result["rowcount"] = len(rows)
                
                # Convert rows to dicts
                for row in rows:
                    row_dict = {}
                    for i, col in enumerate(result["columns"]):
                        value = row[i]
                        # Serialize complex types
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value)
                        row_dict[col] = value
                    result["rows"].append(row_dict)
                
                conn.commit()
                
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def execute_update(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an UPDATE/INSERT/DELETE query.
        
        Args:
            query: SQL query string.
            params: Query parameters.
            
        Returns:
            Dict with rowcount and error info.
        """
        if not self.is_connected():
            return {
                "success": False,
                "error": "Not connected to database",
                "rowcount": 0,
            }
        
        result = {
            "success": True,
            "error": None,
            "rowcount": 0,
        }
        
        try:
            with self.engine.connect() as conn:
                stmt = text(query)
                cursor = conn.execute(stmt, params or {})
                result["rowcount"] = cursor.rowcount
                conn.commit()
                
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def update_row(
        self,
        table: str,
        pk_column: str,
        pk_value: Any,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Update a single row.
        
        Args:
            table: Table name.
            pk_column: Primary key column name.
            pk_value: Primary key value.
            updates: Dict of column->value to update.
            
        Returns:
            Result dict with success status.
        """
        if not updates:
            return {"success": True, "rowcount": 0}
        
        set_clauses = []
        params = {"pk": pk_value}
        
        for i, (col, val) in enumerate(updates.items()):
            set_clauses.append(f"{col} = :val{i}")
            params[f"val{i}"] = val
        
        query = f"UPDATE {table} SET {', '.join(set_clauses)} WHERE {pk_column} = :pk"
        return self.execute_update(query, params)
    
    def delete_row(
        self,
        table: str,
        pk_column: str,
        pk_value: Any,
    ) -> dict[str, Any]:
        """Delete a single row.
        
        Args:
            table: Table name.
            pk_column: Primary key column name.
            pk_value: Primary key value.
            
        Returns:
            Result dict with success status.
        """
        query = f"DELETE FROM {table} WHERE {pk_column} = :pk"
        return self.execute_update(query, {"pk": pk_value})
    
    def get_table_primary_key(self, table_name: str) -> str | None:
        """Get primary key column for a table.
        
        Args:
            table_name: Name of the table.
            
        Returns:
            Primary key column name or None.
        """
        if not self.is_connected():
            return None
        
        try:
            pk = self._inspector.get_pk_constraint(table_name)
            columns = pk.get("constrained_columns", [])
            return columns[0] if columns else None
        except Exception:
            return None
    
    def build_query_from_params(
        self,
        table: str,
        columns: list[str] | None = None,
        where: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int = 100,
    ) -> str:
        """Build a SELECT query from parameters.
        
        Args:
            table: Table name.
            columns: Columns to select (None = all).
            where: Where clauses as dict.
            order_by: Order by clause.
            limit: Limit.
            
        Returns:
            SQL query string.
        """
        cols = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols} FROM {table}"
        
        if where:
            conditions = []
            for col, val in where.items():
                if isinstance(val, list):
                    # Array containment
                    conditions.append(f"{col} @> ARRAY[{', '.join(repr(v) for v in val)}]")
                elif isinstance(val, tuple) and len(val) == 2:
                    # Range
                    conditions.append(f"{col} BETWEEN {repr(val[0])} AND {repr(val[1])}")
                else:
                    conditions.append(f"{col} = {repr(val)}")
            query += " WHERE " + " AND ".join(conditions)
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        query += f" LIMIT {limit}"
        return query
