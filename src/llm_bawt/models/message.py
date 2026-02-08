import time
from typing import Any
try:
    import tiktoken
    _tiktoken_present = True
except ImportError:
    tiktoken = None
    _tiktoken_present = False

class Message:
    """Standard message format for all LLM clients"""

    def __init__(
        self,
        role: str,
        content: str,
        timestamp: float | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        db_id: str | None = None,
    ):
        self.role = role
        self.content = content if content is not None else ""
        self.timestamp = timestamp if timestamp else time.time()
        self.tool_calls = tool_calls  # For assistant messages that make tool calls
        self.tool_call_id = tool_call_id  # For tool result messages
        self.db_id = db_id  # Primary key from the database (when loaded from PostgreSQL)

    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        """Create a Message from a dictionary"""
        role = data.get("role", "user")
        content = cls._extract_content(data)
        timestamp = data.get("timestamp", time.time())
        tool_calls = data.get("tool_calls")
        tool_call_id = data.get("tool_call_id")
        db_id = data.get("db_id")
        return cls(role, content, timestamp, tool_calls, tool_call_id, db_id=db_id)

    @staticmethod
    def _extract_content(data: dict) -> str:
        """Extract content from the data dictionary"""
        content = data.get("content", "")
        if isinstance(content, list):
            return " ".join(item.get("text", "") for item in content if item.get("type") == "text")
        return str(content) # Ensure content is always a string

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        d = {"role": self.role, "content": self.content, "timestamp": self.timestamp}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.db_id:
            d["db_id"] = self.db_id
        return d

    def to_api_format(self) -> dict:
        """Convert to API-compatible format (without timestamp)"""
        # Map internal roles to API-compatible roles
        # 'summary' is stored internally but should be sent as 'system' to the LLM
        api_role = self.role
        if api_role == "summary":
            api_role = "system"

        # Ensure content is never None (OpenAI API rejects null content)
        d = {"role": api_role, "content": self.content or ""}
        
        # Include tool_calls for assistant messages that made tool calls
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": tc.get("arguments", "{}") if isinstance(tc.get("arguments"), str) else __import__("json").dumps(tc.get("arguments", {})),
                    }
                }
                for tc in self.tool_calls
            ]
        
        # Include tool_call_id for tool result messages
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
            
        return d

    def get_token_count(self, encoding, base_overhead: int) -> int:
        """
        Calculates the token count for this message using a given tiktoken encoding.
        Assumes tiktoken is present and encoding is a valid tiktoken encoding object.
        """
        if not _tiktoken_present or not encoding:
            # Fallback or raise error if tiktoken/encoding is not available
            # For now, let's assume a simple character count as a rough estimate
            # or indicate that token counting is not possible.
            # This behavior might need to be more sophisticated depending on requirements.
            # print("Warning: tiktoken not available for accurate token counting.")
            return len(self.content) // 4 # Very rough estimate

        num_tokens = base_overhead
        if self.content: # Ensure content is not None
            num_tokens += len(encoding.encode(self.content))
        return num_tokens