from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.message import Message


class ModelAdapter(ABC):
    """Handles model-specific formatting and output quirks.
    
    Adapters are responsible for:
    - Additional stop sequences (beyond tool format handlers)
    - Output cleaning (removing model-specific artifacts)
    - Message formatting quirks (system role support, etc.)
    """
    
    name: str = "default"
    
    def get_stop_sequences(self) -> list[str]:
        """Model-specific stop sequences (combined with tool handler's)."""
        return []
    
    def clean_output(self, response: str) -> str:
        """Remove model-specific artifacts from output.
        
        Called AFTER streaming completes, BEFORE format handler sanitization.
        Default: passthrough (no cleaning).
        """
        return response
    
    def supports_system_role(self) -> bool:
        """Whether model natively supports system role messages."""
        return True
    
    def transform_messages(self, messages: list["Message"]) -> list["Message"]:
        """Apply model-specific message transformations.
        
        For models that don't support system role, this can merge
        system message into first user message.
        Default: passthrough.
        """
        return messages
