import logging
import re
from .base import ModelAdapter

logger = logging.getLogger(__name__)


class PygmalionAdapter(ModelAdapter):
    """Adapter for Pygmalion/character models (MythoMax, etc).
    
    These models output:
    - Role markers: [HUMAN], [/HUMAN], [INST], [/INST]
    - BBCode: [FONT=Arial], [/FONT], etc.
    - Sometimes try to continue conversation with fake turns
    """
    
    name = "pygmalion"
    
    def get_stop_sequences(self) -> list[str]:
        return [
            "[HUMAN]",
            "[/HUMAN]", 
            "[INST]",
            "[/INST]",
            "### Instruction:",
            "### Human:",
            "<|im_start|>user",
        ]
    
    def clean_output(self, response: str) -> str:
        original = response
        
        # Step 1: Remove content inside role blocks FIRST (includes the tags)
        # This handles hallucinated conversation turns like [HUMAN]fake message[/HUMAN]
        response = re.sub(r"\[HUMAN\].*?\[/HUMAN\]", "", response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r"\[INST\].*?\[/INST\]", "", response, flags=re.DOTALL | re.IGNORECASE)
        
        # Step 2: Remove standalone role markers (if they slipped through stop sequences)
        response = re.sub(r"\[/?HUMAN\]", "", response, flags=re.IGNORECASE)
        response = re.sub(r"\[/?INST\]", "", response, flags=re.IGNORECASE)
        
        # Step 3: Remove BBCode formatting tags: [FONT=Arial], [/FONT], [B], [/B], etc.
        response = re.sub(r"\[\w+(?:=[^\]]+)?\]", "", response)
        response = re.sub(r"\[/\w+\]", "", response)
        
        # Step 4: Clean excessive whitespace
        response = re.sub(r"\n{3,}", "\n\n", response)
        
        result = response.strip()
        
        # Log when content was modified (helps debugging model output issues)
        if result != original:
            removed = len(original) - len(result)
            logger.debug(f"PygmalionAdapter cleaned output: removed {removed} chars "
                        f"({len(original)} -> {len(result)})")
        
        return result
