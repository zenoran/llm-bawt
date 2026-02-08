import logging
import re
from .base import ModelAdapter

logger = logging.getLogger(__name__)


class DolphinAdapter(ModelAdapter):
    """Adapter for Dolphin models (Dolphin3.0-Llama3.1, Dolphin-Qwen, etc).

    Provides output cleaning as a safety net for hallucinated observations.
    Stop sequences are handled by ReActFormatHandler.
    """

    name = "dolphin"

    def clean_output(self, response: str) -> str:
        """Clean any hallucinated observations that slipped through."""
        original = response

        # Remove anything after "Observation:" if it somehow got through
        # This catches hallucinated tool results
        obs_match = re.search(r"\n+Observation\s*:", response, re.IGNORECASE)
        if obs_match:
            response = response[: obs_match.start()]

        result = response.strip()

        if result != original:
            removed = len(original) - len(result)
            logger.debug(
                f"DolphinAdapter cleaned output: removed {removed} chars "
                f"({len(original)} -> {len(result)})"
            )

        return result
