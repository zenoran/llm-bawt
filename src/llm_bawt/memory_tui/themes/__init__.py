"""Theme definitions for memory TUI."""

from textual.theme import Theme

# Codex-like dark theme with cool cyan/green accents.
NEURAL_THEME = Theme(
    name="neural",
    primary="#22D3EE",
    secondary="#0EA5E9",
    accent="#2DD4BF",
    foreground="#E2E8F0",
    background="#020617",
    surface="#0F172A",
    panel="#1E293B",
    boost="#22D3EE",
    warning="#F59E0B",
    error="#EF4444",
    success="#22C55E",
    dark=True,
)

MIDNIGHT_THEME = Theme(
    name="midnight",
    primary="#8B5CF6",
    secondary="#6366F1",
    accent="#A78BFA",
    foreground="#E2E8F0",
    background="#000000",
    surface="#0A0A0A",
    panel="#141414",
    dark=True,
)

MATRIX_THEME = Theme(
    name="matrix",
    primary="#22C55E",
    secondary="#16A34A",
    accent="#4ADE80",
    foreground="#86EFAC",
    background="#000000",
    surface="#050505",
    panel="#0A0A0A",
    dark=True,
)

__all__ = ["NEURAL_THEME", "MIDNIGHT_THEME", "MATRIX_THEME"]
