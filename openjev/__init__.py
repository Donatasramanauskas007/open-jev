"""openjev: one-pass option scoring with a local Gemma model on Apple silicon (MLX)."""
from .scorer import DEFAULT_MODEL, OptionScore, OptionScorer

__all__ = ["DEFAULT_MODEL", "OptionScore", "OptionScorer"]
