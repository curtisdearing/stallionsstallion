"""Provenance-first pregame context — display-only by construction.

Nothing in this package may alter a projected margin, total or win
probability. Two guarantees make that structural rather than aspirational:

  * the package cannot import the modelling modules, and they cannot
    import it (asserted both ways in tests/test_context_leakage.py);
  * whether a factor is excluded, display-only, under revalidation or
    active is read from `data/factor_registry.json` and nowhere else, so
    promotion happens by clearing that gate, never by editing this code.
"""
from .assemble import GameContext, assemble_game_context  # noqa: F401
from .contracts import (Availability, ContextValue, FieldSpec,  # noqa: F401
                        Usage, FIELDS, FAMILIES)
from .registry import PRESERVED_EXCLUSIONS, FactorRegistry  # noqa: F401
from .sources import SourceManifest  # noqa: F401

__all__ = ["GameContext", "assemble_game_context", "Availability",
           "ContextValue", "FieldSpec", "Usage", "FIELDS", "FAMILIES",
           "PRESERVED_EXCLUSIONS", "FactorRegistry", "SourceManifest"]
