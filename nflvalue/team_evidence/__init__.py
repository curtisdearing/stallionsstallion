"""Team-season evidence layer — current-season team profiles, pregame only.

WHAT THIS IS
------------
A display-only evidence layer. It turns the cached CFBD plays/drives/games tables
into per-team-season profiles built from information available *before* each
forecast, with a full provenance envelope on every single number.

WHAT THIS IS NOT
----------------
It is not a rating engine and it is not a forecast module. It produces no projected
margin, no total, no win probability and no point adjustment. Opponent adjustment
is not reimplemented here: rush/pass efficiency and explosiveness are adjusted by
calling the EXISTING `ratings.solve_units`, and every other measure ships
opponent-UNadjusted and says so on the record.

WHY IT IS A SEPARATE PACKAGE
----------------------------
`nflvalue/ratings.py`, `engine.py`, `kalman_engine.py`, `cfb_sim.py`, `cfb_ot.py`
and `config.json` are sha256-pinned by `data/evidence_manifest.json` and back the
repo's only `selected` benchmark. A helper added to any of them would un-certify
that record without changing one published number. So this package consumes them
and writes nothing back — the pattern `nflvalue/cross_division.py` and
`nflvalue/early_strength_candidate.py` already established.

THE ONE THING TO READ BEFORE USING IT
-------------------------------------
No cached CFBD record carries a retrieval time, an effective time, a version or a
hash. Availability of a past stat is therefore a RECONSTRUCTION at (season, week)
granularity, not a proof, and every observation says so in `availability_class`.
Retrospective PREDICTIVE claims off reconstructed observations are blocked by
`contracts.assert_retrospective_claim_allowed`. See `docs/final_score_program/
00_contract.md` §3.
"""
from __future__ import annotations

from .contracts import (  # noqa: F401
    AvailabilityClass,
    EvidenceAvailability,
    EvidenceContractError,
    Missingness,
    Observation,
    RetrospectiveClaimBlocked,
    TeamSeasonProfile,
    assert_retrospective_claim_allowed,
)
from .registry import MeasureRegistry, MeasureSpec, RegistryError  # noqa: F401

__all__ = [
    "AvailabilityClass",
    "EvidenceAvailability",
    "EvidenceContractError",
    "Missingness",
    "Observation",
    "RetrospectiveClaimBlocked",
    "TeamSeasonProfile",
    "assert_retrospective_claim_allowed",
    "MeasureRegistry",
    "MeasureSpec",
    "RegistryError",
]
