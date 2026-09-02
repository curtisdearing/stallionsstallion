"""The one public forecast contract. Schema version 2.

Before this module there were two contracts wearing one filename.  The
early-season writer emitted `pred_margin` + `band_pm`; the weekly loop emitted
`mc_margin` / `mc_margin_sd` / `mc_total` *plus* `pred_margin` and `band_pm` as
aliases; the hub feed relabelled a ridge point estimate as `mc_margin` when the
Monte-Carlo column was absent; and `band_pm` meant rating-parameter uncertainty
(~3 points) in one route and simulation outcome uncertainty (~16-18 points) in
the other -- under the same column name, in the same file, at the same
`schema_version`.  A reader could not tell which it held, and the two are not
the same kind of quantity.

Version 2 removes the ambiguity rather than preserving it:

* **Names say what the number is.**  `projected_home_margin` is the forecast.
  `forecast_margin_sd` is the SD of the predictive distribution that forecast
  came from.  `rating_parameter_sd` is the separate, technical, much smaller
  quantity -- how unsure the rating parameters are -- and is never the
  forecast's uncertainty.  There is no `band_pm`.
* **No Monte-Carlo name for a non-Monte-Carlo number.**  A ridge point estimate
  is `projected_home_margin` produced by `model_family="ridge_preseason_prior"`.
  Nothing anywhere may call it `mc_margin`.
* **Mean, SD and p(home win) come off one object.**  :class:`PredictiveMargin`
  is the only way to supply them, and its constructors derive all three from a
  single distribution.  A caller cannot pass a probability computed from one
  distribution beside an SD computed from another, because there is no
  parameter through which to do it.
* **A number needs validated out-of-sample backing for the week it prices.**
  Every forecast-bearing row names a `forecast_sd_basis`, and
  :class:`BasisRegistry` refuses to issue one that does not cover that week for
  that model family.  Weeks 0-3 had no such evidence at all until
  `scripts/backtest_early.py` produced it (pre-registered 2026-08-30T17:20Z);
  before that the early route was using the week-4-15 residual SD, which is an
  extrapolation, not a validated distribution.
* **Numbers and refusals are mutually exclusive.**  `status` is a four-value
  enum, and only `priced` and `frozen` rows may carry forecast numbers.
  `refused` and `degraded` rows carry none, and must say why.

PAPER-ONLY. Every field here describes what the model projected and how
uncertain it is. Nothing in this contract is, or can be rendered as, a bet, an
edge, a stake, a recommendation or an actionable pick.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd

from . import cfb_calendar

#: Bumped from 1. Version 1 files are rejected, not migrated: the ambiguity
#: this replaces cannot be resolved by reading a v1 row, because `band_pm`
#: does not record which of its two meanings it carried.
SCHEMA_VERSION = 2


class ContractError(RuntimeError):
    """A frame or row does not satisfy the public forecast contract."""


class Status(str, Enum):
    """What this row is. Exactly one of five things, always stated."""

    PRICED = "priced"      #: a live pregame forecast; numbers present
    FROZEN = "frozen"      #: a pre-kickoff forecast republished verbatim
    REFUSED = "refused"    #: the model declined this game; no numbers
    DEGRADED = "degraded"  #: the run could not stand behind a number; none given
    #: a margin model exists for this week but no validated out-of-sample SD
    #: backs it; publishing a number here would be precision the run cannot
    #: support, so none is given (never a rating-parameter band standing in
    #: for it).
    UNCERTAINTY_UNVALIDATED = "uncertainty_unvalidated"

    def __str__(self) -> str:  # so DataFrame cells are plain strings
        return self.value


#: The two statuses that may carry forecast numbers. Everything else must not.
FORECAST_BEARING = frozenset({Status.PRICED.value, Status.FROZEN.value})

#: Every status that must carry a reason.
REASON_BEARING = frozenset({Status.REFUSED.value, Status.DEGRADED.value,
                           Status.UNCERTAINTY_UNVALIDATED.value})

#: Statuses that mean the WHOLE run could not stand behind a number, as
#: distinct from `refused` -- a normal per-game decision made even in a
#: healthy run. `write_snapshot` uses this to confirm a caller's
#: `degraded=True` is actually reflected in the rows it was handed.
RUN_DEGRADED_STATUSES = frozenset(
    {Status.DEGRADED.value, Status.UNCERTAINTY_UNVALIDATED.value})


@dataclass(frozen=True)
class ModelFamily:
    """A family of models allowed to produce public rows."""

    name: str
    supports_totals: bool
    description: str


#: Families this project ships. A row naming anything else is not a row this
#: contract can validate, and validation says so rather than guessing.
FAMILIES: Dict[str, ModelFamily] = {
    "ridge_preseason_prior": ModelFamily(
        "ridge_preseason_prior", supports_totals=False,
        description=("Weeks 0-3. Ridge ratings fitted around the preseason "
                     "prior; a point estimate for margin with an SD taken from "
                     "the registered early-week out-of-sample residuals. It "
                     "runs no simulation, so it projects no total.")),
    "kalman_montecarlo": ModelFamily(
        "kalman_montecarlo", supports_totals=True,
        description=("Weeks 4+. Kalman ratings driving the drive-level Monte "
                     "Carlo; margin, SD, win probability and total all read "
                     "off the same simulated sample.")),
    "ridge_cross_division": ModelFamily(
        "ridge_cross_division", supports_totals=False,
        description=("Any week, FBS-vs-FCS only. The ridge table's FBS "
                     "rating against the pooled __FCS__ pseudo-team's own "
                     "fitted coefficients; SD from the registered "
                     "cross-division out-of-sample residuals "
                     "(scripts/backtest_cross_division.py). Runs no "
                     "simulation, so it projects no total. Rows carrying "
                     "this family sit inside an artifact whose provenance "
                     "names the week's main route; the per-row family says "
                     "which model actually made the number.")),
}

# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #
IDENTITY_COLUMNS = (
    "schema_version", "season", "week", "game_id", "home", "away",
    "kickoff_utc", "neutral_site",
)

#: Derived public labels. Not identity -- both come from `kickoff_utc` through
#: `cfb_calendar`, which is the ONE definition of the Eastern calendar day a
#: game belongs to. They exist because a late-night Saturday kickoff is a
#: Saturday game to a reader and a Sunday game to UTC.
LABEL_COLUMNS = ("slate", "public_date")

STATUS_COLUMNS = ("status", "refusal_reason")

MODEL_COLUMNS = ("model_family", "model_version")

#: Every number a public row may carry. All null together, or all present
#: together except the two that are legitimately optional
#: (`projected_total`, `rating_parameter_sd`).
FORECAST_COLUMNS = (
    "projected_home_margin", "forecast_margin_sd", "p_home_win",
    "projected_total", "rating_parameter_sd",
)

#: The three that must be present on any forecast-bearing row.
REQUIRED_FORECAST_COLUMNS = (
    "projected_home_margin", "forecast_margin_sd", "p_home_win",
)

PROVENANCE_COLUMNS = (
    "forecast_sd_basis", "generated_at_utc", "snapshot_id", "source_digest",
    "config_hash", "frozen_from_stamp",
)

COLUMNS = (IDENTITY_COLUMNS + LABEL_COLUMNS + STATUS_COLUMNS + MODEL_COLUMNS
           + FORECAST_COLUMNS + PROVENANCE_COLUMNS)

#: Version-1 names. Their presence is not a warning, it is a hard failure:
#: a frame carrying them is either a v1 artifact or a v2 artifact something
#: has written v1 fields back into, and both mean a reader cannot know which
#: meaning a number has. Mapped to what replaced them so the error can say so.
BANNED_COLUMNS: Dict[str, str] = {
    "band_pm": ("ambiguous in v1: rating-parameter SD in the ridge route, "
                "simulation margin SD in the weekly route. Use "
                "`forecast_margin_sd` for the predictive SD and "
                "`rating_parameter_sd` for the technical one"),
    "mc_margin": "use `projected_home_margin` (and `model_family` to say how)",
    "mc_margin_sd": "use `forecast_margin_sd`",
    "mc_total": "use `projected_total`",
    "pred_margin": "use `projected_home_margin`",
    "param_sd": "use `rating_parameter_sd`",
    "veto": "use `status` (`refused`) and `refusal_reason`",
    "forecast_status": "use `status` plus `model_family`",
    "engine": "use `model_family` and `model_version`",
    "id": "use `game_id`",
    "neutral": "use `neutral_site`",
    "date": "use `public_date` (and `kickoff_utc` for the instant)",
}

#: Tolerance on the agreement between `p_home_win` and the normal implied by
#: `projected_home_margin` / `forecast_margin_sd`.
#:
#: Not a guess. Over the three registered simulation records
#: (backtest_c_kalman n=1,766; backtest_c n=1,765; backtest_c_2024 n=612 --
#: 4,143 games) the largest observed |p - Phi(mu/sd)| is 0.018 and the 99th
#: percentile is 0.012: the drive-level Monte Carlo's margin distribution is
#: near-Gaussian around the win/loss boundary. 0.05 sits well clear of that
#: real skew while still catching the failure this check exists for -- a
#: probability computed from one distribution published beside an SD from
#: another.
P_COHERENCE_TOL = 0.05


# --------------------------------------------------------------------------- #
# The predictive distribution
# --------------------------------------------------------------------------- #
def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass(frozen=True)
class PredictiveMargin:
    """One predictive distribution over home margin.

    The point of this type is that `mean`, `sd` and `p_home_win` are read off
    the SAME distribution. There is no constructor that accepts a probability
    from the caller, so the incoherence the old artifact shipped -- a win
    probability formed from the week-4-15 residual SD, published beside a
    rating band that described something else entirely -- has no way in.
    """

    mean: float
    sd: float
    p_home_win: float
    basis_id: str
    total: Optional[float] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ContractError(f"projected margin {self.mean!r} is not finite")
        if not math.isfinite(self.sd) or self.sd <= 0:
            raise ContractError(
                f"forecast_margin_sd {self.sd!r} must be finite and positive; "
                "a forecast with no spread is not a forecast")
        if not 0.0 < self.p_home_win < 1.0:
            raise ContractError(
                f"p_home_win {self.p_home_win!r} must lie strictly in (0, 1)")
        if not self.basis_id:
            raise ContractError(
                "a forecast must name the validated out-of-sample basis its "
                "SD comes from")

    @classmethod
    def from_normal(cls, mean: float, sd: float, basis_id: str
                    ) -> "PredictiveMargin":
        """A point estimate plus a validated residual SD, read as a normal.

        This is the early-season ridge route: the model gives a margin, and
        the width comes from how wrong that kind of forecast has been out of
        sample. `p_home_win` follows from those two and nothing else.
        """
        mean, sd = float(mean), float(sd)
        return cls(mean=mean, sd=sd, p_home_win=_phi(mean / sd),
                   basis_id=basis_id, total=None)

    @classmethod
    def from_samples(cls, margins: Sequence[float], basis_id: str,
                     totals: Optional[Sequence[float]] = None
                     ) -> "PredictiveMargin":
        """The simulated sample itself: every statistic from one array."""
        series = pd.Series(list(margins), dtype="float64")
        if len(series) < 2:
            raise ContractError(
                "a simulated predictive distribution needs at least two draws")
        wins = float((series > 0).mean())
        # Overtime resolves ties, so P(margin > 0) is the win probability. A
        # tie in the sample would silently deflate it, so say so instead.
        if bool((series == 0).any()):
            raise ContractError(
                "simulated margins contain exact ties; overtime is supposed to "
                "resolve them, and P(margin > 0) would understate p_home_win")
        return cls(mean=float(series.mean()), sd=float(series.std(ddof=1)),
                   p_home_win=wins, basis_id=basis_id,
                   total=(None if totals is None
                          else float(pd.Series(list(totals),
                                               dtype="float64").mean())))

    @classmethod
    def from_summary(cls, mean: float, sd: float, p_home_win: float,
                     basis_id: str, total: Optional[float] = None
                     ) -> "PredictiveMargin":
        """Rebuild from already-computed summary statistics.

        For readers reconstructing a published row, and for producers whose
        simulator returns summaries rather than the raw sample. Coherence is
        checked rather than assumed -- unlike the other constructors this one
        *can* be handed mismatched inputs, so it refuses them.
        """
        mean, sd, p_home_win = float(mean), float(sd), float(p_home_win)
        implied = _phi(mean / sd) if sd > 0 else float("nan")
        if not math.isfinite(implied) or abs(p_home_win - implied) > P_COHERENCE_TOL:
            raise ContractError(
                f"p_home_win {p_home_win:.4f} is not coherent with "
                f"margin {mean:.3f} / sd {sd:.3f} (implies {implied:.4f}, "
                f"tolerance {P_COHERENCE_TOL}). These must describe one "
                "predictive distribution, not two")
        return cls(mean=mean, sd=sd, p_home_win=p_home_win, basis_id=basis_id,
                   total=(None if total is None else float(total)))


@dataclass(frozen=True)
class Forecast:
    """What a producer supplies for one priced game.

    `model_family` / `model_version` are per-row overrides for a game that
    the artifact's main route did not price -- the cross-division ridge row
    inside a Kalman week, say. Left None, the row inherits the artifact's
    provenance. An override must name a registered family; the validator
    then judges the row (totals, basis) by the family the ROW names, so a
    ridge row cannot borrow the simulation's permission to publish a total.
    """

    predictive: PredictiveMargin
    rating_parameter_sd: Optional[float] = None
    model_family: Optional[str] = None
    model_version: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.model_family is None) != (self.model_version is None):
            raise ContractError(
                "a per-row family override must carry BOTH model_family and "
                "model_version: a family without the fitted model that "
                "produced the number, or the reverse, is half a provenance")
        if self.model_family is not None and self.model_family not in FAMILIES:
            raise ContractError(
                f"unknown per-row model_family {self.model_family!r}; known "
                f"families are {sorted(FAMILIES)}")


@dataclass(frozen=True)
class Provenance:
    """Who produced this artifact, from what, and when."""

    model_family: str
    model_version: str
    generated_at_utc: str
    snapshot_id: Optional[str] = None
    source_digest: Optional[str] = None
    config_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if self.model_family not in FAMILIES:
            raise ContractError(
                f"unknown model_family {self.model_family!r}; known families "
                f"are {sorted(FAMILIES)}")
        if not self.model_version:
            raise ContractError(
                "model_version must identify the fitted model that produced "
                "these rows")
        if not self.generated_at_utc:
            raise ContractError("generated_at_utc is required")


# --------------------------------------------------------------------------- #
# The out-of-sample basis registry
# --------------------------------------------------------------------------- #
class BasisRegistry:
    """The registered out-of-sample evidence, and which weeks it covers.

    This is where "validated" stops being an adjective in a docstring. A
    producer asks for a basis by (family, week); if no usable registered basis
    covers that week for that family, it does not get one, and the contract
    will not let it publish a number.
    """

    def __init__(self, payload: Mapping[str, Any], source: str = "") -> None:
        self._bases = list(payload.get("bases") or [])
        self.source = source
        self.findings = list(payload.get("findings") or [])

    @classmethod
    def load(cls, data_dir: str) -> "BasisRegistry":
        path = os.path.join(data_dir, "forecast_sd_basis.json")
        if not os.path.exists(path):
            raise ContractError(
                f"no forecast-SD basis manifest at {path}. Nothing may be "
                "priced without registered out-of-sample evidence for the week "
                "it prices; run scripts/backtest_early.py")
        with open(path) as fh:
            return cls(json.load(fh), source=path)

    def find(self, model_family: str, week: int) -> Optional[dict]:
        for basis in self._bases:
            if basis.get("model_family") != model_family:
                continue
            if int(week) in {int(w) for w in basis.get("covers_weeks") or ()}:
                return basis
        return None

    def require(self, model_family: str, week: int) -> dict:
        basis = self.find(model_family, week)
        if basis is None:
            raise ContractError(
                f"no registered out-of-sample basis covers week {week} for "
                f"model_family {model_family!r} (manifest: {self.source}). "
                "Publishing a forecast SD for a week no evidence covers is an "
                "extrapolation wearing a forecast's clothes")
        if not basis.get("usable", False):
            raise ContractError(
                f"basis {basis.get('basis_id')!r} covers week {week} but is "
                f"marked unusable: {basis.get('note') or 'see the manifest'}")
        return basis

    def margin_sd(self, model_family: str, week: int) -> Optional[float]:
        """The pooled SD for this family/week, or None if it is per-game."""
        basis = self.require(model_family, week)
        value = basis.get("forecast_margin_sd")
        return None if value is None else float(value)

    def basis_id(self, model_family: str, week: int) -> str:
        return str(self.require(model_family, week)["basis_id"])


# --------------------------------------------------------------------------- #
# The shared artifact builder
# --------------------------------------------------------------------------- #
DEFAULT_REFUSALS = {
    "non_fbs": "Non-FBS matchup: no validated lower-division model",
    "cross_division": "FBS-vs-non-FBS: no validated cross-division model",
    "unrated": "NO RATING: team lacks a validated FBS rating",
}


def _row_skeleton(season: int, week: int, game, provenance: Provenance) -> dict:
    kickoff = getattr(game, "start_date")
    slate, public_date = cfb_calendar.public_slate(season, week, kickoff)
    return {
        "schema_version": SCHEMA_VERSION,
        "season": int(season),
        "week": int(week),
        "game_id": int(game.id),
        "home": str(game.home_team),
        "away": str(game.away_team),
        "kickoff_utc": pd.to_datetime(kickoff, utc=True, errors="raise")
                         .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "neutral_site": bool(game.neutral_site),
        "slate": slate,
        "public_date": public_date,
        "status": None,
        "refusal_reason": None,
        "model_family": provenance.model_family,
        "model_version": provenance.model_version,
        "projected_home_margin": None,
        "forecast_margin_sd": None,
        "p_home_win": None,
        "projected_total": None,
        "rating_parameter_sd": None,
        "forecast_sd_basis": None,
        "generated_at_utc": provenance.generated_at_utc,
        "snapshot_id": provenance.snapshot_id,
        "source_digest": provenance.source_digest,
        "config_hash": provenance.config_hash,
        "frozen_from_stamp": None,
    }


def build_artifact(season: int, week: int, schedule: pd.DataFrame,
                   forecasts: Mapping[int, Forecast],
                   refusals: Mapping[int, str],
                   provenance: Provenance,
                   *, degraded: bool = False,
                   degraded_reason: Optional[str] = None,
                   degraded_status: Status = Status.DEGRADED) -> pd.DataFrame:
    """One public row per scheduled game, in the one public contract.

    Both routes come through here. The early-season ridge producer and the
    in-season simulation producer differ in how they build a
    :class:`PredictiveMargin` and in nothing else -- which is the whole point:
    the schema cannot drift between them, because there is only one place that
    writes it.

    The row set is exactly the schedule: a public artifact that is not its own
    schedule is either hiding a game or inventing one.
    """
    from . import engine as EG  # local: avoids a package-level import cycle

    if degraded and not degraded_reason:
        raise ContractError(
            "a degraded artifact must say what degraded it; publishing "
            "blank numbers with no reason is indistinguishable from a model "
            "that simply declined")
    if degraded_status.value not in RUN_DEGRADED_STATUSES:
        raise ContractError(
            f"degraded_status={degraded_status.value!r} is not a whole-run "
            f"degraded status; expected one of {sorted(RUN_DEGRADED_STATUSES)}")

    family = FAMILIES.get(provenance.model_family)
    if family is None:
        raise ContractError(f"unknown model_family {provenance.model_family!r}")

    rows = []
    for game in schedule.sort_values(["start_date", "id"],
                                     kind="mergesort").itertuples():
        row = _row_skeleton(season, week, game, provenance)
        game_id = int(game.id)
        forecast = None if degraded else forecasts.get(game_id)
        if forecast is not None:
            predictive = forecast.predictive
            row_family = family
            if forecast.model_family is not None:
                row_family = FAMILIES[forecast.model_family]
                row["model_family"] = forecast.model_family
                row["model_version"] = forecast.model_version
            if predictive.total is not None and not row_family.supports_totals:
                raise ContractError(
                    f"model_family {row_family.name!r} runs no simulation and "
                    "cannot project a total, but a total was supplied for "
                    f"game {game_id}")
            row.update({
                "status": Status.PRICED.value,
                "projected_home_margin": round(float(predictive.mean), 3),
                "forecast_margin_sd": round(float(predictive.sd), 3),
                "p_home_win": round(float(predictive.p_home_win), 4),
                "projected_total": (None if predictive.total is None
                                    else round(float(predictive.total), 3)),
                "rating_parameter_sd": (
                    None if forecast.rating_parameter_sd is None
                    else round(float(forecast.rating_parameter_sd), 3)),
                "forecast_sd_basis": predictive.basis_id,
            })
        elif degraded:
            row.update({"status": degraded_status.value,
                        "refusal_reason": degraded_reason})
        else:
            row.update({
                "status": Status.REFUSED.value,
                "refusal_reason": refusals.get(game_id)
                or DEFAULT_REFUSALS["unrated"],
            })
        rows.append(row)

    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    problem = EG.id_coverage_problem(frame["game_id"], schedule["id"])
    if problem:
        raise ContractError(
            f"{problem} — refusing to build a public artifact that is not its "
            "own schedule")
    validate_frame(frame, where="build_artifact")
    return frame


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return value is pd.NA or (isinstance(value, str) and not value.strip())


def validate_frame(frame: pd.DataFrame, *, where: str = "") -> pd.DataFrame:
    """Raise unless every row satisfies the contract. Returns the frame.

    Loud on purpose, and loud *before* anything is written or read downstream.
    A mixed-schema artifact is not a degraded artifact that can be partly
    trusted -- it is one in which no reader can tell what a number means.
    """
    tag = f"{where}: " if where else ""

    if frame is None:
        raise ContractError(f"{tag}no frame at all")

    banned = [c for c in frame.columns if c in BANNED_COLUMNS]
    if banned:
        detail = "; ".join(f"{c} -> {BANNED_COLUMNS[c]}" for c in banned)
        raise ContractError(
            f"{tag}schema-version-1 columns present: {sorted(banned)}. This "
            f"artifact mixes contracts and cannot be read unambiguously. {detail}")

    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ContractError(
            f"{tag}missing contract columns {missing}. Expected exactly "
            f"schema_version {SCHEMA_VERSION}: {list(COLUMNS)}")

    extra = [c for c in frame.columns if c not in COLUMNS]
    if extra:
        raise ContractError(
            f"{tag}unrecognised columns {sorted(extra)}. The public contract "
            "is closed: a field no reader knows how to interpret must not "
            "ride along in a published artifact")

    if frame.empty:
        return frame

    versions = set(frame["schema_version"].dropna().astype(int).unique())
    if versions != {SCHEMA_VERSION}:
        raise ContractError(
            f"{tag}schema_version(s) {sorted(versions)} present; this reader "
            f"handles exactly {SCHEMA_VERSION}. Mixed-schema data is rejected, "
            "never merged")

    seen_ids = set()
    for position, record in enumerate(frame.to_dict("records"), start=1):
        _validate_row(record, f"{tag}row {position}")
        game_id = int(record["game_id"])
        if game_id in seen_ids:
            raise ContractError(
                f"{tag}game_id {game_id} appears more than once; two rows "
                "claiming one game publish an ambiguous identity")
        seen_ids.add(game_id)
    return frame


def _validate_row(record: Mapping[str, Any], where: str) -> None:
    status = record.get("status")
    if status not in {s.value for s in Status}:
        raise ContractError(
            f"{where}: status {status!r} is not one of "
            f"{sorted(s.value for s in Status)}")

    family_name = record.get("model_family")
    family = FAMILIES.get(family_name)
    if family is None:
        raise ContractError(
            f"{where}: unknown model_family {family_name!r}; known families "
            f"are {sorted(FAMILIES)}")
    if _is_null(record.get("model_version")):
        raise ContractError(f"{where}: model_version is required")

    for field in ("season", "week", "game_id"):
        value = record.get(field)
        if _is_null(value) or float(value) != int(float(value)):
            raise ContractError(
                f"{where}: {field}={value!r} must be a whole number; a row "
                "without exact identity names a game that is not the game "
                "priced")
    for field in ("home", "away", "kickoff_utc", "public_date", "slate"):
        if _is_null(record.get(field)):
            raise ContractError(f"{where}: {field} is required")

    bearing = status in FORECAST_BEARING
    present = [c for c in FORECAST_COLUMNS if not _is_null(record.get(c))]

    if not bearing:
        if present:
            raise ContractError(
                f"{where}: status={status} but forecast fields {present} carry "
                "values. A refused or degraded row states that no number is "
                "being published; carrying one anyway is the contradiction "
                "this contract exists to make impossible")
        if _is_null(record.get("refusal_reason")):
            raise ContractError(
                f"{where}: status={status} requires a refusal_reason")
        if not _is_null(record.get("forecast_sd_basis")):
            raise ContractError(
                f"{where}: status={status} must not name a forecast_sd_basis")
        return

    if not _is_null(record.get("refusal_reason")):
        raise ContractError(
            f"{where}: status={status} carries a refusal_reason "
            f"({record['refusal_reason']!r}); a priced row is not a refusal")

    missing = [c for c in REQUIRED_FORECAST_COLUMNS if _is_null(record.get(c))]
    if missing:
        raise ContractError(
            f"{where}: status={status} but {missing} are empty. A priced row "
            "publishes a margin, the spread of the distribution it came from, "
            "and the win probability implied by that same distribution")
    if _is_null(record.get("forecast_sd_basis")):
        raise ContractError(
            f"{where}: status={status} must name the registered "
            "out-of-sample basis its forecast_margin_sd comes from")

    if not family.supports_totals and not _is_null(record.get("projected_total")):
        raise ContractError(
            f"{where}: model_family {family.name!r} runs no simulation and "
            "cannot support projected_total, but one is published")

    # Coherence: the published probability must belong to the published
    # distribution. This is the check that would have caught a win probability
    # formed from the week-4-15 residual SD sitting beside a rating band.
    margin = float(record["projected_home_margin"])
    sd = float(record["forecast_margin_sd"])
    probability = float(record["p_home_win"])
    if not math.isfinite(sd) or sd <= 0:
        raise ContractError(
            f"{where}: forecast_margin_sd {sd!r} must be finite and positive")
    if not 0.0 < probability < 1.0:
        raise ContractError(
            f"{where}: p_home_win {probability!r} must lie strictly in (0, 1)")
    implied = _phi(margin / sd)
    if abs(probability - implied) > P_COHERENCE_TOL:
        raise ContractError(
            f"{where}: p_home_win {probability:.4f} does not belong to the "
            f"published distribution (margin {margin:.3f}, sd {sd:.3f} implies "
            f"{implied:.4f}; tolerance {P_COHERENCE_TOL}). These must be one "
            "predictive distribution, not two")

    for column in FORECAST_COLUMNS:
        value = record.get(column)
        if not _is_null(value) and not math.isfinite(float(value)):
            raise ContractError(
                f"{where}: {column}={value!r} is not finite. NaN and infinity "
                "are not forecasts, and JSON cannot carry them")

    rating_sd = record.get("rating_parameter_sd")
    if not _is_null(rating_sd) and float(rating_sd) < 0:
        raise ContractError(
            f"{where}: rating_parameter_sd {rating_sd!r} cannot be negative")


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
_BOOL_TRUE = {"true", "1", "yes", "t"}
_BOOL_FALSE = {"false", "0", "no", "f", ""}

_NUMERIC = ("projected_home_margin", "forecast_margin_sd", "p_home_win",
            "projected_total", "rating_parameter_sd")


def coerce_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Types a CSV round trip loses, restored. Does not validate.

    Identity casts phrase their own failures. A fractional or infinite id
    makes pandas raise `TypeError`/`OverflowError` from inside `astype`,
    which reaches a consumer as a library error about float64 rather than as
    "this artifact names a game that is not a game" -- and a consumer whose
    stated error contract is `ContractError` would then fail in a way it does
    not document. The read path owns that message.
    """
    out = frame.copy()
    for column in ("schema_version", "season", "week", "game_id"):
        if column not in out.columns:
            continue
        numeric = pd.to_numeric(out[column], errors="coerce")
        finite = numeric.notna()
        bad_inf = finite & ~numeric.apply(math.isfinite)
        if bool(bad_inf.any()):
            raise ContractError(
                f"{column} is not finite in row(s) "
                f"{list(out.index[bad_inf])[:5]}; identity must be an exact "
                "whole number")
        bad_frac = finite & (numeric != numeric.round())
        if bool(bad_frac.any()):
            raise ContractError(
                f"{column} is fractional in row(s) "
                f"{list(out.index[bad_frac])[:5]}; truncating it would name a "
                "different game than the one priced")
        try:
            out[column] = numeric.astype("Int64")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ContractError(
                f"{column} could not be read as an exact whole number: "
                f"{exc}") from None
    for column in _NUMERIC:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if "neutral_site" in out.columns:
        out["neutral_site"] = out["neutral_site"].map(_as_bool)
    for column in ("status", "refusal_reason", "model_family", "model_version",
                   "home", "away", "kickoff_utc", "slate", "public_date",
                   "forecast_sd_basis", "generated_at_utc", "snapshot_id",
                   "source_digest", "config_hash", "frozen_from_stamp"):
        if column in out.columns:
            out[column] = out[column].where(out[column].notna(), None)
    return out


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _BOOL_TRUE:
        return True
    if text in _BOOL_FALSE:
        return False
    raise ContractError(f"{value!r} is not a boolean")


def read_artifact(path: str) -> pd.DataFrame:
    """Read a published artifact and validate it before handing it over.

    Every consumer goes through here. That is the point: the old readers each
    had their own tolerance for a missing or differently-named column, so a
    mixed-schema file could be read three different ways by three consumers
    and disagree with itself in public.
    """
    if not os.path.exists(path):
        raise ContractError(f"no forecast artifact at {path}")
    raw = pd.read_csv(path, keep_default_na=True)
    frame = coerce_frame(raw)
    return validate_frame(frame, where=os.path.basename(path))


def write_artifact(frame: pd.DataFrame, path: str) -> str:
    """Validate, then write atomically. Nothing invalid reaches disk."""
    validate_frame(frame, where=f"write_artifact -> {os.path.basename(path)}")
    tmp = path + ".tmp"
    frame.reindex(columns=list(COLUMNS)).to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def forecast_bearing(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows that actually carry a number."""
    return frame[frame["status"].isin(FORECAST_BEARING)]


def summarise(frame: pd.DataFrame) -> Dict[str, Any]:
    """Counts by status, for manifests and the hub feed."""
    counts = {status.value: int((frame["status"] == status.value).sum())
              for status in Status}
    return {"schema_version": SCHEMA_VERSION, "n_rows": int(len(frame)),
            "by_status": counts,
            "model_families": sorted(set(frame["model_family"].dropna())),
            "bases": sorted(set(frame["forecast_sd_basis"].dropna()))}
