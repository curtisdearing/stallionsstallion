"""One forecast, understandable in twenty seconds. Deterministic, no prose.

The explainer this replaces as the published surface had grown to carry
per-game margin histograms, deciles, evidence tiers, excluded-input rows,
model-vs-market error comparisons, CLV slices and a free-text "why not"
engine — several of which contradicted each other, several of which sourced
numbers from artifacts nobody had verified, and one of which rendered an
evidence tier for inputs that contributed nothing precisely so it could say
so. A reader could not get to the forecast.

A card has exactly four sections and no fifth:

1. **Call** — who is projected ahead, by how much, the home-win probability,
   and the predictive spread. The spread is `forecast_margin_sd` from the
   published contract row, so the number here and the number in the artifact
   cannot disagree.
2. **Why** — the rating differential, the home-field term where one applies,
   and a simulation adjustment ONLY when a simulation actually produced one.
   The components must sum to the published margin; if they do not, the card
   refuses to build rather than showing arithmetic the reader could check and
   find wrong.
3. **Provenance** — model family and version, generated-at, snapshot id,
   source digest, data-as-of.
4. **Caution** — one specific, deterministic reason, chosen by a fixed
   priority. Never empty and never generic when something specific applies.

Rules the module enforces rather than documents:

* **No generated prose.** Every string is a constant or a format of one.
  There is no path by which a model writes a sentence here.
* **No actionable vocabulary.** :func:`assert_non_actionable` walks every
  string in a finished card against a word list and raises. It runs inside
  `build_card`, so a card containing that vocabulary cannot be returned, let
  alone published.
* **Nothing renders without provenance.** A context field, a market
  reference or a why-component with no source is omitted, not shown empty. An
  empty framework tells the reader something is there when nothing is.
* **The market is never described as absent when it is cached.** The card
  either carries a timestamped, explicitly non-actionable reference with its
  source, or states that market comparison is switched off for this surface.
  "No line cached" is not one of the things this module can say.
* **Byte-stable.** Canonical JSON, fixed rounding, no clock reads: every
  timestamp comes from the row. Identical inputs give identical bytes.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import forecast_contract as FC

SCHEMA_VERSION = 1

#: Vocabulary a paper-only research surface must not contain. Matched on word
#: boundaries so "acknowledge" survives and "edge" does not. Checked against
#: the finished card, which is the only place it matters -- a term is fine in
#: this module's own source, and banned in its output.
FORBIDDEN_TERMS = (
    "pick", "picks", "bet", "bets", "betting", "bettor", "wager", "wagers",
    "edge", "edges", "value", "values", "stake", "stakes", "staking",
    "bankroll", "confidence", "confident", "recommend", "recommends",
    "recommended", "recommendation", "recommendations", "lock", "play",
    "plays", "action", "parlay", "tout", "advice",
)
_FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in FORBIDDEN_TERMS) + r")\b",
    re.IGNORECASE)

#: A predictive spread wider than this is called out. Not invented: the
#: registered early-week basis is 16.898 points and the registered week-4-15
#: anchor 16.662, so a game whose own simulated spread exceeds 17 is wider
#: than the typical spread of either registered window.
WIDE_SD_POINTS = 17.0

#: Below this ratio the projection is small relative to its own spread.
#: |margin| < sd/2 corresponds to a home-win probability inside roughly
#: 0.31-0.69 under the normal the contract's coherence check already assumes.
NARROW_MARGIN_RATIO = 0.5


class InsightError(RuntimeError):
    """A card could not be built without saying something untrue."""


# --------------------------------------------------------------------------- #
# Vocabulary guard
# --------------------------------------------------------------------------- #
#: Fields whose strings are DATA carried through verbatim, not text this
#: module wrote. A digest, a stamp or an external source id is not a
#: statement to anybody, and scanning it produces false positives without
#: catching anything.
_OPAQUE_KEYS = frozenset({
    "model_version", "snapshot_id", "source_digest", "forecast_sd_basis",
    "kickoff_utc", "public_date", "generated_at_utc", "data_as_of_utc",
    "observed_at_utc", "frozen_from_stamp", "source_ref", "slate",
})


def _strings(node: Any) -> List[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, Mapping):
        out = []
        for key, item in node.items():
            out.append(str(key))
            if str(key) in _OPAQUE_KEYS:
                continue
            out.extend(_strings(item))
        return out
    if isinstance(node, (list, tuple)):
        out = []
        for item in node:
            out.extend(_strings(item))
        return out
    return []


def assert_non_actionable(card: Any, *,
                          proper_nouns: Sequence[str] = ()) -> Any:
    """Raise if any text in `card` carries actionable vocabulary.

    `proper_nouns` — the teams in this game — are removed before matching.
    They are data, and a team really is called Lock Haven. A guard that
    refuses to publish a forecast because of the opponent's name is a guard
    someone will switch off, so it checks the words this module chose and not
    the words it was handed.
    """
    scrubbed = []
    ordered = sorted((n for n in proper_nouns if n), key=len, reverse=True)
    for text in _strings(card):
        for noun in ordered:
            text = text.replace(noun, " ")
        scrubbed.append(text)
    hits = sorted({m.group(0).lower()
                   for text in scrubbed
                   for m in _FORBIDDEN_RE.finditer(text)})
    if hits:
        raise InsightError(
            f"card contains actionable vocabulary {hits}. This surface "
            "describes what the model projected and how uncertain it is; it "
            "does not tell anyone to do anything")
    return card


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WhyComponent:
    """One signed contribution to the published margin, with its source."""

    key: str
    label: str
    points: float
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "label": self.label,
                "points": round(float(self.points), 2), "source": self.source}


@dataclass(frozen=True)
class MarketReference:
    """A cached market observation, or an explicit statement that it is off.

    There is no third state. The failure this removes is a surface that says
    "no market line cached" while `lines_2026.parquet` sits on disk: the
    reader concludes the data is missing when the truth is that nothing looked.
    """

    state: str                                  # "available" | "disabled"
    home_spread: Optional[float] = None
    observed_at_utc: Optional[str] = None
    source: Optional[str] = None
    n_books: Optional[int] = None
    reason: str = ""

    @classmethod
    def disabled(cls, reason: str) -> "MarketReference":
        if not reason:
            raise InsightError(
                "a disabled market reference must say why it is disabled")
        return cls(state="disabled", reason=reason)

    @classmethod
    def available(cls, *, home_spread: float, observed_at_utc: str,
                  source: str, n_books: Optional[int] = None
                  ) -> "MarketReference":
        missing = [name for name, item in
                   (("observed_at_utc", observed_at_utc), ("source", source))
                   if not item]
        if missing:
            raise InsightError(
                f"a market reference cannot be shown without {missing}; an "
                "undated market number is not a reference, it is a rumour")
        return cls(state="available", home_spread=float(home_spread),
                   observed_at_utc=observed_at_utc, source=source,
                   n_books=(None if n_books is None else int(n_books)))

    def to_dict(self) -> Dict[str, Any]:
        if self.state == "disabled":
            return {"state": "disabled", "reason": self.reason}
        return {
            "state": "available",
            "home_spread": round(float(self.home_spread), 2),
            "observed_at_utc": self.observed_at_utc,
            "source": self.source,
            "n_books": self.n_books,
            "note": ("Reference only, shown with its observation time. This "
                     "surface makes no comparison against it."),
        }


#: The default. Stated, not implied.
MARKET_DISABLED = MarketReference.disabled(
    "Market comparison is intentionally switched off for this surface.")


# --------------------------------------------------------------------------- #
# Section 1 — the call
# --------------------------------------------------------------------------- #
def _favoured(row: Mapping[str, Any]) -> tuple:
    margin = float(row["projected_home_margin"])
    if margin > 0:
        return str(row["home"]), margin
    if margin < 0:
        return str(row["away"]), -margin
    return None, 0.0


def build_call(row: Mapping[str, Any]) -> Dict[str, Any]:
    team, points = _favoured(row)
    sd = float(row["forecast_margin_sd"])
    probability = float(row["p_home_win"])
    headline = ("Level projection: no side projected ahead" if team is None
                else f"{team} projected by {points:.1f}")
    call = {
        "headline": headline,
        "projected_ahead": team,
        "projected_margin": round(points, 1),
        "projected_home_margin": round(float(row["projected_home_margin"]), 2),
        "home_win_probability": round(probability, 3),
        "margin_uncertainty_sd": round(sd, 2),
        "uncertainty_statement": (
            f"Predictive spread of the projected margin: "
            f"{sd:.1f} points (1 SD)."),
    }
    total = row.get("projected_total")
    if total is not None and not (isinstance(total, float) and math.isnan(total)):
        call["projected_total"] = round(float(total), 1)
    return call


# --------------------------------------------------------------------------- #
# Section 2 — why
# --------------------------------------------------------------------------- #
RATING_KEY = "rating_differential"
HOME_FIELD_KEY = "home_field"
SIMULATION_KEY = "simulation"

_LABELS = {
    RATING_KEY: "Rating differential",
    HOME_FIELD_KEY: "Home-field effect",
    SIMULATION_KEY: "Simulation adjustment",
}


def build_why(components: Sequence[WhyComponent], row: Mapping[str, Any],
              *, tolerance: float = 0.05) -> List[Dict[str, Any]]:
    """The components, checked against the number they claim to explain.

    A decomposition that does not add up is worse than none: the reader can
    do the arithmetic, and when it fails they are right to stop believing the
    rest of the card.
    """
    if not components:
        raise InsightError(
            "a priced card must say why. Supply the rating differential, the "
            "home-field term where one applies, and a simulation adjustment "
            "only if a simulation produced one")
    keys = [c.key for c in components]
    unknown = [k for k in keys if k not in _LABELS]
    if unknown:
        raise InsightError(f"unknown why-component(s) {unknown}")
    if len(set(keys)) != len(keys):
        raise InsightError(f"duplicate why-components in {keys}")
    if SIMULATION_KEY in keys:
        family = FC.FAMILIES.get(str(row.get("model_family")))
        if family is None or not family.supports_totals:
            raise InsightError(
                f"model_family {row.get('model_family')!r} runs no simulation, "
                "so it cannot have produced a simulation adjustment")
    if HOME_FIELD_KEY in keys and bool(row.get("neutral_site")):
        raise InsightError(
            "a neutral-site game has no home-field term to show; omit it "
            "rather than rendering a zero the reader has to interpret")

    total = sum(float(c.points) for c in components)
    published = float(row["projected_home_margin"])
    if abs(total - published) > tolerance:
        raise InsightError(
            f"the components sum to {total:.3f} but the published margin is "
            f"{published:.3f}. A card whose arithmetic does not close is a "
            "contradiction on its face")
    return [c.to_dict() for c in components]


def components_from_table(tbl, *, home: str, away: str, neutral: bool,
                          model_version: str,
                          simulated_margin: Optional[float] = None
                          ) -> List[WhyComponent]:
    """Build the components from a solved ratings table.

    The ridge solve carries a separate conference term; it is folded into the
    rating differential rather than shown as its own row, because a reader
    asked to hold four numbers in their head is a reader who has stopped
    reading. The fold is exact, so the sum check still closes.
    """
    from . import explain as EX
    rows = EX.decompose_margin(tbl, home, away, neutral,
                               mc_margin=simulated_margin)
    by_key = {r["key"]: float(r["points"]) for r in rows}
    rating = by_key.get("team_rating", 0.0) + by_key.get("conference", 0.0)
    out = [WhyComponent(key=RATING_KEY, label=_LABELS[RATING_KEY],
                        points=rating, source=model_version)]
    if not neutral:
        out.append(WhyComponent(key=HOME_FIELD_KEY,
                                label=_LABELS[HOME_FIELD_KEY],
                                points=by_key.get("hfa", 0.0),
                                source=model_version))
    elif abs(by_key.get("hfa", 0.0)) > 1e-9:
        raise InsightError(
            "a neutral-site game produced a non-zero home-field term")
    if simulated_margin is not None:
        out.append(WhyComponent(key=SIMULATION_KEY,
                                label=_LABELS[SIMULATION_KEY],
                                points=by_key.get("simulation", 0.0),
                                source="drive-level simulation"))
    return out


def components_cross_division(tbl, *, home: str, away: str, neutral: bool,
                              fbs_is_home: bool, model_version: str
                              ) -> List[WhyComponent]:
    """The why-components of an FBS-vs-FCS projection.

    Same two rows as the ridge card -- rating differential (team deviation
    plus conference term, folded) and home field -- but the FCS side is the
    pooled `__FCS__` pseudo-team, because that is what the number was made
    from. The label says so: a reader must not be led to believe the
    opponent was rated as itself. The sum closes exactly on
    `cross_division.predict_margin`, which is the check `build_why` runs.
    """
    from . import ratings as R
    fbs_team = home if fbs_is_home else away
    h, a = (fbs_team, R.FCS_TEAM) if fbs_is_home else (R.FCS_TEAM, fbs_team)
    team = ((tbl.off_dev.get(h, 0.0) + tbl.def_dev.get(a, 0.0))
            - (tbl.off_dev.get(a, 0.0) + tbl.def_dev.get(h, 0.0)))
    hc, ac = tbl.team_conf.get(h, ""), tbl.team_conf.get(a, "")
    conf = ((tbl.conf_off.get(hc, 0.0) + tbl.conf_def.get(ac, 0.0))
            - (tbl.conf_off.get(ac, 0.0) + tbl.conf_def.get(hc, 0.0)))
    out = [WhyComponent(key=RATING_KEY,
                        label=_LABELS[RATING_KEY] + " (FCS side: pooled FCS pseudo-team)",
                        points=team + conf, source=model_version)]
    if not neutral:
        out.append(WhyComponent(key=HOME_FIELD_KEY,
                                label=_LABELS[HOME_FIELD_KEY],
                                points=tbl.hfa(h, neutral),
                                source=model_version))
    return out


def components_for_rows(priced, table, xd_table, *, fbs,
                        simulated: Optional[Mapping[int, Mapping]] = None
                        ) -> Dict[int, List[WhyComponent]]:
    """Why-components per priced row, from the table that priced THAT row.

    A `ridge_cross_division` row is decomposed against the pooled FCS
    pseudo-team on `xd_table`; every other row on the week's main `table`
    (with the simulation adjustment when a simulation made the number).
    `fbs` is the season's FBS set, the same one the producer classified by.
    """
    out: Dict[int, List[WhyComponent]] = {}
    for entry in priced.to_dict("records"):
        game_id = int(entry["game_id"])
        home, away = str(entry["home"]), str(entry["away"])
        neutral = bool(entry["neutral_site"])
        if str(entry["model_family"]) == "ridge_cross_division":
            out[game_id] = components_cross_division(
                xd_table, home=home, away=away, neutral=neutral,
                fbs_is_home=home in fbs,
                model_version=str(entry["model_version"]))
            continue
        sim = None if not simulated else simulated.get(game_id)
        out[game_id] = components_from_table(
            table, home=home, away=away, neutral=neutral,
            model_version=str(entry["model_version"]),
            simulated_margin=(None if sim is None
                              else float(sim["mc_margin"])))
    return out


# --------------------------------------------------------------------------- #
# Section 3 — provenance
# --------------------------------------------------------------------------- #
def build_provenance(row: Mapping[str, Any],
                     data_as_of_utc: Optional[str] = None) -> Dict[str, Any]:
    out = {
        "model_family": row.get("model_family"),
        "model_version": row.get("model_version"),
        "generated_at_utc": row.get("generated_at_utc"),
        "snapshot_id": row.get("snapshot_id"),
        "source_digest": row.get("source_digest"),
        "data_as_of_utc": data_as_of_utc,
        "forecast_sd_basis": row.get("forecast_sd_basis"),
        "schema_version": int(row.get("schema_version", FC.SCHEMA_VERSION)),
    }
    frozen_from = row.get("frozen_from_stamp")
    if frozen_from:
        out["frozen_from_stamp"] = frozen_from
    return {k: v for k, v in out.items()
            if not (v is None or (isinstance(v, float) and math.isnan(v)))}


# --------------------------------------------------------------------------- #
# Section 4 — caution
# --------------------------------------------------------------------------- #
CAUTIONS = {
    "refused_non_fbs": (
        "Not forecast: the matchup is outside the division this model is "
        "validated on."),
    "refused_cross_division": (
        "Not forecast: no validated cross-division model exists for this "
        "matchup."),
    "refused_unrated": (
        "Not forecast: at least one team has no validated rating yet."),
    "refused_no_pregame_record": (
        "Not forecast: kickoff passed with no pre-kickoff record for this "
        "game, and a number produced after kickoff is not a forecast."),
    "refused_other": "Not forecast: the model declined this game.",
    "degraded_inputs": (
        "No number published: the inputs behind this forecast did not pass "
        "their health check on the run that produced this row."),
    "uncertainty_unvalidated": (
        "No number published: a margin model exists for this game, but no "
        "validated out-of-sample evidence backs its uncertainty this run."),
    "frozen_pre_kickoff": (
        "Frozen before kickoff and republished unchanged. The reasoning "
        "behind it is not recomputed after kickoff, so no breakdown is shown."),
    "cross_division_pooled": (
        "Cross-division: the FCS opponent is projected as the pooled FCS "
        "pseudo-team, not as itself. Out of sample the FBS side has beaten "
        "this projection on average; that bias is registered in the basis "
        "manifest and deliberately not corrected here."),
    "early_season_prior": (
        "Early season: the preseason prior still dominates the rating, and "
        "the registered out-of-sample window for these weeks is small."),
    "uncertainty_exceeds_projection": (
        "The projected margin is small relative to its own predictive "
        "spread, so the projected outcome is close to even."),
    "wide_uncertainty": (
        "The predictive spread is wide even by college-football standards."),
    "paper_only": (
        "Paper-only research output: a projection and its uncertainty, "
        "nothing more."),
}


def build_caution(row: Mapping[str, Any]) -> Dict[str, Any]:
    """One specific reason, by fixed priority. Never empty, never vague when
    something concrete applies."""
    status = str(row.get("status"))
    reason = str(row.get("refusal_reason") or "")

    if status == FC.Status.REFUSED.value:
        lowered = reason.lower()
        if "non-fbs matchup" in lowered:
            code = "refused_non_fbs"
        elif "cross-division" in lowered:
            code = "refused_cross_division"
        elif "no rating" in lowered:
            code = "refused_unrated"
        elif "pregame snapshot" in lowered:
            code = "refused_no_pregame_record"
        else:
            code = "refused_other"
        return {"code": code, "text": CAUTIONS[code], "detail": reason}

    if status == FC.Status.DEGRADED.value:
        return {"code": "degraded_inputs", "text": CAUTIONS["degraded_inputs"],
                "detail": reason}

    if status == FC.Status.UNCERTAINTY_UNVALIDATED.value:
        return {"code": "uncertainty_unvalidated",
                "text": CAUTIONS["uncertainty_unvalidated"], "detail": reason}

    if status == FC.Status.FROZEN.value:
        return {"code": "frozen_pre_kickoff",
                "text": CAUTIONS["frozen_pre_kickoff"]}

    if str(row.get("model_family")) == "ridge_cross_division":
        # Above the early-season note on purpose: the pooled-opponent caveat
        # is the one specific to THIS number, whatever week it is.
        return {"code": "cross_division_pooled",
                "text": CAUTIONS["cross_division_pooled"]}

    week = int(row.get("week", 0))
    from . import engine as EG
    if EG.is_early_season(week):
        return {"code": "early_season_prior",
                "text": CAUTIONS["early_season_prior"]}

    margin = abs(float(row["projected_home_margin"]))
    sd = float(row["forecast_margin_sd"])
    if margin < NARROW_MARGIN_RATIO * sd:
        return {"code": "uncertainty_exceeds_projection",
                "text": CAUTIONS["uncertainty_exceeds_projection"]}
    if sd > WIDE_SD_POINTS:
        return {"code": "wide_uncertainty", "text": CAUTIONS["wide_uncertainty"]}
    return {"code": "paper_only", "text": CAUTIONS["paper_only"]}


# --------------------------------------------------------------------------- #
# Optional display-only context
# --------------------------------------------------------------------------- #
def build_context(context) -> List[Dict[str, Any]]:
    """Only fields whose source is actually there. Empty means omitted.

    The context layer refuses to construct an available field without its
    whole provenance envelope, so "availability is OK" already means the
    source, its identifier and its observation time exist. What this adds is
    the display rule: an unavailable field is not rendered as a row saying
    unavailable. A framework with nothing in it reads as data that failed to
    load, which is a different and worse claim than saying nothing.
    """
    if context is None:
        return []
    values = getattr(context, "values", None)
    if not values:
        return []
    out = []
    for item in values:
        if getattr(item.availability, "value", item.availability) != "ok":
            continue
        if not (item.source and item.source_url_or_id and item.observed_at):
            continue
        out.append({
            "key": item.key,
            "label": item.label,
            "display": item.display,
            "source": item.source,
            "source_ref": item.source_url_or_id,
            "observed_at_utc": item.observed_at.isoformat(),
            "display_only": True,
        })
    return sorted(out, key=lambda entry: entry["key"])


# --------------------------------------------------------------------------- #
# The card
# --------------------------------------------------------------------------- #
DISCLAIMER = ("Paper-only research output. This describes what the model "
              "projected and how uncertain it is.")


def build_card(row: Mapping[str, Any], *,
               components: Optional[Sequence[WhyComponent]] = None,
               market: Optional[MarketReference] = None,
               context=None,
               data_as_of_utc: Optional[str] = None) -> Dict[str, Any]:
    """One card for one contract row. Deterministic; reads no clock."""
    status = str(row.get("status"))
    if status not in {s.value for s in FC.Status}:
        raise InsightError(f"unknown row status {status!r}")

    card: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "game_id": int(row["game_id"]),
        "season": int(row["season"]),
        "week": int(row["week"]),
        "matchup": {
            "home": str(row["home"]),
            "away": str(row["away"]),
            "neutral_site": bool(row.get("neutral_site")),
            "kickoff_utc": row.get("kickoff_utc"),
            "public_date": row.get("public_date"),
            "slate": row.get("slate"),
        },
        "status": status,
    }

    if status == FC.Status.PRICED.value:
        card["call"] = build_call(row)
        card["why"] = build_why(components or (), row)
    elif status == FC.Status.FROZEN.value:
        # The numbers stand; the reasoning is not re-derived after kickoff.
        card["call"] = build_call(row)
    # refused / degraded carry neither a call nor a why, by contract.

    card["provenance"] = build_provenance(row, data_as_of_utc)
    card["market_reference"] = (market or MARKET_DISABLED).to_dict()

    context_rows = build_context(context)
    if context_rows:
        card["context"] = context_rows

    card["caution"] = build_caution(row)
    card["disclaimer"] = DISCLAIMER
    return assert_non_actionable(
        card, proper_nouns=(str(row["home"]), str(row["away"])))


def canonical_json(card: Mapping[str, Any]) -> str:
    """Byte-stable serialisation: same inputs, same bytes, always."""
    return json.dumps(card, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def render_card_text(card: Mapping[str, Any]) -> str:
    """A compact fixed-width rendering. Same four sections, same order."""
    matchup = card["matchup"]
    site = " (neutral site)" if matchup["neutral_site"] else ""
    lines = [f"{matchup['away']} @ {matchup['home']}{site}  "
             f"[{matchup['public_date']} · {matchup['slate']}]",
             f"status: {card['status']}"]
    call = card.get("call")
    if call:
        lines += [
            f"  {call['headline']}",
            f"  home win probability {call['home_win_probability']:.3f}",
            f"  {call['uncertainty_statement']}",
        ]
    if card.get("why"):
        lines.append("  why:")
        for component in card["why"]:
            lines.append(f"    {component['points']:+7.2f}  "
                         f"{component['label']}  [{component['source'][:12]}]")
    provenance = card["provenance"]
    lines.append("  provenance: "
                 + ", ".join(f"{k}={provenance[k]}" for k in
                             ("model_family", "model_version",
                              "generated_at_utc", "snapshot_id",
                              "source_digest", "data_as_of_utc")
                             if k in provenance))
    market = card["market_reference"]
    lines.append("  market: " + (market["reason"] if market["state"] == "disabled"
                                 else f"{market['home_spread']:+.1f} from "
                                      f"{market['source']} observed "
                                      f"{market['observed_at_utc']}"))
    if card.get("context"):
        lines.append("  context (display only): "
                     + "; ".join(f"{c['label']} {c['display']}"
                                 for c in card["context"]))
    lines.append(f"  caution [{card['caution']['code']}]: "
                 f"{card['caution']['text']}")
    return "\n".join(lines)


def build_week(rows: Sequence[Mapping[str, Any]], *,
               components_by_game: Optional[Mapping[int, Sequence[WhyComponent]]] = None,
               market_by_game: Optional[Mapping[int, MarketReference]] = None,
               context_by_game: Optional[Mapping[int, Any]] = None,
               data_as_of_utc: Optional[str] = None) -> Dict[str, Any]:
    """Every card for a published week, in schedule order."""
    components_by_game = components_by_game or {}
    market_by_game = market_by_game or {}
    context_by_game = context_by_game or {}
    cards = []
    for row in rows:
        game_id = int(row["game_id"])
        cards.append(build_card(
            row, components=components_by_game.get(game_id),
            market=market_by_game.get(game_id),
            context=context_by_game.get(game_id),
            data_as_of_utc=data_as_of_utc))
    by_status: Dict[str, int] = {}
    for card in cards:
        by_status[card["status"]] = by_status.get(card["status"], 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "season": int(rows[0]["season"]) if rows else None,
        "week": int(rows[0]["week"]) if rows else None,
        "n_cards": len(cards),
        "by_status": by_status,
        "cards": cards,
    }
