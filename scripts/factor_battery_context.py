#!/usr/bin/env python3
"""Context-layer factor battery — run 1, pre-registered 2026-08-28.

The registration is `data/factor_prereg_2026-08-28.json`, written and
appended to `_worklog.md` BEFORE this file was executed. It fixes the
hypothesis list, the cohorts, the outcome, the n-gates, the matching and
the multiplicity correction. This script may not alter any of them.

Reuses `nflvalue.factor_battery` unchanged for the gate machinery:
matched-control event tests, residual regression for continuous factors,
Benjamini-Hochberg across exactly the registered list, and the registry
read/write. Nothing here promotes a factor; a pass is recorded as
display-only and wiring one into a probability stays a separate,
human-reviewed decision.

DEVIATION CAUGHT AND CORRECTED BEFORE ANY RESULT WAS RECORDED
-------------------------------------------------------------
The first execution of this script computed Benjamini-Hochberg over only
the hypotheses that produced a p-value (m=4), silently dropping the
underpowered one. The registration says m=5 and "underpowered hypotheses
are reported as underpowered, never dropped to shrink m". A smaller m
loosens every threshold, and it mattered: `rest_days_diff` (p=0.031764)
clears the m=4 threshold of 0.0375 and FAILS the registered m=5 threshold
of 0.0300. The correction is implemented below — the underpowered
hypothesis stays in the family at p=1.0 — and it moved the result AWAY
from a pass, which is the direction that tells you it was not a
convenience. Nothing from the m=4 execution was recorded in the registry.

ERRATUM, recorded rather than corrected in place
------------------------------------------------
The registration's `special_teams_field_position_diff` entry states the
formula (our field position after their punt, MINUS the field position
they get after ours) and then labels the hypothesised direction
"positive". Those disagree: start_yards_to_goal is distance to the
opponent's goal, so better special teams means a LOWER received value and
a HIGHER conceded one, i.e. a more NEGATIVE differential. The formula is
the operative definition and is implemented exactly as registered; the
direction label was wrong. The test is two-sided, so the pass/reject
decision is unaffected — only the reading of the sign. Flipping the
formula after the fact to match the label would be editing a frozen
registration, which this project does not do.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflvalue import factor_battery as FB  # noqa: E402
from nflvalue import evidence as EV  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PREREG_PATH = os.path.join(DATA, "factor_prereg_2026-08-28.json")
SEASONS = [2021, 2022, 2023, 2024]
TODAY = "2026-08-28"

#: frozen by the registration — this list is the BH family, m = 5.
HYPOTHESES: List[str] = ["short_week", "rest_days_diff", "coach_year2",
                         "returning_production_diff",
                         "special_teams_field_position_diff"]
#: the project's registered gates, copied unchanged from factor_battery.
MIN_TREATED = 40
MIN_CONTINUOUS_N = 800
BH_Q = 0.05

DISPLAY = "display_only"
PUNT_RESULTS = ("PUNT",)


#: Interpretive notes attached AFTER the run. They change no p-value and
#: no verdict; they record how each result must be read. Written down
#: rather than left in a session transcript.
CAVEATS: Dict[str, str] = {
    "short_week":
        "UNDERPOWERED, which is not the same as absent: 16 treated "
        "team-games against a 40-pair gate. A true short week (<=6 days "
        "rest while the opponent has >=7) is rare in weeks 4-15 of the CFB "
        "calendar, which is the only window the walk-forward backtest "
        "covers. This is a statement about the sample, not about football "
        "— the hypothesis is neither supported nor refuted.",
    "rest_days_diff":
        "REJECTED at the registered threshold (p=0.031764 vs BH crit "
        "0.0300 at m=5). It WOULD have passed against a denominator of the "
        "4 hypotheses that produced a p-value (crit 0.0375); the first "
        "execution of this script had exactly that bug and it was "
        "corrected before anything was recorded. A borderline result that "
        "survives only under a looser denominator is not a finding.",
    "coach_year2":
        "REJECTED cleanly and with power: p=0.515 on 835 matched pairs, "
        "effect +0.54 pts. Read beside the registered coaching_change_y1 "
        "(-2.66 pts, passed 2026-07-30): first-year disruption registers, "
        "a distinct second-year recovery does not.",
    "returning_production_diff":
        "CAVEAT THAT CHANGES THE READING, recorded after the run: the "
        "registration failed to declare that this variable is ENDOGENOUS "
        "to the prediction it is scored against. `ratings.solve` uses "
        "`preseason_prior` as the ridge PRIOR MEAN, and that prior is "
        "built from talent + returning production with "
        "prior_w_returning=1.0, so returning production is already inside "
        "pred_b at every week. The slope is also WRONG-SIGNED against the "
        "registered direction (-3.83 vs hypothesised positive). The "
        "parsimonious reading is therefore NOT 'returning production "
        "predicts residuals' but 'the preseason prior over-weights "
        "returning production and the residual corrects it' — a "
        "model-calibration finding, not an external signal. Stays "
        "display_only. Re-fitting prior_w_returning is a separate "
        "exercise needing its own pre-registration and a walk-forward "
        "search; never a guessed constant (AGENT_BUILD_PROMPT section 8).",
    "special_teams_field_position_diff":
        "SIGN READS AS INTENDED once the module ERRATUM is applied: the "
        "registered differential is (field position received after their "
        "punt) minus (conceded after ours), and start_yards_to_goal is "
        "distance to the opponent's goal, so BETTER special teams is a "
        "more NEGATIVE differential. A negative slope therefore means "
        "better net punt play RAISES the residual — the substantive "
        "direction hypothesised. Magnitude 0.12 pts per yard of net punt "
        "field position, so a fraction of a point at realistic "
        "differentials. OVERLAP DECLARED IN ADVANCE: correlated with the "
        "registered field_position_diff (slope +0.6165, passed "
        "2026-07-30), which mixes offence-earned and special-teams-earned "
        "field position. Not independent evidence; must not be counted "
        "twice.",
}


def travel_census(games: pd.DataFrame,
                  eval_ids=None) -> Dict[str, int]:
    """The census fraction that needs no venue data. COUNTS ONLY.

    research/circadian-timezone-factor-plan.md pre-registers a slice
    census before any modelling, and this run does not override it. Zones
    crossed, travel distance and body-clock hour all need venue
    coordinates, which are behind the cfbd_venues approval gate, so only
    the kickoff-hour slices are countable today. No effect is estimated.
    """
    def _buckets(frame, prefix):
        ts = pd.to_datetime(frame["start_date"], utc=True, format="ISO8601",
                            errors="coerce")
        hour = ts.dt.tz_convert("America/New_York").dt.hour
        return {
            f"{prefix}_total": int(len(frame)),
            f"{prefix}_kick_before_noon_et": int((hour < 12).sum()),
            f"{prefix}_kick_1200_1559_et": int(((hour >= 12) & (hour < 16)).sum()),
            f"{prefix}_kick_1600_1959_et": int(((hour >= 16) & (hour < 20)).sum()),
            f"{prefix}_kick_2000_2159_et": int(((hour >= 20) & (hour < 22)).sum()),
            f"{prefix}_kick_2200_et_or_later": int((hour >= 22).sum()),
        }

    out = _buckets(games, "full_2021_2024_schedule")
    if eval_ids is not None:
        out.update(_buckets(games[games["id"].isin(set(eval_ids))],
                            "walk_forward_evaluation_set"))
    out["n_gate_for_a_kill_check_slice"] = 143
    return out


# --------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------- #
def _parquet(kind: str, seasons=SEASONS) -> pd.DataFrame:
    frames = []
    for s in seasons:
        p = os.path.join(DATA, "cfbd", f"{kind}_{s}.parquet")
        if os.path.exists(p):
            frames.append(pd.read_parquet(p))
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame())


def load_preds() -> pd.DataFrame:
    """The evaluation set, chosen ONLY through the evidence manifest.

    This read `backtest_c.csv` + `backtest_c_2024.csv` directly. Both are
    pre-correction artifacts, so every factor result computed from them is a
    statement about a walk-forward that leaked pace and OT — and a factor
    battery is precisely where a leaked residual turns into a false positive.
    With no artifact selected this raises rather than falling back: a battery
    run over archival evidence, labelled as though it graded the current
    model, is worse than no run.
    """
    frame, record, notice = EV.benchmark_or_notice(ROOT, EV.ROLE_WK4PLUS)
    if frame is None:
        raise SystemExit(
            "factor battery refused: no benchmark artifact is selected.\n"
            f"{notice}")
    print(f"[battery] evaluation set: {record.artifact_id} "
          f"({record.n_rows} rows, method {record.method})")
    return frame[["id", "season", "week", "home", "away", "pred_b", "margin"]]


# --------------------------------------------------------------------- #
# H1 / H2 — rest and short week
# --------------------------------------------------------------------- #
def rest_days(games: pd.DataFrame) -> pd.DataFrame:
    """Days between each side's previous kickoff and this one.

    Reads only kickoff timestamps, which are published before the season.
    A team's first game of a season has no prior kickoff and yields NaN
    rather than a filled-in default.
    """
    g = games.copy()
    g["ts"] = pd.to_datetime(g["start_date"], utc=True, format="ISO8601",
                             errors="coerce")
    parts = []
    for side, col in (("home", "home_team"), ("away", "away_team")):
        p = g[["id", "season", "ts", col]].rename(columns={col: "team"})
        p["side"] = side
        parts.append(p)
    long = pd.concat(parts, ignore_index=True)
    long = long.sort_values(["season", "team", "ts", "id"], kind="mergesort")
    long["prev_ts"] = long.groupby(["season", "team"])["ts"].shift(1)
    long["rest"] = (long["ts"] - long["prev_ts"]).dt.total_seconds() / 86400.0
    wide = long.pivot_table(index="id", columns="side", values="rest",
                            aggfunc="first")
    for side in ("home", "away"):
        if side not in wide.columns:
            wide[side] = np.nan
    # A team's first game of a season has no prior kickoff. Reindex over
    # every scheduled game so those rows survive as NaN instead of being
    # dropped — a missing value must stay visible, not vanish.
    wide = wide.reindex(pd.Index(g["id"].unique(), name="id"))
    out = wide.reset_index()[["id", "home", "away"]]
    out.columns = ["id", "home_rest", "away_rest"]
    return out.sort_values("id").reset_index(drop=True)


def short_week_events(rest: pd.DataFrame) -> pd.DataFrame:
    """One treated row per side on <=6 days rest facing >=7.

    Games where BOTH sides are short carry no differential and are
    excluded, as registered.
    """
    r = rest.dropna(subset=["home_rest", "away_rest"])
    home_short = (r["home_rest"] <= 6.0) & (r["away_rest"] >= 7.0)
    away_short = (r["away_rest"] <= 6.0) & (r["home_rest"] >= 7.0)
    ev = pd.concat([
        pd.DataFrame({"id": r.loc[home_short, "id"], "side": "home"}),
        pd.DataFrame({"id": r.loc[away_short, "id"], "side": "away"}),
    ], ignore_index=True)
    return ev.sort_values(["id", "side"]).reset_index(drop=True)


def rest_days_diffs(rest: pd.DataFrame) -> pd.DataFrame:
    r = rest.dropna(subset=["home_rest", "away_rest"]).copy()
    r["diff"] = r["home_rest"] - r["away_rest"]
    return r[["id", "diff"]].reset_index(drop=True)


# --------------------------------------------------------------------- #
# H3 — second-year head coach
# --------------------------------------------------------------------- #
def coach_year2_events(coaches: pd.DataFrame,
                       games: pd.DataFrame) -> pd.DataFrame:
    """Same head coach as last season, and last season was his first."""
    if coaches.empty:
        return pd.DataFrame(columns=["id", "side"])
    cur = coaches[["season", "team", "coach"]].copy()
    prev = coaches[coaches["first_season"].fillna(False).astype(bool)][
        ["season", "team", "coach"]].copy()
    prev["season"] = prev["season"] + 1
    y2 = cur.merge(prev, on=["season", "team", "coach"], how="inner")
    keys = set(zip(y2["season"], y2["team"]))
    rows = []
    for g in games.itertuples():
        for side, team in (("home", g.home_team), ("away", g.away_team)):
            if (int(g.season), team) in keys:
                rows.append({"id": int(g.id), "side": side})
    return (pd.DataFrame(rows, columns=["id", "side"])
            .sort_values(["id", "side"]).reset_index(drop=True))


# --------------------------------------------------------------------- #
# H4 — returning production
# --------------------------------------------------------------------- #
def returning_diffs(returning: pd.DataFrame,
                    games: pd.DataFrame) -> pd.DataFrame:
    if returning.empty:
        return pd.DataFrame(columns=["id", "diff"])
    r = returning[["season", "team", "percent_ppa"]]
    g = games[["id", "season", "home_team", "away_team"]]
    m = (g.merge(r, left_on=["season", "home_team"],
                 right_on=["season", "team"], how="left")
          .rename(columns={"percent_ppa": "home_pct"}).drop(columns=["team"])
          .merge(r, left_on=["season", "away_team"],
                 right_on=["season", "team"], how="left")
          .rename(columns={"percent_ppa": "away_pct"}).drop(columns=["team"]))
    m["diff"] = m["home_pct"] - m["away_pct"]
    return m[["id", "diff"]].dropna(subset=["diff"]).reset_index(drop=True)


# --------------------------------------------------------------------- #
# H5 — net punt-driven field position, walk-forward
# --------------------------------------------------------------------- #
def usable_drives(drives: pd.DataFrame) -> pd.DataFrame:
    """Gate-3 convention: no garbage time, regulation periods only."""
    d = drives
    if "garbage_time" in d.columns:
        d = d[~d["garbage_time"].fillna(False).astype(bool)]
    if "start_period" in d.columns:
        d = d[d["start_period"] <= 4]
    return d.reset_index(drop=True)


def _punt_events(drives: pd.DataFrame) -> pd.DataFrame:
    """Per team-week: field position received after an opponent punt, and
    field position conceded to the opponent after our own punt."""
    d = usable_drives(drives).sort_values(["game_id", "drive_number"],
                                          kind="mergesort").copy()
    grp = d.groupby("game_id", sort=False)
    d["prev_result"] = grp["drive_result"].shift(1)
    d["prev_offense"] = grp["offense"].shift(1)
    d["prev_number"] = grp["drive_number"].shift(1)
    after_punt = (d["prev_result"].isin(PUNT_RESULTS)
                  & (d["prev_offense"] == d["defense"])
                  & (d["drive_number"] - d["prev_number"] == 1))
    ev = d[after_punt]
    recv = (ev.groupby(["season", "week", "offense"])["start_yards_to_goal"]
              .mean().rename("recv").reset_index()
              .rename(columns={"offense": "team"}))
    give = (ev.groupby(["season", "week", "defense"])["start_yards_to_goal"]
              .mean().rename("give").reset_index()
              .rename(columns={"defense": "team"}))
    return recv.merge(give, on=["season", "week", "team"], how="outer")


def _walk_forward_net(punts: pd.DataFrame, season: int, week: int,
                      team: str):
    past = punts[(punts["season"] == season) & (punts["week"] < week)
                 & (punts["team"] == team)]
    if past.empty:
        return None
    recv, give = past["recv"].mean(), past["give"].mean()
    if pd.isna(recv) or pd.isna(give):
        return None
    return float(recv - give)          # as registered; see module ERRATUM


def special_teams_diffs(drives: pd.DataFrame,
                        games: pd.DataFrame) -> pd.DataFrame:
    punts = _punt_events(drives)
    rows = []
    for g in games.itertuples():
        h = _walk_forward_net(punts, int(g.season), int(g.week), g.home_team)
        a = _walk_forward_net(punts, int(g.season), int(g.week), g.away_team)
        rows.append({"id": int(g.id),
                     "diff": None if h is None or a is None else h - a})
    return (pd.DataFrame(rows, columns=["id", "diff"])
            .dropna(subset=["diff"]).reset_index(drop=True))


def special_teams_rows_used(drives: pd.DataFrame, games: pd.DataFrame,
                            game_id: int) -> Dict:
    """Exactly which drives fed one game's value — for the leakage test."""
    g = games[games["id"] == game_id].iloc[0]
    d = usable_drives(drives)
    past = d[(d["season"] == int(g.season)) & (d["week"] < int(g.week))
             & (d["offense"].isin([g.home_team, g.away_team])
                | d["defense"].isin([g.home_team, g.away_team]))]
    return {"weeks": sorted(set(past["week"].tolist())),
            "game_ids": sorted(set(past["game_id"].tolist()))}


# --------------------------------------------------------------------- #
def frames_handed_to_evaluator() -> List[pd.DataFrame]:
    games = _parquet("games")
    rest = rest_days(games)
    return [load_preds(), short_week_events(rest), rest_days_diffs(rest),
            coach_year2_events(_parquet("coaches", [2020] + SEASONS), games),
            returning_diffs(_parquet("returning"), games),
            special_teams_diffs(_parquet("drives"), games)]


# --------------------------------------------------------------------- #
def seed_registry() -> int:
    """Register every hypothesis as UNTESTED before it is tested.

    `factor_battery.registry_entry` refuses to evaluate a factor that is
    not already in the registry — the project's rule that a hypothesis is
    declared before it is run. Seeding is part of the registration step,
    not part of the run: it writes definitions and `untested`, never a
    result.
    """
    prereg = json.load(open(PREREG_PATH, encoding="utf-8"))
    entries = FB.load_registry(os.path.join(DATA, "factor_registry.json"))
    known = {e["factor_id"] for e in entries}
    added = []
    for h in prereg["hypotheses"]:
        if h["id"] in known:
            continue
        entries.append({
            "factor_id": h["id"],
            "description": h["definition"],
            "bucket": 3, "status": "untested", "effect_size": None,
            "ci": None, "n": None,
            "source_doc": "data/factor_prereg_2026-08-28.json",
            "source_measurement": (
                f"REGISTERED {TODAY}, not yet run. Shape {h['shape']}; "
                f"n-gate {h['n_gate']}; direction hypothesised "
                f"{h['direction_hypothesised']}."),
            "test": "scripts/factor_battery_context.py",
            "last_validated": None,
            "next_validation_due": None,
            "revalidation_history": [],
        })
        added.append(h["id"])
    for fam in prereg["families_recorded_but_not_tested"]:
        fid = f"{fam}__not_tested_2026_08_28"
        if fid in known:
            continue
        entries.append({
            "factor_id": fid,
            "description": f"Context family '{fam}' — declared in the "
                           f"{TODAY} registration as recorded-but-not-tested.",
            "bucket": 3, "status": "untested", "effect_size": None,
            "ci": None, "n": None,
            "source_doc": "data/factor_prereg_2026-08-28.json",
            "source_measurement": f"REGISTERED {TODAY}, not run.",
            "test": None, "last_validated": None,
            "next_validation_due": None, "revalidation_history": [],
        })
        added.append(fid)
    FB.save_registry(entries, os.path.join(DATA, "factor_registry.json"))
    print("seeded as untested:", added or "(nothing new)")
    return 0


def main() -> int:
    prereg = json.load(open(PREREG_PATH, encoding="utf-8"))
    registered = [h["id"] for h in prereg["hypotheses"]]
    if registered != HYPOTHESES:
        raise SystemExit("hypothesis list drifted from the registration")

    entries = FB.load_registry(os.path.join(DATA, "factor_registry.json"))
    for fid in HYPOTHESES:
        FB.assert_not_excluded(entries, fid)

    preds = load_preds()
    games = _parquet("games")
    games = games[games["season"].isin(SEASONS)]
    rest = rest_days(games)

    results, pvals = {}, {}

    ev = short_week_events(rest)
    r = FB.evaluate_event_factor("short_week", ev, preds,
                                 min_treated=MIN_TREATED)
    results["short_week"] = r
    if not np.isnan(r.p_value):
        pvals["short_week"] = r.p_value

    r = FB.evaluate_continuous_factor("rest_days_diff", rest_days_diffs(rest),
                                      preds, min_n=MIN_CONTINUOUS_N)
    results["rest_days_diff"] = r
    if not np.isnan(r.p_value):
        pvals["rest_days_diff"] = r.p_value

    coaches = _parquet("coaches", [2020] + SEASONS)
    r = FB.evaluate_event_factor("coach_year2",
                                 coach_year2_events(coaches, games), preds,
                                 min_treated=MIN_TREATED)
    results["coach_year2"] = r
    if not np.isnan(r.p_value):
        pvals["coach_year2"] = r.p_value

    r = FB.evaluate_continuous_factor(
        "returning_production_diff",
        returning_diffs(_parquet("returning"), games), preds,
        min_n=MIN_CONTINUOUS_N)
    results["returning_production_diff"] = r
    if not np.isnan(r.p_value):
        pvals["returning_production_diff"] = r.p_value

    r = FB.evaluate_continuous_factor(
        "special_teams_field_position_diff",
        special_teams_diffs(_parquet("drives"), games), preds,
        min_n=MIN_CONTINUOUS_N)
    results["special_teams_field_position_diff"] = r
    if not np.isnan(r.p_value):
        pvals["special_teams_field_position_diff"] = r.p_value

    # BH across exactly the registered list. An underpowered hypothesis
    # produced no p-value, but it STAYS IN THE FAMILY at p=1.0 so the
    # denominator remains m=5, as registered. Dropping it would shrink m
    # and loosen every threshold — the wrong direction, and the first run
    # of this script did exactly that (see the DEVIATION note below).
    family = {fid: pvals.get(fid, 1.0) for fid in HYPOTHESES}
    bh = FB.benjamini_hochberg(family, q=BH_Q)
    if len(family) != len(HYPOTHESES):
        raise SystemExit("BH family drifted from the registered list")
    return _record(entries, results, bh, prereg)


def _record(entries, results, bh, prereg) -> int:
    lines = []
    for fid in HYPOTHESES:
        r = results[fid]
        note = getattr(r, "note", "")
        verdict = ("underpowered" if note.startswith("n-gate")
                   else ("pass" if bh.get(fid, {}).get("pass") else "reject"))
        if hasattr(r, "effect_pts"):
            eff, n = r.effect_pts, r.n_treated
            shape = "event"
        else:
            eff, n = r.slope, r.n
            shape = "continuous"
        p = r.p_value
        crit = bh.get(fid, {}).get("q_crit")
        measurement = (
            f"{verdict.upper()} {TODAY} ({shape}; "
            f"effect {eff if eff == eff else float('nan'):.4f}, "
            f"p={p if p == p else float('nan'):.5g}, n={n}"
            + (f", BH crit {crit:.5g}" if crit is not None else "")
            + (f"; {note}" if note else "")
            + "). Pre-registered in data/factor_prereg_2026-08-28.json; "
            "BH q<0.05 across exactly the 5 registered hypotheses. "
            "display_only — promotion is a separate reviewed decision.")
        entry = FB.registry_entry(entries, fid)
        entry.update({
            "description": next(h["definition"] for h in prereg["hypotheses"]
                                if h["id"] == fid),
            "bucket": 3,
            "status": DISPLAY,
            "effect_size": None if eff != eff else round(float(eff), 4),
            "ci": None,
            "n": int(n) if n == n else None,
            "source_doc": "data/factor_prereg_2026-08-28.json",
            "source_measurement": measurement,
            "test": "scripts/factor_battery_context.py",
            "last_validated": TODAY,
            "next_validation_due": "2026 season end",
        })
        if fid in CAVEATS:
            entry["source_measurement"] += " || " + CAVEATS[fid]
        entry.setdefault("revalidation_history", []).append(
            {"date": TODAY, "hypothesis": fid, "result": verdict,
             "p": None if p != p else round(float(p), 6)})
        lines.append(f"{fid:36s} {verdict:12s} effect={eff:9.4f} "
                     f"p={p:9.5g} n={n}")

    for fam, why in prereg["families_recorded_but_not_tested"].items():
        entry = FB.registry_entry(entries, f"{fam}__not_tested_2026_08_28")
        entry.update({
            "description": f"Context family '{fam}' — recorded, not tested in "
                           "the 2026-08-28 battery.",
            "bucket": 3, "status": DISPLAY, "effect_size": None, "ci": None,
            "n": None,
            "source_doc": "data/factor_prereg_2026-08-28.json",
            "source_measurement": f"NOT TESTED {TODAY}. {why}",
            "test": None, "last_validated": TODAY,
            "next_validation_due": None,
        })
        if fam == "travel_timezone":
            census = travel_census(_parquet("games"),
                                   eval_ids=load_preds()["id"])
            entry["source_measurement"] += (
                " || CENSUS (counts only, no effect estimated), the fraction "
                "derivable without venue data: " + json.dumps(census)
                + ". Zones crossed, travel distance and body-clock hour "
                "remain uncountable until the cfbd_venues gate is opened.")
        entry.setdefault("revalidation_history", [])
        lines.append(f"{fam:36s} not_tested")

    FB.save_registry(entries, os.path.join(DATA, "factor_registry.json"))
    report = "\n".join(lines)
    with open(os.path.join(DATA, "factor_battery_context_report.txt"), "w") as fh:
        fh.write(f"context-layer factor battery — run 1, {TODAY}\n"
                 f"pre-registered: data/factor_prereg_2026-08-28.json\n"
                 f"BH q<{BH_Q} across m={len(HYPOTHESES)}\n\n{report}\n")
    print(report)
    return 0


if __name__ == "__main__":
    if "--seed" in sys.argv:
        raise SystemExit(seed_registry())
    raise SystemExit(main())
