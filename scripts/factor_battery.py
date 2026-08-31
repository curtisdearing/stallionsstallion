#!/usr/bin/env python3
"""Workstream E — run the situational factor battery (gate 6).

PRE-REGISTERED before the first run (2026-07-30):

  * evaluation set: every game with a walk-forward prediction —
    data/backtest_c.csv (2021-2023) + data/backtest_c_2024.csv. The outcome
    is always the walk-forward residual (margin - pred_b), so a factor can
    only register with information the rating missed.
  * hypotheses in THIS run (BH q<0.05 corrected across exactly this list):
      1. backup_qb            — week starter != established primary (>=3
                                prior starts); event, matched control
      2. coaching_change_y1   — first-season HC (hireDate-inferred); event
      3. coaching_change_y1_weak_prior — same, subset where the treated
                                team's walk-forward rating is below its
                                season median (the conditional version
                                situational-factors.md asks for)
      4. finishing_drives_diff — walk-forward pts-per-trip-inside-40 diff
      5. field_position_diff   — walk-forward avg starting-YTG diff
  * n-gates: events >= 40 matched pairs; continuous >= 800 games. Factor
    values need >= 30 prior drives / >= 3 prior starts to exist at all.
  * registry outcome: PASS -> status `display_only` + passed-gate note
    (wiring into a probability is a separate decision, never this script);
    FAIL -> status `display_only` with rejected note. Either way the
    numbers land in the registry with `last_validated` set. Nothing here
    may touch `gated_active`.

Expect mostly rejections. A clean rejection list is the deliverable.
"""

from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflvalue import cfb_sim, factor_battery as FB, ingest  # noqa: E402
from nflvalue import evidence as EV  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEASONS = [2021, 2022, 2023, 2024]
TODAY = "2026-07-30"


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


def backup_qb_events(preds: pd.DataFrame) -> pd.DataFrame:
    qb = ingest.load_qb_starts(SEASONS)
    starters = (qb[qb["is_week_starter"]]
                .sort_values(["season", "week"]))
    events = []
    for (season, team), grp in starters.groupby(["season", "team"], observed=True):
        seq = grp[["week", "athlete_id"]].drop_duplicates("week")
        counts: dict = {}
        for r in seq.itertuples():
            prior_total = sum(counts.values())
            primary = max(counts, key=counts.get) if counts else None
            if (primary is not None and counts[primary] >= 3
                    and r.athlete_id != primary):
                events.append({"season": season, "week": int(r.week),
                               "team": team})
            counts[r.athlete_id] = counts.get(r.athlete_id, 0) + 1
    ev = pd.DataFrame(events)
    rows = []
    for g in preds.itertuples():
        for side, team in (("home", g.home), ("away", g.away)):
            hit = ev[(ev["season"] == g.season) & (ev["week"] == g.week)
                     & (ev["team"].astype(str) == str(team))]
            if len(hit):
                rows.append({"id": g.id, "side": side})
    return pd.DataFrame(rows)


def coaching_events(preds: pd.DataFrame) -> pd.DataFrame:
    co = ingest.load_coaches(SEASONS)
    first = {(int(r.season), str(r.team)) for r in co.itertuples()
             if bool(r.first_season)}
    rows = []
    for g in preds.itertuples():
        for side, team in (("home", g.home), ("away", g.away)):
            if (int(g.season), str(team)) in first:
                rows.append({"id": g.id, "side": side, "season": g.season,
                             "team": team})
    return pd.DataFrame(rows)


def drive_factor_diffs(preds: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward finishing-drives and field-position differentials."""
    drives = ingest.load_drives(SEASONS)
    d, _ = cfb_sim.classify_drives(drives)
    d = d[d["start_period"] <= 4]
    d["trip"] = d["end_yards_to_goal"] <= 40
    d["trip_pts"] = np.where(d["trip"], d["off_pts"], np.nan)

    # per (season, team, week): that week's own numbers, then expanding
    # PRIOR-week means (shift(1) then expanding — the only legitimate view)
    tw = (d.groupby(["season", "offense", "week"], observed=True)
          .agg(n_drives=("off_pts", "size"),
               trips=("trip", "sum"),
               trip_pts_sum=("trip_pts", "sum"),
               sytg_sum=("start_yards_to_goal", "sum"))
          .reset_index()
          .sort_values(["season", "offense", "week"]))
    g = tw.groupby(["season", "offense"], observed=True)
    for c in ("n_drives", "trips", "trip_pts_sum", "sytg_sum"):
        # cumsum-through-current minus current = sum of STRICTLY EARLIER
        # weeks, computed inside the group (a cross-group shift would leak
        # one team's tail into the next team's head).
        tw[f"cum_{c}"] = g[c].cumsum() - tw[c]
    tw["fin"] = tw["cum_trip_pts_sum"] / tw["cum_trips"].replace(0, np.nan)
    tw["fp"] = -(tw["cum_sytg_sum"] / tw["cum_n_drives"].replace(0, np.nan))
    tw.loc[tw["cum_n_drives"] < 30, ["fin", "fp"]] = np.nan
    key = {(int(r.season), str(r.offense), int(r.week)): (r.fin, r.fp)
           for r in tw.itertuples()}

    def look(season, team, week):
        # The row keyed at week w holds cumulative values through w-1 (its
        # own week already subtracted). For a game at week W use the row at
        # the largest w <= W that exists (byes leave gaps; walking back only
        # ever makes the view STALER, never leaky).
        for w in range(week, 0, -1):
            v = key.get((int(season), str(team), w))
            if v is not None:
                return v
        return (np.nan, np.nan)

    fin_rows, fp_rows = [], []
    for gm in preds.itertuples():
        fh, ph = look(gm.season, gm.home, gm.week)
        fa, pa = look(gm.season, gm.away, gm.week)
        fin_rows.append({"id": gm.id, "diff": fh - fa})
        fp_rows.append({"id": gm.id, "diff": ph - pa})
    return pd.DataFrame(fin_rows), pd.DataFrame(fp_rows)


def main() -> int:
    preds = load_preds()
    registry = FB.load_registry()
    for fid in ("backup_qb", "coaching_change_y1", "finishing_drives_diff",
                "field_position_diff"):
        FB.assert_not_excluded(registry, fid)

    results = {}

    qb_ev = backup_qb_events(preds)
    results["backup_qb"] = FB.evaluate_event_factor("backup_qb", qb_ev, preds)

    co_ev = coaching_events(preds)
    results["coaching_change_y1"] = FB.evaluate_event_factor(
        "coaching_change_y1", co_ev[["id", "side"]], preds)

    # conditional: treated team rated below the season median rating that week
    if len(co_ev):
        m = co_ev.merge(preds, on="id")
        side_pred = np.where(m["side"] == "home", m["pred_b"], -m["pred_b"])
        weak = m.loc[side_pred < np.median(side_pred), ["id", "side"]]
        results["coaching_change_y1_weak_prior"] = FB.evaluate_event_factor(
            "coaching_change_y1_weak_prior", weak, preds)

    fin, fp = drive_factor_diffs(preds)
    results["finishing_drives_diff"] = FB.evaluate_continuous_factor(
        "finishing_drives_diff", fin, preds)
    results["field_position_diff"] = FB.evaluate_continuous_factor(
        "field_position_diff", fp, preds)

    pvals = {k: r.p_value for k, r in results.items()
             if r.p_value == r.p_value}          # drop NaN (n-gate fails)
    bh = FB.benjamini_hochberg(pvals, q=0.05) if pvals else {}

    lines = [f"=== FACTOR BATTERY (gate 6) — {TODAY}, eval 2021-2024 "
             f"walk-forward residuals, BH q<0.05 across {len(pvals)} "
             "hypotheses ==="]
    for k, r in results.items():
        if hasattr(r, "effect_pts"):
            desc = (f"event  n={r.n_treated:4d} effect {r.effect_pts:+.2f} pts "
                    f"(treated {r.treated_mean:+.2f} vs control "
                    f"{r.control_mean:+.2f})")
        else:
            desc = (f"cont.  n={r.n:4d} slope {r.slope:+.3f} r {r.r:+.3f}")
        verdict = "n-gate FAIL" if r.note else (
            "PASS" if bh.get(k, {}).get("pass") else "REJECTED")
        p_str = "nan" if r.p_value != r.p_value else f"{r.p_value:.4f}"
        lines.append(f"{k:32s} {desc}  p={p_str}  -> {verdict} {r.note}")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(ROOT, "data", "factor_battery_report.txt"), "w") as fh:
        fh.write(report + "\n")

    # Registry updates — never gated_active from here. Sub-hypotheses (e.g.
    # the _weak_prior conditional) append to history and the note but must
    # NOT overwrite the base entry's headline effect/n — the first run did
    # exactly that and had to be hand-corrected.
    for k, r in results.items():
        base_id = k.replace("_weak_prior", "")
        is_sub = k != base_id
        try:
            e = FB.registry_entry(registry, base_id)
        except FB.FactorError:
            continue
        passed = bh.get(k, {}).get("pass", False)
        n = getattr(r, "n_treated", getattr(r, "n", 0))
        eff = getattr(r, "effect_pts", getattr(r, "slope", None))
        note = (f"{k}: {'PASSED gate' if passed else 'REJECTED'} "
                f"{TODAY} (p={r.p_value if r.p_value == r.p_value else 'nan'}, "
                f"n={n}); {r.note or ''} "
                + ("wiring into a probability is a separate decision — status "
                   "stays display_only until then" if passed else
                   "kept as recorded rejection"))
        if is_sub:
            e["source_measurement"] = (e.get("source_measurement") or "") \
                + " || " + note.strip()
        else:
            e["status"] = "display_only"
            e["effect_size"] = (None if eff is None or eff != eff
                                else round(float(eff), 4))
            e["n"] = int(n)
            e["source_measurement"] = note.strip()
        e["last_validated"] = TODAY
        e["next_validation_due"] = "2026 season end"
        e.setdefault("revalidation_history", []).append(
            {"date": TODAY, "hypothesis": k,
             "result": "pass" if passed else "reject",
             "p": None if r.p_value != r.p_value else round(float(r.p_value), 5)})
    FB.save_registry(registry)
    print(f"\nregistry updated: data/factor_registry.json "
          f"({len(registry)} entries). Nothing was promoted to gated_active.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
