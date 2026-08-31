#!/usr/bin/env python3
"""backup_qb_pregame — the pregame-knowable QB-change variant.

PRE-REGISTERED in _worklog.md (2026-07-30) BEFORE this script was written or
run. Registered protocol, restated:

  * TREATED in week W iff, using ONLY weeks < W: the team's season-primary
    QB (most week-starts through W-1) has >= 3 starts AND the team's most
    recent game before W was started by a DIFFERENT QB. Who starts week W
    is deliberately not consulted (unknowable pregame here); starter
    returns dilute toward zero — the honest direction.
  * outcome/matching/machinery identical to the battery (signed walk-forward
    residual, matched control, Welch t, n-gate 40 pairs).
  * SINGLE hypothesis, alpha 0.05, no sweeps, no subsets.
  * result lands in the registry as `backup_qb_pregame` either way; a PASS
    is still display_only. A FAIL alongside the retro proxy's PASS means the
    reverse-causation caveat was the story — that ships as the conclusion.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from nflvalue import factor_battery as FB, ingest  # noqa: E402
from nflvalue import evidence as EV  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEASONS = [2021, 2022, 2023, 2024]
TODAY = "2026-07-30"
ALPHA = 0.05


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


def pregame_events(preds: pd.DataFrame) -> pd.DataFrame:
    qb = ingest.load_qb_starts(SEASONS)
    starters = qb[qb["is_week_starter"]].sort_values(["season", "week"])
    treated_team_weeks = set()
    for (season, team), grp in starters.groupby(["season", "team"],
                                                observed=True):
        seq = grp[["week", "athlete_id"]].drop_duplicates("week") \
                                         .sort_values("week")
        weeks = list(seq.itertuples())
        counts: dict = {}
        for i, r in enumerate(weeks):
            # State BEFORE this week's game: counts over weeks[0..i-1].
            if i > 0:
                primary = max(counts, key=counts.get) if counts else None
                last_starter = weeks[i - 1].athlete_id
                if (primary is not None and counts[primary] >= 3
                        and last_starter != primary):
                    treated_team_weeks.add((int(season), int(r.week),
                                            str(team)))
            counts[r.athlete_id] = counts.get(r.athlete_id, 0) + 1
    rows = []
    for g in preds.itertuples():
        for side, team in (("home", g.home), ("away", g.away)):
            if (int(g.season), int(g.week), str(team)) in treated_team_weeks:
                rows.append({"id": g.id, "side": side})
    return pd.DataFrame(rows)


def main() -> int:
    preds = load_preds()
    registry = FB.load_registry()
    events = pregame_events(preds)
    res = FB.evaluate_event_factor("backup_qb_pregame", events, preds,
                                   min_treated=40)
    passed = (res.p_value == res.p_value and res.p_value < ALPHA
              and not res.note)
    print(f"backup_qb_pregame: n_treated={res.n_treated} matched pairs, "
          f"effect {res.effect_pts:+.2f} pts (treated {res.treated_mean:+.2f} "
          f"vs control {res.control_mean:+.2f}), p={res.p_value:.4f} "
          f"-> {'PASS' if passed else 'FAIL/REJECT'} {res.note}")
    retro = FB.registry_entry(registry, "backup_qb")
    print(f"retro proxy for comparison: effect {retro['effect_size']} "
          f"(n={retro['n']}) — divergence here is the caveat speaking.")

    entry = {
        "factor_id": "backup_qb_pregame",
        "description": ("Pregame-knowable QB change: entered week W with the "
                        "most recent game started by a non-primary QB "
                        "(intent-to-treat; starter returns dilute to zero)."),
        "bucket": 3,
        "status": "display_only",
        "effect_size": (None if res.effect_pts != res.effect_pts
                        else round(float(res.effect_pts), 4)),
        "ci": None,
        "n": int(res.n_treated),
        "source_doc": "research/situational-factors.md",
        "source_measurement": (
            f"backup_qb_pregame: {'PASSED single pre-registered hypothesis' if passed else 'REJECTED'} "
            f"{TODAY} (p={res.p_value:.5f}, n={res.n_treated} pairs, effect "
            f"{res.effect_pts:+.2f} pts). Pre-registration in _worklog.md "
            f"{TODAY}. Compare retro proxy backup_qb "
            f"({retro['effect_size']} pts): the gap between the two is the "
            "size of the in-game reverse-causation contamination."),
        "test": "scripts/factor_qb_pregame.py",
        "last_validated": TODAY,
        "next_validation_due": "2026 season end",
        "revalidation_history": [{"date": TODAY,
                                  "hypothesis": "backup_qb_pregame",
                                  "result": "pass" if passed else "reject",
                                  "p": round(float(res.p_value), 5)}],
    }
    existing = [e for e in registry if e["factor_id"] == "backup_qb_pregame"]
    if existing:
        existing[0].update(entry)
    else:
        registry.append(entry)
    FB.save_registry(registry)
    print("registry updated (display_only either way; nothing gated_active).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
