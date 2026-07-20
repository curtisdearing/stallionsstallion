#!/usr/bin/env python3
"""Preseason projections for 2026 week 0 and week 1 (CFBD labels both week 1).

HONESTY HEADER — read before quoting any number below.
  * As of 2026-07-19 CFBD serves NO 2026 talent and NO 2026 returning
    production (both endpoints return empty). The preseason prior therefore
    contains ONLY the previous-season rating component, diluted exactly as
    `preseason_prior` would dilute it (talent/returning z-scores enter as 0),
    i.e. prior = prior_strength * z(prev_net) / 3. Our own research says
    returning production is the DOMINANT week-0 signal — and it is missing.
    These are "last year's team + HFA" projections, nothing more.
  * 2026 conference realignment inside the solve is 2025's map (ratings are
    credited to the conference a team played in last season).
  * Teams new to FBS in 2026 have no rating and are VETOED, not guessed at.
  * Week-0/week-1 games are the least predictable of the season. Bands are
    wide because they should be.
  * PAPER-ONLY. Nothing here recommends a bet (gate 7 has not run).

Writes data/predictions_2026_wk01.csv and prints the table.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflvalue import ingest, ratings as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "predictions_2026_wk01.csv")
HISTORY = [2019, 2020, 2021, 2022, 2023, 2024, 2025]


def main() -> int:
    params = R.load_params()
    games = R.game_frame(HISTORY)

    # 2025 final ratings -> prev_net for the 2026 prior.
    prev_tbl = R.solve(games, 2025, 25, params)
    prev_frame = prev_tbl.to_frame()
    prev_net = {r.team: float(r.net) for r in prev_frame.itertuples() if not r.veto}

    # 2026 teams: conference + FBS flags from the 2026 /teams pull.
    teams26 = ingest.load("teams", [2026])
    fbs26 = {str(r.team) for r in teams26.itertuples() if bool(r.is_fbs)}
    conf26 = {str(r.team): (str(r.conference) or "Independent")
              for r in teams26.itertuples()}

    # Prior injection mirroring preseason_prior's composite with talent and
    # returning UNAVAILABLE (their z-terms are 0): strength * z(prev) * w_p/tot.
    vals = np.array(list(prev_net.values()), dtype=float)
    mu, sd = float(vals.mean()), float(vals.std(ddof=1))
    z = {t: (v - mu) / sd for t, v in prev_net.items()}
    by_conf: dict = {}
    for t, v in z.items():
        by_conf.setdefault(conf26.get(t, "Independent"), []).append(v)
    conf_mean = {c: float(np.mean(v)) for c, v in by_conf.items()}
    w_tot = params.prior_w_talent + params.prior_w_returning + params.prior_w_prev_rating
    scale = params.prior_strength * params.prior_w_prev_rating / w_tot
    prior = {t: scale * z.get(t, conf_mean.get(conf26.get(t, "Independent"), 0.0))
             for t in fbs26 | set(prev_net)}

    tbl = R.solve(games, 2026, 1, params, prior=prior)
    frame = tbl.to_frame().set_index("team")

    # Forecast sigma comes from OUT-OF-SAMPLE walk-forward residuals
    # (data/backtest_b.csv, 2021-2023, n=1,765) — NOT from tbl.residual_sd,
    # which is the in-sample weighted solve residual (~6 pts) and would make a
    # 7-point favourite look like an 86% winner. Empirically ~16.5 pts, which
    # matches the CFB literature for margin sd around a point spread.
    bt = pd.read_csv(os.path.join(ROOT, "data", "backtest_b.csv"))
    sigma = float((bt["pred"] - bt["margin"]).std(ddof=1))

    sched = ingest.load("games", [2026])
    sched = sched[sched["week"] == 1].sort_values("start_date")

    rows = []
    for r in sched.itertuples():
        date = str(r.start_date)[:10]
        label = "wk0" if date < "2026-09-01" else "wk1"
        home, away = str(r.home_team), str(r.away_team)
        h = home if home in fbs26 else R.FCS_TEAM
        a = away if away in fbs26 else R.FCS_TEAM
        note = ""
        if a == R.FCS_TEAM:
            note = "opp pooled as FCS"
        if h == R.FCS_TEAM:
            note = "home pooled as FCS"
        try:
            m = tbl.predict_margin(h, a, bool(r.neutral_site))
            veto = bool(frame.loc[h, "veto"]) if h in frame.index else True
            veto = veto or (bool(frame.loc[a, "veto"]) if a in frame.index else True)
            band = float(frame.loc[h, "net_band"]) + float(frame.loc[a, "net_band"]) \
                if h in frame.index and a in frame.index else float("nan")
            p_home = 0.5 * (1.0 + math.erf(m / (sigma * math.sqrt(2))))
            rows.append({"slate": label, "date": date, "home": home, "away": away,
                         "neutral": bool(r.neutral_site), "pred_margin": round(m, 1),
                         "p_home_win": round(p_home, 3),
                         "band_pm": round(band, 1) if band == band else None,
                         "veto": veto, "note": note})
        except R.RatingsError as e:
            if R.FCS_TEAM in (h, a):
                # Not an error: the pooled FCS pseudo-team is excluded from
                # connectivity ON PURPOSE, so the model refuses to price
                # FBS-vs-FCS games rather than guess at them (same rule the
                # backtest scores by).
                why = "FBS-vs-FCS: not priced, by design"
            else:
                why = f"NO RATING ({type(e).__name__}): new-to-FBS or unseen team"
            rows.append({"slate": label, "date": date, "home": home, "away": away,
                         "neutral": bool(r.neutral_site), "pred_margin": None,
                         "p_home_win": None, "band_pm": None, "veto": True,
                         "note": why})

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} games; sigma={sigma:.1f})")
    for lab in ("wk0", "wk1"):
        d = df[df["slate"] == lab]
        print(f"\n=== {lab}: {len(d)} games, {int(d['veto'].sum())} vetoed ===")
        print(d.to_string(index=False))
    print("\nPrior = last-season component ONLY (no 2026 talent/returning at "
          "CFBD yet). Paper-only; nothing here recommends a bet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
