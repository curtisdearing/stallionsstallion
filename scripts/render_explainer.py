#!/usr/bin/env python3
"""Render the Workstream F explainer for one historical week, walk-forward.

Everything on the page is computed exactly as the gate-3 backtest would have
computed it AT that week: the rating table sees only earlier games, the drive
rates are fit on drives labelled strictly before the week, and the market
columns come from the cached closing medians. The page itself is read-only by
construction (F8) — it is rendered from a serialized explanation object after
all numbers are final.

Usage:
    python3 scripts/render_explainer.py --season 2023 --week 10
    -> data/explainer_{season}_wk{week}.html
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflvalue import cfb_ot, cfb_sim, explain, gameline as GL  # noqa: E402
from nflvalue import engine as EG, ingest, ratings as R  # noqa: E402
from nflvalue import evidence as EV  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_START = 2019


def rate_models_before(games, drives_c, params, season, week):
    """Fit DriveRateModel/PaceModel on drives strictly before (season, week),
    each labelled with its OWN week's table — the gate-3 discipline."""
    pool = []
    for s in sorted(games["season"].unique()):
        if s > season:
            continue
        for w in sorted(games.loc[games["season"] == s, "week"].unique()):
            if s == season and w >= week:
                break
            try:
                tbl = EG.solve_production_table(games, int(s), int(w), params)
            except (R.RatingsError, np.linalg.LinAlgError):
                # Includes EarlySeasonPriorError: the first history season has
                # no prior year, so its early weeks cannot be labelled the way
                # production prices them. Skipped, never mislabelled.
                continue
            gw = games[(games["season"] == s) & (games["week"] == w)]
            dw = drives_c[(drives_c["season"] == s) & (drives_c["week"] == w)]
            lab = cfb_sim.label_week_drives(dw, gw, tbl)
            if len(lab):
                pool.append(lab)
    train = pd.concat(pool, ignore_index=True)
    past = drives_c[(drives_c["season"] < season)
                    | ((drives_c["season"] == season)
                       & (drives_c["week"] < week))]
    return cfb_sim.DriveRateModel.fit(train), cfb_sim.PaceModel.fit(past)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2023)
    ap.add_argument("--week", type=int, default=10)
    ap.add_argument("--n-sims", type=int, default=4000)
    args = ap.parse_args()

    seasons = list(range(HISTORY_START, args.season + 1))
    params = R.load_params()
    games = R.game_frame(seasons)
    drives = ingest.load_drives(seasons)
    drives_c, _ = cfb_sim.classify_drives(drives)
    drives_c = drives_c[drives_c["start_period"] <= 4]
    ot = cfb_ot.fit_ot_params(ingest.load_drives(range(2021, 2026)))
    rates, pace = rate_models_before(games, drives_c, params,
                                     args.season, args.week)
    # The page IS a published forecast, so it is priced by the production
    # policy — the same table the weekly loop would publish for this week.
    tbl = EG.solve_production_table(games, args.season, args.week, params)

    lines = ingest.load_lines([args.season])
    med_spread = (lines.dropna(subset=["spread"])
                  .groupby("game_id")["spread"].median())
    med_total = (lines.dropna(subset=["over_under"])
                 .groupby("game_id")["over_under"].median())

    # Model MAE comes from the SELECTED benchmark or from nowhere. Reading
    # data/backtest_c.csv directly is what let a pre-correction figure keep
    # appearing in the explainer after the method it measured was replaced.
    bt, bench, notice = EV.benchmark_or_notice(ROOT, EV.ROLE_WK4PLUS)
    model_mae = (None if bt is None
                 else float((bt["mc_margin"] - bt["margin"]).abs().mean()))
    if bt is None:
        print(f"[explainer] benchmark unavailable: {notice}")
    clv = pd.read_csv(os.path.join(ROOT, "data", "clv_log_backtest.csv"))
    clv_slices = clv.groupby("slice")["clv_pts"].agg(["count", "mean"])

    rng = np.random.default_rng(20260730)
    wk = games[(games["season"] == args.season) & (games["week"] == args.week)]
    exps, skipped = [], []
    for g in wk.itertuples():
        if not (g.home_fbs and g.away_fbs):
            continue
        try:
            e_h, e_a = cfb_sim.side_edges(tbl, g.home_team, g.away_team,
                                          bool(g.neutral_site))
        except R.DisconnectedTeamError as exc:
            skipped.append(f"{g.away_team} @ {g.home_team}: {exc}")
            continue
        psd = cfb_sim.param_margin_sd(tbl, g.home_team, g.away_team)
        sim = cfb_sim.simulate_game(
            rng, e_h, e_a, rates, pace, ot, n=args.n_sims, param_sd=psd,
            param_rng=(np.random.default_rng([20260730, int(g.id)])
                       if psd > 0.0 else None))
        mkt = med_spread.get(g.id)
        exps.append(explain.build_explanation(
            tbl=tbl,
            game={"id": g.id, "season": g.season, "week": g.week,
                  "home_team": g.home_team, "away_team": g.away_team,
                  "neutral_site": bool(g.neutral_site)},
            sim=sim,
            market=None if mkt is None else -float(mkt),
            market_total=med_total.get(g.id),
            model_mae=model_mae,
            slice_tag=GL.slice_tag(str(g.home_conf), str(g.away_conf),
                                   int(g.week), bool(g.neutral_site)),
            clv_slices=clv_slices))

    # Gate-5 checks, asserted at render time — fail loud, never render wrong
    for e in exps:
        s = explain.contributions_sum(e)
        assert abs(s - e["mc_margin"]) < 1e-6, \
            f"contributions do not sum for game {e['game_id']}: {s} vs {e['mc_margin']}"
        assert e["why_not"], f"empty why-not panel for game {e['game_id']} — bug"

    html_text = explain.render_week_html(
        exps, clv_slices=clv_slices,
        title=f"stallionsstallion — {args.season} week {args.week} explainer",
        generated_note=(f"walk-forward as of {args.season} wk{args.week}; "
                        f"{len(exps)} games simulated, {len(skipped)} skipped "
                        "as unratable (refused, not guessed)"))
    out = os.path.join(ROOT, "data",
                       f"explainer_{args.season}_wk{args.week:02d}.html")
    with open(out, "w") as fh:
        fh.write(html_text)
    print(f"[explainer] wrote {out} — {len(exps)} games, "
          f"{len(skipped)} skipped")
    for s in skipped:
        print(f"  skipped: {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
