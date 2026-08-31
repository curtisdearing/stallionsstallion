#!/usr/bin/env python3
"""Gate 3 — walk-forward backtest of the drive-level Monte Carlo (Workstream C).

Pre-registered BEFORE the first run (2026-07-30, matching backtest_b's window
so the two layers are graded on the same games):

  * eval seasons: 2021, 2022, 2023 pooled; weeks >= 4
  * burn-in for drive-rate labelling: 2019-2020 (never graded)
  * 2024 and 2025 are DELIBERATELY untouched — they stay a virgin holdout for
    Workstreams G (bakeoff) and D. Do not add them here because the numbers
    disappoint; that is exactly the failure mode the pre-registration exists
    to prevent.
  * metrics: margin MAE + correlation vs actual (head-to-head vs the median
    closing spread on the same games), Brier of P(home win) vs the de-vigged
    median closing moneyline on the same games, calibration by decile,
    simulated-margin-SD sanity vs the CFB residual scale, MC-vs-B agreement.

Honesty notes:
  * The model is NOT expected to beat the closing line. If it does, suspect a
    bug before an edge (BUILD_SPEC 0; backtest_b pooled: model 13.16 vs
    market 12.24 MAE).
  * Games either side of which is vetoed/unratable are skipped, not guessed.

Historical note on the REGISTERED 2026-07-30 run (data/backtest_c.csv, still
the published record, untouched by the correction below): its OT parameters
were fit league-level on the full 2021-2025 drives window, which was
IN-WINDOW relative to the eval seasons and was a recorded, bounded caveat at
the time (OT decides ~2% of sims; the exposure was reported, not hidden) —
but PaceModel was ALSO, undocumented, fit on the full pooled multi-season
frame. Both were leaks against the walk-forward guarantee the model claims.

2026-08-29 (Prompt 5A) correction: `walk_forward_backtest` now fits PaceModel
and OT params per evaluated week from only `season < S OR (season == S AND
week < W)` — never season-only, never the full window. This script no longer
pre-fits OT; it passes `ot=None` so the walk-forward fit applies. A week
whose pre-cutoff history cannot support an OT/pace fit yet is skipped, not
guessed — inevitable for the very earliest 2021+ weeks, since the OT-rules
era itself starts in 2021. Re-running this script with the fix WILL NOT
reproduce the registered 2026-07-30 numbers bit-for-bit; it must write to a
separately versioned, separately pre-registered output file, never
data/backtest_c.csv (see out_path()).

Writes data/backtest_c.csv (one row per simulated game, resumable per season:
re-invoke until exit 0; exit 3 = more seasons to do) and prints the gate
report. NOTHING HERE RECOMMENDS A BET.
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflvalue import cfb_ot, cfb_sim, engine as EG, ingest, oddsmath, ratings as R  # noqa: E402
from nflvalue import evidence as EV  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Files whose contents determine what a run produces. Hashed into the run
#: record so a later reader can tell whether the artifact is still the output
#: of the code it names — a benchmark whose producing method has changed
#: underneath it is not evidence about anything in particular.
CODE_FILES = (
    "scripts/backtest_c.py", "nflvalue/cfb_sim.py", "nflvalue/cfb_ot.py",
    "nflvalue/ratings.py", "nflvalue/kalman_engine.py", "nflvalue/engine.py",
    "config.json",
)

#: The method this script implements RIGHT NOW. Bump it when the fitting
#: procedure changes, so two artifacts produced by different methods can
#: never be compared as though they were the same measurement.
METHOD = "strict_walk_forward_pace_ot_v2_20260829"


def out_path(engine: str, variant: Optional[str] = None) -> str:
    """Engine-suffixed artifact. The registered 2026-07-30 ridge run stays
    data/backtest_c.csv; a non-incumbent engine writes its own file so the
    gate-3 record is never overwritten by a re-run.

    `variant`, when given, ALWAYS forces a suffixed filename — even for
    `engine == "ridge"` — so a corrected/candidate run (e.g. the 2026-08-29
    pace/OT walk-forward fix) can never silently land on data/backtest_c.csv
    just because it happens to use the incumbent engine. Use a dated variant
    name for anything that isn't the original registered run.
    """
    if variant:
        return os.path.join(ROOT, "data", f"backtest_c_{variant}.csv")
    name = "backtest_c.csv" if engine == "ridge" else f"backtest_c_{engine}.csv"
    return os.path.join(ROOT, "data", name)

BURN_IN = [2019, 2020]
EVAL_SEASONS = [2021, 2022, 2023]
FIRST_WEEK = 4
N_SIMS = 4000


def market_frames(seasons):
    """Median closing spread and de-vigged median-book ML win prob per game."""
    lines = ingest.load_lines(seasons)
    med_spread = (lines.dropna(subset=["spread"])
                  .groupby("game_id")["spread"].median())
    ml = lines.dropna(subset=["home_moneyline", "away_moneyline"]).copy()
    probs = {}
    for gid, grp in ml.groupby("game_id"):
        ps = []
        for r in grp.itertuples():
            dh = oddsmath.american_to_decimal(float(r.home_moneyline))
            da = oddsmath.american_to_decimal(float(r.away_moneyline))
            ph, _ = oddsmath.devig_multiplicative([dh, da])
            ps.append(ph)
        probs[gid] = float(np.median(ps))
    med_total = (lines.dropna(subset=["over_under"])
                 .groupby("game_id")["over_under"].median())
    return med_spread, probs, med_total


def run_season(season: int, engine: str, on_skip=None) -> pd.DataFrame:
    """OT params are NOT pre-fit here any more (2026-08-29, Prompt 5A
    correction): the old single `fit_ot_params` call over the full
    2021-2025 window fed every evaluated season, including 2021, an OT model
    that had seen 2022-2025 data it could not possibly have had pregame.
    `ot=None` makes `walk_forward_backtest` fit OT itself, per evaluated
    week, from only `season < S OR (season == S AND week < W)` — the same
    preregistered cutoff as pace and the drive rates. A diagnostic-only OT
    fit over all history available AS OF this season (still season-bounded,
    for a quick console sanity check) is printed but never touches the sim.
    """
    params = R.load_params()
    hist = sorted(set(BURN_IN) | {s for s in EVAL_SEASONS if s <= season})
    games = R.game_frame(hist)
    drives = ingest.load_drives(hist)
    try:
        diag_ot = cfb_ot.fit_ot_params(drives)
        print(f"[gate3] OT params, diagnostic only (history through season "
              f"{season}; the actual sim fits OT walk-forward per week): "
              f"{diag_ot.as_dict()}")
    except cfb_ot.OTFitError as exc:
        print(f"[gate3] OT diagnostic fit unavailable yet ({exc}); the sim "
              "will skip weeks it cannot fit OT for, same as always.")
    burn = [s for s in hist if s < season]
    df = cfb_sim.walk_forward_backtest(
        games, drives, params, ot=None, eval_seasons=[season],
        burn_in_seasons=burn, first_week=FIRST_WEEK, n_sims=N_SIMS,
        engine=engine, on_skip=on_skip,
        progress=lambda s, w, pool, done: print(
            f"  walked {s} wk{w:<2d} pool={pool} simulated={done}", flush=True)
        if w % 5 == 0 else None)
    return df


def read_artifact(out: str) -> pd.DataFrame:
    """The artifact so far. A file that exists but holds no rows is a real
    state under the corrected method — an eval season can be excluded
    entirely — so it reads as an empty typed frame, not an error."""
    if not os.path.exists(out) or os.path.getsize(out) <= 1:
        return pd.DataFrame(columns=list(cfb_sim.BACKTEST_COLUMNS))
    try:
        return pd.read_csv(out)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(cfb_sim.BACKTEST_COLUMNS))


def input_files(seasons) -> list:
    """Every cached file a run of this window reads."""
    out = []
    for season in sorted(set(seasons)):
        for kind in ("games", "drives"):
            rel = os.path.join("data", "cfbd", f"{kind}_{season}.parquet")
            if os.path.exists(os.path.join(ROOT, rel)):
                out.append(rel)
    return out


def build_run_record(artifact_id: str, out: str, engine: str,
                     exclusions: dict, seasons_done: list) -> dict:
    """Everything needed to trust, reproduce or distrust this artifact."""
    from datetime import datetime, timezone
    rel_out = os.path.relpath(out, ROOT)
    frame = read_artifact(out)
    seasons_used = sorted(set(BURN_IN) | set(EVAL_SEASONS))
    return {
        "artifact_id": artifact_id,
        "role": EV.ROLE_WK4PLUS,
        "status": EV.Status.CANDIDATE.value,
        "path": rel_out,
        "method": METHOD,
        "description": ("corrected strict walk-forward gate-3 run: PaceModel "
                        "and OT parameters fitted per evaluated week from "
                        "season < S or (season == S and week < W) only"),
        "engine_policy": {
            "margin_engine_config": EG.margin_engine(),
            "run_engine": engine,
            "early_season_weeks": list(EG.EARLY_SEASON_WEEKS),
            "early_season_engine": EG.EARLY_SEASON_ENGINE,
            "policy_changed_by_this_run": False,
        },
        "seasons": seasons_used,
        "weeks": {"eval_seasons": EVAL_SEASONS, "first_week": FIRST_WEEK,
                  "burn_in_seasons": BURN_IN,
                  "seasons_completed": sorted(seasons_done)},
        "seed": int(inspect.signature(cfb_sim.walk_forward_backtest)
                    .parameters["seed"].default),
        "n_rows": int(len(frame)),
        "n_sims_per_game": N_SIMS,
        "exclusions": exclusions,
        "code_hashes": {rel: EV.sha256_file(os.path.join(ROOT, rel))
                        for rel in CODE_FILES
                        if os.path.exists(os.path.join(ROOT, rel))},
        "input_hashes": {rel: EV.sha256_file(os.path.join(ROOT, rel))
                         for rel in input_files(seasons_used)},
        "schema": EV.csv_header(out),
        "content_sha256": EV.sha256_file(out),
        "generated_at_utc": datetime.now(timezone.utc)
                                    .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics": None,
        "selection_rule": (
            "selectable only when the run covers the pre-registered window "
            "(eval seasons 2021-2023, weeks >= 4) with the exclusions "
            "accounted for and a human records the acceptance. Coverage "
            "shortfall is not a reason to widen the window."),
        "not_selected_reason": "",
    }


def report(df: pd.DataFrame, engine: str) -> None:
    med_spread, ml_probs, med_total = market_frames(EVAL_SEASONS)
    df = df.copy()
    df["market"] = [-med_spread.get(g) if g in med_spread.index else np.nan
                    for g in df["id"]]
    df["market_p"] = [ml_probs.get(g, np.nan) for g in df["id"]]
    df["market_total"] = [med_total.get(g, np.nan) for g in df["id"]]
    df["won"] = (df["margin"] > 0).astype(float)

    print(f"\n=== GATE 3 — drive-level MC, walk-forward, weeks >= "
          f"{FIRST_WEEK}, pre-registered eval {EVAL_SEASONS} "
          f"[margin engine: {engine}] ===")
    for season in EVAL_SEASONS + ["pooled"]:
        d = df if season == "pooled" else df[df["season"] == season]
        if d.empty:
            continue
        both = d.dropna(subset=["market"])
        mae = (d["mc_margin"] - d["margin"]).abs().mean()
        corr = float(np.corrcoef(d["mc_margin"], d["margin"])[0, 1])
        mae_h2h = (both["mc_margin"] - both["margin"]).abs().mean()
        mae_mkt = (both["market"] - both["margin"]).abs().mean()
        print(f"{season}: n={len(d)} | MC margin MAE {mae:.2f} corr {corr:.3f}"
              f" || h2h n={len(both)}: MC {mae_h2h:.2f} vs market {mae_mkt:.2f}")

    mlb = df.dropna(subset=["market_p"])
    brier_model = float(((df["p_home_win"] - df["won"]) ** 2).mean())
    brier_model_mlset = float(((mlb["p_home_win"] - mlb["won"]) ** 2).mean())
    brier_market = float(((mlb["market_p"] - mlb["won"]) ** 2).mean())
    print(f"\nBrier (home win): model {brier_model:.4f} on all n={len(df)} | "
          f"same-set h2h n={len(mlb)}: model {brier_model_mlset:.4f} vs "
          f"de-vigged closing ML {brier_market:.4f} (coin flip = 0.2500)")

    # Calibration by predicted-probability decile
    df["bucket"] = pd.cut(df["p_home_win"], np.linspace(0, 1, 11))
    cal = df.groupby("bucket", observed=True).agg(
        n=("won", "size"), pred=("p_home_win", "mean"), obs=("won", "mean"))
    print("\ncalibration (pred bucket -> observed home-win rate):")
    for b, r in cal.iterrows():
        print(f"  {str(b):24s} n={int(r['n']):4d}  pred {r['pred']:.3f}  "
              f"obs {r['obs']:.3f}")

    resid_sd = float((df["mc_margin"] - df["margin"]).std())
    print(f"\ndistribution sanity: mean simulated margin SD "
          f"{df['mc_margin_sd'].mean():.2f} vs realised residual SD "
          f"{resid_sd:.2f} (CFB literature ~16-17; NFL ~13 — must be wider)")
    agree = float(np.corrcoef(df["mc_margin"], df["pred_b"])[0, 1])
    slope = float(np.polyfit(df["pred_b"], df["mc_margin"], 1)[0])
    print(f"MC vs B agreement: corr {agree:.3f}, slope {slope:.3f} "
          "(slope << 1 means the binned rates flatten the rating signal)")

    tot = df.dropna(subset=["market_total"])
    mae_t = (df["mc_total"] - df["total"]).abs().mean()
    mae_t_mkt = (tot["market_total"] - tot["total"]).abs().mean()
    mae_t_h2h = (tot["mc_total"] - tot["total"]).abs().mean()
    print(f"\ntotals (bonus, not gated): MC MAE {mae_t:.2f} | h2h n={len(tot)}: "
          f"MC {mae_t_h2h:.2f} vs market total {mae_t_mkt:.2f}")

    ot_share = float(df["p_ot"].mean())
    print(f"\nOT caveat quantified: mean P(OT) {ot_share:.3f}; a ±0.15 error in "
          f"the small-n 2-pt rate shifts P(home win) by < {0.15 * ot_share:.4f}"
          " — below Brier resolution at this n. In-window OT fit recorded.")
    print("\nThe model is NOT expected to beat the closing line; if it does, "
          "suspect a bug before an edge. Paper-only until gate 7.")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default=None,
                    help="margin engine (default: config margin_engine)")
    ap.add_argument("--variant", default=None,
                    help="write data/backtest_c_{variant}.csv instead of the "
                        "registered artifact, regardless of --engine "
                        "(required for any non-registered/corrected run)")
    args = ap.parse_args()
    engine = args.engine or EG.margin_engine()
    out = out_path(engine, variant=args.variant)
    artifact_id = (f"backtest_c_{args.variant}" if args.variant
                   else f"backtest_c_{engine}_registered")
    record_path = os.path.join(EV.run_record_dir(ROOT), f"{artifact_id}.json")
    exclusions = {"by_cause": {}, "weeks": [], "games": []}
    prior_record = {}
    if args.variant and os.path.exists(record_path):
        import json as _json
        with open(record_path) as fh:
            prior_record = _json.load(fh)
        exclusions = prior_record.get("exclusions", exclusions)

    def on_skip(kind, season, week, game_id, reason):
        exclusions["by_cause"][kind] = exclusions["by_cause"].get(kind, 0) + 1
        entry = {"kind": kind, "season": season, "week": week,
                 "reason": reason[:160]}
        if game_id is None:
            exclusions["weeks"].append(entry)
        else:
            exclusions["games"].append({**entry, "game_id": game_id})

    # Resume tracking cannot key off rows in the artifact any more: under the
    # corrected cutoff an eval season can contribute ZERO rows, and "no rows
    # for 2021" would then mean "2021 not done yet" forever. Completed seasons
    # are recorded explicitly.
    done_rows = read_artifact(out)
    completed = set(int(s) for s in done_rows["season"].dropna().unique()) \
        if len(done_rows) else set()
    completed |= set(int(s) for s in prior_record.get("weeks", {})
                     .get("seasons_completed", []))

    for season in EVAL_SEASONS:
        if season in completed:
            continue
        df = run_season(season, engine, on_skip=on_skip)
        if len(df):
            df.to_csv(out, mode="a", header=not os.path.exists(out),
                      index=False)
        elif not os.path.exists(out):
            # Nothing simulatable yet, but the artifact must still exist with
            # its schema: an unreadable file is worse than an empty one.
            df.to_csv(out, index=False)
        completed.add(season)
        print(f"[gate3] season {season}: {len(df)} simulated games written; "
              f"exclusions so far {exclusions['by_cause']}")
        if args.variant:
            # A candidate run records its own provenance as it goes, so an
            # interrupted resume never loses the exclusion accounting.
            EV.write_run_record(ROOT, build_run_record(
                artifact_id, out, engine, exclusions, sorted(completed)))
        return 3

    if args.variant:
        path = EV.write_run_record(ROOT, build_run_record(
            artifact_id, out, engine, exclusions, sorted(completed)))
        print(f"[gate3] run record: {os.path.relpath(path, ROOT)}")
    frame = read_artifact(out)
    if frame.empty:
        print("[gate3] NO SIMULATED GAMES: every evaluated week was excluded. "
              "Reporting nothing rather than a metric over zero rows.")
        return 0
    report(frame, engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
