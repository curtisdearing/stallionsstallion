#!/usr/bin/env python3
"""Workstream I — the weekly run loop (WEEKLY_LOOP_SPEC Phase 1, plus the
simulate/explain stages that unblocked when Workstreams C and F landed).

Cadence is CFB-shaped, not the inherited NFL wed/t90/tuesday
(`scripts/auto_weekly.py` stays as reference — its NFL job names encode a
Sun/Mon game week that CFB does not have):

  --job week-open   Tuesday: freshness-gated ingest refresh, warm ratings
                    refit, incremental honesty report, simulate + render the
                    upcoming week's explainer, notify.
  --job pregame     Tue-Sat, run as often as scheduled: cheap freshness
                    check; re-simulates + re-renders only if the cache moved.
  --job grade       Sat night -> Sunday, idempotent: refresh finals for the
                    current week, update the honesty report, notify.

Spec furniture, all load-bearing:
  * single-writer lock  data/.weekly.lock (PID+UTC+job; stale after 2h)
  * run manifest        data/manifests/{season}_wk{week}_{stamp}.json —
                        config hash, data-as-of per table, git sha, seed,
                        call tally before/after, per-stage pass/fail
  * stage sentinels     data/manifests/_done_{season}_wk{week}_{stage}
                        (a killed run resumes at the next stage, never
                        restarts the week)
  * budget guard        refuses (loud) if the projected calls would push the
                        monthly tally within 50 of the 1,000 cap
  * degraded, never silent: a failed freshness gate still writes a manifest
                        (degraded=true) and still notifies, saying so.

Phase-2 note (2026-07-30): the spec's blend + CLV stages are DELIBERATELY
ABSENT — the Phase-2 gate FAILED on 2021-2023 and on the 2024 holdout (the
blend never beat the close; see backtest_d*.py), so there is no blend to run
and no bet-candidate log to write. The simulate stage publishes the model's
own distributions with the explainer's full why-not honesty instead. If a
future D passes on new data, its stages slot between refit and notify
exactly as the spec designed. PAPER-ONLY throughout; nothing here is a
betting signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflvalue import cfb_ot, cfb_sim, explain, freshness, gameline as GL  # noqa: E402
from nflvalue import engine as EG, ingest, ratings as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MANIFESTS = os.path.join(DATA, "manifests")
LOCK_PATH = os.path.join(DATA, ".weekly.lock")
LOCK_STALE_HOURS = 2.0
BUDGET_HEADROOM = 50
SEED = 20260730
HISTORY_START = 2019


class LoopError(RuntimeError):
    pass


def utcstamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------- #
# Lock
# --------------------------------------------------------------------------- #
def acquire_lock(job: str) -> bool:
    if os.path.exists(LOCK_PATH):
        try:
            with open(LOCK_PATH) as fh:
                held = json.load(fh)
            age_h = (datetime.now(timezone.utc)
                     - datetime.fromisoformat(held["utc"])).total_seconds() / 3600
        except Exception:  # noqa: BLE001
            age_h = float("inf")
        if age_h < LOCK_STALE_HOURS:
            print(f"[lock] live lock held by pid={held.get('pid')} "
                  f"job={held.get('job')} age={age_h:.2f}h — exiting clean")
            return False
        print(f"[lock] stale lock ({age_h:.1f}h old) — overriding")
    os.makedirs(DATA, exist_ok=True)
    with open(LOCK_PATH, "w") as fh:
        json.dump({"pid": os.getpid(), "job": job,
                   "utc": datetime.now(timezone.utc).isoformat()}, fh)
    return True


def release_lock() -> None:
    try:
        os.unlink(LOCK_PATH)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Manifest + sentinels
# --------------------------------------------------------------------------- #
def config_hash() -> str:
    h = hashlib.sha256()
    for name in ("config.json", "data/rating_params.json"):
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def data_as_of() -> dict:
    out = {}
    cfbd = os.path.join(DATA, "cfbd")
    if os.path.isdir(cfbd):
        for fn in sorted(os.listdir(cfbd)):
            if fn.endswith(".parquet"):
                kind = fn.rsplit("_", 1)[0]
                ts = os.path.getmtime(os.path.join(cfbd, fn))
                out[kind] = max(out.get(kind, 0.0), ts)
    return {k: datetime.fromtimestamp(v, timezone.utc).isoformat()
            for k, v in out.items()}


def sentinel_path(season: int, week: int, stage: str) -> str:
    return os.path.join(MANIFESTS, f"_done_{season}_wk{week:02d}_{stage}")


def stage_done(season: int, week: int, stage: str) -> bool:
    return os.path.exists(sentinel_path(season, week, stage))


def mark_done(season: int, week: int, stage: str) -> None:
    os.makedirs(MANIFESTS, exist_ok=True)
    with open(sentinel_path(season, week, stage), "w") as fh:
        fh.write(utcstamp() + "\n")


# --------------------------------------------------------------------------- #
# Season/week detection — self-scheduling, offseason-safe
# --------------------------------------------------------------------------- #
def current_season_week(now=None):
    """(season, next_week, in_season). Offseason -> in_season=False."""
    now = now or datetime.now(timezone.utc)
    season = now.year if now.month >= 2 else now.year - 1
    games = ingest.load_games([season])
    if games.empty or "start_date" not in games.columns:
        return season, 1, False
    dates = pd.to_datetime(games["start_date"], errors="coerce", utc=True)
    played = games["home_points"].notna()
    future = dates > now
    if not future.any() and played.all():
        return season, int(games["week"].max()), False
    if future.all():
        days_to_first = (dates.min() - pd.Timestamp(now)).days
        return season, int(games.loc[future, "week"].min()), days_to_first <= 7
    return season, int(games.loc[future, "week"].min()), True


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_budget_guard(projected_calls: int) -> None:
    used = ingest.calls_used_this_month()
    if used + projected_calls > ingest.CALL_BUDGET_MONTHLY - BUDGET_HEADROOM:
        raise LoopError(
            f"budget guard: {used} used + ~{projected_calls} projected would "
            f"land within {BUDGET_HEADROOM} of the "
            f"{ingest.CALL_BUDGET_MONTHLY}/month cap — refusing, loudly")


def stage_refresh(season: int) -> dict:
    res = ingest.refresh_season(season, checkpoint=True)
    drv = ingest.refresh_drives(season, checkpoint=True) \
        if os.path.exists(os.path.join(DATA, "cfbd", f"games_{season}.parquet")) \
        else None
    return {"season_refresh": str(res), "stale": bool(res.stale),
            "drives_refresh": (str(drv) if drv else "skipped"),
            "calls": res.calls + (drv.calls if drv else 0)}


def stage_freshness(season: int) -> dict:
    games = ingest.load_games([season])
    now = datetime.now(timezone.utc)
    feeds = [freshness.Feed(
        name="games",
        timestamp=datetime.fromtimestamp(os.path.getmtime(
            os.path.join(DATA, "cfbd", f"games_{season}.parquet")),
            timezone.utc) if os.path.exists(
            os.path.join(DATA, "cfbd", f"games_{season}.parquet")) else None,
        n_records=len(games), schema_ok=True, load_bearing=True)]
    return freshness.gate(feeds, as_of=now,
                          staleness_hours={"games": 24 * 8})


def stage_refit(season: int, week: int) -> dict:
    params = R.load_params()
    games = R.game_frame(list(range(HISTORY_START, season + 1)))
    tbl = EG.solve_table(games, season, week, params)
    frame = tbl.to_frame()
    ingest._atomic_write_parquet(
        frame, os.path.join(DATA, f"ratings_{season}.parquet"))
    return {"fingerprint": tbl.fingerprint(), "n_teams": len(frame),
            "n_games_fit": tbl.n_games_fit, "week": week}


def stage_honesty(season: int) -> dict:
    """Incremental model-vs-close honesty report over completed weeks."""
    params = R.load_params()
    games = R.game_frame(list(range(HISTORY_START, season + 1)))
    lines = ingest.load_lines([season])
    med = (lines.dropna(subset=["spread"]).groupby("game_id")["spread"].median()
           if len(lines) else pd.Series(dtype=float))
    season_games = games[(games["season"] == season)
                         & games["home_points"].notna()]
    recs = []
    for w in sorted(season_games["week"].unique()):
        if w < 2:
            continue
        try:
            tbl = EG.solve_table(games, season, int(w), params)
        except (R.RatingsError, np.linalg.LinAlgError):
            continue
        for g in season_games[season_games["week"] == w].itertuples():
            if not (g.home_fbs and g.away_fbs):
                continue
            try:
                pred = tbl.predict_margin(g.home_team, g.away_team,
                                          bool(g.neutral_site))
            except R.DisconnectedTeamError:
                continue
            spread = med.get(g.id)
            recs.append({"season": season, "week": int(w), "id": int(g.id),
                         "pred": pred, "margin": float(g.margin),
                         "market": (-float(spread)
                                    if spread is not None
                                    and not np.isnan(spread) else np.nan)})
    df = pd.DataFrame(recs)
    out = os.path.join(DATA, f"honesty_{season}.csv")
    if df.empty:
        return {"n": 0, "note": "no completed gradable games yet"}
    df.to_csv(out, index=False)
    both = df.dropna(subset=["market"])
    return {"n": len(df),
            "model_mae": round(float((df["pred"] - df["margin"]).abs().mean()), 3),
            "h2h_n": len(both),
            "model_mae_h2h": (round(float((both["pred"] - both["margin"])
                                          .abs().mean()), 3) if len(both) else None),
            "market_mae": (round(float((both["market"] - both["margin"])
                                       .abs().mean()), 3) if len(both) else None),
            "path": os.path.basename(out)}


def week_prediction_rows(season: int, week: int, upcoming: pd.DataFrame,
                         simulated: dict, refusals: dict) -> pd.DataFrame:
    """Render one public-contract row for every scheduled game.

    The model may honestly refuse a matchup, but that refusal must be visible
    rather than silently disappearing from the weekly artifact or hub feed.
    """
    rows = []
    for game in upcoming.sort_values(["start_date", "id"], kind="mergesort").itertuples():
        game_id = int(game.id)
        date = str(game.start_date)[:10]
        row = {
            "season": season,
            "week": week,
            "id": game_id,
            "slate": "wk0" if week == 1 and date < f"{season}-09-01" else f"wk{week}",
            "date": date,
            "home": str(game.home_team),
            "away": str(game.away_team),
            "neutral": bool(game.neutral_site),
        }
        sim = simulated.get(game_id)
        if sim is not None:
            row.update({
                "mc_margin": sim["mc_margin"],
                "mc_margin_sd": sim["mc_margin_sd"],
                "mc_total": sim["mc_total"],
                "p_home_win": sim["p_home_win"],
                "param_sd": sim["param_sd"],
                "pred_margin": sim["mc_margin"],
                "band_pm": sim["mc_margin_sd"],
                "veto": False,
                "note": None,
            })
        else:
            note = refusals.get(game_id)
            if note is None:
                if bool(game.home_fcs) and bool(game.away_fcs):
                    note = "Non-FBS matchup: no validated lower-division model"
                elif bool(game.home_fcs) or bool(game.away_fcs):
                    note = "FBS-vs-non-FBS: no validated cross-division model"
                else:
                    note = "NO RATING: team lacks a validated FBS rating"
            row.update({
                "mc_margin": None,
                "mc_margin_sd": None,
                "mc_total": None,
                "p_home_win": None,
                "param_sd": None,
                "pred_margin": None,
                "band_pm": None,
                "veto": True,
                "note": note,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def stage_simulate(season: int, week: int) -> dict:
    """Simulate the upcoming week and render its explainer. Paper-only."""
    params = R.load_params()
    seasons = list(range(HISTORY_START, season + 1))
    games = R.game_frame(seasons)
    upcoming_all = ingest.load_games([season])
    upcoming = upcoming_all[(upcoming_all["week"] == week)
                            & upcoming_all["home_points"].isna()]
    if upcoming.empty:
        return {"n": 0, "note": "no unplayed games in target week"}
    drives = ingest.load_drives(seasons)
    drives_c, _ = cfb_sim.classify_drives(drives)
    drives_c = drives_c[drives_c["start_period"] <= 4]
    ot = cfb_ot.fit_ot_params(ingest.load_drives(range(2021, season + 1)))

    pool = []
    for s in seasons:
        for w in sorted(games.loc[games["season"] == s, "week"].unique()):
            if s == season and w >= week:
                break
            try:
                tbl_w = EG.solve_table(games, int(s), int(w), params)
            except (R.RatingsError, np.linalg.LinAlgError):
                continue
            gw = games[(games["season"] == s) & (games["week"] == w)]
            dw = drives_c[(drives_c["season"] == s) & (drives_c["week"] == w)]
            lab = cfb_sim.label_week_drives(dw, gw, tbl_w)
            if len(lab):
                pool.append(lab)
    rates = cfb_sim.DriveRateModel.fit(pd.concat(pool, ignore_index=True))
    pace = cfb_sim.PaceModel.fit(drives_c)
    tbl = EG.solve_table(games, season, week, params)

    teams = ingest.load_teams([season])
    conf = {str(r.team): str(r.conference) for r in teams.itertuples()}
    rng = np.random.default_rng(SEED)
    simulated, refusals, exps, skipped = {}, {}, [], 0
    bt = pd.read_csv(os.path.join(DATA, "backtest_c.csv"))
    model_mae = float((bt["mc_margin"] - bt["margin"]).abs().mean())
    clv = pd.read_csv(os.path.join(DATA, "clv_log_backtest.csv"))
    clv_slices = clv.groupby("slice")["clv_pts"].agg(["count", "mean"])
    for g in upcoming.itertuples():
        game_id = int(g.id)
        home, away = str(g.home_team), str(g.away_team)
        if bool(g.home_fcs) or bool(g.away_fcs):
            skipped += 1
            continue
        try:
            e_h, e_a = cfb_sim.side_edges(tbl, home, away,
                                          bool(g.neutral_site))
        except R.DisconnectedTeamError as exc:
            skipped += 1
            refusals[game_id] = (
                f"NO RATING ({type(exc).__name__}): new-to-FBS or unseen team")
            continue
        psd = cfb_sim.param_margin_sd(tbl, home, away)
        sim = cfb_sim.simulate_game(
            rng, e_h, e_a, rates, pace, ot, n=4000, param_sd=psd,
            param_rng=(np.random.default_rng([SEED, game_id])
                       if psd > 0.0 else None))
        simulated[game_id] = {
            "mc_margin": sim["margin_mean"],
            "mc_margin_sd": sim["margin_sd"],
            "mc_total": sim["total_mean"],
            "p_home_win": sim["p_home_win"],
            "param_sd": psd,
        }
        exps.append(explain.build_explanation(
            tbl=tbl, game={"id": g.id, "season": season, "week": week,
                           "home_team": home, "away_team": away,
                           "neutral_site": bool(g.neutral_site)},
            sim=sim, market=None, market_total=None, model_mae=model_mae,
            slice_tag=GL.slice_tag(conf.get(home, ""), conf.get(away, ""),
                                   week, bool(g.neutral_site)),
            clv_slices=clv_slices))
    artifact_rows = week_prediction_rows(season, week, upcoming, simulated, refusals)
    artifact_rows.to_csv(
        os.path.join(DATA, f"predictions_{season}_wk{week:02d}.csv"),
        index=False)
    if simulated:
        html_text = explain.render_week_html(
            exps, clv_slices=clv_slices,
            title=f"stallionsstallion — {season} week {week} (upcoming)",
            generated_note=(f"weekly loop {utcstamp()}; {len(simulated)} games, "
                            f"{skipped} refused as unratable"))
        with open(os.path.join(DATA,
                               f"explainer_{season}_wk{week:02d}.html"),
                  "w") as fh:
            fh.write(html_text)
    return {"n": len(simulated), "skipped": skipped,
            "n_schedule": len(artifact_rows)}


def stage_recap(season: int, week: int) -> dict:
    """Weekly scoreboard (WEEKLY_SCOREBOARD_SPEC): grade the PUBLISHED
    projections for the most recently completed week; append/revisit the
    hypothesis ledger. Read-only w.r.t. the model by construction."""
    from nflvalue import scoreboard
    target = week - 1
    if target < 1:
        return {"n": 0, "note": "no completed week to recap yet"}
    try:
        r = scoreboard.build_recap(season, target)
    except scoreboard.ScoreboardError as exc:
        return {"n": 0, "note": str(exc)[:200]}
    return {"n": r["n_graded"], "week_recapped": target,
            "model_mae": r["model_mae"], "market_mae": r["market_mae"],
            "ledger_new": r["ledger_new"],
            "path": f"recaps/{season}_wk{target:02d}.md"}


def stage_clv_capture(season: int, week: int) -> dict:
    """Close-free live CLV capture for the completed week: the PUBLISHED
    MC margin (pregame, never recomputed) vs same-book open, scored against
    that book's close. Appends data/clv_live_{season}.csv. PAPER-ONLY
    research log — Workstream K's raw material, never a recommendation."""
    from nflvalue import gameline as GL
    target = week - 1
    pub_path = os.path.join(DATA, f"predictions_{season}_wk{target:02d}.csv")
    if target < 1 or not os.path.exists(pub_path):
        return {"n": 0, "note": "no published projections for completed week"}
    pub = pd.read_csv(pub_path)
    games = R.game_frame([season])
    lines = ingest.load_lines([season])
    wk_lines = lines[lines["game_id"].isin(set(pub["id"]))]
    signal = dict(zip(pub["id"].astype(int), pub["mc_margin"]))
    log = GL.build_clv_log(games, wk_lines, signal, min_edge_pts=2.0)
    if log.empty:
        return {"n": 0, "note": "no open+close candidates this week"}
    out = os.path.join(DATA, f"clv_live_{season}.csv")
    if os.path.exists(out):
        prev = pd.read_csv(out)
        key = set(zip(prev["game_id"], prev["book"]))
        log = log[[(g, b) not in key for g, b in
                   zip(log["game_id"], log["book"])]]
        if len(log):
            log.to_csv(out, mode="a", header=False, index=False)
    else:
        log.to_csv(out, index=False)
    return {"n": int(len(log)), "week_captured": target,
            "mean_clv": (None if log.empty
                         else round(float(log["clv_pts"].mean()), 3))}


def notify(manifest: dict) -> dict:
    """Discord if configured; clean skip otherwise. Fail-of-stage notifies too."""
    try:
        from nflvalue import notify as N
        url = N.resolve_webhook()
        if not url:
            return {"status": "skipped", "reason": "no webhook configured"}
        import urllib.request
        body = {"content": (
            f"stallionsstallion weekly [{manifest['job']}] "
            f"{manifest['season']} wk{manifest['week']} — "
            f"{'DEGRADED' if manifest.get('degraded') else 'ok'}; stages: "
            + ", ".join(f"{k}:{'ok' if v.get('ok') else 'FAIL'}"
                        for k, v in manifest["stages"].items())
            + ". PAPER-ONLY research output.")}
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=15)
        return {"status": "sent"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reason": str(exc)[:200]}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(job: str, force_week: int = None) -> int:
    season, week, in_season = current_season_week()
    if force_week:
        week, in_season = force_week, True
    stamp = utcstamp()
    manifest = {"job": job, "season": season, "week": week, "stamp": stamp,
                "in_season": in_season, "config_hash": config_hash(),
                "git_sha": git_sha(), "seed": SEED,
                "calls_before": ingest.calls_used_this_month(),
                "data_as_of": data_as_of(), "degraded": False, "stages": {}}

    def finish(code: int) -> int:
        manifest["calls_after"] = ingest.calls_used_this_month()
        os.makedirs(MANIFESTS, exist_ok=True)
        path = os.path.join(MANIFESTS, f"{season}_wk{week:02d}_{stamp}.json")
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[manifest] {path}")
        return code

    if not in_season:
        manifest["stages"]["offseason_check"] = {
            "ok": True, "note": "no games within 7 days — clean offseason exit"}
        print(f"[loop] {season} wk{week}: offseason — exiting clean")
        return finish(0)

    if not acquire_lock(job):
        return 0
    try:
        stages = {"week-open": ("refresh", "freshness", "refit", "honesty",
                                "simulate", "notify"),
                  "pregame": ("freshness", "simulate", "notify"),
                  "grade": ("refresh", "freshness", "honesty", "recap",
                            "clv_capture", "notify")}[job]
        stage_budget_guard(25 if "refresh" in stages else 0)
        for st in stages:
            if st != "notify" and stage_done(season, week, f"{job}_{st}"):
                manifest["stages"][st] = {"ok": True, "note": "sentinel: done earlier"}
                continue
            try:
                if st == "refresh":
                    out = stage_refresh(season)
                    manifest["degraded"] |= out.get("stale", False)
                elif st == "freshness":
                    out = stage_freshness(season)
                    if not out.get("publish", False):
                        manifest["degraded"] = True
                elif st == "refit":
                    out = stage_refit(season, week)
                elif st == "honesty":
                    out = stage_honesty(season)
                elif st == "recap":
                    out = stage_recap(season, week)
                elif st == "clv_capture":
                    out = stage_clv_capture(season, week)
                elif st == "simulate":
                    out = stage_simulate(season, week)
                else:
                    out = notify(manifest)
                manifest["stages"][st] = {"ok": True, **out}
                if st != "notify":
                    mark_done(season, week, f"{job}_{st}")
            except Exception as exc:  # noqa: BLE001
                manifest["stages"][st] = {"ok": False, "error": str(exc)[:300]}
                manifest["degraded"] = True
                notify(manifest)
                print(f"[loop] stage {st} FAILED: {exc}", file=sys.stderr)
                return finish(1)
        return finish(0)
    finally:
        release_lock()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", choices=("week-open", "pregame", "grade"),
                    required=True)
    ap.add_argument("--week", type=int, default=None,
                    help="force a target week (testing/backfill)")
    return run(ap.parse_args().job, force_week=ap.parse_args().week)


if __name__ == "__main__":
    raise SystemExit(main())
