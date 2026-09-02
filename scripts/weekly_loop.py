#!/usr/bin/env python3
"""Workstream I — the weekly run loop (WEEKLY_LOOP_SPEC Phase 1, plus the
simulate/explain stages that unblocked when Workstreams C and F landed).

Cadence is CFB-shaped, not the inherited NFL wed/t90/tuesday
(`scripts/auto_weekly.py` stays as reference — its NFL job names encode a
Sun/Mon game week that CFB does not have):

  --job week-open   Tuesday: health-gated ingest refresh, warm ratings
                    refit, incremental honesty report, simulate + render the
                    upcoming week's explainer, notify.
  --job pregame     Tue-Sat, run as often as scheduled: cheap input-health
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
  * FAIL CLOSED: the input-health gate (nflvalue/data_health.py) checks every
                        input this job's engine actually depends on, and a
                        blocked capability HALTS the run before simulation.
                        Nothing is published, the `latest` pointer does not
                        move, no artifact is overwritten, and the public
                        status becomes a non-numeric "data unavailable".
                        Exit status 3 says so distinctly.

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
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflvalue import cfb_ot, cfb_sim, explain, gameline as GL  # noqa: E402
from nflvalue import cfb_calendar, engine as EG, ingest, ratings as R  # noqa: E402
from nflvalue import snapshots as SN  # noqa: E402
from nflvalue import forecast_contract as FC  # noqa: E402
from nflvalue import producers as PR  # noqa: E402
from nflvalue import cross_division as XD  # noqa: E402
from nflvalue import data_health as DH  # noqa: E402
from nflvalue import insight as IN  # noqa: E402
from nflvalue import atomic  # noqa: E402
from nflvalue import evidence as EV  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MANIFESTS = os.path.join(DATA, "manifests")
LOCK_PATH = os.path.join(DATA, ".weekly.lock")
LOCK_STALE_HOURS = 2.0
BUDGET_HEADROOM = 50

#: Exit status for a run that stopped because its inputs were not healthy.
#: Distinct from 1 (a stage died) on purpose: this run did exactly what it
#: was supposed to do, and the operator needs to tell the two apart.
EXIT_UNHEALTHY_INPUTS = 3
SEED = 20260730
HISTORY_START = 2019
REFRESH_CALLS = 25

# Ordered stages per job.  `pregame` deliberately has NO refresh stage: its
# repeat-suppression is decided from the snapshot store (nflvalue/snapshots.py),
# so running it five times between Tuesday and Saturday costs zero CFBD calls.
JOB_STAGES = {
    "week-open": ("refresh", "freshness", "refit", "honesty", "simulate",
                  "notify"),
    "pregame": ("freshness", "simulate", "notify"),
    "grade": ("refresh", "freshness", "honesty", "recap", "clv_capture",
              "notify"),
}

# Jobs whose "already done this week" sentinel is meaningful.  pregame is
# excluded on purpose: a per-week sentinel is exactly the bug that stopped a
# Saturday run from reacting to a Friday injury report or to a slate kicking
# off.  Its idempotency is fingerprint-based instead.
SENTINEL_JOBS = ("week-open", "grade")


def projected_calls(job: str) -> int:
    """CFBD calls a job may spend, for the budget guard."""
    return REFRESH_CALLS if "refresh" in JOB_STAGES[job] else 0


def sentinel_applies(job: str, stage: str) -> bool:
    return job in SENTINEL_JOBS and stage != "notify"


class LoopError(RuntimeError):
    pass


KICKED_OFF_NOTE = ("KICKED OFF: the pregame window for this game closed; its "
                   "frozen pregame snapshot stands")


def utcstamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------- #
# Lock
# --------------------------------------------------------------------------- #
def _read_lock() -> tuple:
    """(held payload, age in hours). Unreadable counts as infinitely old."""
    try:
        with open(LOCK_PATH) as fh:
            held = json.load(fh)
        age_h = (datetime.now(timezone.utc)
                 - datetime.fromisoformat(held["utc"])).total_seconds() / 3600
        return held, age_h
    except Exception:  # noqa: BLE001 - a lock we cannot read is a stale lock
        return {}, float("inf")


def acquire_lock(job: str) -> bool:
    """Take the single-writer lock, atomically.

    `os.open(..., O_CREAT | O_EXCL)` is one syscall that both tests for the
    lock and takes it. The previous exists-then-write had a window between
    the two in which a second run could see no lock, and both would proceed
    to publish the same week -- which is precisely the scenario the lock is
    for, and precisely when two runs fire at once.

    A stale lock is broken by unlinking and retrying the same exclusive
    create, so if two runs race to take over an abandoned lock, exactly one
    of them wins and the other exits clean rather than sharing it.
    """
    os.makedirs(DATA, exist_ok=True)
    payload = json.dumps({"pid": os.getpid(), "job": job,
                          "utc": datetime.now(timezone.utc).isoformat()})
    for attempt in (0, 1):
        try:
            fd = os.open(LOCK_PATH,
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            held, age_h = _read_lock()
            if age_h < LOCK_STALE_HOURS:
                print(f"[lock] live lock held by pid={held.get('pid')} "
                      f"job={held.get('job')} age={age_h:.2f}h — exiting clean")
                return False
            if attempt == 0:
                print(f"[lock] stale lock ({age_h:.1f}h old) — overriding")
                try:
                    os.unlink(LOCK_PATH)
                except FileNotFoundError:
                    pass                # another run got there first; retry
                continue
            print("[lock] lost the race to break a stale lock — exiting clean")
            return False
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
        return True
    return False


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


def stage_done(season: int, week: int, stage: str,
               fingerprint: str = None) -> bool:
    """Has this stage already run for this week?

    With a `fingerprint`, the sentinel is honoured only while that fingerprint
    still matches — which is how `grade` reopens as more games go final instead
    of staying "done" for the rest of the week.
    """
    path = sentinel_path(season, week, stage)
    if not os.path.exists(path):
        return False
    if fingerprint is None:
        return True
    try:
        with open(path) as fh:
            stored = json.load(fh).get("fingerprint")
    except (OSError, ValueError):
        return False        # pre-fingerprint sentinel: re-run rather than guess
    return stored == fingerprint


def mark_done(season: int, week: int, stage: str,
              fingerprint: str = None) -> None:
    os.makedirs(MANIFESTS, exist_ok=True)
    with open(sentinel_path(season, week, stage), "w") as fh:
        json.dump({"utc": utcstamp(), "fingerprint": fingerprint}, fh)


def finals_fingerprint(games: pd.DataFrame, week: int) -> str:
    """Digest of which games in `week` have final scores.

    Moves the moment another game goes final, so an interrupted grade run — or
    one that ran before the night slate finished — reopens instead of being
    suppressed for the rest of the week.
    """
    if games is None or games.empty:
        return SN.source_digest({"finals": []})
    wk = games[games["week"] == week] if "week" in games.columns else games
    done = wk[wk["home_points"].notna() & wk["away_points"].notna()]
    rows = sorted([int(r.id), float(r.home_points), float(r.away_points)]
                  for r in done.itertuples())
    return SN.source_digest({"finals": rows})


def loop_version() -> dict:
    """Code+config identity: a change here may legitimately move a forecast."""
    return {"config_hash": config_hash(), "git_sha": git_sha(), "seed": SEED}


def simulatable_ids(games: pd.DataFrame, now: datetime) -> list:
    """Games that may still be forecast: kickoff strictly in the future.

    BUILD_SPEC section 0.4 — a "pregame" number computed after kickoff is not a
    pregame number.  This filter is what makes that structural rather than
    aspirational.
    """
    return SN.pending_game_ids(games, now)


def pregame_decision(season: int, week: int, *, games: pd.DataFrame = None,
                     degraded: bool = False, now: datetime = None,
                     force: bool = False) -> dict:
    """Should this pregame tick do real work?

    Pure and free: reads the schedule already on disk plus the snapshot store,
    and spends no API budget.
    """
    now = now or datetime.now(timezone.utc)
    if games is None:
        games = ingest.load_games([season])
    wk = games[games["week"] == week] if "week" in games.columns else games
    digest, _ = SN.collect_source_digest(DATA, season, week, games,
                                         version=loop_version())
    window = SN.kickoff_window(wk, now)
    out = SN.decide(DATA, season, week, source_digest=digest, window=window,
                    degraded=degraded, force=force)
    out.update({"source_digest": digest, "window": window,
                "n_pending": len(SN.pending_game_ids(wk, now))})
    return out


def assert_schedule_id_coverage(rows: pd.DataFrame, schedule: pd.DataFrame,
                                where: str) -> None:
    """The published rows must BE the week's schedule — exactly, once each —
    with one narrow exception: a row for a FROZEN game (one carrying a
    `frozen_from_stamp`) is allowed even if that game has since vanished from the
    schedule entirely. A game that kicked off, froze, and was later dropped
    by the provider must not permanently wedge every future publish; its
    historical frozen row still belongs in the artifact (one row per
    scheduled-or-frozen game), it just isn't in `schedule` any more.

    Checked at the writer and again after freezing/splicing, because those are
    the two places a row can be duplicated or lost, and a public artifact that
    quietly omits a game is worse than a run that fails.
    """
    columns = getattr(rows, "columns", [])
    row_ids = rows["game_id"] if "game_id" in columns else []
    sched_ids = (schedule["id"]
                 if "id" in getattr(schedule, "columns", []) else [])
    frozen_ids = (rows.loc[rows["frozen_from_stamp"].notna(), "game_id"]
                  if "frozen_from_stamp" in columns else [])
    allowed_ids = set(int(i) for i in sched_ids) | set(int(i) for i in frozen_ids)
    problem = EG.id_coverage_problem(row_ids, allowed_ids)
    if problem:
        raise LoopError(
            f"{where}: {problem} — refusing to publish a week whose rows are "
            "not its schedule")


def assemble_week_artifact(season: int, week: int, schedule: pd.DataFrame,
                           rows: pd.DataFrame, now: datetime,
                           data_dir: str = None) -> tuple:
    """Freeze started games into already-built contract rows.

    `rows` arrives from `forecast_contract.build_artifact` -- there is no
    second place that decides what a public row looks like. This function only
    freezes and splices.

    Order matters: freeze BEFORE splicing, and freeze only from snapshots taken
    strictly before each game's own kickoff, so a started game's published
    number provably predates that kickoff.
    """
    data_dir = data_dir or DATA
    SN.freeze_kicked_off(data_dir, season, week, schedule, now,
                         model_family=PR.production_family(week))
    return SN.apply_frozen(rows, SN.load_frozen(data_dir, season, week))


# --------------------------------------------------------------------------- #
# Season/week detection — self-scheduling, offseason-safe
# --------------------------------------------------------------------------- #
# How long after a kickoff results may still be landing. Matches the 7-day
# window the preseason branch already uses, and it is what separates "the
# slate just finished, grade it" from "the season is over".
RESULTS_SETTLING = timedelta(days=7)


def _week_or_none(weeks):
    """Lowest real week number in `weeks`, or None if there is no real one.

    CFBD occasionally serves a game with no week assigned. `int(NaN)` raises,
    and this runs before the loop has a manifest to record anything in, so a
    null week must skip a row rather than kill the run. Returning None rather
    than a guessed number keeps the caller honest: a week we cannot determine
    is not week 1.
    """
    clean = pd.to_numeric(weeks, errors="coerce").dropna()
    return int(clean.min()) if len(clean) else None


def _last_week(games):
    """Highest real week number in the frame, or 1 if the frame has none."""
    clean = pd.to_numeric(games["week"], errors="coerce").dropna()
    return int(clean.max()) if len(clean) else 1


def current_season_week(now=None):
    """(season, week, in_season) for the week the loop is IN.

    Total over every schedule shape this project has seen, because it runs on
    the first line of `run()`, outside the per-stage try/except: an exception
    here leaves no manifest, no notification, and no exit code the runner can
    classify.

    The shape that used to raise `int(NaN)` is not exotic and is not the
    "night game's final has not landed" case it might look like. CFBD leaves a
    handful of games permanently unscored every year -- lower-division
    fixtures and weather cancellations; 4 in 2023, 2 in 2024, 2 in 2025 in
    this repo's own caches -- so `played.all()` is never true for a real
    season. Once the last game of the season had kicked off, the old code
    reached its final line with an empty future selection and raised, every
    January.

    Scoredness alone therefore cannot separate "the slate just finished and
    results are landing" from "the season ended months ago"; recency can, and
    that is what `RESULTS_SETTLING` is for. Without it the loop reports
    `in_season` on an October week all through January, and `pregame` -- which
    is deliberately not sentinel-guarded and whose launchd job fires year
    round -- re-simulates that week on every fire.
    """
    now = now or datetime.now(timezone.utc)
    season = now.year if now.month >= 2 else now.year - 1
    games = ingest.load_games([season])
    if games.empty or "start_date" not in games.columns:
        return season, 1, False
    if "week" not in games.columns or "home_points" not in games.columns:
        return season, 1, False

    dates = pd.to_datetime(games["start_date"], errors="coerce", utc=True)
    played = games["home_points"].notna()
    future = dates > pd.Timestamp(now)

    if not future.any():
        # Nothing left to play. Either a slate just finished and its results
        # are still landing, or the season is over. An unparseable date column
        # lands here too, with nothing recent -- and that is the right answer:
        # with no kickoff times, no game can be frozen at its own kickoff, so
        # the honest move is not to run rather than to publish fresh numbers
        # for games that may already have started.
        settling = ~played & (dates > pd.Timestamp(now) - RESULTS_SETTLING)
        if settling.any():
            week = _week_or_none(games.loc[settling, "week"])
            if week is not None:
                return season, week, True
        return season, _last_week(games), False

    week = _week_or_none(games.loc[future, "week"])
    if week is None:
        # Games ahead, but not one of them says which week it belongs to.
        # Nothing can be priced against a week we cannot name.
        return season, _last_week(games), False
    if future.all():
        days_to_first = (dates.min() - pd.Timestamp(now)).days
        return season, week, days_to_first <= 7
    return season, week, True


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


def stage_freshness(season: int, week: int, job: str, *,
                    now: datetime = None) -> dict:
    """Dependency-aware input health for THIS job, at THIS week, on THIS engine.

    The version this replaces checked the games file, hardcoded
    `schema_ok=True`, and returned a `publish` flag the run then ignored. It
    could not see a missing drives cache, an empty team table, or a benchmark
    record the explainer was about to quote — and passing it was not required
    to publish anyway.

    Writes the public status file on every run, healthy or not, so the last
    thing a reader saw is never staler than the last thing that happened.
    """
    now = now or datetime.now(timezone.utc)
    report = DH.evaluate(DATA, season, week, job,
                         PR.production_family(week), as_of=now)
    DH.write_status(DATA, report, published=False)
    return report.to_dict()


def stage_refit(season: int, week: int) -> dict:
    params = R.load_params()
    games = R.game_frame(list(range(HISTORY_START, season + 1)))
    tbl = EG.solve_production_table(games, season, week, params)
    frame = tbl.to_frame()
    ingest._atomic_write_parquet(
        frame, os.path.join(DATA, f"ratings_{season}.parquet"))
    return {"fingerprint": tbl.fingerprint(), "n_teams": len(frame),
            "n_games_fit": tbl.n_games_fit, "week": week,
            "engine": EG.production_margin_engine(week)}


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
            tbl = EG.solve_production_table(games, season, int(w), params)
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


def mc_forecasts(simulated: dict, registry: FC.BasisRegistry,
                 week: int) -> dict:
    """Simulation summaries -> contract forecasts, coherence enforced.

    `PredictiveMargin.from_summary` refuses a win probability that does not
    belong to the (mean, sd) beside it, so the simulator cannot publish three
    numbers drawn from two distributions. The registry supplies the basis id,
    and refuses to supply one for a week its registered evidence does not
    cover.
    """
    if not simulated:
        # Nothing to price. Asking the registry for evidence covering a week
        # this call is not going to price would refuse an artifact that
        # publishes no numbers at all -- which is a legitimate thing to
        # publish, and is exactly what the all-kicked-off path does.
        return {}
    basis_id = registry.basis_id("kalman_montecarlo", week)
    forecasts = {}
    for game_id, sim in simulated.items():
        forecasts[int(game_id)] = FC.Forecast(
            predictive=FC.PredictiveMargin.from_summary(
                mean=sim["mc_margin"], sd=sim["mc_margin_sd"],
                p_home_win=sim["p_home_win"], basis_id=basis_id,
                total=sim["mc_total"]),
            rating_parameter_sd=sim.get("param_sd"))
    return forecasts


def default_refusal(game) -> str:
    """Why a game the simulator skipped is not being priced."""
    if bool(game.home_fcs) and bool(game.away_fcs):
        return FC.DEFAULT_REFUSALS["non_fbs"]
    if bool(game.home_fcs) or bool(game.away_fcs):
        return FC.DEFAULT_REFUSALS["cross_division"]
    return FC.DEFAULT_REFUSALS["unrated"]


def simulation_artifact(season: int, week: int, schedule: pd.DataFrame,
                        simulated: dict, refusals: dict,
                        provenance: FC.Provenance,
                        registry: FC.BasisRegistry, *,
                        degraded: bool = False,
                        degraded_reason: str = None,
                        extra_forecasts: dict = None) -> pd.DataFrame:
    """The in-season route's rows, in the one public contract.

    `extra_forecasts` are contract forecasts the simulation did not make --
    the cross-division ridge rows, each carrying its own per-row family --
    merged after the simulated ones, never over them.
    """
    extra = dict(extra_forecasts or {})
    reasons = dict(refusals)
    for game in schedule.itertuples():
        game_id = int(game.id)
        if game_id in simulated or game_id in reasons or game_id in extra:
            continue
        reasons[game_id] = default_refusal(game)
    if degraded:
        return FC.build_artifact(season, week, schedule, {}, reasons,
                                 provenance, degraded=True,
                                 degraded_reason=degraded_reason)
    try:
        forecasts = mc_forecasts(simulated, registry, week)
        forecasts = {**{k: v for k, v in extra.items() if k not in forecasts},
                     **forecasts}
    except FC.ContractError as exc:
        # The simulator produced summaries; no validated out-of-sample SD
        # backs them. Same honest degrade as the ridge route -- one status,
        # both routes -- rather than a crash or a number this run cannot
        # stand behind.
        return FC.build_artifact(
            season, week, schedule, {}, {}, provenance,
            degraded=True,
            degraded_reason=f"UNCERTAINTY UNVALIDATED: {exc}",
            degraded_status=FC.Status.UNCERTAINTY_UNVALIDATED)
    return FC.build_artifact(season, week, schedule, forecasts, reasons,
                             provenance)


def provenance_for(season: int, week: int, model_version: str, *,
                   source_digest: str = None) -> FC.Provenance:
    """Who produced this week's rows, from what code and config."""
    return FC.Provenance(
        model_family=PR.production_family(week),
        model_version=model_version,
        generated_at_utc=utcstamp(),
        source_digest=source_digest,
        config_hash=config_hash())


def digest_and_window(season: int, week: int, schedule: pd.DataFrame,
                      now: datetime, decision: dict) -> tuple:
    """(source digest, kickoff window) — computed before the rows, because
    the rows now carry the digest as provenance."""
    if decision is None:
        digest, _ = SN.collect_source_digest(
            DATA, season, week, schedule, version=loop_version())
        return digest, SN.kickoff_window(schedule, now)
    return decision["source_digest"], decision["window"]


def stage_simulate(season: int, week: int, *, now: datetime = None,
                   decision: dict = None, degraded: bool = False,
                   degraded_reason: str = None) -> dict:
    """Forecast the week, render its explainer, and leave an immutable
    snapshot behind. Paper-only.

    The published artifact covers the WHOLE week's schedule, not just its
    unplayed games: a game that has already kicked off keeps the final pregame
    row it was published with (frozen, with provenance) instead of silently
    vanishing from the file the scoreboard later grades.
    """
    now = now or datetime.now(timezone.utc)
    params = R.load_params()
    seasons = list(range(HISTORY_START, season + 1))
    games = R.game_frame(seasons)
    upcoming_all = ingest.load_games([season])
    # DENOMINATOR: every scheduled game in the week, played or not.
    schedule = upcoming_all[upcoming_all["week"] == week]
    if schedule.empty:
        return {"n": 0, "note": "no games scheduled in target week"}
    registry = FC.BasisRegistry.load(DATA)
    digest, window = digest_and_window(season, week, schedule, now, decision)

    # ROUTE. Weeks 0-3 are priced by the ridge/preseason policy and weeks 4+
    # by the Kalman simulation -- the registered 2026-08-10 decision, and the
    # engine's own scope note. Until now that split was achieved by running
    # two scripts in the right order, which is not a policy, it is a habit:
    # whichever writer ran last owned the file, and the two wrote different
    # schemas. The branch lives in code now, and both sides come back with
    # rows in the one contract.
    if EG.is_early_season(week):
        return _publish_early_week(season, week, schedule, games, params,
                                   registry, now, digest, window,
                                   degraded, degraded_reason)

    # NUMERATOR: only games that are still genuinely pregame may be simulated.
    live = set(simulatable_ids(schedule, now)) & set(
        int(i) for i in schedule.loc[schedule["home_points"].isna(), "id"])
    if not live:
        provenance = provenance_for(
            season, week, model_version="none: nothing left to price",
            source_digest=digest)
        built = simulation_artifact(
            season, week, schedule, {},
            {int(g.id): KICKED_OFF_NOTE for g in schedule.itertuples()},
            provenance, registry, degraded=degraded,
            degraded_reason=degraded_reason)
        rows, n_frozen = assemble_week_artifact(
            season, week, schedule, built, now)
        return _publish_week(season, week, rows, None, now, digest, window,
                             degraded, degraded_reason, schedule,
                             n_sim=0, skipped=int(len(schedule)),
                             n_frozen=n_frozen, components={})
    # Pace and OT are bounded to the SAME preregistered walk-forward cutoff
    # as the rate model below (2026-08-29, Prompt 5A correction): strictly
    # before this target week, never merely "through this season". Without
    # the week-level half of that filter, a `--week` backfill run (this
    # loop's own documented testing/backfill path) could silently train on
    # same-season drives from weeks that had, by the time of the backfill,
    # already kicked off or gone final.
    drives = ingest.load_drives(seasons)
    hist_mask = ((drives["season"] < season)
                | ((drives["season"] == season) & (drives["week"] < week)))
    drives_hist = drives[hist_mask]
    drives_c, _ = cfb_sim.classify_drives(drives_hist)
    drives_c = drives_c[drives_c["start_period"] <= 4]
    ot = cfb_ot.fit_ot_params(drives_hist)

    pool = []
    for s in seasons:
        for w in sorted(games.loc[games["season"] == s, "week"].unique()):
            if s == season and w >= week:
                break
            try:
                tbl_w = EG.solve_production_table(games, int(s), int(w),
                                                  params)
            except (R.RatingsError, np.linalg.LinAlgError):
                # Includes EarlySeasonPriorError: the first history season has
                # no prior year, so its early weeks cannot be labelled the way
                # production prices them. Skipped, never mislabelled.
                continue
            gw = games[(games["season"] == s) & (games["week"] == w)]
            dw = drives_c[(drives_c["season"] == s) & (drives_c["week"] == w)]
            lab = cfb_sim.label_week_drives(dw, gw, tbl_w)
            if len(lab):
                pool.append(lab)
    rates = cfb_sim.DriveRateModel.fit(pd.concat(pool, ignore_index=True))
    pace = cfb_sim.PaceModel.fit(drives_c)
    tbl = EG.solve_production_table(games, season, week, params)

    teams = ingest.load_teams([season])
    conf = {str(r.team): str(r.conference) for r in teams.itertuples()}
    simulated, refusals, exps, skipped = {}, {}, [], 0
    # Selected-benchmark-or-nothing. A pre-correction MAE quoted beside a
    # corrected forecast is a claim the pipeline is not entitled to make.
    bench_frame, bench_record, bench_notice = EV.benchmark_or_notice(
        ROOT, EV.ROLE_WK4PLUS)
    model_mae = (None if bench_frame is None else
                 float((bench_frame["mc_margin"]
                        - bench_frame["margin"]).abs().mean()))
    if bench_frame is None:
        print(f"[loop] benchmark unavailable: {bench_notice}")
    clv = pd.read_csv(os.path.join(DATA, "clv_log_backtest.csv"))
    clv_slices = clv.groupby("slice")["clv_pts"].agg(["count", "mean"])
    for g in schedule.itertuples():
        game_id = int(g.id)
        home, away = str(g.home_team), str(g.away_team)
        if game_id not in live:
            # Already kicked off (or already final). Never compute a "pregame"
            # number now — the frozen pre-kickoff snapshot is the only honest
            # answer, and assemble_week_artifact splices it back in.
            skipped += 1
            refusals[game_id] = KICKED_OFF_NOTE
            continue
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
        # Per-game rng, derived only from (seed, game id) — never from
        # iteration order — so a game's forecast cannot depend on which
        # other games are in `schedule` or what order they arrive in.
        game_rng = np.random.default_rng([SEED, game_id, 0])
        sim = cfb_sim.simulate_game(
            game_rng, e_h, e_a, rates, pace, ot, n=4000, param_sd=psd,
            param_rng=(np.random.default_rng([SEED, game_id, 1])
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
    provenance = provenance_for(season, week,
                                model_version=tbl.fingerprint(),
                                source_digest=digest)
    # FBS-vs-FCS games: never simulated (the Kalman pools FCS too, but no
    # simulation of it is validated); priced from the cross-division ridge
    # table under the registered cross-division basis, still-pregame games
    # only, each row naming its own family. Without a usable basis the
    # standing refusal comes back, exactly as before.
    xd_table = XD.solve_table(games, season, week, params)
    xd_forecasts, xd_refusals = PR.cross_division_forecasts(
        xd_table, schedule, registry, week, fbs=PR.fbs_teams(season),
        only_ids=set(live))
    refusals = {**xd_refusals, **refusals}
    built = simulation_artifact(season, week, schedule, simulated, refusals,
                                provenance, registry, degraded=degraded,
                                degraded_reason=degraded_reason,
                                extra_forecasts=xd_forecasts)
    skipped -= len(xd_forecasts)
    components = IN.components_for_rows(
        built[built["status"] == FC.Status.PRICED.value], tbl, xd_table,
        fbs=PR.fbs_teams(season), simulated=simulated)
    artifact_rows, n_frozen = assemble_week_artifact(
        season, week, schedule, built, now)

    html_text = None
    if simulated:
        html_text = explain.render_week_html(
            exps, clv_slices=clv_slices,
            title=f"stallionsstallion — {season} week {week} (upcoming)",
            generated_note=(f"weekly loop {utcstamp()}; {len(simulated)} games, "
                            f"{skipped} refused or past kickoff, "
                            f"{n_frozen} frozen at kickoff"))
        with open(os.path.join(DATA,
                               f"explainer_{season}_wk{week:02d}.html"),
                  "w") as fh:
            fh.write(html_text)

    return _publish_week(season, week, artifact_rows, html_text, now,
                         digest, window, degraded, degraded_reason, schedule,
                         n_sim=len(simulated), skipped=skipped,
                         n_frozen=n_frozen, components=components)


def _publish_early_week(season: int, week: int, schedule: pd.DataFrame,
                        games: pd.DataFrame, params, registry, now: datetime,
                        digest: str, window: str, degraded: bool,
                        degraded_reason: str) -> dict:
    """Weeks 0-3: the ridge/preseason route, through the shared producer."""
    table_out = {}
    built = PR.build_early_week(
        season, week, games=games, schedule=schedule,
        generated_at_utc=utcstamp(), data_dir=DATA, params=params,
        registry=registry, source_digest=digest, config_hash=config_hash(),
        degraded=degraded, degraded_reason=degraded_reason,
        table_out=table_out)
    table = table_out.get("table")
    components = {}
    if table is not None:
        priced = built[built["status"] == FC.Status.PRICED.value]
        components = IN.components_for_rows(
            priced, table, table, fbs=PR.fbs_teams(season))
    rows, n_frozen = assemble_week_artifact(season, week, schedule, built, now)
    n_priced = int((rows["status"] == FC.Status.PRICED.value).sum())
    return _publish_week(season, week, rows, None, now, digest, window,
                         degraded, degraded_reason, schedule,
                         n_sim=n_priced,
                         skipped=int(len(rows)) - n_priced,
                         n_frozen=n_frozen, components=components)


def _publish_week(season: int, week: int, rows: pd.DataFrame,
                  html_text, now: datetime, digest: str, window: str,
                  degraded: bool, degraded_reason: str,
                  schedule: pd.DataFrame,
                  *, n_sim: int, skipped: int, n_frozen: int,
                  components: dict = None) -> dict:
    """Immutable snapshot first, then the flat `latest` file the site reads.

    The snapshot is the record; the flat artifact is a convenience copy of the
    newest snapshot and is the only thing ever overwritten.

    Nothing invalid reaches either one: the rows are validated here, again by
    `write_snapshot`, and again by `write_artifact` on the way to disk. A
    contract mismatch therefore fails BEFORE any public output exists, which
    is the property the old pipeline lacked -- it wrote first and the readers
    each coped differently afterwards.
    """
    # Post-assembly, pre-publication: freezing and splicing happen between the
    # writer and here, so the coverage contract is re-checked against the same
    # schedule the artifact claims to cover.
    assert_schedule_id_coverage(rows, schedule, where="_publish_week")

    stamp = utcstamp()
    body = rows.copy()
    # The snapshot id is only knowable at publish time; it is provenance, not
    # forecast, so stamping it here changes no number. Frozen rows keep the
    # snapshot id they were published under.
    fresh = body["frozen_from_stamp"].isna()
    body.loc[fresh, "snapshot_id"] = stamp
    FC.validate_frame(body, where="_publish_week")

    snap = SN.write_snapshot(
        DATA, season, week, stamp, body,
        source_digest=digest, window=window, degraded=degraded,
        degraded_reason=degraded_reason, taken_at=now,
        explainer_html=html_text)

    latest_csv = os.path.join(DATA, f"predictions_{season}_wk{week:02d}.csv")
    published = FC.read_artifact(os.path.join(snap["path"], "predictions.csv"))
    FC.write_artifact(published, latest_csv)
    insights = _publish_insights(season, week, published, components)

    return {"n": n_sim, "skipped": skipped, "n_schedule": len(rows),
            "n_scheduled_games": int(len(schedule)), "n_frozen": n_frozen,
            "snapshot": snap["stamp"], "window": window,
            "source_digest": digest, "degraded": bool(degraded),
            "schema_version": FC.SCHEMA_VERSION,
            "contract": FC.summarise(body),
            "insights": insights}


def _publish_insights(season: int, week: int, published: pd.DataFrame,
                      components: dict = None) -> dict:
    """The four-section card per game, written beside the artifact.

    Built from the PUBLISHED rows, so a card cannot disagree with the
    artifact it describes, and from the same solved table that priced the
    week, so the "why" is the reasoning that actually produced the number.
    Nothing here reads a clock: every timestamp on a card comes from the row.
    """
    data_as_of = None
    games_cache = os.path.join(DATA, "cfbd", f"games_{season}.parquet")
    if os.path.exists(games_cache):
        data_as_of = datetime.fromtimestamp(
            os.path.getmtime(games_cache), timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
    payload = IN.build_week(published.to_dict("records"),
                            components_by_game=components or {},
                            data_as_of_utc=data_as_of)
    path = os.path.join(DATA, f"insights_{season}_wk{week:02d}.json")
    atomic.write_json(path, payload)
    return {"path": os.path.basename(path), "n_cards": payload["n_cards"],
            "by_status": payload["by_status"]}


def stage_recap(season: int, week: int) -> dict:
    """Weekly scoreboard (WEEKLY_SCOREBOARD_SPEC): grade the PUBLISHED
    projections for the most recently completed week; append/revisit the
    hypothesis ledger. Read-only w.r.t. the model by construction.

    `week` is the week the loop is IN; the recap targets `week - 1`.  Every
    return names that target (`target_week`) so a zero-row recap in the
    manifest says which week it graded — `current_season_week()` keeps a
    week current until its last game is played, so a Sunday grade run before
    a Monday game targets the week before the one that just finished.
    """
    from nflvalue import scoreboard
    target = week - 1
    if target < 1:
        return {"n": 0, "target_week": target,
                "note": "no completed week to recap yet"}
    try:
        r = scoreboard.build_recap(season, target)
    except scoreboard.ScoreboardError as exc:
        return {"n": 0, "target_week": target, "note": str(exc)[:200]}
    return {"n": r["n_graded"], "target_week": target, "week_recapped": target,
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
        return {"n": 0, "target_week": target,
                "note": "no published projections for completed week"}
    # Validated read. A CLV log built from an artifact whose columns cannot be
    # interpreted unambiguously is research raw material that quietly means
    # something other than it says.
    pub = FC.read_artifact(pub_path)
    priced = FC.forecast_bearing(pub)
    if priced.empty:
        return {"n": 0, "target_week": target,
                "note": "no forecast-bearing rows in the published week"}
    games = R.game_frame([season])
    lines = ingest.load_lines([season])
    wk_lines = lines[lines["game_id"].isin(set(pub["game_id"].astype(int)))]
    signal = dict(zip(priced["game_id"].astype(int),
                      priced["projected_home_margin"]))
    log = GL.build_clv_log(games, wk_lines, signal, min_edge_pts=2.0)
    if log.empty:
        return {"n": 0, "target_week": target,
                "note": "no open+close candidates this week"}
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
    return {"n": int(len(log)), "target_week": target, "week_captured": target,
            "model_families": sorted(set(priced["model_family"])),
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
def _halt_unhealthy(manifest: dict, stages, stage: str, health: dict,
                    season: int, week: int, job: str, finish) -> int:
    """Stop the run before anything can be published. Say exactly why.

    Nothing downstream of here writes: no snapshot, no advance of the `latest`
    pointer, no rewrite of the published artifact. The only output is the
    manifest, the alert, and a public status that carries no numbers at all —
    which is the honest thing to show when the inputs behind the numbers did
    not check out.
    """
    reasons = health.get("reasons") or ["inputs did not pass the health check"]
    manifest["degraded"] = True
    manifest["halted"] = True
    manifest["halt_reason"] = "; ".join(reasons)[:1000]
    manifest["blocked_capabilities"] = health.get("blocked_capabilities", [])
    manifest["stages"][stage] = {"ok": False, **health}
    remaining = list(stages)[list(stages).index(stage) + 1:]
    for later in remaining:
        manifest["stages"][later] = {
            "ok": False, "skipped": True,
            "reason": ("halted: inputs unhealthy — nothing was published, "
                       "the latest pointer did not move, and no artifact was "
                       "overwritten")}
    notify(manifest)
    print(f"[loop] {season} wk{week} {job}: HALTED — "
          f"{manifest['halt_reason']}", file=sys.stderr)
    print(f"[loop] public status: {DH.STATUS_UNAVAILABLE} "
          f"(data/{DH.PUBLIC_STATUS_FILE})", file=sys.stderr)
    return finish(EXIT_UNHEALTHY_INPUTS)


def run(job: str, force_week: int = None) -> int:
    season, week, in_season = current_season_week()
    if force_week is not None:
        # `if force_week:` was the test here, so --week 0 fell through to the
        # auto-detected week: the operator asked for one week, got another,
        # and the run exited 0 without saying so.  Weeks start at 1 (week 0
        # slates are carried inside CFBD week 1), so anything below that is a
        # refusal, not a default.
        if force_week < 1:
            raise LoopError(
                f"--week {force_week}: the season starts at week 1, and a "
                "week below it cannot be run. CFBD carries the week-0 slate "
                "inside week 1.")
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
        stages = JOB_STAGES[job]
        stage_budget_guard(projected_calls(job))
        grade_fp = None
        if job == "grade":
            try:
                grade_fp = finals_fingerprint(ingest.load_games([season]),
                                              week - 1)
            except Exception:  # noqa: BLE001
                grade_fp = None     # cannot fingerprint -> re-run, never skip
        decision = None
        for st in stages:
            if (sentinel_applies(job, st)
                    and stage_done(season, week, f"{job}_{st}",
                                   fingerprint=grade_fp)):
                manifest["stages"][st] = {"ok": True, "note": "sentinel: done earlier"}
                continue
            try:
                if st == "refresh":
                    out = stage_refresh(season)
                    # A stale refresh is recorded, not acted on: whether the
                    # CACHE is usable is the health gate's call, and a failed
                    # CFBD call over a cache that is still inside its
                    # staleness window is not a reason to withhold anything.
                    manifest["refresh_stale"] = bool(out.get("stale", False))
                elif st == "freshness":
                    out = stage_freshness(season, week, job)
                    manifest["input_health"] = out
                    manifest["suppressed_insights"] = out.get(
                        "optional_unavailable", [])
                    if not out.get("healthy", False):
                        return _halt_unhealthy(manifest, stages, st, out,
                                               season, week, job, finish)
                elif st == "refit":
                    out = stage_refit(season, week)
                elif st == "honesty":
                    out = stage_honesty(season)
                elif st == "recap":
                    out = stage_recap(season, week)
                elif st == "clv_capture":
                    out = stage_clv_capture(season, week)
                elif st == "simulate":
                    if job == "pregame":
                        decision = pregame_decision(season, week)
                        manifest["decision"] = {
                            k: decision.get(k) for k in
                            ("run", "reason", "window", "source_digest",
                             "n_pending")}
                        if not decision["run"]:
                            manifest["stages"][st] = {
                                "ok": True, "skipped": True,
                                "reason": decision["reason"]}
                            continue
                    # No `degraded=` from here any more. A run whose inputs
                    # are unhealthy has already halted above without touching
                    # the store; a run that reaches this point has inputs that
                    # passed, so publishing numbers is honest.
                    out = stage_simulate(season, week, decision=decision)
                else:
                    out = notify(manifest)
                manifest["stages"][st] = {"ok": True, **out}
                if sentinel_applies(job, st):
                    mark_done(season, week, f"{job}_{st}", fingerprint=grade_fp)
            except Exception as exc:  # noqa: BLE001
                manifest["stages"][st] = {"ok": False, "error": str(exc)[:300]}
                manifest["degraded"] = True
                notify(manifest)
                print(f"[loop] stage {st} FAILED: {exc}", file=sys.stderr)
                return finish(1)
        simulate = manifest["stages"].get("simulate") or {}
        DH.mark_published(
            DATA, published=bool(simulate.get("ok")
                                 and not simulate.get("skipped")
                                 and simulate.get("n_schedule")))
        return finish(0)
    finally:
        release_lock()


def _week_arg(value: str) -> int:
    """argparse type for --week: a real season week, or a clear refusal."""
    try:
        week = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"week must be a whole number, got {value!r}")
    if week < 1:
        raise argparse.ArgumentTypeError(
            f"week must be 1 or greater, got {week} — the season starts at "
            "week 1 and CFBD carries the week-0 slate inside it")
    return week


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", choices=("week-open", "pregame", "grade"),
                    required=True)
    ap.add_argument("--week", type=_week_arg, default=None,
                    help=("force the week the loop is IN (testing/backfill). "
                          "week-open and pregame price this week; grade "
                          "recaps and captures CLV for week-1, so "
                          "`--job grade --week N` grades week N-1"))
    return ap


def main_args(argv=None):
    """Parsed arguments, so the CLI contract is testable without running a job."""
    return build_parser().parse_args(argv)


def main() -> int:
    args = main_args()
    return run(args.job, force_week=args.week)


if __name__ == "__main__":
    raise SystemExit(main())
