#!/usr/bin/env python3
"""Re-fit Workstream B's parameters on the full FBS-wide 2019-2023 sample.

Why this exists (2026-07-19): B's parameters were searched on n=42 gradable
games (MWC-only 2023) and four landed on grid boundaries — PROVISIONAL by the
project's own record. The backfill is now complete (2019-2023 FBS-wide), so
this script re-runs the coordinate-wise walk-forward search with the error
pooled across THREE evaluation seasons (2021, 2022, 2023), each predicted
week-by-week from strictly-prior games with 2019-2020 as warm history.

RESUMABLE BY DESIGN: the execution sandbox hard-kills any process at ~45s, so
every (param, value, season) evaluation is appended to the trace CSV the
moment it finishes, and a fresh invocation replays the trace instead of
recomputing. Invoke repeatedly (`timeout 38 python3 scripts/refit_b.py`)
until it exits 0. Exit 3 = more work to do, re-invoke.

The grid is nflvalue.ratings.DEFAULT_GRID, unmodified. In particular
fcs_weight stays capped at 0.5 — a search that wants FCS blowouts at face
value is the search being wrong, not the cap (tests enforce this).

Writes:
    data/refit_b_trace.csv       every (param, value, season, mae, n) row
    data/rating_params.json      the selected params — ONLY when search completes
    data/refit_b_report.txt      boundary report + per-season final numbers
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from nflvalue import ratings as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = os.path.join(ROOT, "data", "refit_b_trace.csv")
REPORT = os.path.join(ROOT, "data", "refit_b_report.txt")

HISTORY = [2019, 2020, 2021, 2022, 2023]
EVAL_SEASONS = [2021, 2022, 2023]   # pre-registered before any run; do not grow
BUDGET_S = 33                        # leave headroom under the 45s hard kill


def _load_trace() -> pd.DataFrame:
    if os.path.exists(TRACE):
        return pd.read_csv(TRACE)
    return pd.DataFrame(columns=["param", "value", "season", "mae", "n"])


def _append_trace(row: dict) -> None:
    df = _load_trace()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    tmp = TRACE + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, TRACE)


def main() -> int:
    t0 = time.time()
    games = R.game_frame(HISTORY)
    trace = _load_trace()
    params = R.RatingParams()

    for name, values in R.DEFAULT_GRID.items():
        # Replay: pick the best value for every param whose rows are complete.
        done = trace[trace["param"] == name] if len(trace) else trace
        complete = (len(done) == len(values) * len(EVAL_SEASONS))
        if complete:
            agg = done.groupby("value").apply(
                lambda d: pd.Series({"mae": (d["mae"] * d["n"]).sum() / max(d["n"].sum(), 1),
                                     "n": d["n"].sum()}), include_groups=False)
            best_val = float(agg["mae"].idxmin())
            # ints stored as floats in CSV; cast back where the grid is int-ish
            if all(float(v) == int(v) for v in values):
                best_val = type(values[0])(best_val)
            params = R.replace(params, **{name: best_val})
            continue

        # Incomplete param: compute missing (value, season) cells until budget.
        for v in values:
            for season in EVAL_SEASONS:
                have = done[(done["value"] == float(v)) & (done["season"] == season)] \
                    if len(done) else done
                if len(have):
                    continue
                if time.time() - t0 > BUDGET_S:
                    print(f"[refit] budget reached mid-'{name}'; re-invoke to resume")
                    return 3
                cand = R.replace(params, **{name: v})
                mae, n = R.walk_forward_error(games, season, cand)
                _append_trace({"param": name, "value": float(v),
                               "season": season, "mae": mae, "n": n})
                done = _load_trace()
                done = done[done["param"] == name]
                print(f"[refit] {name}={v} season={season}: MAE {mae:.3f} n={n}")
        return 3  # param just finished (or partially); re-invoke to replay + continue

    # ---- search complete: boundary report + final numbers ------------------ #
    lines = ["refit_b: coordinate-wise walk-forward search, error pooled over "
             f"{EVAL_SEASONS} (history from {HISTORY[0]})", ""]
    for name, values in R.DEFAULT_GRID.items():
        chosen = getattr(params, name)
        mark = ""
        if chosen in (min(values), max(values)):
            edge = "LOW" if chosen == min(values) else "HIGH"
            mark = f"  << BOUNDARY ({edge}) — treat as unfit"
        lines.append(f"  {name:<16} = {chosen}{mark}")
    lines.append("")
    for season in EVAL_SEASONS:
        mae, n = R.walk_forward_error(games, season, params)
        lines.append(f"  season {season}: walk-forward MAE {mae:.3f} on n={n}")
    report = "\n".join(lines)
    print(report)
    with open(REPORT, "w") as fh:
        fh.write(report + "\n")
    R.save_params(params)
    print(f"[refit] wrote {R.params_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
