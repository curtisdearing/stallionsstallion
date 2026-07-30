"""Workstream G bakeoff machinery — walk-forward and update-direction tests."""

import numpy as np
import pandas as pd
import pytest

from nflvalue import bakeoff as BK


def _games(weeks=8, seed=2):
    teams = tuple(f"T{i}" for i in range(8))
    strengths = {t: 10.0 - 3.0 * i for i, t in enumerate(teams)}
    rng = np.random.default_rng(seed)
    rows, gid = [], 100
    for season in (2023, 2024):
        for wk in range(1, weeks + 1):
            for i, h in enumerate(teams):
                a = teams[(i + wk) % len(teams)]
                if a == h:
                    continue
                m = (strengths[h] - strengths[a]) / 2 + 2.5 + rng.normal(0, 10)
                rows.append({"id": gid, "season": season, "week": wk,
                             "home_team": h, "away_team": a,
                             "home_points": 28 + m / 2, "away_points": 28 - m / 2,
                             "neutral_site": False, "home_fbs": True,
                             "away_fbs": True, "margin": m})
                gid += 1
    return pd.DataFrame(rows)


def test_elo_learns_the_ordering():
    games = _games()
    m = BK.EloModel(BK.EloParams(k=24, hfa_elo=50, pts_per_elo=0.05))
    for g in games.sort_values(["season", "week", "id"]).itertuples():
        m.update(g)
    assert m.elo["T0"] > m.elo["T7"], "Elo must learn the strength ordering"


def test_gelo_mov_multiplier_moves_ratings_faster():
    games = _games()
    slow = BK.EloModel(BK.EloParams(k=20, mov=False))
    fast = BK.EloModel(BK.EloParams(k=20, mov=True))
    for g in games.sort_values(["season", "week", "id"]).itertuples():
        slow.update(g)
        fast.update(g)
    spread_slow = slow.elo["T0"] - slow.elo["T7"]
    spread_fast = fast.elo["T0"] - fast.elo["T7"]
    assert spread_fast > spread_slow, \
        "the MOV multiplier exists to move ratings faster on blowouts"


def test_kalman_updates_toward_the_residual_and_shrinks_variance():
    m = BK.KalmanModel(BK.KalmanParams())
    g = _games().iloc[0]
    before_var = m.p.prior_sd ** 2
    m.update(next(_games().head(1).itertuples()))
    t = str(g["home_team"])
    assert m.var[t] < before_var, "an observation must shrink uncertainty"


def test_online_runner_is_walk_forward_by_construction():
    """Poison the final week; earlier predictions must not move."""
    games = _games()
    def run(df):
        m = BK.KalmanModel(BK.KalmanParams())
        return BK.run_online_model(m, df, eval_seasons=[2023, 2024],
                                   first_week=2)
    base = run(games)
    poisoned = games.copy()
    last = (poisoned["season"] == 2024) & (poisoned["week"] == poisoned["week"].max())
    poisoned.loc[last, "margin"] = 999.0
    after = run(poisoned)
    cut = after["week"] < poisoned.loc[last, "week"].iloc[0]
    early_a = after[(after["season"] == 2024) & cut].reset_index(drop=True)
    early_b = base[(base["season"] == 2024)
                   & (base["week"] < poisoned.loc[last, "week"].iloc[0])
                   ].reset_index(drop=True)
    pd.testing.assert_frame_equal(early_a, early_b), \
        "LEAKAGE: poisoning the final week moved earlier online predictions"


def test_burn_in_fit_never_sees_eval_seasons():
    games = _games()
    cfg, mae = BK.fit_online_params(
        games, lambda c: BK.KalmanModel(BK.KalmanParams(**c)),
        grid=[{"obs_sd": 12.0, "week_sd": 0.5, "season_sd": 5.0,
               "prior_sd": 10.0, "hfa": 2.5}],
        burn_seasons=(2023,), first_week=2, min_preds=30)
    poisoned = games.copy()
    fut = poisoned["season"] == 2024
    poisoned.loc[fut, "margin"] = 999.0
    cfg2, mae2 = BK.fit_online_params(
        poisoned, lambda c: BK.KalmanModel(BK.KalmanParams(**c)),
        grid=[{"obs_sd": 12.0, "week_sd": 0.5, "season_sd": 5.0,
               "prior_sd": 10.0, "hfa": 2.5}],
        burn_seasons=(2023,), first_week=2, min_preds=30)
    assert mae == pytest.approx(mae2), \
        "burn-in fitting must be blind to the eval seasons"
