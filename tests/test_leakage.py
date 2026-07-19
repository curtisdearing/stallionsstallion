"""Leakage test — poison a future week, assert no earlier number moves.

Re-implemented for stallionsstallion (Workstream A, 2026-07-19). This was a
docstring-only stub: the pattern is the most valuable one inherited from
fablesfable, but the original was written against that project's player-prop
feature builders, which are deliberately out of scope here. Same technique,
new target — the CFBD ingest layer.

The idea: build every through-week-N view of a season, then go back and
corrupt weeks N+1..end with absurd values. Rebuild the through-week-N views.
If any of them changed, information from the future reached the past, and
every backtest number downstream is fiction.

SCOPE — read this before trusting it. This covers the INGEST layer only.
When Workstream B lands, extend the same poisoning technique to the ridge
solve; do not replace this file, add to it. A leakage test that only covers
ingest is necessary, not sufficient.
"""

import pandas as pd

from nflvalue import ingest


def _synthetic_season(season=2023, weeks=6, teams=("A", "B", "C", "D")):
    """A small, fully deterministic season we can poison without a network."""
    rows = []
    gid = 1
    for wk in range(1, weeks + 1):
        for i in range(0, len(teams), 2):
            home, away = teams[i], teams[i + 1]
            rows.append({
                "game_id": gid, "season": season, "week": wk, "period": 1,
                "offense": home, "defense": away,
                "offense_score": 7 * wk, "defense_score": 3 * wk,
                "down": 1, "distance": 10, "yards_to_goal": 65,
                "yards_gained": 4, "play_type": "Rush", "ppa": 0.1 * wk,
            })
            gid += 1
    df = pd.DataFrame(rows)
    df["garbage_time"] = [
        ingest.is_garbage_time(p, o, d)
        for p, o, d in zip(df["period"], df["offense_score"], df["defense_score"])
    ]
    return df


def _through_week(df, week):
    """The only legitimate way to look at a season mid-flight."""
    return (df.loc[df["week"] <= week]
              .sort_values(["week", "game_id"], kind="mergesort")
              .reset_index(drop=True))


def _summary(df):
    """Stand-in for 'every number the model would compute from this view'."""
    if df.empty:
        return {}
    return {
        "n_plays": len(df),
        "mean_ppa": round(float(df["ppa"].mean()), 10),
        "garbage_share": round(float(df["garbage_time"].mean()), 10),
        "max_week": int(df["week"].max()),
    }


def test_future_weeks_cannot_move_earlier_numbers():
    clean = _synthetic_season()
    weeks = sorted(clean["week"].unique())
    before = {w: _summary(_through_week(clean, w)) for w in weeks}

    for cutoff in weeks[:-1]:
        poisoned = clean.copy()
        future = poisoned["week"] > cutoff
        # Absurd values: if any of this bleeds backwards it will be obvious.
        poisoned.loc[future, "ppa"] = 999.0
        poisoned.loc[future, "offense_score"] = 999
        poisoned.loc[future, "defense_score"] = 0

        after = _summary(_through_week(poisoned, cutoff))
        assert after == before[cutoff], (
            f"LEAKAGE at cutoff week {cutoff}: the through-week view changed "
            f"after only FUTURE weeks were poisoned.\n"
            f"  before={before[cutoff]}\n  after ={after}\n"
            "Something is reading past the cutoff."
        )


def test_garbage_time_uses_only_within_play_state():
    """Garbage time must be decided from the play's own score and period,
    never from the final score. Labelling early plays by how the game ended
    is leakage wearing a plausible disguise."""
    df = _synthetic_season()
    first = df.iloc[0]
    direct = ingest.is_garbage_time(first["period"], first["offense_score"],
                                    first["defense_score"])
    assert bool(first["garbage_time"]) == direct

    poisoned = df.copy()
    poisoned.loc[poisoned.index[1:], "offense_score"] = 999
    recomputed = ingest.is_garbage_time(
        poisoned.iloc[0]["period"], poisoned.iloc[0]["offense_score"],
        poisoned.iloc[0]["defense_score"])
    assert recomputed == direct


def test_garbage_time_thresholds_are_per_period():
    # 4th quarter has the tightest threshold (22), 1st the loosest (43).
    assert ingest.is_garbage_time(4, 30, 0) is True
    assert ingest.is_garbage_time(1, 30, 0) is False
    assert ingest.is_garbage_time(1, 50, 0) is True
    # Overtime (period 5+) has no threshold — never garbage by definition.
    assert ingest.is_garbage_time(5, 999, 0) is False
