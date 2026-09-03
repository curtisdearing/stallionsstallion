"""The real-bet ledger: log a bet Curtis actually placed, grade it once the
game is final. Distinct from board_ledger.py -- this only ever holds games
someone actually bet, any kind (spread or moneyline), any division."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nflvalue import forecast_contract as FC  # noqa: E402


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BL = _load("bet_log", "scripts/bet_log.py")

SEASON, WEEK = 2026, 1


def _row(game_id, home, away, kickoff, **extra):
    base = {c: None for c in FC.COLUMNS}
    base.update({
        "schema_version": FC.SCHEMA_VERSION, "season": SEASON, "week": WEEK,
        "game_id": game_id, "home": home, "away": away,
        "kickoff_utc": kickoff, "neutral_site": False,
        "slate": "wk1", "public_date": kickoff[:10],
        "status": FC.Status.PRICED.value,
        "model_family": "ridge_preseason_prior", "model_version": "fp",
        "projected_home_margin": 7.0, "forecast_margin_sd": 16.0,
        "p_home_win": 0.6688,
        "forecast_sd_basis": "early_ridge_wk1_3_2021_2025",
        "generated_at_utc": "2026-08-30T00:00:00Z", "snapshot_id": "S1",
        "source_digest": "digest", "config_hash": "cfg",
    })
    base.update(extra)
    return base


def _artifact(tmp_path):
    rows = [
        _row(1, "Home U", "Away U", "2026-09-05T23:30:00Z"),
        _row(2, "Neutral A", "Neutral B", "2026-09-05T20:00:00Z",
             projected_home_margin=-4.0, p_home_win=0.4013),
        _row(3, "Big State", "Tiny College", "2026-09-01T16:00:00Z",
             model_family="ridge_cross_division",
             forecast_sd_basis="cross_division_ridge_wk1_15_2021_2025",
             projected_home_margin=30.0, forecast_margin_sd=17.9,
             p_home_win=0.9533),
    ]
    frame = pd.DataFrame(rows, columns=list(FC.COLUMNS))
    path = tmp_path / f"predictions_{SEASON}_wk{WEEK:02d}.csv"
    FC.write_artifact(frame, str(path))
    return str(path)


def _games_parquet(tmp_path, rows):
    path = tmp_path / f"games_{SEASON}.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return str(path)


BEFORE_KICKOFF = pd.Timestamp("2026-09-05T12:00:00Z")


# --------------------------------------------------------------------------- #
# --add
# --------------------------------------------------------------------------- #
NO_GAMES = "/nonexistent/no_games.parquet"


def test_add_logs_a_moneyline_bet_with_the_game_looked_up_from_the_artifact(tmp_path):
    art = _artifact(tmp_path)
    log = tmp_path / "bet_log.csv"
    rc = BL.add(SEASON, WEEK, 1, "moneyline", "Home U", -150.0, 1.0, "MyBook",
                "", artifact_path=art, log_path=str(log),
                games_path=NO_GAMES, now=BEFORE_KICKOFF)
    assert rc == 0
    df = pd.read_csv(log)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["bet_id"] == 1
    assert row["away"] == "Away U" and row["home"] == "Home U"
    assert row["kind"] == "moneyline" and row["side"] == "Home U"
    assert row["price"] == -150.0 and row["stake_units"] == 1.0
    assert row["book"] == "MyBook"


def test_add_logs_a_spread_bet_and_assigns_increasing_bet_ids(tmp_path):
    art = _artifact(tmp_path)
    log = tmp_path / "bet_log.csv"
    BL.add(SEASON, WEEK, 1, "moneyline", "Home U", -150.0, 1.0, "MyBook", "",
          artifact_path=art, log_path=str(log), games_path=NO_GAMES,
          now=BEFORE_KICKOFF)
    BL.add(SEASON, WEEK, 2, "spread", "Neutral A +6.0", -110.0, 2.0, "MyBook",
          "value swap", artifact_path=art, log_path=str(log),
          games_path=NO_GAMES, now=BEFORE_KICKOFF)
    df = pd.read_csv(log)
    assert list(df["bet_id"]) == [1, 2]
    assert df.iloc[1]["side"] == "Neutral A +6.0"
    assert df.iloc[1]["note"] == "value swap"


def test_add_refuses_a_game_id_not_on_the_weeks_artifact(tmp_path):
    art = _artifact(tmp_path)
    log = tmp_path / "bet_log.csv"
    with pytest.raises(SystemExit, match="not on the season"):
        BL.add(SEASON, WEEK, 999, "moneyline", "Home U", -150.0, 1.0, "", "",
              artifact_path=art, log_path=str(log), games_path=NO_GAMES,
              now=BEFORE_KICKOFF)


def test_add_refuses_a_side_that_is_neither_team(tmp_path):
    art = _artifact(tmp_path)
    log = tmp_path / "bet_log.csv"
    with pytest.raises(SystemExit, match="is not"):
        BL.add(SEASON, WEEK, 1, "moneyline", "Some Other Team", -150.0, 1.0,
              "", "", artifact_path=art, log_path=str(log),
              games_path=NO_GAMES, now=BEFORE_KICKOFF)


def test_add_refuses_a_bet_logged_after_kickoff(tmp_path):
    art = _artifact(tmp_path)
    log = tmp_path / "bet_log.csv"
    after_kickoff = pd.Timestamp("2026-09-05T23:31:00Z")
    with pytest.raises(SystemExit, match="kicked off"):
        BL.add(SEASON, WEEK, 1, "moneyline", "Home U", -150.0, 1.0, "", "",
              artifact_path=art, log_path=str(log), games_path=NO_GAMES,
              now=after_kickoff)


def test_add_refuses_a_game_whose_final_score_is_already_cached(tmp_path):
    art = _artifact(tmp_path)
    log = tmp_path / "bet_log.csv"
    games = _games_parquet(tmp_path, [
        {"id": 3, "home_points": 40, "away_points": 3}])
    with pytest.raises(SystemExit, match="already has a final score"):
        BL.add(SEASON, WEEK, 3, "moneyline", "Big State", -1000.0, 1.0, "",
              "", artifact_path=art, log_path=str(log), games_path=games,
              now=pd.Timestamp("2026-09-01T00:00:00Z"))


def test_add_is_first_write_wins_on_an_identical_bet(tmp_path, capsys):
    art = _artifact(tmp_path)
    log = tmp_path / "bet_log.csv"
    BL.add(SEASON, WEEK, 1, "moneyline", "Home U", -150.0, 1.0, "MyBook", "",
          artifact_path=art, log_path=str(log), games_path=NO_GAMES,
          now=BEFORE_KICKOFF)
    BL.add(SEASON, WEEK, 1, "moneyline", "Home U", -150.0, 1.0, "MyBook", "",
          artifact_path=art, log_path=str(log), games_path=NO_GAMES,
          now=BEFORE_KICKOFF)
    df = pd.read_csv(log)
    assert len(df) == 1
    assert "already logged this bet" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# --grade
# --------------------------------------------------------------------------- #
def _logged(tmp_path, rows):
    log = tmp_path / "bet_log.csv"
    pd.DataFrame(rows, columns=BL.BET_LOG_COLUMNS).to_csv(log, index=False)
    return str(log)


def test_grade_scores_a_moneyline_win_and_computes_profit(tmp_path):
    log = _logged(tmp_path, [
        {"bet_id": 1, "season": SEASON, "week": WEEK, "game_id": 1,
         "kickoff_utc": "2026-09-05T23:30:00Z", "away": "Away U",
         "home": "Home U", "kind": "moneyline", "side": "Home U",
         "price": -150.0, "stake_units": 2.0, "book": "MyBook",
         "placed_at_utc": "2026-09-05T12:00:00Z", "note": ""},
    ])
    games = _games_parquet(tmp_path, [
        {"id": 1, "home_points": 30, "away_points": 10}])
    out = tmp_path / "graded.csv"
    rc = BL.grade(log_path=log, out_path=str(out), cfbd_dir=str(tmp_path))
    assert rc == 0
    graded = pd.read_csv(out)
    row = graded.iloc[0]
    assert row["result"] == "win"
    assert row["profit_units"] == pytest.approx(2.0 * (100.0 / 150.0))


def test_grade_scores_a_spread_loss_and_a_push(tmp_path):
    log = _logged(tmp_path, [
        {"bet_id": 1, "season": SEASON, "week": WEEK, "game_id": 1,
         "kickoff_utc": "x", "away": "Away U", "home": "Home U",
         "kind": "spread", "side": "Home U -7.0", "price": -110.0,
         "stake_units": 1.0, "book": "", "placed_at_utc": "x", "note": ""},
        {"bet_id": 2, "season": SEASON, "week": WEEK, "game_id": 2,
         "kickoff_utc": "x", "away": "Away U", "home": "Home U",
         "kind": "spread", "side": "Away U +7.0", "price": -110.0,
         "stake_units": 1.0, "book": "", "placed_at_utc": "x", "note": ""},
    ])
    games = _games_parquet(tmp_path, [
        {"id": 1, "home_points": 20, "away_points": 17},   # margin +3, -7 loses
        {"id": 2, "home_points": 24, "away_points": 17},   # margin +7, dog +7 pushes
    ])
    out = tmp_path / "graded.csv"
    BL.grade(log_path=log, out_path=str(out), cfbd_dir=str(tmp_path))
    graded = pd.read_csv(out).set_index("bet_id")
    assert graded.loc[1, "result"] == "loss"
    assert graded.loc[1, "profit_units"] == pytest.approx(-1.0)
    assert graded.loc[2, "result"] == "push"
    assert graded.loc[2, "profit_units"] == pytest.approx(0.0)


def test_grade_leaves_bets_on_unplayed_games_pending(tmp_path, capsys):
    log = _logged(tmp_path, [
        {"bet_id": 1, "season": SEASON, "week": WEEK, "game_id": 1,
         "kickoff_utc": "x", "away": "Away U", "home": "Home U",
         "kind": "moneyline", "side": "Home U", "price": -150.0,
         "stake_units": 1.0, "book": "", "placed_at_utc": "x", "note": ""},
    ])
    games = _games_parquet(tmp_path, [
        {"id": 99, "home_points": None, "away_points": None}])
    out = tmp_path / "graded.csv"
    rc = BL.grade(log_path=log, out_path=str(out), cfbd_dir=str(tmp_path))
    assert rc == 0
    assert "0 played yet" in capsys.readouterr().out
    assert not out.exists()


def test_grade_reports_are_split_by_kind(tmp_path, capsys):
    log = _logged(tmp_path, [
        {"bet_id": 1, "season": SEASON, "week": WEEK, "game_id": 1,
         "kickoff_utc": "x", "away": "Away U", "home": "Home U",
         "kind": "moneyline", "side": "Home U", "price": -150.0,
         "stake_units": 1.0, "book": "", "placed_at_utc": "x", "note": ""},
        {"bet_id": 2, "season": SEASON, "week": WEEK, "game_id": 2,
         "kickoff_utc": "x", "away": "Away U", "home": "Home U",
         "kind": "spread", "side": "Home U -7.0", "price": -110.0,
         "stake_units": 1.0, "book": "", "placed_at_utc": "x", "note": ""},
    ])
    games = _games_parquet(tmp_path, [
        {"id": 1, "home_points": 30, "away_points": 10},
        {"id": 2, "home_points": 20, "away_points": 17},
    ])
    out = tmp_path / "graded.csv"
    BL.grade(log_path=log, out_path=str(out), cfbd_dir=str(tmp_path))
    output = capsys.readouterr().out
    assert "moneyline " in output
    assert "spread    " in output
    assert "overall" in output
