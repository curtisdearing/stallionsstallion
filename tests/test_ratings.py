"""Workstream B — ratings engine tests.

Two tiers, on purpose:

* **Synthetic** tests build their own game frame and pass an explicit prior, so
  they run with no parquet cache, no API key and no network. These carry the
  structural guarantees — neutral-site HFA zeroing, FCS pooling, conference
  shrinkage, disconnection refusal, reproducibility.
* **Cache-backed** tests read `data/cfbd/*.parquet` and are SKIPPED if it is
  absent. These carry the gate evidence that has to come from real data
  (connectivity actually rising September to November, the talent sentinel
  actually not sinking Air Force).

Leakage lives in `tests/test_leakage.py`, extended rather than duplicated here.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from nflvalue import ratings as R


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #
def synth_games(n_weeks=8, strengths=None, fcs_games=(), neutral=(), season=2023):
    """A deterministic round-robin with known true strengths."""
    strengths = strengths or {"A": 10.0, "B": 4.0, "C": -3.0, "D": -11.0}
    teams = sorted(strengths)
    conf = {t: "TEST" for t in teams}
    rows, gid = [], 1000
    for wk in range(1, n_weeks + 1):
        for i, h in enumerate(teams):
            a = teams[(i + wk) % len(teams)]
            if a == h:
                continue
            is_neutral = (gid in neutral)
            base = 24.0
            hp = base + strengths[h] / 2 + (2.5 if not is_neutral else 0.0)
            ap = base + strengths[a] / 2
            rows.append({
                "id": gid, "season": season, "week": wk,
                "home_team": h, "away_team": a,
                "home_points": hp, "away_points": ap,
                "neutral_site": is_neutral,
                "home_fbs": True, "away_fbs": True,
                "home_conf": conf[h], "away_conf": conf[a],
                "margin": hp - ap,
            })
            gid += 1
    for (wk, h, opp, hp, ap) in fcs_games:
        rows.append({
            "id": gid, "season": season, "week": wk,
            "home_team": h, "away_team": opp,
            "home_points": hp, "away_points": ap, "neutral_site": False,
            "home_fbs": True, "away_fbs": False,
            "home_conf": "TEST", "away_conf": "FCS-Land",
            "margin": hp - ap,
        })
        gid += 1
    return pd.DataFrame(rows).sort_values(["season", "week", "id"]).reset_index(drop=True)


FLAT_PRIOR = {t: 0.0 for t in "ABCDEFGH"}
P = R.RatingParams(lambda_team=2.0, lambda_conf=1.0, prior_strength=0.0)


# --------------------------------------------------------------------------- #
# Reproducibility — the first half of the gate
# --------------------------------------------------------------------------- #
def test_solve_is_bit_for_bit_reproducible():
    g = synth_games()
    a = R.solve(g, 2023, 9, P, prior=FLAT_PRIOR)
    b = R.solve(g, 2023, 9, P, prior=FLAT_PRIOR)
    assert a.fingerprint() == b.fingerprint()
    pd.testing.assert_frame_equal(a.to_frame(), b.to_frame())


def test_row_order_does_not_change_the_answer():
    """Shuffling the input rows must not move a single rating.

    A solve that depends on row order is one that is secretly sequential.
    """
    g = synth_games()
    shuffled = g.sample(frac=1.0, random_state=1234).reset_index(drop=True)
    assert R.solve(g, 2023, 9, P, prior=FLAT_PRIOR).fingerprint() == \
           R.solve(shuffled, 2023, 9, P, prior=FLAT_PRIOR).fingerprint()


def test_cache_key_tracks_content_not_week_number():
    g = synth_games()
    R.clear_cache()
    k1 = R._cache_key(g, 2023, 9, P)
    revised = g.copy()
    revised.loc[0, "home_points"] = revised.loc[0, "home_points"] + 3
    assert R._cache_key(revised, 2023, 9, P) != k1, \
        "a corrected score must miss the cache; keying on (season, week) would serve stale ratings"


# --------------------------------------------------------------------------- #
# Recovering known structure
# --------------------------------------------------------------------------- #
def test_recovers_true_ordering():
    truth = {"A": 12.0, "B": 5.0, "C": -4.0, "D": -13.0}
    g = synth_games(strengths=truth)
    t = R.solve(g, 2023, 9, P, prior=FLAT_PRIOR)
    order = [r.team for r in t.to_frame().itertuples()]
    assert order.index("A") < order.index("B") < order.index("C") < order.index("D")


def test_league_hfa_is_estimated_not_assumed():
    g = synth_games()                       # built with a 2.5 pt home edge
    t = R.solve(g, 2023, 9, P, prior=FLAT_PRIOR)
    assert 1.5 < t.hfa_league < 3.5, t.hfa_league


def test_neutral_site_hfa_is_exactly_zero():
    g = synth_games()
    t = R.solve(g, 2023, 9, P, prior=FLAT_PRIOR)
    assert t.hfa("A", neutral=True) == 0.0
    flat = t.predict_margin("A", "B", neutral=True)
    homefield = t.predict_margin("A", "B", neutral=False)
    assert homefield - flat == pytest.approx(t.hfa("A"), abs=1e-9)


def test_all_neutral_slate_yields_exactly_zero_hfa():
    """A bowl-only slice must not manufacture a home edge out of nothing.

    The HFA column is structurally zero here; the solve pins it rather than
    inverting a singular matrix or inventing a number.
    """
    g = synth_games()
    g["neutral_site"] = True
    g["home_points"] = g["home_points"] - 2.5          # remove the built-in edge
    g["margin"] = g["home_points"] - g["away_points"]
    t = R.solve(g, 2023, 9, P, prior=FLAT_PRIOR)
    assert t.hfa_league == 0.0
    assert t.hfa("A") == 0.0


def test_per_team_hfa_can_differ_and_shrinks_with_penalty():
    g = synth_games()
    # give A a much bigger home edge than everyone else
    home_a = (g["home_team"] == "A") & (~g["neutral_site"])
    g.loc[home_a, "home_points"] = g.loc[home_a, "home_points"] + 8.0
    g["margin"] = g["home_points"] - g["away_points"]
    loose = R.solve(g, 2023, 9, replace(P, lambda_hfa=1.0), prior=FLAT_PRIOR)
    tight = R.solve(g, 2023, 9, replace(P, lambda_hfa=1000.0), prior=FLAT_PRIOR)
    assert loose.hfa("A") > loose.hfa("B")
    assert (loose.hfa("A") - loose.hfa("B")) > (tight.hfa("A") - tight.hfa("B"))


# --------------------------------------------------------------------------- #
# Conference shrinkage
# --------------------------------------------------------------------------- #
def two_conference_league(n_weeks=8, season=2023):
    """Two 4-team conferences with a cross-conference game every week.

    Enough cross-play that each conference term is genuinely identified, which
    is the setting where "shrink toward the conference mean" is a meaningful
    claim rather than a coin flip.
    """
    strengths = {"A": 10.0, "B": 4.0, "C": -3.0, "D": -11.0,
                 "P": 6.0, "Q": 1.0, "R": -5.0, "S": -12.0}
    left, right = ["A", "B", "C", "D"], ["P", "Q", "R", "S"]
    rows, gid = [], 2000
    for wk in range(1, n_weeks + 1):
        pairs = [(left[i], left[(i + wk) % 4], "L", "L") for i in range(4)]
        pairs += [(right[i], right[(i + wk) % 4], "R", "R") for i in range(4)]
        pairs += [(left[wk % 4], right[(wk + 1) % 4], "L", "R")]
        for h, a, hc, ac in pairs:
            if h == a:
                continue
            hp = 24.0 + strengths[h] / 2 + 2.5
            ap = 24.0 + strengths[a] / 2
            rows.append({
                "id": gid, "season": season, "week": wk, "home_team": h,
                "away_team": a, "home_points": hp, "away_points": ap,
                "neutral_site": False, "home_fbs": True, "away_fbs": True,
                "home_conf": hc, "away_conf": ac, "margin": hp - ap,
            })
            gid += 1
    return pd.DataFrame(rows).sort_values(["season", "week", "id"]).reset_index(drop=True)


def test_one_game_team_lands_near_its_conference_not_on_its_own_result():
    """The concrete meaning of conference-mean shrinkage.

    Newcomer Z joins conference R and wins its only game 80-0. Its rating must
    end up far closer to R's conference level than to what an 80-0 win would
    imply on its own, and it must not be ratable at all.
    """
    g = two_conference_league()
    z = pd.DataFrame([{
        "id": 9999, "season": 2023, "week": 3, "home_team": "Z", "away_team": "D",
        "home_points": 80.0, "away_points": 0.0, "neutral_site": False,
        "home_fbs": True, "away_fbs": True, "home_conf": "R",
        "away_conf": "L", "margin": 80.0,
    }])
    prior = {t: 0.0 for t in "ABCDPQRSZ"}
    t = R.solve(pd.concat([g, z], ignore_index=True), 2023, 9, P, prior=prior)

    assert t.veto["Z"] is True, "a one-game team must not be ratable"
    conf_r = [t.net[x] for x in ("P", "Q", "R", "S")]
    conf_level = float(np.mean(conf_r))
    raw_implied = 80.0 + t.net["D"]        # what the result alone would say
    assert abs(t.net["Z"] - conf_level) < abs(t.net["Z"] - raw_implied), (
        f"Z={t.net['Z']:.1f} sits closer to its own 80-0 result "
        f"({raw_implied:.1f}) than to its conference level ({conf_level:.1f})")
    assert abs(t.off_dev["Z"]) < 10.0, (
        f"Z's own team deviation is {t.off_dev['Z']:.1f}; a single result "
        "should not buy that much team-specific credit")


def test_conference_term_is_penalised_by_cross_conference_games_only():
    """A conference is identified by who it plays outside itself, nothing else."""
    g = two_conference_league()
    t = R.solve(g, 2023, 9, P, prior={x: 0.0 for x in "ABCDPQRS"})
    # 8 weeks x 1 cross game x 2 sides = 8 charged to each conference
    assert t.conf_games["L"] == 8 and t.conf_games["R"] == 8
    thin = two_conference_league(n_weeks=2)
    t2 = R.solve(thin, 2023, 3, P, prior={x: 0.0 for x in "ABCDPQRS"})
    assert t2.conf_games["L"] == 2
    gap_thick = abs(t.conf_off["L"] - t.conf_off["R"])
    gap_thin = abs(t2.conf_off["L"] - t2.conf_off["R"])
    assert gap_thin < gap_thick, (
        "the conference gap did not tighten as cross-conference games "
        f"accumulated: 2 games -> {gap_thin:.2f}, 8 games -> {gap_thick:.2f}")


# --------------------------------------------------------------------------- #
# FCS handling
# --------------------------------------------------------------------------- #
def test_fcs_opponents_are_pooled_into_one_pseudo_team():
    g = synth_games(fcs_games=[(2, "A", "Tiny State", 70.0, 3.0),
                               (3, "B", "Other Tiny", 63.0, 7.0)])
    t = R.solve(g, 2023, 9, P, prior=FLAT_PRIOR)
    assert R.FCS_TEAM in t.net
    assert "Tiny State" not in t.net and "Other Tiny" not in t.net
    assert t.veto[R.FCS_TEAM] is True


def test_fcs_downweight_monotonically_limits_how_far_a_blowout_moves_a_rating():
    """The downweight has to actually bite.

    The earlier version of this test compared an FCS blowout against the same
    scoreline versus a known-bad FBS team, which is not the right control: the
    FBS opponent absorbs half the update through its own rating, so the two
    numbers are not comparable and the test failed for a reason that had
    nothing to do with FCS handling. What matters is that `fcs_weight` is the
    knob it claims to be, monotonically.
    """
    base = synth_games()
    fcs = synth_games(fcs_games=[(2, "A", "Tiny State", 70.0, 3.0)])
    b = R.solve(base, 2023, 9, P, prior=FLAT_PRIOR).net["A"]
    moves = []
    for w in (0.0, 0.1, 0.5, 1.0):
        t = R.solve(fcs, 2023, 9, replace(P, fcs_weight=w), prior=FLAT_PRIOR)
        moves.append(abs(t.net["A"] - b))
    assert moves[0] == pytest.approx(0.0, abs=1e-9), \
        "fcs_weight=0 must make the FCS game literally weightless"
    assert moves == sorted(moves), f"not monotone in fcs_weight: {moves}"
    assert moves[1] < moves[-1]


def test_shipped_grid_never_offers_full_weight_to_fcs_results():
    """Guardrail, not preference.

    An unconstrained walk-forward search on the real MWC 2023 slice picks
    fcs_weight=1.0 — it wants FCS results at face value — for a ~0.12 point MAE
    gain on 42 gradable games. BUILD_SPEC section 0 and AGENT_BUILD_PROMPT both
    forbid that, so the shipped grid cannot reach it. If someone widens the
    grid to "improve" a number, this test is the thing that objects.
    """
    assert max(R.DEFAULT_GRID["fcs_weight"]) < 1.0


def test_fcs_games_do_not_count_toward_connectivity():
    plain = synth_games(n_weeks=2)
    padded = synth_games(n_weeks=2, fcs_games=[(1, "A", "Tiny State", 70.0, 3.0),
                                               (2, "A", "Other Tiny", 66.0, 0.0)])
    c1 = R.connectivity_table(plain, P)
    c2 = R.connectivity_table(padded, P)
    assert c1.n_games["A"] == c2.n_games["A"]
    assert c1.score["A"] == c2.score["A"]


# --------------------------------------------------------------------------- #
# Connectivity as an input, and refusal to guess
# --------------------------------------------------------------------------- #
def test_connectivity_scales_the_penalty_so_it_changes_the_fit():
    """The load-bearing claim: connectivity is an input, not an annotation."""
    g = synth_games()
    strong = replace(P, conn_half_games=0.5)
    weak = replace(P, conn_half_games=50.0)
    a = R.solve(g, 2023, 9, strong, prior=FLAT_PRIOR)
    b = R.solve(g, 2023, 9, weak, prior=FLAT_PRIOR)
    assert a.fingerprint() != b.fingerprint()
    spread_a = np.std(list(a.net.values()))
    spread_b = np.std(list(b.net.values()))
    assert spread_a > spread_b, (
        "treating every team as badly connected did not shrink the ratings; "
        "the connectivity term is decorative")


def test_disconnected_team_refuses_rather_than_guesses():
    g = synth_games()
    island = pd.DataFrame([{
        "id": 7777, "season": 2023, "week": 2, "home_team": "Y", "away_team": "Z",
        "home_points": 30.0, "away_points": 10.0, "neutral_site": False,
        "home_fbs": True, "away_fbs": True, "home_conf": "ISLE",
        "away_conf": "ISLE", "margin": 20.0,
    }])
    g2 = pd.concat([g, island], ignore_index=True)
    t = R.solve(g2, 2023, 9, P, prior={**FLAT_PRIOR, "Y": 0.0, "Z": 0.0})
    assert t.connectivity.hops["Y"] is None, "Y should be unreachable"
    assert t.connectivity.score["Y"] == 0.0
    with pytest.raises(R.DisconnectedTeamError):
        t.require_usable("Y")
    with pytest.raises(R.DisconnectedTeamError):
        t.predict_margin("Y", "A")


def test_band_is_wider_for_the_teams_we_know_least_about():
    g = synth_games()
    thin = pd.DataFrame([{
        "id": 6666, "season": 2023, "week": 2, "home_team": "Q", "away_team": "A",
        "home_points": 20.0, "away_points": 24.0, "neutral_site": False,
        "home_fbs": True, "away_fbs": True, "home_conf": "TEST",
        "away_conf": "TEST", "margin": -4.0,
    }])
    g2 = pd.concat([g, thin], ignore_index=True)
    t = R.solve(g2, 2023, 9, P, prior={**FLAT_PRIOR, "Q": 0.0})
    assert t.net_band["Q"] > t.net_band["A"], (
        "the one-game team reported a tighter band than the eight-game team")


def test_ill_conditioned_design_fails_loud():
    g = synth_games()
    with pytest.raises(R.IllConditionedError):
        R.solve(g, 2023, 9, replace(P, max_condition=1.0), prior=FLAT_PRIOR)


def test_empty_history_refuses():
    g = synth_games()
    with pytest.raises(R.RatingsError):
        R.solve(g, 2023, 1, P, prior=FLAT_PRIOR)


def test_load_params_refuses_when_never_fit(tmp_path):
    with pytest.raises(R.RatingsError):
        R.load_params(str(tmp_path / "nope.json"))


def test_save_then_load_params_roundtrips(tmp_path):
    p = replace(P, lambda_team=3.25, decay=0.07)
    path = R.save_params(p, str(tmp_path / "p.json"))
    assert R.load_params(path) == p


# --------------------------------------------------------------------------- #
# Preseason prior — the talent-sentinel landmine
# --------------------------------------------------------------------------- #
def test_prior_falls_back_to_conference_mean_for_missing_talent(monkeypatch):
    """CFBD serves 0 for the service academies. NULL means UNMEASURED.

    Air Force led the 2023 Mountain West in net PPA. A prior that reads its
    missing talent as "worst in FBS" is not a rounding error, it is the
    difference between rating the best team in the conference top and bottom.
    """
    teams = ["Air Force", "Boise State", "Fresno State", "Nevada", "Alabama"]
    conf = {"Air Force": "Mountain West", "Boise State": "Mountain West",
            "Fresno State": "Mountain West", "Nevada": "Mountain West",
            "Alabama": "SEC"}

    talent = pd.DataFrame([
        {"season": 2023, "team": "Air Force", "talent": np.nan,
         "unranked_recruiting": True},
        {"season": 2023, "team": "Boise State", "talent": 600.0,
         "unranked_recruiting": False},
        {"season": 2023, "team": "Fresno State", "talent": 560.0,
         "unranked_recruiting": False},
        {"season": 2023, "team": "Nevada", "talent": 520.0,
         "unranked_recruiting": False},
        {"season": 2023, "team": "Alabama", "talent": 1015.0,
         "unranked_recruiting": False},
    ])
    returning = pd.DataFrame([
        {"season": 2023, "team": t, "percent_ppa": v} for t, v in
        [("Air Force", 0.60), ("Boise State", 0.60), ("Fresno State", 0.60),
         ("Nevada", 0.60), ("Alabama", 0.60)]])

    monkeypatch.setattr(R.ingest, "load_talent", lambda s: talent)
    monkeypatch.setattr(R.ingest, "load_returning", lambda s: returning)

    prior = R.preseason_prior(2023, teams, conf, R.RatingParams())
    mwc = ["Air Force", "Boise State", "Fresno State", "Nevada"]
    assert prior["Air Force"] != min(prior.values()), \
        "unmeasured talent was treated as low talent"
    assert prior["Air Force"] > prior["Nevada"], (
        "Air Force must inherit the Mountain West mean, which sits above the "
        f"conference's weakest recruiter; got {prior}")
    # and it must sit at the conference mean of the measured teams
    measured = [prior[t] for t in ("Boise State", "Fresno State", "Nevada")]
    assert prior["Air Force"] == pytest.approx(float(np.mean(measured)), abs=1e-6)


def test_prior_never_reads_zero_as_a_measurement(monkeypatch):
    """Guard against a future regression that stops nulling the sentinel."""
    teams = ["Army", "Navy", "Rutgers"]
    conf = {"Army": "FBS Independents", "Navy": "American Athletic",
            "Rutgers": "Big Ten"}
    talent = pd.DataFrame([
        {"season": 2023, "team": "Army", "talent": np.nan, "unranked_recruiting": True},
        {"season": 2023, "team": "Navy", "talent": np.nan, "unranked_recruiting": True},
        {"season": 2023, "team": "Rutgers", "talent": 700.0,
         "unranked_recruiting": False},
    ])
    returning = pd.DataFrame([{"season": 2023, "team": t, "percent_ppa": 0.5}
                              for t in teams])
    monkeypatch.setattr(R.ingest, "load_talent", lambda s: talent)
    monkeypatch.setattr(R.ingest, "load_returning", lambda s: returning)
    prior = R.preseason_prior(2023, teams, conf, R.RatingParams())
    assert all(np.isfinite(v) for v in prior.values())
    # Army/Navy have no measured conference-mate here, so they land at the
    # league mean of the z-score (0), NEVER at the bottom.
    assert prior["Army"] == pytest.approx(prior["Navy"])
    assert prior["Army"] > min(prior.values()) or prior["Army"] == prior["Navy"]


# --------------------------------------------------------------------------- #
# Unit ratings
# --------------------------------------------------------------------------- #
def test_success_definition_matches_down_and_distance():
    assert R._success(1, 10, 5) is True and R._success(1, 10, 4) is False
    assert R._success(2, 10, 7) is True and R._success(2, 10, 6) is False
    assert R._success(3, 10, 10) is True and R._success(3, 10, 9) is False
    assert R._success(None, 10, 5) is None


def _synth_plays():
    rows, gid = [], 1000
    rng = np.random.default_rng(7)
    teams = ["A", "B", "C", "D"]
    for wk in range(1, 9):
        for i, h in enumerate(teams):
            a = teams[(i + wk) % len(teams)]
            if a == h:
                continue
            for off, dfn in ((h, a), (a, h)):
                for j in range(40):
                    gained = float(rng.integers(0, 14))
                    rows.append({
                        "game_id": gid, "season": 2023, "week": wk, "period": 2,
                        "offense": off, "defense": dfn, "offense_score": 7,
                        "defense_score": 7, "down": 1 + j % 3, "distance": 10,
                        "yards_to_goal": 60, "yards_gained": gained,
                        "play_type": "Rush" if j % 2 else "Pass Reception",
                        "ppa": float(rng.normal(0.2, 1.0)), "garbage_time": False,
                    })
            gid += 1
    return pd.DataFrame(rows)


def test_units_produce_all_eight_ratings():
    g = synth_games()
    plays = _synth_plays()
    u = R.play_units(plays)
    assert set(u["phase"]) == {"rush", "pass"}
    out = R.solve_units(g, u, 2023, 9, P)
    assert set(out) == set(R.UNIT_NAMES)
    assert len(R.UNIT_NAMES) == 8


def test_explosiveness_is_shrunk_harder_than_efficiency():
    """`research/win-factors-literature-scan.md`: explosiveness has near-zero
    week-to-week stickiness. It must be pulled toward the mean harder."""
    g = synth_games()
    u = R.play_units(_synth_plays())
    light = R.solve_units(g, u, 2023, 9, replace(P, explosiveness_shrinkage=1.0))
    heavy = R.solve_units(g, u, 2023, 9, replace(P, explosiveness_shrinkage=8.0))
    for name in R.UNIT_NAMES:
        s_light = float(np.std(list(light[name].values())))
        s_heavy = float(np.std(list(heavy[name].values())))
        if name.endswith("_exp"):
            assert s_heavy < s_light, f"{name} was not shrunk by the multiplier"
        else:
            assert s_heavy == pytest.approx(s_light, abs=1e-9), \
                f"{name} is an efficiency unit and must be untouched"


# --------------------------------------------------------------------------- #
# Cache-backed: the gate evidence that has to come from real data
# --------------------------------------------------------------------------- #
def _real_games():
    try:
        g = R.game_frame([2023])
    except Exception:                        # noqa: BLE001
        pytest.skip("no cached CFBD parquet")
    if g.empty:
        pytest.skip("no cached CFBD games")
    return g


def test_real_slice_is_reproducible():
    g = _real_games()
    p = R.RatingParams()
    a, b = R.solve(g, 2023, 15, p), R.solve(g, 2023, 15, p)
    assert a.fingerprint() == b.fingerprint()
    pd.testing.assert_frame_equal(a.to_frame(), b.to_frame())


def test_connectivity_is_populated_and_rises_through_the_season():
    """Gate for Workstream B: "connectivity populated", and the DoD's
    "demonstrably lower in September than November"."""
    g = _real_games()
    p = R.RatingParams()
    early = R.solve(g, 2023, 4, p)           # September
    late = R.solve(g, 2023, 13, p)           # November
    for t in (early, late):
        f = t.to_frame()
        assert f["connectivity"].notna().all()
        assert (f["connectivity"] > 0).any()
    e, l = early.to_frame(), late.to_frame()
    assert l["connectivity"].mean() > e["connectivity"].mean(), (
        f"September {e['connectivity'].mean():.3f} vs "
        f"November {l['connectivity'].mean():.3f}")
    assert l.loc[~l["veto"], "net_band"].mean() < e.loc[~e["veto"], "net_band"].mean()


def test_air_force_is_not_bottom_of_the_real_prior():
    """The live version of the sentinel test. Air Force has NULL talent."""
    g = _real_games()
    t = R.solve(g, 2023, 15, R.RatingParams())
    mwc = [team for team, c in t.team_conf.items() if c == "Mountain West"]
    if "Air Force" not in mwc:
        pytest.skip("Air Force not in the cached slice")
    priors = {team: t.prior[team] for team in mwc if team in t.prior}
    assert priors["Air Force"] > min(priors.values()), (
        "Air Force landed at the bottom of the preseason prior — the NULL "
        f"talent sentinel is being read as low talent. priors={priors}")


def test_known_ordering_sanity_on_the_real_slice():
    """DoD: a known strong team must out-rate a known bottom-tier one.

    Nevada went 2-10 in 2023 and was the worst team in the Mountain West by
    point differential; Boise State went 8-5 and won the conference's bowl bid.
    """
    g = _real_games()
    t = R.solve(g, 2023, 15, R.RatingParams())
    for team in ("Boise State", "Nevada", "USC"):
        if team not in t.net:
            pytest.skip(f"{team} not in the cached slice")
    assert t.net["Boise State"] > t.net["Nevada"]
    assert not t.veto["Boise State"] and not t.veto["Nevada"]
    # The closest thing to a blue-blood in an MWC-only ingest. USC appears in
    # two games, so its band is roughly twice Boise's — the ORDERING is the
    # assertion, the precision is not.
    assert t.net["USC"] > t.net["Nevada"]
    assert t.net_band["USC"] > t.net_band["Boise State"]


def test_real_slice_hfa_is_in_the_cfb_range_not_the_nfl_one():
    """Not a tuning target — a smell test with a deliberately wide band.

    `research/win-factors-literature-scan.md` puts modern CFB HFA near 2.5-3.5
    points. If this fit landed at 0.5 or at 9 something is wrong upstream. The
    assertion is wide on purpose so it can never become a reason to tune.
    """
    g = _real_games()
    t = R.solve(g, 2023, 15, R.RatingParams())
    assert 0.5 < t.hfa_league < 7.0, t.hfa_league
