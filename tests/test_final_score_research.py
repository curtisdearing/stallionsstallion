"""Research-only guarantees; fixtures need neither a CFBD cache nor a market."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from nflvalue import final_score_research as F
from nflvalue import ratings as R, cfb_sim as S
from nflvalue.cfb_ot import OTParams


def games():
    rows = []
    for s in (2019, 2020, 2021):
        for w in range(1, 7):
            for h, a in [('A','B'), ('B','C'), ('C','A')]:
                hp = 21 + 2 * w + (h == 'A') * 7
                ap = 14 + w + (a == 'B') * 3
                rows.append(dict(id=len(rows), season=s, week=w,
                    home_team=h, away_team=a, home_points=hp, away_points=ap,
                    margin=hp-ap, home_fbs=True, away_fbs=True,
                    home_conf='SEC', away_conf='ACC', neutral_site=w%2==0))
    return pd.DataFrame(rows)


def fit(g=None, week=3, decay=.02, ridge=30):
    return F.fit_total(games() if g is None else g, 2021, week,
                       window=3, population='p4', ridge=ridge, decay=decay)


def test_market_and_future_results_or_prose_cannot_change_forecast():
    g = games(); original = fit(g)
    poisoned = g.assign(sportsbook_line=9999, generated_news='future winner A')
    poisoned.loc[(poisoned.season==2021)&(poisoned.week>=3),
                 ['home_points','away_points','margin']] = 999
    changed = fit(poisoned)
    assert original.table.fingerprint() == changed.table.fingerprint()
    assert F.expected_scores(original,'A','B',7,False) == F.expected_scores(changed,'A','B',7,False)
    changed_past = g.copy(); changed_past.loc[changed_past.season==2020,'home_points'] += 20
    assert fit(changed_past).table.fingerprint() != original.table.fingerprint()


def test_uncentering_reproduces_training_score_level_with_free_intercept():
    f = fit(); g = f.games; weighted_mean = np.average(
        g.home_points+g.away_points, weights=R._row_weights(g,2021,3,f.table.params))
    totals = [F.ridge_total(f,r.home_team,r.away_team,r.neutral_site)[0] for r in g.itertuples()]
    assert np.average(totals,weights=R._row_weights(g,2021,3,f.table.params)) == pytest.approx(weighted_mean,abs=1e-9)


def test_uncentering_handles_historical_conference_changes():
    g=games(); g.loc[(g.season==2019)&(g.home_team=='A'),'home_conf']='Big Ten'
    f=fit(g); t=f.table
    raw=[]
    for r in f.games.itertuples():
        raw.append(2*f.intercept + t.off_dev[r.home_team]+t.def_dev[r.away_team]
            +t.off_dev[r.away_team]+t.def_dev[r.home_team]
            +t.conf_off[r.home_conf]+t.conf_def[r.away_conf]
            +t.conf_off[r.away_conf]+t.conf_def[r.home_conf]
            +(t.hfa_league+t.hfa_team[r.home_team] if not r.neutral_site else 0))
    w=R._row_weights(f.games,2021,3,t.params)
    assert np.average(raw,weights=w)==pytest.approx(np.average(f.games.home_points+f.games.away_points,weights=w),abs=1e-9)


@pytest.mark.parametrize('margin',[-150,-7.5,0,7.5,150])
def test_expected_means_are_coherent_not_discrete_draws(margin):
    p=F.expected_scores(fit(),'A','B',margin,False)
    assert p['home']>=0 and p['away']>=0
    assert p['home']-p['away']==pytest.approx(margin)
    assert p['home']+p['away']==pytest.approx(p['total'])
    assert 'samples' not in p


def test_unseen_team_has_explicit_research_reference_fallback():
    f=fit(); p=F.expected_scores(f,'NEW','B',7,True)
    assert p['fallback'] and p['total']==pytest.approx(f.mean_home+f.mean_away)


def test_fit_window_and_population_are_explicit_and_prior_read_is_impossible(monkeypatch):
    monkeypatch.setattr(R,'preseason_prior',lambda *a,**k:pytest.fail('external prior read'))
    f=F.fit_total(games(),2021,1,window=1,population='p4',ridge=30,decay=0)
    assert set(f.games.season)=={2020}
    assert set(F.allowed_games(games().assign(closing_spread=6)).columns)==set(F.GAME_COLUMNS)


def test_team_name_familiarity_has_no_numerical_effect():
    g=games(); original=F.expected_scores(fit(g),'A','B',7,False)
    renamed=g.replace({'A':'famous team','B':'unknown team','C':'third team'})
    changed=F.expected_scores(fit(renamed),'famous team','unknown team',7,False)
    assert changed['total']==pytest.approx(original['total'],abs=1e-9)


def test_stronger_ridge_reduces_single_extreme_game_influence():
    g=games(); p=g.copy(); p.loc[(p.season==2021)&(p.week==2)&(p.home_team=='A'),'home_points'] += 42
    shifts=[]
    for ridge in [3,3000]:
        before=F.expected_scores(fit(g,ridge=ridge),'A','B',7,False)['total']
        after=F.expected_scores(fit(p,ridge=ridge),'A','B',7,False)['total']
        shifts.append(abs(after-before))
    assert shifts[1]<shifts[0]


def test_train_residual_covariance_and_margin_total_identity():
    predicted=np.array([[20,10],[22,18],[15,27],[30,12]],float)
    actual=np.array([[17,14],[23,19],[20,21],[32,9]],float)
    u=F.fit_uncertainty(predicted,actual)
    c=np.cov((actual-predicted).T,ddof=1)
    assert np.allclose(u['covariance'],c)
    assert u['margin_sd']**2==pytest.approx(c[0,0]+c[1,1]-2*c[0,1])
    assert u['total_sd']**2==pytest.approx(c[0,0]+c[1,1]+2*c[0,1])
    assert u['distribution_support']=='continuous_error_approximation_not_discrete_scores'


def test_inner_selection_rejects_outer_or_future_rows():
    d=pd.DataFrame(dict(season=[2020,2021],week=[1,1],ridge=[30,30],decay=[0,0],loss=[10.,0.]))
    with pytest.raises(ValueError,match='outer'):
        F.select_hyperparameters(d,2021)


def test_return_turnover_shrinkage_single_event_influence_is_bounded():
    for n in [1,10,100]:
        raw_delta=1/n
        for prior_n in [4,8,16]:
            delta=F.shrunk_rate(1,n,.02,prior_n)-F.shrunk_rate(0,n,.02,prior_n)
            assert delta==pytest.approx(1/(n+prior_n)) and delta<raw_delta


def test_continuous_margin_adjustment_is_not_a_joint_football_score_distribution():
    probs=np.zeros((2,len(S.CLASSES))); probs[0,S.CLASSES.index((0,0))]=1; probs[1,S.CLASSES.index((7,0))]=1
    rates=S.DriveRateModel(np.array([-1.,1.]),probs); pace=S.PaceModel(np.array([[6,6]])); ot=OTParams(.4,.2,{7:1.},.5)
    base=S.simulate_game(np.random.default_rng(20260908),1,-1,rates,pace,ot)
    adj=S.simulate_game(np.random.default_rng(20260908),1,-1,rates,pace,ot,param_sd=3.6,param_rng=np.random.default_rng(20260909))
    assert np.array_equal(base['total'],adj['total'])
    assert base['margin_mean']==pytest.approx(adj['margin_mean'])
    h=(adj['total']+adj['margin'])/2; a=(adj['total']-adj['margin'])/2
    assert np.any(a<0) and np.any(h!=np.round(h))
    assert np.all((base['total']-base['margin'])/2>=0)


def test_official_final_scores_are_used_including_overtime():
    g=games(); g.loc[0,['home_points','away_points']]=[34,31]
    # No extra OT increment or regulation-only substitution is permitted here.
    h=F.allowed_games(g)
    assert h.loc[0,'home_points']==34 and h.loc[0,'away_points']==31


def test_every_forecast_probability_belongs_to_exactly_one_bin():
    p=np.arange(0,11)/10
    from scipy.special import ndtri
    margin=ndtri(np.clip(p,1e-12,1-1e-12))
    m=F.loss_metrics(30+margin/2,30-margin/2,np.full(11,31),np.full(11,29),1,1,1,2)
    assert sum(b['n'] for b in m['win']['reliability'])==11


def test_runner_outer_results_do_not_change_selection_or_uncertainty():
    from scripts.final_score_experiment import select_and_calibrate
    rows=[]; anchors=[]
    for season in range(2020,2026):
        for i in range(6):
            anchors.append(dict(season=season,week=1,id=season*10+i,home='A',away='B',
                actual_home=20+i*i,actual_away=17+3*i,p4=True))
            for ridge in [3,3000]:
                rows.append(dict(variant='C1_p4_all',season=season,week=1,id=season*10+i,
                    ridge=ridge,decay=.02,loss=float(i+ridge/3000),
                    home=25.,away=20.,reference_home=25.,reference_away=20.,
                    simple_home=24.,simple_away=21.,frozen_state_home=25.,frozen_state_away=20.))
    grid=pd.DataFrame(rows); a=pd.DataFrame(anchors)
    original,fits,_=select_and_calibrate(grid,a)
    poisoned=a.copy(); poisoned.loc[poisoned.season>=2023,['actual_home','actual_away']]=99
    changed,after,_=select_and_calibrate(grid,poisoned)
    assert [f for f in fits if f['outer_season']<=2023]==[f for f in after if f['outer_season']<=2023]
    cols=['model','season','id','home','away','home_sd','away_sd','margin_sd','total_sd']
    pd.testing.assert_frame_equal(original.loc[original.season<=2023,cols],changed.loc[changed.season<=2023,cols])
