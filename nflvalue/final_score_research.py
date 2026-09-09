"""Isolated C1 score-level adapter. Never imported by a production entry point.

This is a new research score capability, not incumbent final scores. It reuses
ratings.solve; no market, news, file reader, or alternate rating engine lives
here. Historical week cutoffs do not establish historical receipt timestamps.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.special import ndtr
from . import ratings as R

VERSION = 'c1-ridge-total-v1'
P4 = {'ACC', 'Big Ten', 'Big 12', 'SEC'}
GAME_COLUMNS = ('id','season','week','home_team','away_team','home_points',
                'away_points','home_fbs','away_fbs','home_conf','away_conf','neutral_site')


def allowed_games(games):
    """Positive schema; caller's ancillary market/prose/outcome-derived columns disappear."""
    out=games.loc[:,GAME_COLUMNS].copy()
    if out.duplicated(['season','id']).any():
        raise ValueError('duplicate game identity')
    return out.sort_values(['season','week','id'],kind='stable').reset_index(drop=True)


def history(games, season, week, window, population):
    g=allowed_games(games)
    keep=(g.season<season)|((g.season==season)&(g.week<week))
    if window != 'all':
        keep &= g.season>=season-int(window)
    keep &= g.home_fbs & g.away_fbs
    if population=='p4':
        keep &= g.home_conf.isin(P4)|g.away_conf.isin(P4)
    elif population!='fbs':
        raise ValueError('unknown training population')
    out=g.loc[keep].dropna(subset=['home_points','away_points']).copy()
    if out.empty:
        raise ValueError('no training games')
    return out


@dataclass
class TotalFit:
    table: R.RatingsTable
    games: pd.DataFrame
    intercept: float
    mean_home: float
    mean_away: float


def fit_total(games, season, week, *, window, population, ridge, decay):
    hist=history(games,season,week,window,population)
    params=R.RatingParams(lambda_team=ridge,lambda_conf=ridge,lambda_hfa=ridge,
                          decay=decay,prior_strength=0)
    table=R.solve(hist,season,week,params,prior={})
    # R.solve centers penalized columns, but exports centered mu and raw
    # coefficients. Recover their shared origin; mu alone is NOT the raw
    # score intercept. Historical rows need their own conference membership.
    shifts=[]
    for r in hist.itertuples():
        shifts.append(table.off_dev[r.home_team]+table.def_dev[r.away_team]
            +table.off_dev[r.away_team]+table.def_dev[r.home_team]
            +table.conf_off[r.home_conf]+table.conf_def[r.away_conf]
            +table.conf_off[r.away_conf]+table.conf_def[r.home_conf]
            +(table.hfa_team[r.home_team] if not r.neutral_site else 0))
    weights=R._row_weights(hist,season,week,params)
    intercept=table.mu-float(np.average(shifts,weights=weights))/2
    return TotalFit(table,hist,intercept,float(hist.home_points.mean()),float(hist.away_points.mean()))


def ridge_total(fit, home, away, neutral, home_conf=None, away_conf=None):
    t=fit.table
    if home not in t.off or away not in t.off:
        return fit.mean_home+fit.mean_away, True
    hc=home_conf or t.team_conf[home]; ac=away_conf or t.team_conf[away]
    if hc not in t.conf_off or ac not in t.conf_off:
        return fit.mean_home+fit.mean_away, True
    value=(2*fit.intercept+t.off_dev[home]+t.def_dev[away]+t.off_dev[away]+t.def_dev[home]
           +t.conf_off[hc]+t.conf_def[ac]+t.conf_off[ac]+t.conf_def[hc]
           +(t.hfa_league+t.hfa_team[home] if not neutral else 0))
    return float(value),False


def from_total_margin(total, margin):
    if not np.isfinite([total,margin]).all():
        raise ValueError('nonfinite expected score')
    total=float(max(total,abs(margin)))
    return dict(home=(total+margin)/2,away=(total-margin)/2,total=total)


def expected_scores(fit,home,away,margin,neutral,home_conf=None,away_conf=None):
    raw,fallback=ridge_total(fit,home,away,neutral,home_conf,away_conf)
    return dict(**from_total_margin(raw,margin),raw_total=raw,fallback=fallback,
                mean_constraint_applied=raw<abs(margin))


def select_hyperparameters(inner, outer_season):
    if (inner.season>=outer_season).any():
        raise ValueError('outer or future outcomes entered inner selection')
    if inner.empty:
        return dict(ridge=3000,decay=.05,unfit=True)
    means=inner.groupby(['ridge','decay']).loss.mean()
    best=means.idxmin()
    b=inner[(inner.ridge==best[0])&(inner.decay==best[1])].groupby(['season','week']).loss.mean()
    se=float(b.std(ddof=1)/np.sqrt(len(b))) if len(b)>1 else 0.
    allowed=means[means<=float(means.min())+se].reset_index()
    chosen=allowed.sort_values(['ridge','decay'],ascending=False).iloc[0]
    return dict(ridge=int(chosen.ridge),decay=float(chosen.decay),unfit=False,
                inner_best_loss=float(means.min()),one_se=se)


def fit_uncertainty(predicted, actual):
    residual=np.asarray(actual,float)-np.asarray(predicted,float)
    if residual.ndim!=2 or residual.shape[1]!=2 or len(residual)<3 or not np.isfinite(residual).all():
        raise ValueError('need finite paired chronological training residuals')
    cov=np.cov(residual.T,ddof=1)
    if np.linalg.eigvalsh(cov).min()<=0:
        raise ValueError('singular paired uncertainty; unavailable, no guessed SD')
    return dict(covariance=cov.tolist(),n=len(residual),mean_residual=residual.mean(axis=0).tolist(),
        home_sd=float(np.sqrt(cov[0,0])),away_sd=float(np.sqrt(cov[1,1])),
        margin_sd=float(np.sqrt(cov[0,0]+cov[1,1]-2*cov[0,1])),
        total_sd=float(np.sqrt(cov[0,0]+cov[1,1]+2*cov[0,1])),
        distribution_support='continuous_error_approximation_not_discrete_scores')


def shrunk_rate(events, exposure, prior_rate, prior_exposure):
    """Structural influence probe only; no return/turnover feature is activated."""
    if not 0<=events<=exposure or prior_exposure<=0 or not 0<=prior_rate<=1:
        raise ValueError('invalid rate inputs')
    return (events+prior_rate*prior_exposure)/(exposure+prior_exposure)


def loss_metrics(home,away,actual_home,actual_away,home_sd,away_sd,margin_sd,total_sd):
    h,a,yh,ya=map(lambda x:np.asarray(x,float),(home,away,actual_home,actual_away))
    out={'n':len(h)}
    for name,mean,y,sd in [('home',h,yh,home_sd),('away',a,ya,away_sd),
                          ('margin',h-a,yh-ya,margin_sd),('total',h+a,yh+ya,total_sd)]:
        sd=np.broadcast_to(np.asarray(sd,float),len(h)); e=mean-y; z=(y-mean)/sd
        out[name]=dict(mae=float(abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),bias=float(e.mean()),
            absolute_error_p90=float(np.quantile(abs(e),.9)),absolute_error_p95=float(np.quantile(abs(e),.95)),
            crps=float(np.mean(sd*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi)))),
            negative_mass=float(ndtr(-mean/sd).mean()) if name!='margin' else None)
        for level,q in [(50,.6744897501960817),(80,1.2815515655446004),(95,1.959963984540054)]:
            out[name][f'coverage{level}']=float((abs(z)<=q).mean())
            out[name][f'width{level}']=float((2*q*sd).mean())
    p=ndtr((h-a)/np.asarray(margin_sd)); y=(yh>ya).astype(float); q=np.clip(p,1e-12,1-1e-12)
    out['team_mae']=(out['home']['mae']+out['away']['mae'])/2
    out['win']=dict(brier=float(np.mean((p-y)**2)),log_loss=float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q))),
                    probability='Gaussian margin approximation, identity calibration, SD learned on inner residuals',reliability=[])
    bin_index=np.minimum((p*10).astype(int),9)
    for i in range(10):
        lo=i/10; ix=bin_index==i; n=int(ix.sum())
        if not n: continue
        rate=float(y[ix].mean()); den=1+1.96**2/n
        center=(rate+1.96**2/(2*n))/den; width=1.96*np.sqrt(rate*(1-rate)/n+1.96**2/(4*n*n))/den
        out['win']['reliability'].append(dict(lo=float(lo),n=n,p=float(p[ix].mean()),observed=rate,wilson95=[center-width,center+width]))
    out['joint_energy_score']=None; out['discrete_pit']=None; out['exact_score_probability']=None
    return out
