#!/usr/bin/env python3
"""Immutable, cached-input C1 development experiment; never calls weekly writers.

Run: .venv/bin/python scripts/final_score_experiment.py --output NEW_DIRECTORY
The directory must not exist. No API calls, market files, rendered notes, or
evidence-layer profiles enter the calculation. See c1_preregistration.json.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, ndtr, logit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nflvalue import ratings as R, engine as EG, final_score_research as F, forecast_contract as FC
from docs.final_score_program.reproduce_audit import scores as baseline_scores

RIDGES=(3,30,300,3000); DECAYS=(0.,.02,.05)
POPS=('p4','fbs'); WINDOWS=(1,3,'all'); SEED=20260908


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p,value):
    Path(p).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def supported_baselines():
    """Recompute approved record summaries without optional dirty-worktree files."""
    original=json.loads((ROOT/'docs/final_score_program/audit_evidence.json').read_text())
    bases=FC.BasisRegistry.load(str(ROOT/'data')); out={}
    for name,col,family,total in [('early','pred_margin','ridge_preseason_prior',False),
            ('cross_division','pred_margin','ridge_cross_division',False),
            ('c_corrected_wf_kalman_20260830','mc_margin',None,True)]:
        path=ROOT/f'data/backtest_{name}.csv'; old=original['baselines'][name]
        if sha(path)!=old['sha256']: raise RuntimeError(f'Baseline source changed: {name}')
        cols=['season','week','id','home','away','margin',col]
        if total: cols+=['mc_margin_sd','mc_total','mc_total_sd','p_home_win','total']
        d=pd.read_csv(path,usecols=cols)
        sd=d.mc_margin_sd.to_numpy() if total else bases.margin_sd(family,1)
        rec=dict(input=str(path.relative_to(ROOT)),sha256=sha(path),full=baseline_scores(d,col,sd,total))
        if rec['full']!=old['full']: raise RuntimeError(f'Baseline metrics changed: {name}')
        if name!='cross_division':
            games=pd.concat([pd.read_parquet(ROOT/f'data/cfbd/games_{s}.parquet',columns=['season','id','home_conference','away_conference']) for s in sorted(d.season.unique())])
            meta=d.merge(games,on=['season','id'],validate='one_to_one')
            sub=d.loc[(meta.home_conference.isin(F.P4)|meta.away_conference.isin(F.P4)).to_numpy()]
            rec['p4_involved']=baseline_scores(sub,col,sub.mc_margin_sd.to_numpy() if total else sd,total)
            if rec['p4_involved']!=old['p4_involved']: raise RuntimeError(f'Baseline P4 metrics changed: {name}')
        rec['status']=old['manifest_status'];out[name]=rec
    return dict(baselines=out,selected_later_benchmark=original['selected_later_benchmark'],
                metrics_exactly_match_audit=True,incumbent_early_team_scores=None,incumbent_early_total=None,
                certified_later_incumbent_scores=None,corrected_later_record_is_unselected_candidate=True)


def anchors(g):
    rows=[]; refused=[]; params=R.load_params()
    for s in range(2020,2026):
        for w in (1,2,3):
            hist=g[(g.season<s)|((g.season==s)&(g.week<w))]
            table=EG.solve_production_table(hist,s,w,params)
            target=g[(g.season==s)&(g.week==w)&g.home_fbs&g.away_fbs]
            for r in target.itertuples():
                try: margin=table.predict_margin(r.home_team,r.away_team,r.neutral_site)
                except R.RatingsError:
                    refused.append(dict(season=s,week=w,id=int(r.id),reason='incumbent_unratable')); continue
                rows.append(dict(season=s,week=w,id=int(r.id),home=r.home_team,away=r.away_team,
                    home_conf=r.home_conf,away_conf=r.away_conf,neutral=bool(r.neutral_site),
                    actual_home=float(r.home_points),actual_away=float(r.away_points),anchor=float(margin),
                    p4=r.home_conf in F.P4 or r.away_conf in F.P4,
                    cutoff_id=f'{s}_wk{w:02d}_strictly_earlier_week',
                    availability='reconstructed_week_granular_not_historical_receipt_verified'))
    out=pd.DataFrame(rows)
    selected=pd.read_csv(ROOT/'data/backtest_early.csv',usecols=['season','id','pred_margin'])
    paired=selected.merge(out,on=['season','id'],validate='one_to_one',how='left')
    if len(paired)!=724 or paired.anchor.isna().any() or not np.array_equal(paired.pred_margin.to_numpy(),paired.anchor.to_numpy()):
        raise RuntimeError('Selected incumbent replay parity failed; stop before candidate evaluation')
    if len(out[out.season>=2021])!=len(selected):
        raise RuntimeError('Incumbent cohort changed; stop and version the protocol')
    return out,refused,dict(n=724,max_absolute_delta=0.,selected_sha256=sha(ROOT/'data/backtest_early.csv'))


def forecast_grid(g,a):
    rows=[]
    for pop in POPS:
        for window in WINDOWS:
            variant=f'C1_{pop}_{window}'
            for ridge in RIDGES:
                for decay in DECAYS:
                    for s in range(2020,2026):
                        frozen_fit=None
                        for w in (1,2,3):
                            fit=F.fit_total(g,s,w,window=window,population=pop,ridge=ridge,decay=decay)
                            if w==1: frozen_fit=fit
                            target=a[(a.season==s)&(a.week==w)]
                            current=fit.games[fit.games.season==s]
                            for r in target.itertuples():
                                pred=F.expected_scores(fit,r.home,r.away,r.anchor,r.neutral,r.home_conf,r.away_conf)
                                ref=F.from_total_margin(fit.mean_home+fit.mean_away,r.anchor)
                                frozen=F.expected_scores(frozen_fit,r.home,r.away,r.anchor,r.neutral,r.home_conf,r.away_conf)
                                rows.append(dict(variant=variant,ridge=ridge,decay=decay,season=s,week=w,id=r.id,
                                    home=pred['home'],away=pred['away'],raw_total=pred['raw_total'],
                                    fallback=pred['fallback'],constrained=pred['mean_constraint_applied'],
                                    reference_home=ref['home'],reference_away=ref['away'],
                                    simple_home=fit.mean_home,simple_away=fit.mean_away,
                                    frozen_state_home=frozen['home'],frozen_state_away=frozen['away'],
                                    frozen_state_raw_total=frozen['raw_total'],frozen_state_fallback=frozen['fallback'],
                                    frozen_state_constrained=frozen['mean_constraint_applied'],
                                    fit_games=len(fit.games),
                                    fit_fingerprint=fit.table.fingerprint(),
                                    home_current_games=int(((current.home_team==r.home)|(current.away_team==r.home)).sum()),
                                    away_current_games=int(((current.home_team==r.away)|(current.away_team==r.away)).sum()),
                                    loss=(abs(pred['home']-r.actual_home)+abs(pred['away']-r.actual_away))/2))
            print(f'computed {variant} chronological grid',flush=True)
    return pd.DataFrame(rows)


def calibration_diagnostic(d):
    p=ndtr((d.home-d.away)/d.margin_sd); y=(d.actual_home>d.actual_away).astype(float).to_numpy()
    x=np.column_stack([np.ones(len(d)),logit(np.clip(p,1e-9,1-1e-9))])
    result=minimize(lambda b:float(np.sum(np.logaddexp(0,x@b)-y*(x@b))),[0.,1.],method='BFGS')
    b=result.x; q=expit(x@b); h=x.T@((q*(1-q))[:,None]*x)
    cov=np.linalg.inv(h); se=np.sqrt(np.diag(cov))
    return dict(intercept=float(b[0]),slope=float(b[1]),wald95=[[float(z-1.96*s),float(z+1.96*s)] for z,s in zip(b,se)],
        semantics='Held-out calibration diagnostic only, never applied to forecasts; Wald intervals assume independent games, paired decision uses week/season resampling',optimizer_success=bool(result.success))


def summary(d):
    out=F.loss_metrics(d.home,d.away,d.actual_home,d.actual_away,d.home_sd,d.away_sd,d.margin_sd,d.total_sd)
    out['win']['calibration']=calibration_diagnostic(d)
    out['fallback_n']=int(d.fallback.sum()); out['constrained_mean_n']=int(d.constrained.sum())
    return out


def select_and_calibrate(grid,a):
    records=[]; fits=[]; inner_search=[]
    for variant in sorted(grid.variant.unique()):
        v=grid[grid.variant==variant]
        for season in range(2021,2026):
            train=v[v.season<season].merge(a[['season','id','actual_home','actual_away','p4']],on=['season','id'],validate='many_to_one')
            # Selection is scored on primary P4 games for BOTH training populations.
            selected=F.select_hyperparameters(train[train.p4],season)
            for (ridge,decay), sub in train[train.p4].groupby(['ridge','decay']):
                inner_search.append(dict(variant=variant,outer_season=season,ridge=int(ridge),decay=float(decay),n=len(sub),mean_team_mae=float(sub.loss.mean())))
            inner=train[(train.ridge==selected['ridge'])&(train.decay==selected['decay'])&train.p4]
            test=v[(v.season==season)&(v.ridge==selected['ridge'])&(v.decay==selected['decay'])].merge(
                a.drop(columns=['home','away']),on=['season','week','id'],validate='one_to_one')
            # Fit covariance using only inner chronological residuals. These
            # same residuals selected hyperparameters; this limitation is
            # inside training, not a use of outer labels for calibration.
            models={}
            for kind,cols in [('candidate',('home','away')),('reference',('reference_home','reference_away')),
                              ('simple',('simple_home','simple_away')),('frozen_state',('frozen_state_home','frozen_state_away'))]:
                u=F.fit_uncertainty(inner[list(cols)].to_numpy(),inner[['actual_home','actual_away']].to_numpy())
                models[kind]=u
                for r in test.to_dict('records'):
                    rec=dict(r); rec['model']=variant if kind=='candidate' else f'{variant}__{kind}'
                    rec['home']=r[cols[0]]; rec['away']=r[cols[1]]
                    if kind in ('reference','simple'):
                        rec['raw_total']=r['simple_home']+r['simple_away']
                        rec['fallback']=False
                        rec['constrained']=bool(kind=='reference' and rec['raw_total']<abs(r.get('anchor',0.)))
                    elif kind=='frozen_state':
                        rec['raw_total']=r.get('frozen_state_raw_total',rec['home']+rec['away'])
                        rec['fallback']=r.get('frozen_state_fallback',False)
                        rec['constrained']=r.get('frozen_state_constrained',False)
                    rec['adaptation_semantics']='total_state_frozen_margin_updated' if kind=='frozen_state' else 'weekly_total_state_update'
                    rec.update({k:u[k] for k in ['home_sd','away_sd','margin_sd','total_sd']})
                    rec['score_covariance']=u['covariance'][0][1]
                    rec['calibration_max_season']=int(inner.season.max()); rec['calibration_n']=len(inner)
                    rec['error_home']=rec['home']-r['actual_home']; rec['error_away']=rec['away']-r['actual_away']
                    rec['error_margin']=rec['error_home']-rec['error_away']; rec['error_total']=rec['error_home']+rec['error_away']
                    rec['team_loss']=(abs(rec['error_home'])+abs(rec['error_away']))/2
                    rec['loss']=rec['team_loss']
                    rec['model_version']=F.VERSION
                    records.append(rec)
            fits.append(dict(variant=variant,outer_season=season,selection=selected,
                max_selection_season=int(train.season.max()),inner_n=len(inner),uncertainty=models))
    return pd.DataFrame(records),fits,inner_search


def paired_comparisons(rows):
    d=rows[rows.p4]
    ref=d[d.model=='C1_p4_all__reference'].set_index(['season','week','id'])
    variants=sorted(x for x in d.model.unique() if '__' not in x)
    diffs=pd.DataFrame(index=ref.index)
    for model in variants:
        c=d[d.model==model].set_index(['season','week','id']).reindex(ref.index)
        if c.team_loss.isna().any(): raise ValueError('unpaired comparison')
        diffs[model]=ref.team_loss-c.team_loss
    rng=np.random.default_rng(SEED); b=diffs.groupby(['season','week']).sum(); counts=diffs.groupby(['season','week']).size()
    draws=rng.integers(len(b),size=(20000,len(b)))
    boot=b.to_numpy()[draws].sum(axis=1)/counts.to_numpy()[draws].sum(axis=1)[:,None]
    centers=diffs.mean().to_numpy(); simultaneous=float(np.quantile(np.max(abs(boot-centers),axis=1),.95))
    sb=diffs.groupby('season').sum(); sn=diffs.groupby('season').size(); draws=rng.integers(len(sb),size=(20000,len(sb)))
    seasonboot=sb.to_numpy()[draws].sum(axis=1)/sn.to_numpy()[draws].sum(axis=1)[:,None]
    out={}
    for j,model in enumerate(variants):
        folds=diffs[model].groupby('season').mean()
        out[model]=dict(n=len(diffs),improvement_points_per_team=float(centers[j]),
            week_block95=np.quantile(boot[:,j],[.025,.975]).tolist(),
            simultaneous_six95=[float(centers[j]-simultaneous),float(centers[j]+simultaneous)],
            season_block95=np.quantile(seasonboot[:,j],[.025,.975]).tolist(),
            per_season_improvement={str(k):float(v) for k,v in folds.items()},
            leave_one_season_out={str(s):float(diffs.loc[diffs.index.get_level_values('season')!=s,model].mean()) for s in folds.index},
            nonnegative_folds=int((folds>=0).sum()),
            meets_descriptive_primary_screen=bool(centers[j]>=.5 and (folds>=0).sum()>=4),
            independently_supported_improvement=False)
    # Window contrasts use identical actual evaluation games and fixed anchors.
    contrasts={}
    for pop in POPS:
        for short in (1,3):
            delta=diffs[f'C1_{pop}_all']-diffs[f'C1_{pop}_{short}']
            j=variants.index(f'C1_{pop}_all'); k=variants.index(f'C1_{pop}_{short}')
            contrasts[f'{pop}_all_minus_{short}_improvement']=dict(mean=float(delta.mean()),
                week_block95=np.quantile(boot[:,j]-boot[:,k],[.025,.975]).tolist(),
                season_block95=np.quantile(seasonboot[:,j]-seasonboot[:,k],[.025,.975]).tolist())
    return dict(reference='C1_p4_all__reference (new research reference; NOT incumbent final scores)',
        bootstrap_unit='season-week',draws=20000,seed=SEED,comparisons=out,window_contrasts=contrasts)


def postmortems(rows,a):
    names=a[['season','id','home','away']].rename(columns={'home':'home_team','away':'away_team'})
    d=rows[~rows.model.str.contains('__')].merge(names,on=['season','id'],validate='many_to_one').copy()
    d['actual_margin']=d.actual_home-d.actual_away; d['actual_total']=d.actual_home+d.actual_away
    d['predicted_margin']=d.home-d.away; d['predicted_total']=d.home+d.away
    # Missing diagnostics are explicit; final score error alone cannot identify
    # a personnel, pace, efficiency or field-position cause.
    for field in ['pace_discrepancy','scoring_efficiency_discrepancy','field_position_discrepancy','personnel_discrepancy','roster_change_status']:
        d[field]='unavailable_no_reconciled_pregame_and_actual_pair'
    d['news_coverage']='unavailable_no_approved_corpus'
    d['diagnostic_category']='unexplained_outcome_variation_not_causal_attribution'
    d.loc[d.fallback,'diagnostic_category']='data_coverage_fallback_plus_unexplained_variation'
    d['small_sample']=(d.home_current_games<4)|(d.away_current_games<4)
    return d


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True,type=Path); args=parser.parse_args()
    out=args.output
    if out.exists(): raise SystemExit('Refusing to overwrite existing experiment directory')
    out.mkdir(parents=True)
    sources=[ROOT/'data'/f for f in ['backtest_early.csv','backtest_cross_division.csv','backtest_c_corrected_wf_kalman_20260830.csv','rating_params.json','evidence_manifest.json','forecast_sd_basis.json']]
    sources += [ROOT/'data/cfbd'/f'{kind}_{s}.parquet' for kind in ['games','teams','talent','returning'] for s in range(2019,2026)]
    sources += [ROOT/p for p in ['nflvalue/final_score_research.py','scripts/final_score_experiment.py','nflvalue/ratings.py','nflvalue/engine.py','nflvalue/ingest.py','nflvalue/forecast_contract.py','docs/final_score_program/c1_preregistration.json','docs/final_score_program/reproduce_audit.py','docs/final_score_program/audit_evidence.json']]
    manifest=dict(schema='stallionsstallion.final_score_development.v1',run_id=out.name,
        created_at_utc=datetime.now(timezone.utc).isoformat(),source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},protocol='final-score-c1-20260909-v1',
        forecast_availability='development_reconstruction_week_granular',no_market_inputs=True,activation=False)
    write_json(out/'manifest.json',manifest)
    # Reproduce historical records without altering their canonical artifacts.
    write_json(out/'supported_baselines.json',supported_baselines())
    g=R.game_frame(range(2019,2026)); a,refused,parity=anchors(g)
    a.to_csv(out/'anchors.csv',index=False); write_json(out/'incumbent_parity.json',dict(**parity,refusals=refused))
    print('incumbent replay parity: all724 exact',flush=True)
    grid=forecast_grid(g,a)
    rows,fits,inner=select_and_calibrate(grid,a)
    rows.to_csv(out/'forecasts.csv',index=False,float_format='%.10g')
    write_json(out/'fits.json',fits); pd.DataFrame(inner).to_csv(out/'all_inner_searches.csv',index=False)
    metrics={}
    for model,d in rows.groupby('model'):
        metrics[model]={'all_fbs':summary(d),'p4':summary(d[d.p4]),'seasons':{str(s):summary(x[x.p4]) for s,x in d.groupby('season')}}
    write_json(out/'metrics.json',metrics)
    paired=paired_comparisons(rows); write_json(out/'paired_comparisons.json',paired)
    pm=postmortems(rows,a); pm.to_csv(out/'postmortems.csv',index=False,float_format='%.10g')
    # Coverage proxy fixed before selection: incumbent preseason |margin| or
    # conference must never masquerade as an actual media prominence measure.
    coverage=dict(eligible_early_games=len(a[a.season>=2021])+len([r for r in refused if r['season']>=2021]),
        priced_early_games=int((a.season>=2021).sum()),p4_priced=int(((a.season>=2021)&a.p4).sum()),
        all_small_sample=bool(pm.small_sample.all()),verified_roster_strata=None,verified_news_strata=None,
        prominent_less_covered_comparison=None,reason='No preregistered independently sourced prominence labels or approved news corpus; conference slices are not media coverage.',
        conference_pair_counts=a[a.season>=2021].groupby(['home_conf','away_conf']).size().rename('n').reset_index().to_dict('records'))
    write_json(out/'coverage.json',coverage)
    write_json(out/'decision.json',dict(verdict='retain_incumbent',production_replacement=False,shadow_activation=False,
        primary_screen_candidates=[k for k,v in paired['comparisons'].items() if v['meets_descriptive_primary_screen']],
        independently_supported_improvement=False,
        reasons=['Inspected history remains development; no independent prospective window',
                 'Incumbent early scores/total unavailable; comparisons concern new research score capability',
                 'No supported discrete joint score law; continuous approximation has negative-score support',
                 'Later replay, C2/C3 feature provenance, verified personnel/news, event-feature ablations remain blocked']))
    write_json(out/'artifact_hashes.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(paired['comparisons'],indent=2),flush=True)


if __name__=='__main__': main()
