"""Read-only architecture audit probes. No fitting, forecast writes, or API calls.

Run from the repository with its existing .venv; prints JSON to stdout.
Historical summaries are descriptive development evidence, never promotion.
The synthetic support probe establishes a possibility, not a real-game rate.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nflvalue import cfb_sim, forecast_contract as FC  # noqa: E402
from nflvalue.cfb_ot import OTParams  # noqa: E402

P4 = {"ACC", "Big Ten", "Big 12", "SEC"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def error(pred, actual):
    e = np.asarray(pred, float) - np.asarray(actual, float)
    return dict(n=len(e), mae=float(abs(e).mean()),
                rmse=float(np.sqrt((e * e).mean())), bias=float(e.mean()))


def distribution(mean, sd, actual):
    z = (np.asarray(actual) - np.asarray(mean)) / np.asarray(sd)
    pdf = np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
    return dict(approximation="Gaussian from summaries; not empirical joint scores",
                crps=float(np.mean(sd * (z * (2 * ndtr(z) - 1)
                                        + 2 * pdf - 1 / np.sqrt(np.pi)))),
                negative_log_density=float(np.mean(.5 * z * z + np.log(sd)
                                                   + .5 * np.log(2 * np.pi))),
                coverage80=float(np.mean(abs(z) <= 1.2815515655446004)),
                coverage95=float(np.mean(abs(z) <= 1.959963984540054)),
                standardized_residual_sd=float(np.std(z, ddof=1)))


def scores(d, mean_col, sd, total=False):
    m, y = d[mean_col].to_numpy(float), d.margin.to_numpy(float)
    sd = np.broadcast_to(np.asarray(sd, float), len(d))
    p = d.p_home_win.to_numpy(float) if 'p_home_win' in d else ndtr(m / sd)
    win = (y > 0).astype(float)
    safe = np.clip(p, 1e-9, 1 - 1e-9)
    bins = []
    for i in range(10):
        ix = (p >= i / 10) & ((p < (i + 1) / 10) if i < 9 else (p <= 1))
        if not ix.any():
            continue
        n, obs = int(ix.sum()), float(win[ix].mean())
        den = 1 + 1.96**2 / n
        center = (obs + 1.96**2 / (2*n)) / den
        width = 1.96 * np.sqrt(obs*(1-obs)/n + 1.96**2/(4*n*n)) / den
        bins.append(dict(lo=i/10, hi=(i+1)/10, n=n, predicted=float(p[ix].mean()),
                         observed=obs, wilson95=[float(center-width), float(center+width)]))
    out = dict(n=len(d), seasons=sorted(map(int, d.season.unique())),
               weeks=sorted(map(int, d.week.unique())), margin=error(m, y),
               win=dict(brier=float(((p-win)**2).mean()),
                        log_loss=float(-(win*np.log(safe)+(1-win)*np.log1p(-safe)).mean()),
                        observed_home_win=float(win.mean()), mean_p_home=float(p.mean()),
                        reliability=bins), margin_distribution=distribution(m, sd, y),
               home_score=None, away_score=None, team_score_mae=None, total=None,
               joint_score_distribution=None)
    if total:
        t, yt = d.mc_total.to_numpy(float), d.total.to_numpy(float)
        out.update(home_score=error((t+m)/2, (yt+y)/2),
                   away_score=error((t-m)/2, (yt-y)/2), total=error(t, yt),
                   score_mean_status="algebraic means from existing supported MC M/T; unselected corrected record, not published incumbent scores",
                   total_distribution=distribution(t, d.mc_total_sd.to_numpy(float), yt))
        out['team_score_mae'] = (out['home_score']['mae'] + out['away_score']['mae']) / 2
    else:
        out['unavailable_reason'] = 'route has no supported total or team-score forecast'
    return out


def main():
    evidence = json.loads((ROOT/'data/evidence_manifest.json').read_text())
    files = ['nflvalue/'+n+'.py' for n in ['engine', 'ratings', 'kalman_engine', 'cfb_sim',
             'explain', 'forecast_contract', 'early_challenger', 'performance_mc',
             'producers', 'snapshots', 'bakeoff', 'ingest', 'cfb_ot']]
    files += ['scripts/'+n+'.py' for n in ['weekly_loop', 'render_picks', 'release_check']]
    files += ['data/'+n+'.json' for n in ['context_sources', 'factor_registry',
              'rating_params', 'forecast_sd_basis', 'evidence_manifest']]
    out = dict(audit_version=1, historical_status='2019-2025 development; 2021-2025 already inspected',
               bias_sign='forecast minus actual; margin home minus away',
               p4_scope='both FBS in historical record; at least one season-specific ACC/Big Ten/Big 12/SEC. Historical Pac-12 excluded; no retroactive membership.',
               audit_source_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
               source_hashes={p:sha(ROOT/p) for p in files}, baselines={})
    bases = FC.BasisRegistry.load(str(ROOT/'data'))
    routes = [('early', 'pred_margin', bases.margin_sd('ridge_preseason_prior', 1), False),
              ('cross_division', 'pred_margin', bases.margin_sd('ridge_cross_division', 1), False),
              ('c_corrected_wf_kalman_20260830', 'mc_margin', None, True)]
    for name, col, sd, has_total in routes:
        path = ROOT/f'data/backtest_{name}.csv'
        d = pd.read_csv(path)
        assert not d.duplicated(['season', 'id']).any()
        rec = next((r for r in evidence['artifacts'] if r['path'] == str(path.relative_to(ROOT))), None)
        games = pd.concat([pd.read_parquet(ROOT/f'data/cfbd/games_{s}.parquet')
                           for s in sorted(d.season.unique())])
        meta = d.merge(games[['season','id','home_team','away_team','home_conference','away_conference']],
                       on=['season','id'], how='left', validate='one_to_one')
        assert meta.home_team.eq(meta.home).all() and meta.away_team.eq(meta.away).all()
        p4 = meta.home_conference.isin(P4) | meta.away_conference.isin(P4)
        result = dict(input=str(path.relative_to(ROOT)), sha256=sha(path),
                      manifest_status=rec['status'] if rec else 'not in evidence manifest',
                      manifest_hash_matches=(sha(path)==rec['content_sha256']) if rec else None,
                      full=scores(d, col, d.mc_margin_sd.to_numpy(float) if has_total else sd, has_total))
        if name != 'cross_division':
            sub = d.loc[p4.to_numpy()]
            result['p4_involved'] = scores(sub, col, sub.mc_margin_sd.to_numpy(float) if has_total else sd, has_total)
        if not has_total:
            result['sd_caveat'] = 'published pooled historical residual SD evaluated on its own development rows; not nested OOF calibration'
        out['baselines'][name] = result
    out['selected_later_benchmark'] = [r['artifact_id'] for r in evidence['artifacts']
        if r['role']=='benchmark_wk4plus' and r['status']=='selected']
    out['later_basis_accepts_archival_source'] = bases.require('kalman_montecarlo', 4)['source']
    pub = pd.read_csv(ROOT/'data/predictions_2026_wk01.csv')
    out['current_artifact'] = dict(sha256=sha(ROOT/'data/predictions_2026_wk01.csv'),
                                  n=len(pub), status=pub.status.value_counts().to_dict(),
                                  n_with_total=int(pub.projected_total.notna().sum()),
                                  families=pub.model_family.value_counts().to_dict())
    probs = np.zeros((2, len(cfb_sim.CLASSES)))
    probs[0, cfb_sim.CLASSES.index((0, 0))] = 1
    probs[1, cfb_sim.CLASSES.index((7, 0))] = 1
    rates = cfb_sim.DriveRateModel(np.array([-1., 1.]), probs)
    pace = cfb_sim.PaceModel(np.array([[6, 6]]))
    ot = OTParams(.4, .2, {7:1.}, .5)  # unused: regulation is 42-0 on every path
    base = cfb_sim.simulate_game(np.random.default_rng(20260908), 1, -1, rates, pace, ot)
    shifted = cfb_sim.simulate_game(np.random.default_rng(20260908), 1, -1, rates, pace,
                                  ot, param_sd=3.6, param_rng=np.random.default_rng(20260909))
    h, a = (shifted['total']+shifted['margin'])/2, (shifted['total']-shifted['margin'])/2
    out['support_reproducer'] = dict(kind='synthetic valid 42-0 scoring fixture; not estimated production incidence',
        n=4000, param_sd=3.6, total_identical=bool(np.array_equal(base['total'], shifted['total'])),
        margin_mean_delta=shifted['margin_mean']-base['margin_mean'],
        noninteger_score_paths=int((~np.isclose(h, np.round(h), atol=1e-10, rtol=0)).sum()),
        negative_score_paths=int(((h < 0) | (a < 0)).sum()),
        min_implied_score=float(min(h.min(), a.min())))
    out['cache_inventory'] = {}
    for kind in ['games','teams','talent','returning','plays_clean','drives','coaches','qb_starts','portal','venues']:
        out['cache_inventory'][kind] = {}
        for s in range(2019, 2027):
            p = ROOT/f'data/cfbd/{kind}_{s}.parquet'
            if p.exists():
                out['cache_inventory'][kind][str(s)] = dict(rows=pq.read_metadata(p).num_rows, sha256=sha(p))
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
