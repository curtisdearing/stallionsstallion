#!/usr/bin/env python3
"""Verify saved C1 evidence without private caches; --inputs also verifies inputs.

CSV gzip transport is lossless. Canonical artifact hashes cover decompressed
bytes, so compression never changes a frozen forecast.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import io
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def read_bytes(directory,name):
    p=directory/name
    return p.read_bytes() if p.exists() else gzip.decompress((directory/(name+'.gz')).read_bytes())


def verify(directory,inputs=False):
    directory=Path(directory)
    hashes=json.loads((directory/'artifact_hashes.json').read_text())
    for name,digest in hashes.items():
        assert hashlib.sha256(read_bytes(directory,name)).hexdigest()==digest,name
    manifest=json.loads((directory/'manifest.json').read_text())
    if inputs:
        for name,digest in manifest['input_hashes'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    d=pd.read_csv(io.BytesIO(read_bytes(directory,'forecasts.csv')))
    anchors=pd.read_csv(directory/'anchors.csv').set_index(['season','id'])
    metrics=json.loads((directory/'metrics.json').read_text())
    assert not d.duplicated(['model','season','id','cutoff_id']).any()
    assert (d.home>=0).all() and (d.away>=0).all()
    assert (d.calibration_max_season<d.season).all()
    c=d[~d.model.str.contains('__')]
    assert c.model.nunique()==6 and len(c)==6*724
    eligible=anchors.reset_index().query('season >= 2021')
    keys=['season','week','id']
    expected=set(map(tuple,eligible[keys].to_numpy()))
    expected_p4=set(map(tuple,eligible.loc[eligible.p4,keys].to_numpy()))
    for model,sub in d.groupby('model'):
        assert set(map(tuple,sub[keys].to_numpy()))==expected, f'{model}: cohort differs'
        assert set(map(tuple,sub.loc[sub.p4,keys].to_numpy()))==expected_p4, f'{model}: P4 cohort differs'
        for scope,z in [('all_fbs',sub),('p4',sub[sub.p4])]:
            mae=float((abs(z.home-z.actual_home)+abs(z.away-z.actual_away)).mean()/2)
            assert abs(mae-metrics[model][scope]['team_mae'])<1e-7,(model,scope)
        assert np.allclose(sub.home_sd**2+sub.away_sd**2-2*sub.score_covariance,sub.margin_sd**2,atol=1e-6,rtol=0)
        assert np.allclose(sub.home_sd**2+sub.away_sd**2+2*sub.score_covariance,sub.total_sd**2,atol=1e-6,rtol=0)
    for model,sub in c.groupby('model'):
        indexed=sub.set_index(['season','id'])
        assert np.allclose(indexed.home-indexed.away,anchors.loc[indexed.index,'anchor'],atol=1e-7,rtol=0)
        assert len(sub[sub.p4])==495
    fits=json.loads((directory/'fits.json').read_text())
    assert len(fits)==30 and all(not f['selection']['unfit'] and f['max_selection_season']<f['outer_season'] for f in fits)
    decision=json.loads((directory/'decision.json').read_text())
    assert decision['verdict']=='retain_incumbent' and not decision['shadow_activation']
    return dict(artifact_hashes_verified=len(hashes),input_hashes_verified=len(manifest['input_hashes']) if inputs else None,
        forecasts=len(d),candidate_rows=len(c),candidate_primary_games=495,variants=6,folds=30,
        paired_scores_covariance_metrics_and_chronology='pass',private_caches_required=inputs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--inputs',action='store_true');a=p.parse_args()
    print(json.dumps(verify(a.directory,a.inputs),indent=2))
