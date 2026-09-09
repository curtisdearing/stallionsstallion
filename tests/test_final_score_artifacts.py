"""Saved research evidence must verify without private source caches."""
from pathlib import Path
import hashlib
import json
import shutil
import pandas as pd
import pytest
from scripts.verify_final_score_run import verify

RUN=Path(__file__).resolve().parents[1]/'data/final_score_program/20260909T024200Z-c1-v1'


def test_committed_run_verifies_without_private_caches():
    result=verify(RUN)
    assert result['candidate_rows']==4344 and result['candidate_primary_games']==495


def test_equal_counts_cannot_hide_different_reference_games(tmp_path):
    # Give the utility an internally rehashed but wrongly paired reference.
    # Count checks and score-metric checks alone cannot catch this corruption.
    copy=tmp_path/'run';shutil.copytree(RUN,copy)
    source=copy/'forecasts.csv'
    d=pd.read_csv(source if source.exists() else copy/'forecasts.csv.gz')
    ix=d.index[d.model.str.endswith('__simple')][0]
    d.loc[ix,'id']=9999999999
    d.to_csv(source,index=False)
    hashes=json.loads((copy/'artifact_hashes.json').read_text())
    hashes['forecasts.csv']=hashlib.sha256(source.read_bytes()).hexdigest()
    (copy/'artifact_hashes.json').write_text(json.dumps(hashes))
    with pytest.raises(AssertionError,match='cohort'):
        verify(copy)
