"""Freshness gate (nflvalue/freshness.py): stale/missing data must halt or
downgrade -- never silently proceed (PHASE1_HANDSOFF_DESIGN.md H1/H3/H10)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.freshness import Feed, cap_confidence, gate  # noqa: E402

AS_OF = "2023-11-08T12:00:00Z"


def test_all_fresh_publishes_high():
    feeds = [Feed("injuries", "2023-11-08T06:00:00Z", n_records=32),
             Feed("fantasy", "2023-11-07T12:00:00Z", n_records=500)]
    g = gate(feeds, as_of=AS_OF)
    assert g["publish"] is True
    assert g["confidence_cap"] == "high"
    assert not g["reasons"]


def test_stale_load_bearing_feed_blocks_publish():
    feeds = [Feed("injuries", "2023-11-01T06:00:00Z", n_records=32)]  # 7 days old
    g = gate(feeds, as_of=AS_OF)
    assert g["publish"] is False
    assert "injuries" in g["stale_feeds"]
    assert any("staleness threshold" in r for r in g["reasons"])


def test_missing_timestamp_blocks_publish():
    g = gate([Feed("injuries", None, n_records=32)], as_of=AS_OF)
    assert g["publish"] is False
    assert "injuries" in g["missing_feeds"]


def test_schema_failure_blocks_publish():
    g = gate([Feed("injuries", "2023-11-08T06:00:00Z", n_records=32,
                   schema_ok=False, detail="missing 'injuries' key")], as_of=AS_OF)
    assert g["publish"] is False
    assert any("schema" in r for r in g["reasons"])


def test_empty_feed_blocks_publish():
    g = gate([Feed("injuries", "2023-11-08T06:00:00Z", n_records=0)], as_of=AS_OF)
    assert g["publish"] is False


def test_non_load_bearing_degrades_but_publishes():
    feeds = [Feed("injuries", "2023-11-08T06:00:00Z", n_records=32),
             Feed("news", "2023-10-01T00:00:00Z", n_records=5, load_bearing=False)]
    g = gate(feeds, as_of=AS_OF)
    assert g["publish"] is True
    assert g["confidence_cap"] == "low"


def test_future_dated_feed_flags_leakage_and_blocks():
    g = gate([Feed("injuries", "2023-11-09T06:00:00Z", n_records=32)], as_of=AS_OF)
    assert g["leakage_suspected"] is True
    assert g["publish"] is False
    assert "injuries" in g["future_dated_feeds"]


def test_custom_thresholds_respected():
    feeds = [Feed("fantasy", "2023-11-05T12:00:00Z", n_records=100)]  # 72h old
    assert gate(feeds, as_of=AS_OF)["publish"] is True                 # default 72h: exactly at edge
    assert gate(feeds, as_of=AS_OF, staleness_hours={"fantasy": 24.0})["publish"] is False


def test_cap_confidence():
    assert cap_confidence("high", "low") == "low"
    assert cap_confidence("medium", "high") == "medium"
    assert cap_confidence("low", "high") == "low"
