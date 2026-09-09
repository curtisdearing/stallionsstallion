"""Personnel facts: dedup, unknown-vs-healthy, unsourced refusal, prose pricing."""
from __future__ import annotations

import datetime as dt
import os

import pytest

from nflvalue.context.sources import SourceManifest
from nflvalue.team_evidence import news as NEWS
from nflvalue.team_evidence.contracts import EvidenceAvailability, Missingness

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PUB = dt.datetime(2026, 9, 4, 15, 30, tzinfo=dt.timezone.utc)
KICK = dt.datetime(2026, 9, 6, 23, 0, tzinfo=dt.timezone.utc)

QUOTE = ("Head coach said Friday that starting quarterback J. Smith will not play "
         "against State, citing a lower-body injury sustained in practice.")


def fact(fact_id="f1", *, published=PUB, quote=QUOTE, source_id="qb_availability_feed",
         url=None, status="out", confidence="reported", team="A",
         uncertainty="single beat report; the school has not issued a designation",
         **over):
    kw = dict(
        fact_id=fact_id, subject_type="player", subject_id="cfbd:player:1234",
        subject_name="J. Smith", team=team, season=2026, week=2,
        claim="availability", status=status, status_confidence=confidence,
        published_at=published, retrieved_at=published + dt.timedelta(minutes=10),
        source_id=source_id,
        source_url=url if url is not None else f"https://example.test/{fact_id}",
        citation_quote=quote, uncertainty=uncertainty)
    kw.update(over)
    return NEWS.PersonnelFact(**kw)


# ------------------------------------------------------------ bounded schema
def test_a_fact_is_bounded_by_closed_vocabularies():
    with pytest.raises(NEWS.NewsContractError):
        fact(claim="vibes")
    with pytest.raises(NEWS.NewsContractError):
        fact(status="probably fine")
    with pytest.raises(NEWS.NewsContractError):
        fact(confidence="pretty sure")
    with pytest.raises(NEWS.NewsContractError):
        fact(subject_type="mascot")


def test_a_fact_needs_identity_a_citation_and_an_uncertainty_note():
    for field, blank in (("subject_id", ""), ("subject_name", ""), ("team", ""),
                         ("citation_quote", ""), ("uncertainty", "  ")):
        with pytest.raises(NEWS.NewsContractError):
            fact(**{field: blank})


def test_publication_and_retrieval_times_must_be_timezone_aware():
    with pytest.raises(NEWS.NewsContractError):
        fact(published=dt.datetime(2026, 9, 4, 15, 30))


def test_the_status_vocabulary_is_the_one_the_approval_record_demands():
    """`data/context_sources.json` specifies confirmed/probable/doubtful/out for
    qb_availability_feed, NOT the NFL module's OK/RISK/OUT."""
    man = SourceManifest.load(DATA)
    record = man.get("qb_availability_feed")["approval"]["decision_record"].lower()
    for token in ("confirmed", "probable", "doubtful", "out"):
        assert token in record
    assert set(NEWS.STATUSES) >= {"confirmed_available", "probable", "questionable",
                                  "doubtful", "out", "unknown"}


# --------------------------------------------------------- unsourced refusal
def test_model_recall_cannot_become_a_fact():
    for bad in ("llm_recall", "gpt_memory", "claude", "model_recall",
                "assistant_training_data"):
        with pytest.raises(NEWS.UnsourcedFactError):
            fact(source_id=bad)


def test_a_fact_with_no_url_is_refused():
    with pytest.raises(NEWS.UnsourcedFactError):
        fact(url="")


# ---------------------------------------------------------- prose may not price
def test_prose_may_not_price_a_fact():
    for bad in ("Losing him is worth about 3 points on the spread.",
                "Adjust the line -2.5 points for this.",
                "That is a 4 point swing.",
                "Move the total 3.5 points down."):
        with pytest.raises(NEWS.ProsePricingError):
            NEWS.assert_no_point_adjustment(bad)
    with pytest.raises(NEWS.ProsePricingError):
        fact(quote="He is out, which is worth 6 points to the line.")
    with pytest.raises(NEWS.ProsePricingError):
        fact(uncertainty="uncertain, maybe +2 points either way")


def test_ordinary_football_prose_is_not_mistaken_for_pricing():
    for ok in (QUOTE,
               "He scored 14 points in the opener.",
               "The team is 3-0.",
               "Listed as questionable on the Wednesday report."):
        NEWS.assert_no_point_adjustment(ok)


# ------------------------------------------------------------------ dedup
def test_duplicate_stories_do_not_multiply_effects():
    wire = [fact(f"copy{i}", published=PUB + dt.timedelta(minutes=i))
            for i in range(6)]
    out = NEWS.deduplicate(wire)
    assert len(out) == 1
    assert out[0].evidence_weight == 1.0
    assert len(out[0].duplicate_of) == 5


def test_the_canonical_copy_is_the_earliest_publication():
    late = fact("wire", published=PUB + dt.timedelta(hours=3))
    early = fact("original", published=PUB)
    out = NEWS.deduplicate([late, early])
    assert out[0].fact_id == "original"
    assert out[0].duplicate_of == ("wire",)


def test_syndication_survives_cosmetic_rewriting():
    """Wire copies differ in dateline and punctuation, not in the claim sentence."""
    a = fact("a", quote=QUOTE)
    b = fact("b", quote="  " + QUOTE.replace(",", "").upper() + "  ")
    assert NEWS.quote_fingerprint(a.citation_quote) == \
        NEWS.quote_fingerprint(b.citation_quote)
    assert len(NEWS.deduplicate([a, b])) == 1


def test_two_genuinely_different_reports_are_not_collapsed():
    a = fact("a", status="out")
    b = fact("b", status="questionable",
             quote="Coach said Smith is questionable and will be a game-time call.")
    assert len(NEWS.deduplicate([a, b])) == 2


def test_dedup_is_deterministic():
    facts = [fact(f"c{i}", published=PUB) for i in range(5)]
    assert [f.fact_id for f in NEWS.deduplicate(facts)] == \
        [f.fact_id for f in NEWS.deduplicate(list(reversed(facts)))]


def test_evidence_weight_cannot_be_inflated():
    with pytest.raises(NEWS.NewsContractError):
        NEWS.PersonnelFact(**{**fact().core_fields(), "evidence_weight": 6.0})


# ------------------------------------------------------- unknown, not healthy
def test_absence_of_a_report_is_unknown_not_healthy(monkeypatch, tmp_path):
    import json
    raw = json.load(open(os.path.join(DATA, "context_sources.json")))
    raw["sources"]["qb_availability_feed"]["approval"].update(
        {"status": "approved", "approved_by": "test", "approved_at": "2026-09-09"})
    (tmp_path / "context_sources.json").write_text(json.dumps(raw))
    man = SourceManifest.load(str(tmp_path))

    obs, facts = NEWS.collect("A", 2026, 2, data_dir=str(tmp_path), manifest=man,
                              facts=[])
    assert facts == []
    assert obs.value is None
    assert obs.availability is EvidenceAvailability.UNAVAILABLE_NO_DATA
    assert obs.reason == NEWS.NO_REPORT_REASON
    blob = f"{obs.availability.value} {obs.reason}".lower()
    for word in ("healthy", "available", "cleared", "full strength", "ok"):
        assert not blob.startswith(word)
    assert "unknown" in obs.reason


def test_an_unapproved_feed_refuses_before_any_fact_is_read():
    """Against the real manifest, where the feed is `not_requested` today."""
    obs, facts = NEWS.collect("A", 2026, 2, data_dir=DATA,
                              facts=[fact("should_be_ignored")])
    assert facts == []
    assert obs.value is None
    assert obs.availability in (EvidenceAvailability.UNAVAILABLE_NOT_APPROVED,
                                EvidenceAvailability.UNAVAILABLE_NO_SOURCE)
    assert obs.missingness is Missingness.SOURCE_NOT_APPROVED
    assert obs.availability is not EvidenceAvailability.OK


def test_collect_drops_facts_published_at_or_after_kickoff(tmp_path):
    import json
    raw = json.load(open(os.path.join(DATA, "context_sources.json")))
    raw["sources"]["qb_availability_feed"]["approval"].update(
        {"status": "approved", "approved_by": "test", "approved_at": "2026-09-09"})
    (tmp_path / "context_sources.json").write_text(json.dumps(raw))
    man = SourceManifest.load(str(tmp_path))
    late = fact("late", published=KICK + dt.timedelta(minutes=1))
    obs, facts = NEWS.collect("A", 2026, 2, data_dir=str(tmp_path), manifest=man,
                              facts=[late], kickoff=KICK)
    assert facts == []
    assert obs.reason == NEWS.NO_REPORT_REASON


# ------------------------------------------------------- attention is not quality
def test_attention_signals_are_refused():
    for bad in ("article_count", "story count", "media attention", "AP rank",
                "coaches poll", "mention_volume", "name recognition",
                "team familiarity"):
        with pytest.raises(NEWS.NewsContractError):
            NEWS.assert_not_an_attention_signal(bad)


def test_the_forbidden_list_names_the_specific_traps():
    for token in ("article_count", "ap_rank", "media_attention",
                  "team_familiarity"):
        assert token in NEWS.FORBIDDEN_DERIVED_SIGNALS
