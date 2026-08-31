# Gate-3 corrected strict walk-forward — selection audit, 2026-08-30

**Status: NOT SELECTED — either artifact.** `data/evidence_manifest.json` is
unchanged by this audit: both candidates keep `status: "candidate"`, no
`selection_acknowledgement` is written, and `benchmark_wk4plus` stays
unselected. This report only determines whether the pre-registered rule is
satisfied; it promotes nothing, computes no new metric, and retunes no code.

## The rule applied

Recorded on each candidate record itself in `data/evidence_manifest.json`
(`generated_at_utc: 2026-08-30T19:38:07Z`), identically for both
`backtest_c_corrected_wf_kalman_20260830` and
`backtest_c_corrected_wf_ridge_20260830`, before any human read the metrics:

> "selectable only when the run covers the pre-registered window (eval
> seasons 2021-2023, weeks >= 4) with the exclusions accounted for and a
> human records the acceptance. Coverage shortfall is not a reason to widen
> the window."

This is the only recorded selection criterion for either artifact. No Brier,
MAE, or other metric threshold governs selection — `nflvalue/evidence.py`
enforces this structurally: `Manifest.select` never picks a "better" record,
only a human-written `status: "selected"`, and it refuses to honor that
status at all if `code_hashes` is empty (the archival records can never
qualify). `data/factor_prereg_2026-08-28.json` is a **separate** registration
governing five roster/rest/special-teams factor hypotheses under BH-FDR
correction — it does not govern these two backtest artifacts and was not
applied here.

## Reproducible calculation

Pre-registered window: 3 eval seasons (2021, 2022, 2023) x weeks 4-15
(the upper bound every exclusions list and every prior gate-3 record in this
manifest uses; no week 16+ appears anywhere) = **36 season-weeks**.

| | kalman | ridge |
|---|---|---|
| `content_sha256` (manifest, verified against disk) | `39c303b2…` | `004c0f6c…` |
| `n_rows` | 429 | 429 |
| seasons evaluated | {2023} | {2023} |
| weeks evaluated | 7–15 (9 weeks) | 7–15 (9 weeks) |
| season-weeks covered / required | 9 / 36 = **25.0%** | 9 / 36 = **25.0%** |
| weeks excluded, `week_pace_or_ot_unfittable` | 27 | 27 |
| games excluded, `game_not_fbs_v_fbs` | 8 | 8 |
| 9 covered + 27 excluded | = 36 ✓ | = 36 ✓ |

Comparable full-window pre-correction record for the same `eval_seasons
[2021, 2022, 2023]`: `backtest_c_ridge_20260730`, n=1765. 429 / 1765 = **24.3%**
of that volume. `git`/hash provenance verified by running
`nflvalue.evidence.Manifest.verify_all()` against the live tree: all six
records (three archival, three current) return `"ok"` — content, code, and
input hashes all match what the manifest claims, so this is not a stale or
edited artifact.

**Verdict: the rule requires the full pre-registered window; observed
coverage is 25% of season-weeks and 24.3% of the comparable game volume,
missing 2021 and 2022 in their entirety. The rule is not satisfied, for
either artifact, identically.**

## Root cause — not a caching gap

`nflvalue/cfb_ot.py`: `RULES_ERA_FIRST_SEASON = 2021` (NCAA changed OT rules
starting 2021; 2019-2020 OT possessions are a different game and are never
counted) and `min_possessions=100` to fit OT rates at all. Under the strict
walk-forward constraint — fit only on weeks strictly before the one being
evaluated — the 100-possession floor is not reached until partway through
2023 (the exclusions list quotes the running count for every excluded week:
2021 wk4 = 7 possessions … 2022 wk8 = 97, still short; 2022 wk9-15 then fail
a *different* floor, 3 mandatory 2-pt attempts). **This cannot be fixed by
caching more history**: possessions from before 2021 are not "2021+-era" OT
data by definition, so no amount of additional 2019/2020 cache moves 2021's
count. Two of the three pre-registered seasons may be permanently
unfittable under this exact method.

## Evidence boundaries

- `data/calibration_report.txt` already states "current correction awaiting
  revalidation" and reports no numeric benchmark claim — checked against the
  live manifest and found accurate; nothing here required a fix.
- The three archival records (`backtest_c_ridge_20260730`,
  `backtest_c_kalman_20260730`, `backtest_c_2024_holdout`) describe
  `walk_forward_pace_ot_pooled_v1_20260730` — PaceModel/OT fit on the pooled
  window, the in-window leak the strict correction exists to remove. Cited
  above only as historical volume context; not used, and not citable, as
  validation of the corrected method or of the model now running.
- `nflvalue.evidence.Manifest.select("benchmark_wk4plus")` still raises
  `BenchmarkUnavailable` (re-confirmed by direct execution against the live
  tree during this audit). No selected benchmark exists for weeks 4+, so no
  claim that the current model is "accurate" over weeks 4+ is supported by
  anything in this evidence set.

## The one decision needed from Curtis

The pre-registered window (2021-2023, weeks >= 4) appears **structurally
unsatisfiable** under the strict walk-forward method with a 100-possession
OT floor — 27 of 36 season-weeks fail permanently, not from a missing cache.

**Should the pre-registered eval window for `benchmark_wk4plus` be replaced
by a new, freshly-dated pre-registration scoped to the window that is
actually fittable under this method (on this evidence: 2023 wk7 onward,
extended forward as 2024/2025 seasons complete), rather than continuing to
measure the current candidates against a 2021-2023 window they cannot
structurally cover — or should the OT fit floor/method itself be revisited
first?** Until you decide, both candidates stay `candidate`, no benchmark is
selected for weeks 4+, and every consumer correctly keeps showing "awaiting
revalidation."
