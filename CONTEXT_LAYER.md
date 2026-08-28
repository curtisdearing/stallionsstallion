# Pregame context layer — provenance-first, display-only

**What it is.** One declared surface of pregame context for a game, where
every value carries where it came from, when it was observed, what game it
is effective for, and how much of the league it covers — and where a value
that is missing, stale, conflicting, low-coverage, unapproved or observed
after kickoff says so instead of quietly becoming a number.

**What it is not.** It is not a factor. Nothing here may alter a projected
margin, total or win probability. Promotion is governed by
`data/factor_registry.json` and its existing gate (n-gate + matched control
+ BH q<0.05), which this layer reads and never writes.

---

## Files

| Path | Role |
|---|---|
| `data/context_sources.json` | Every source, its provenance contract, freshness and coverage policy, and its **approval gate**. |
| `nflvalue/context/contracts.py` | `FieldSpec` table (60 fields, 10 families), `ContextValue`, availability/usage vocabularies, the kickoff guard, the forbidden-column set. |
| `nflvalue/context/sources.py` | Loads the manifest; the approval gate, freshness and coverage policy live here. |
| `nflvalue/context/registry.py` | Reads `factor_registry.json`. The only thing that decides a factor's status. |
| `nflvalue/context/providers.py` | One provider per source. Reads finished frames; can only describe. |
| `nflvalue/context/assemble.py` | Builds a `GameContext` after the projection is final. |
| `nflvalue/context/panel.py` | The explainer panel: used / display-only / excluded / missing. |
| `nflvalue/context/fixtures.py` | Synthetic frames so the tests never need the real cache. |
| `tests/test_context_contracts.py` | Provenance, registry-as-SSOT, exclusions, degraded states. |
| `tests/test_context_leakage.py` | Import isolation both ways, kickoff guard, closing-line denial, poison tests. |
| `tests/test_context_panel.py` | The four buckets, escaping, gates rendered. |
| `tests/test_explain_context_hook.py` | The `explain.py` hook is purely additive. |

## The two structural guarantees

1. **It cannot reach a projection.** `nflvalue/context/*` may not import
   `ratings`, `cfb_sim`, `engine`, `kalman_engine`, `montecarlo`,
   `gameline`, `bakeoff`, `learn`, `model`, `factors`, `oddsmath`,
   `killcheck` or `clv`; and none of those may import the context package.
   Both directions are asserted by AST inspection in
   `test_context_leakage.py`. This is the `game_notes.py` / F8 pattern
   applied as an import-graph rule rather than a convention.
2. **A value exists or it does not.** `ContextValue.available` refuses to
   build without the whole envelope (`source`, `source_url_or_id`,
   `observed_at`, `effective_for_game`, `confidence`, `coverage`);
   `ContextValue.unavailable` is the only other constructor and cannot
   carry a number. There is no defaulting path, no mean fill, no zero fill.
   Disagreeing sources return `CONFLICTING`, never an average.

## Display buckets

`used_by_model` requires **both** a `gated_active` registry entry **and** a
field that declares `model_consumed`. A field cannot promote itself, and an
unregistered field is display-only by default — never active.

`excluded` comes straight from the registry, so
`raw_time_of_possession`, `penalty_counts`, `raw_turnover_margin`,
`raw_red_zone_td_pct` and `raw_pressure_rate` stay excluded and are
rendered with the reason they were excluded. `registry.py` refuses to load
if any of the five is absent from the registry.

`missing` covers `unavailable_no_source`, `unavailable_not_approved`,
`unavailable_no_data`, `stale`, `conflicting`, `low_coverage` and
`refused_post_kickoff`. Each renders its own state and note.

## Approval gates — what is open

A source whose approval status is not `approved` cannot emit a value,
whatever code exists for it. Opening a gate means editing
`data/context_sources.json` with a named approver, a date and a decision
record. Five are open:

| Source | Why it is closed | What an approval needs |
|---|---|---|
| `weather_forecast` | No forecast provider, and adding one is a new external dependency. CFBD's weather is *observed*, which is post-kickoff for a pregame packet. | named provider, licence check, refresh cadence, a `forecast_issued_at` field distinct from `observed_at`, venue coordinates (needs `cfbd_venues` first), pre-registered evaluation |
| `qb_availability_feed` | No approved live feed. `qb_starts_*` is retrospective; its pregame variant `backup_qb_pregame` was **rejected** (p=0.075, n=745). | named provider, licence check, update cadence, confirmed/probable/doubtful/out vocabulary, pre-registered evaluation |
| `unit_availability_feed` | No approved source for OL or key-unit availability. | same five |
| `cfbd_venues` | Endpoint exists on the existing client (1 bulk call) but nothing is cached. Gates venue name, altitude, true travel distance, time-zone delta and body-clock hour. | cache it, then approve |
| `cfbd_portal` | Endpoint exists; nothing cached, and no portal effect has ever been measured here. | cache it, then approve — with no effect estimate attached |

`cfbd_coaches` is **approved** but has no 2026 rows cached, so coaching
continuity renders `low_coverage` until a refresh writes
`coaches_2026.parquet`.

## The market field

Opening prices only, and only as a diagnostic reference. The blend that
would let a market price legitimately touch a model number failed
out-of-sample twice and is disabled, and this project carries a retraction
for a CLV finding that was closing-line look-ahead. `lines_*.parquet`
carries the live price under the plain names `spread` and `over_under` —
those become the closing line as kickoff approaches, so they are in
`FORBIDDEN_COLUMNS` and `MarketOpenDiagnostic.opening_only` narrows the
frame before `read` can touch it. A test poisons them and asserts the
context is unchanged.

## Using it

```python
from nflvalue.context import assemble_game_context

ctx = assemble_game_context(
    game=game_row, frames=frames, data_dir=DATA, now=utcnow(),
    projection=finished_projection,          # copied read-only, never mutated
    observed_at=manifest["data_as_of"],      # provenance from the run manifest
)
```

Then, optionally, in the week explainer:

```python
exp  = explain.build_explanation(..., context=ctx)
html = explain.render_week_html(exps, ..., contexts={game_id: ctx})
```

Omit both and `explain.py` renders exactly what it rendered before the hook
existed — verified byte-identical against the pre-hook module, and kept
that way by `test_explain_context_hook.py`, which strips the panel between
its `<!--ctx:start-->` / `<!--ctx:end-->` sentinels and compares.

`frames` keys: `games`, `teams` (two seasons, for realignment and
new-to-FBS), `talent`, `returning`, `lines`, `ratings`, `drives`,
`coaches`, `venues`, `portal`. A frame that is absent renders its family
unavailable; it is never a crash and never a silent omission.

## Status

Built 2026-08-28. 59 tests green, ruff E,F clean. Nothing is promoted, no
provider was added, no dependency was added, and no CFBD call was spent.
PAPER-ONLY: gate 7 has not run.
