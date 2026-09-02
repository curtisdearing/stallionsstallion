# The board

Created 2026-09-02. The board is the last mile the project never had: it takes
the published forecast artifact, joins it to the market, and shows the
disagreement, ranked, with the pregame context around each game.

Everything here is **unvalidated**. Nothing in this repo has demonstrated an
edge against a closing number on live 2026 data — the CLV slice for these weeks
is n=0. The board exists to be graded, and the ledger is what grades it.

## Running it

```bash
python3 scripts/weekly_loop.py --job pregame --week N   # produce the forecast
python3 scripts/build_board.py  --season 2026 --week N  # join the market
python3 scripts/board_ledger.py --record --season 2026 --week N   # freeze the leans
# ... games are played ...
python3 scripts/weekly_loop.py --job grade --week $((N+1))  # pull finals, recap week N
python3 scripts/board_ledger.py --grade --season 2026       # score the record
```

`--week` on the grade job is the week the loop is **in**, and the recap and
CLV stages target the week before it — so grading week N means
`--job grade --week N+1`. Getting this wrong is silent, and silent in two
different ways: `--week 1` targets week 0 and grades nothing, while
`--week N` for any later N re-grades week N-1, reports a plausible non-zero
n, and leaves week N ungraded. Check `target_week` in the manifest to see
which week a run actually graded.

`board_ledger.py --grade` does not share that offset. It takes a season, not
a week, and grades every ledger row whose game has a final score — the count
it prints as "pending" is the rows it skipped. It re-scores every already
graded row each time and rewrites `data/board_ledger_graded.csv` whole, so
running it twice is safe. The refresh stage of any grade run pulls finals
regardless of which week it recaps.

`--record` must run **before kickoff**. It skips games that have already
started, and once a game is in the ledger its line and its lean are frozen: a
re-run never updates them. That is not a convenience, it is the leak guard —
letting a re-run refresh the line would drift the ledger toward the close and
manufacture CLV that was never available.

## Sign conventions

CFBD quotes `spread` from the **home** side, so a `-7.0` means the home team is
favoured by 7.

| quantity             | definition                                  |
| -------------------- | ------------------------------------------- |
| `market_home_margin` | `-spread` — the market's expected home margin |
| `model_home_margin`  | the artifact's `projected_home_margin`      |
| `edge_pts`           | `model_home_margin - market_home_margin`    |
| positive `edge_pts`  | the lean is on the **home** side            |
| negative `edge_pts`  | the lean is on the **away** side            |

`tests/test_build_board.py` pins all three axes against hand-computed cases,
and inverting the market sign flips 31 of 43 leans — the convention is
load-bearing and tested.

## What is actually in the number

Two factors: league home-field advantage and the per-team deviation from it.
That is the whole of it. Weeks 1–3 are priced by the registered
`ridge_preseason_prior` path, whose only season-2026 inputs are recruiting
talent and returning production.

Every `ctx_*` column — surface, dome, capacity, elevation, travel miles,
timezone shift, visitor body-clock kickoff, rest differential — is
**display-only**. The factor registry is the authority and it lists these as
untested or display-only. They are shown because they are the things a human
wants to see before trusting a number, not because they move it. Do not
describe a lean as being "because of" one of them.

## The scale diagnostic — read this before trusting a big edge

Regressing the model's margin on the market's, on the first board:

```
model = 0.807 × market + 0.92     (r = 0.921, n = 43)
```

A slope of 0.807 means the model is **compressed by about 19%** against the
market. That is what a ridge fit does — it shrinks toward the mean — and its
consequence is mechanical: on any large spread the model will read low, so the
board leans **underdog**. On the first board it did so on 27 of 43 games, and
on every game with a market margin above 14.

This is the single most likely way the board is wrong. One bias reported
twenty times is not twenty edges.

The board therefore carries two columns, not one:

- `edge_pts` — the registered forecast path, unmodified.
- `edge_pts_scale_adj` — the same forecast rescaled onto the market's slope.

The rescaled number is **never substituted** for the registered one. Silently
swapping it in would be exactly the unregistered tunable this project refuses,
and it would also be untestable. Instead the ledger grades both, on the same
games from the same frozen lines, and the season answers the question from
outcomes: if the compression is a real edge, `result_raw` wins; if it is a
scale artefact, `result_adj` wins.

## Status

Paper-only. The gate-7 kill-check has not run, so no row here is a bet, a
stake, or a recommendation. A week is not evidence; neither is a month. The
ledger is the only thing that will ever change that, which is why it exists
before any of the picks do.
