"""NFL value-betting engine.

A self-contained, dependency-free toolkit that:
  * pulls odds from multiple sportsbooks (or realistic demo data),
  * removes the bookmaker margin ("de-vigs") to estimate fair probabilities,
  * adjusts those probabilities with situational factors
    (injuries, weather, revenge spots, matchups),
  * finds positive expected-value (+EV) bets and sizes them with Kelly,
  * and LEARNS: it grades finished games and nudges its factor weights
    toward whatever actually predicts winners.

Only the Python standard library is used, so nothing needs installing.
"""

__version__ = "1.0.0"
