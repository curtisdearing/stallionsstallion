"""STUB — not part of this project's scope.

Copied from fablesfable/tests/test_market_config.py during the initial
game-line-only duplication (2026-07-17). All four of its tests exercise
the player-prop market vocabulary: they import
``nflvalue.config.prop_markets_internal`` / ``prop_markets_external``,
both of which do a function-local import of
``nflvalue.sources.oddsapi_props`` — a module deliberately EXCLUDED from
this port along with the rest of the prop stack.

Two notes for whoever reads this next:

1. `_SETUP.md` originally listed this file under "clean and portable
   as-is". That was wrong — it was prop-entangled all along, it just
   wasn't caught because the suite had never been run end-to-end in this
   project. Corrected 2026-07-18. The same sweep found
   `test_availability.py` was fine but was missing two ESPN fixture
   JSONs that hadn't been copied over (now copied), and that
   `test_notify_secrets_app.py` only needed two prop scripts removed
   from a parametrize list rather than full stubbing.

2. The prop-market functions in `nflvalue/config.py` that this tested
   are still present but dead in this project. They are harmless — the
   `oddsapi_props` import is function-local, so `config.py` imports
   cleanly and the Phase 0 smoke-test path is unaffected. They were left
   in place rather than removed because removing live code from
   `config.py` is a bigger call than a test sweep should make; flagged
   for a future cleanup pass instead.

Can't be deleted from this vault mount, so it's stubbed rather than left
silently failing. Original:
nfl-sim/fablesfable/tests/test_market_config.py.
"""
