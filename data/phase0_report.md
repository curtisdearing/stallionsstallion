# Phase 0 smoke test — PASS
mode=selftest  conference=MWC(fixture)  year=2023
CFBD calls used: 0 / 1000 monthly budget


--- /games ---
  [PASS] games returned: 3 games
  [PASS] weeks populated: weeks [1]..[3] (3 distinct)
  [PASS] final scores present: 100% of games have home points

--- /plays ---
  [PASS] plays returned: 6 plays
  [PASS] PPA present: 67% of plays have non-null PPA (CFB EPA equivalent)
  [PASS] down/distance/yardline populated: 83% rows have down+distance+yardline
  [PASS] play type groupable: 6 distinct play types (need this for std/passing-downs splits)

--- /lines ---
  [PASS] line rows returned: 2 game-line rows
  [PASS] multi-book coverage: 1/2 games have >=2 books (needed for de-vig/consensus)
  [PASS] spread present: 3 book-rows carry a spread
  [PASS] total present: 3 book-rows carry an over/under
  [PASS] moneyline present: 3 book-rows carry a moneyline (may be sparse for G5 — verify)

--- /talent ---
  [PASS] talent rows returned: 8 teams league-wide
  [PASS] talent scale sane (measured values only): range 4.33..1015.43 over 5 measured teams; 3 sentinel-0 teams excluded as missing
  [PASS] talent sentinel set is exactly the known academies: talent==0 for ['Air Force', 'Army', 'Navy'] (expected ['Air Force', 'Army', 'Navy']) — these must ingest as NULL + unranked_recruiting=True, never as 0.0
  [PASS] conference teams covered: 3/6 of the conference's teams have a talent row

Manual spot-check game (compare to ESPN/Sports-Reference):
  2023-09-02T18:00:00.000Z  UCF 31 @ Boise State 24
