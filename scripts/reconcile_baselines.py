#!/usr/bin/env python3
"""Workstream N — external baseline reconciliation.

Compares this project's end-of-season net ratings against three public
systems served by CFBD (SP+, SRS, Elo — 3 calls/season) and reports
correlation, rank agreement, and — the more valuable part — the BIGGEST
DISAGREEMENTS, each investigated rather than just logged.

Purpose runs both directions (BUILD_SPEC §18): catch silent data corruption
on our side, and flag where we are confidently different from every public
system — which occasionally means an edge and usually means a bug.

Writes data/baseline_reconciliation_{year}.txt.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from nflvalue import ingest, ratings as R  # noqa: E402
from nflvalue.sources.cfbd import CFBDClient  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISAGREE_Z = 1.5      # |z difference| beyond this gets investigated


def our_final_ratings(year: int) -> pd.DataFrame:
    games = R.game_frame(list(range(2019, year + 1)))
    params = R.load_params()
    last_week = int(games.loc[games["season"] == year, "week"].max())
    tbl = R.solve_cached(games, year, last_week + 1, params)
    rows = [{"team": t, "ours": v,
             "connectivity": tbl.connectivity.score.get(t, 0.0),
             "veto": bool(tbl.veto.get(t, True))}
            for t, v in tbl.net.items() if t != "FCS_TEAM"]
    return pd.DataFrame(rows)


def externals(year: int, client=None) -> pd.DataFrame:
    c = client or CFBDClient()
    sp = pd.DataFrame([{"team": r["team"], "sp": r["rating"]}
                       for r in c._get("/ratings/sp", {"year": year})
                       if r.get("team") != "nationalAverages"])
    srs = pd.DataFrame([{"team": r["team"], "srs": r["rating"]}
                        for r in c._get("/ratings/srs", {"year": year})])
    elo = pd.DataFrame([{"team": r["team"], "elo": r["elo"]}
                        for r in c._get("/ratings/elo", {"year": year})])
    ingest._record_calls(c.calls)
    return sp.merge(srs, on="team", how="outer").merge(elo, on="team",
                                                       how="outer")


def z(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--cached-json", default=None,
                    help="path to a pre-pulled {sp,srs,elo} JSON (0 calls)")
    args = ap.parse_args()

    ours = our_final_ratings(args.year)
    if args.cached_json:
        import json
        raw = json.load(open(args.cached_json))
        ext = (pd.DataFrame([{"team": r["team"], "sp": r["rating"]}
                             for r in raw["sp"]
                             if r.get("team") != "nationalAverages"])
               .merge(pd.DataFrame([{"team": r["team"], "srs": r["rating"]}
                                    for r in raw["srs"]]), on="team", how="outer")
               .merge(pd.DataFrame([{"team": r["team"], "elo": r["elo"]}
                                    for r in raw["elo"]]), on="team", how="outer"))
    else:
        ext = externals(args.year)

    m = ours[~ours["veto"]].merge(ext, on="team", how="inner")
    lines = [f"=== BASELINE RECONCILIATION {args.year} — ours vs SP+/SRS/Elo "
             f"(n={len(m)} rated FBS teams) ==="]
    for col, name in (("sp", "SP+"), ("srs", "SRS"), ("elo", "Elo")):
        d = m.dropna(subset=[col])
        pear = float(np.corrcoef(d["ours"], d[col])[0, 1])
        spear = float(stats.spearmanr(d["ours"], d[col]).statistic)
        top_o = set(d.nlargest(25, "ours")["team"])
        top_e = set(d.nlargest(25, col)["team"])
        lines.append(f"{name:4s}: pearson {pear:.3f}  spearman {spear:.3f}  "
                     f"top-25 overlap {len(top_o & top_e)}/25  (n={len(d)})")

    m["z_ours"] = z(m["ours"])
    m["z_ext"] = z(m[["sp", "srs"]].apply(z).mean(axis=1))
    m["dz"] = m["z_ours"] - m["z_ext"]
    big = m.reindex(m["dz"].abs().sort_values(ascending=False).index).head(8)
    lines.append("\nbiggest disagreements (ours vs public consensus, z-units)"
                 " — each explained, not just logged:")
    for r in big.itertuples():
        direction = "we are HIGHER" if r.dz > 0 else "we are LOWER"
        expl = []
        if r.connectivity < 0.6:
            expl.append(f"thin connectivity ({r.connectivity:.2f}) — our "
                        "number rests on few comparable games")
        if abs(r.dz) >= DISAGREE_Z and not expl:
            expl.append("no data-quality excuse found: our decayed 5-season "
                        "graph vs their single-season/priors weighting is "
                        "the likely cause — flag for the H drift review")
        if not expl:
            expl.append("within normal methodological spread")
        lines.append(f"  {r.team:22s} dz={r.dz:+.2f} ({direction}; "
                     f"ours {r.ours:+.1f}, SP+ {r.sp:+.1f}) — {expl[0]}")

    lines.append("\nReading: high-but-not-1.0 correlation is the healthy "
                 "zone. 1.00 would mean we built a copy of the public "
                 "consensus; anything under ~0.85 would mean someone is "
                 "broken. Disagreements above are inputs to Workstream H, "
                 "not evidence of an edge.")
    report = "\n".join(lines)
    print(report)
    out = os.path.join(ROOT, "data",
                       f"baseline_reconciliation_{args.year}.txt")
    with open(out, "w") as fh:
        fh.write(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
