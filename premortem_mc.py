#!/usr/bin/env python3
"""Monte Carlo bankroll stress-test for the NFL Value Engine premortem.

Everything here is at standard -110 juice (decimal 1.9091): win +0.9091u, lose -1u.
Break-even win rate at -110 is 110/210 = 0.523810.

Parts:
  A. Flat-stake bankroll trajectories under different TRUE win rates -> P(profit), P(ruin), percentiles
  B. Kelly failure mode: what happens when you OVERESTIMATE your edge (Kelly assumes known p)
  C. Time-to-detect: bets needed to prove a small edge vs break-even (analytic + sim check)
  D. NFL sample realism: convert (C) into SEASONS given realistic bets/season
"""
import numpy as np, json, math

RNG = np.random.default_rng(20260701)
DEC = 1.9091            # -110 decimal odds
WIN = DEC - 1.0         # +0.9091 on a win
BE  = 110/210           # 0.523810 break-even win prob at -110
NT  = 20000             # trials per scenario

def sim_flat(p, nbets, unit_frac=0.01, start=100.0, ntrials=NT, rng=RNG):
    """Flat stake = unit_frac of STARTING bankroll, fixed unit. Ruin = equity hits 0."""
    unit = start*unit_frac
    wins = rng.random((ntrials, nbets)) < p
    step = np.where(wins, WIN*unit, -unit)
    equity = start + np.cumsum(step, axis=1)
    finals = equity[:, -1]
    ruined = (equity <= 0).any(axis=1)           # touched zero at any point
    return {
        "p": p, "nbets": nbets, "unit_frac": unit_frac,
        "ev_per_bet_pct": round((p*DEC-1)*100, 3),
        "median": round(float(np.median(finals)), 1),
        "mean": round(float(finals.mean()), 1),
        "p5": round(float(np.percentile(finals, 5)), 1),
        "p95": round(float(np.percentile(finals, 95)), 1),
        "prob_profit": round(float((finals > start).mean()), 4),
        "prob_ruin": round(float(ruined.mean()), 4),
        "prob_down_50pct": round(float((finals < start*0.5).mean()), 4),
    }

def kelly_frac(p, dec=DEC):
    b = dec - 1.0
    return max(0.0, (p*dec - 1.0)/b)

def sim_kelly(p_true, p_est, kmult, nbets, start=100.0, ntrials=NT, rng=RNG):
    """Proportional Kelly on ESTIMATED p, outcomes drawn from TRUE p.
    Ruin proxy = bankroll falls below 5% of start (practical wipeout)."""
    f = kmult*kelly_frac(p_est)
    bank = np.full(ntrials, start)
    minbank = np.full(ntrials, start)
    for _ in range(nbets):
        stake = f*bank
        wins = rng.random(ntrials) < p_true
        bank = bank + np.where(wins, WIN*stake, -stake)
        minbank = np.minimum(minbank, bank)
    return {
        "p_true": p_true, "p_est": p_est, "kmult": kmult, "kelly_f_pct": round(f*100,2),
        "nbets": nbets,
        "median": round(float(np.median(bank)),1),
        "mean": round(float(bank.mean()),1),
        "p5": round(float(np.percentile(bank,5)),1),
        "p95": round(float(np.percentile(bank,95)),1),
        "prob_profit": round(float((bank>start).mean()),4),
        "prob_below_5pct": round(float((minbank < start*0.05).mean()),4),
        "prob_below_50pct": round(float((minbank < start*0.50).mean()),4),
    }

def n_to_detect(p1, p0=BE, alpha=0.05, power=0.80):
    """One-sided sample size to reject H0: p<=p0 when true rate is p1."""
    za = 1.6449; zb = 0.8416   # z(0.95), z(0.80)
    num = (za*math.sqrt(p0*(1-p0)) + zb*math.sqrt(p1*(1-p1)))**2
    return math.ceil(num/((p1-p0)**2))

def sim_detect(p1, n, p0=BE, ntrials=NT, rng=RNG):
    """Empirical power: P(observed rate clears the 95% one-sided bar) at n bets.
    Uses binomial draws (no giant array)."""
    wins = rng.binomial(n, p1, size=ntrials)
    rate = wins/n
    bar = p0 + 1.6449*math.sqrt(p0*(1-p0)/n)   # 95% one-sided critical rate
    return round(float((rate > bar).mean()), 3)

out = {"constants": {"decimal_odds": DEC, "break_even": round(BE,6), "trials": NT}}

# ---------- Part A: flat-stake trajectories ----------
A = []
for p in (0.500, BE, 0.530, 0.540, 0.550):
    for nb in (500, 2000):
        A.append(sim_flat(p, nb))
out["A_flat"] = A

# ---------- Part B: Kelly overestimation failure ----------
B = []
# (i) honest edge, half vs full Kelly
B.append(sim_kelly(0.540, 0.540, 0.5, 1000))
B.append(sim_kelly(0.540, 0.540, 1.0, 1000))
# (ii) OVERFIT: you think 56% but truth is below break-even (51.5%)
B.append(sim_kelly(0.515, 0.560, 1.0, 1000))
B.append(sim_kelly(0.515, 0.560, 0.5, 1000))
# (iii) OVERFIT milder: think 55%, truly 52% (still below BE)
B.append(sim_kelly(0.520, 0.550, 0.5, 1000))
out["B_kelly"] = B

# ---------- Part C: time-to-detect edge ----------
C = []
for p1 in (0.530, 0.540, 0.550, 0.560, 0.570):
    n = n_to_detect(p1)
    C.append({"true_p": p1, "edge_over_BE_pts": round((p1-BE)*100,2),
              "n_bets_needed": n, "empirical_power_at_n": sim_detect(p1, min(n, 200000))})
out["C_detect"] = C

# ---------- Part D: seasons to significance ----------
# Empirical anchor from THEIR backtest: ~150 spread bets/season at 3% EV threshold.
# The >60% Discord filter is stricter, so show a realistic range.
D = []
for label, bets_yr in (("aggressive ~150/yr", 150), ("moderate ~80/yr", 80), ("selective ~40/yr", 40)):
    row = {"cadence": label, "bets_per_season": bets_yr}
    for p1 in (0.540, 0.550, 0.560):
        row[f"seasons_for_{int(p1*100)}pct"] = round(n_to_detect(p1)/bets_yr, 1)
    D.append(row)
out["D_seasons"] = D

print(json.dumps(out, indent=2))
with open("premortem_mc_results.json","w") as f: json.dump(out, f, indent=2)

# ---------- console summary ----------
print("\n================  READABLE SUMMARY  ================")
print(f"Break-even win rate @ -110: {BE*100:.2f}%   (win pays +{WIN:.4f}u)")
print("\nA) FLAT 1u (=1% of 100u bank), fixed unit, ruin=touch 0:")
print(f"{'true p':>7}{'EV/bet':>8}{'bets':>6}{'median':>8}{'P(profit)':>10}{'P(ruin)':>9}{'P(-50%)':>9}")
for r in A:
    print(f"{r['p']*100:6.2f}%{r['ev_per_bet_pct']:+7.2f}%{r['nbets']:6}{r['median']:8.1f}"
          f"{r['prob_profit']*100:9.1f}%{r['prob_ruin']*100:8.1f}%{r['prob_down_50pct']*100:8.1f}%")
print("\nB) KELLY (proportional; ruin proxy = dip below 5% of start), 1000 bets:")
print(f"{'p_true':>7}{'p_est':>7}{'kmult':>6}{'stake%':>7}{'median':>8}{'P(profit)':>10}{'P(<5%)':>8}{'P(<50%)':>9}")
for r in B:
    print(f"{r['p_true']*100:6.1f}%{r['p_est']*100:6.1f}%{r['kmult']:6.2f}{r['kelly_f_pct']:6.2f}%"
          f"{r['median']:8.1f}{r['prob_profit']*100:9.1f}%{r['prob_below_5pct']*100:7.1f}%{r['prob_below_50pct']*100:8.1f}%")
print("\nC) BETS NEEDED to prove edge vs 52.38% break-even (95% one-sided, 80% power):")
for r in C:
    print(f"  true {r['true_p']*100:.1f}% (+{r['edge_over_BE_pts']:.2f} pts): "
          f"{r['n_bets_needed']:>7,} bets   (empirical power {r['empirical_power_at_n']})")
print("\nD) SEASONS to significance at realistic NFL volumes:")
for r in D:
    print(f"  {r['cadence']:>20}: 54%->{r['seasons_for_54pct']}yr  55%->{r['seasons_for_55pct']}yr  56%->{r['seasons_for_56pct']}yr")
