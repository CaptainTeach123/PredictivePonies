"""Verification #1 of R1a@100 (place_longshot_lay_sp), execution / settlement sensitivities.

    python scripts/bf2_verify_place_longshot_lay_sp_1_r1a100_exec.py [--dir data/raw/betfair/split] [--frac 1.0]

Same universe and bets as scripts/bf2_verify_place_longshot_lay_sp_1_r1a100.py (original loader, imported
read-only). Checks:
  X1 G4 at 10% commission and with commission netted per market (conservative SE)
  X2 price impact: our SP lay matched at a place BSP that is 1 tick / +1% / +2% / +5% longer
  X3 dead-heat settlement worst case for the layer (every placed runner in a dead-heat race counts f = 1)
  X4 data consistency: WIN winner not flagged placed; races dropped by the settled filter
  X5 bets per month, races with >= 3 lays, and the share of ROI from the highest place BSPs
Prints summary numbers only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("v1", ROOT / "scripts/bf2_verify_place_longshot_lay_sp_1_r1a100.py")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)
orig = v1.orig
out, r6, summarize, se_cons, lay_liab = v1.out, v1.r6, v1.summarize, v1.se_cons, v1.lay_liab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data/raw/betfair/split"))
    ap.add_argument("--frac", type=float, default=1.0)
    a = ap.parse_args()
    d = Path(a.dir)
    frames = []
    for cc in ("au", "gb", "ie"):
        j, st = orig.load_country(cc, d, a.frac, 7)
        out({"section": "X4_loader_counts", "cc": cc, "identical_runner_sets": st["identical_runner_sets"],
             "settled_n_ge_nw_plus_1": st["settled_n_ge_nw_plus_1"],
             "dropped_by_settled_filter": st["identical_runner_sets"] - st["settled_n_ge_nw_plus_1"],
             "place_dead_heat_races": st["place_dead_heat_races"]})
        frames.append(v1.add_keys(j, d, cc))
    df = pd.concat(frames, ignore_index=True)
    del frames
    u = df[df["win_pre_all"] & df["win_pre_book"].between(0.97, 1.03)]
    g = u[(u["pre_win"] >= 100) & u["bsp_pl"].notna()].copy().reset_index(drop=True)
    g["_px"] = g["bsp_pl"]
    P, F = g["bsp_pl"].to_numpy(), g["f_pl"].to_numpy()
    base = summarize(g, P, F, "X0_base_conservative_se", se_fn=se_cons)

    # X1 commission 10%, and market-netted at 8% and 10%
    summarize(g, P, F, "X1_commission_10pct_conservative_se", se_fn=se_cons, c=0.10)
    for c in (0.08, 0.10):
        gross = (1 - F) - F * (P - 1)
        w = gross / (P - 1)
        net = pd.Series(w).groupby(g["race"].to_numpy()).sum()
        net = np.where(net > 0, net * (1 - c), net)
        out({"section": f"X1_market_netted_{int(c * 100)}pct", "roi": r6(net.sum() / len(P))})

    # X2 price impact
    for lab, P2 in (("one_tick_longer", orig.tick_longer(P)), ("plus_1pct", P * 1.01), ("plus_2pct", P * 1.02),
                    ("plus_5pct", P * 1.05)):
        s = summarize(g, P2, F, f"X2_impact_{lab}", se_fn=se_cons, verbose=False)
        out({"section": f"X2_impact_{lab}", **{k: s.get(k) for k in ("roi8", "se", "z", "halves", "halves_z", "last12",
                                                                       "g4")}})
    # break-even multiplicative impact
    lo, hi = 1.0, 1.5
    for _ in range(40):
        mid = (lo + hi) / 2
        if lay_liab(P * mid, F).mean() > 0:
            lo = mid
        else:
            hi = mid
    out({"section": "X2_break_even_place_bsp_multiplier", "multiplier": r6(lo)})

    # X3 dead-heat worst case for the layer
    dh = (g["n_placed"] > g["n_winners"]).to_numpy()
    F3 = np.where(dh & (F > 0), 1.0, F)
    out({"section": "X3_dead_heat", "bets_in_dead_heat_races": int(dh.sum()), "placed_bets_in_dh_races": int((dh & (F > 0)).sum())})
    summarize(g, P, F3, "X3_dead_heat_worst_case_f1", se_fn=se_cons)

    # X4 consistency: WIN winner must be placed
    inc = ((g["f_win"] > 0) & (g["f_pl"] == 0)).sum()
    allu = ((u["f_win"] > 0) & (u["f_pl"] == 0)).sum()
    out({"section": "X4_win_winner_not_placed", "R1a100_bets": int(inc), "R1a_universe_runners": int(allu),
         "R1a100_bets_that_won_win": int((g["f_win"] > 0).sum())})

    # X5 concentration
    r = lay_liab(P, F)
    order = np.argsort(-P)
    for share in (0.01, 0.05, 0.10):
        k = int(len(P) * share)
        keep = np.ones(len(P), bool)
        keep[order[:k]] = False
        out({"section": f"X5_drop_top_{int(share * 100)}pct_place_bsp", "roi8": r6(r[keep].mean()),
             "min_place_bsp_dropped": r6(P[order[k - 1]])})
    nr = g.groupby("race").size()
    out({"section": "X5_shape", "bets_per_month": r6(len(g) / g["month"].nunique()),
         "share_bets_in_races_with_3plus_lays": r6((g["race"].map(nr) >= 3).mean()),
         "mean_backer_stake_per_unit_liability": r6(np.mean(1 / (P - 1))),
         "expected_profit_per_year_per_10_liability": r6(10 * r.sum() / (len(g["month"].unique()) / 12)),
         "base_roi8": base.get("roi8")})


if __name__ == "__main__":
    main()
