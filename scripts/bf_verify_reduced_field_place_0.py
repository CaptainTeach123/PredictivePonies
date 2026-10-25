"""Adversarial verification of the FORMAL positive reported by bf_reduced_field_place.py:
lay place runners with e = q_harv(lambda_m) * ltp_place - 1 < -0.10 on C1 (3 places, 8-9 runners), settled at the
place last-traded price (+10.1% +/- 1.5%, 19,656 bets). Lens: leakage and data quality.

    python scripts/bf_verify_reduced_field_place_0.py [--dir data/raw/betfair/split] [--set C1]

Re-uses the original module's loading, pairing, Harville grid, walk-forward lambda path and evaluation frame
(imported, not copied) so the evaluation set is identical, then:
  1. REPRO of the cell (stake and liability ROI, per-bet SE, race-clustered SE, halves at the same date).
  2. Duplicate-market and join checks on the C1 MAIN set (place markets per (event_id, start), per WIN market).
  3. Staleness: ltp_t30 == ltp_pre share (no trade in the last 30 min) for rule rows vs all C1 rows; WIN drift
     ltp_win/t30_win; place BSP/LTP ratio; e recomputed at the place BSP.
  4. Executability: lay at LTP vs lay at BSP on the SAME rows; winsorised BSP; SP lay with a price limit;
     fresh/stale splits (traded in the last 30 min or not); rows with/without BSP; price bands; thresholds; years.
Prints summary numbers only (JSON lines); never horse names or rows.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("rfp", ROOT / "scripts" / "bf_reduced_field_place.py")
rfp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rfp)

out = rfp.out
r4 = rfp.r4
COMMS = rfp.COMMS
C_TEST = rfp.C_TEST


def lay_ret(price, f, c):
    return rfp.lay_stake_ret(price, f, c)


def cell(price, f, first, race, label: str, settlement: str, extra: dict | None = None, skipped: int = 0) -> dict:
    """Lay 1 unit of backer's stake: ROI at c = 0/5/8%, per-bet SE, race-clustered SE, halves, per-liability ROI."""
    price = np.asarray(price, float); f = np.asarray(f, float); first = np.asarray(first, bool); race = np.asarray(race)
    n = len(price)
    res = {"cell": label, "settlement": settlement, "bets": int(n), "skipped": int(skipped), "n_placed": int((f > 0).sum())}
    if n < 2:
        res.update({"roi_c5": None, "se": None})
        out(res)
        return res
    for c in COMMS:
        res[f"roi_c{int(c * 100)}"] = r4(lay_ret(price, f, c).mean())
    r5 = lay_ret(price, f, C_TEST)
    m = r5.mean(); se = r5.std(ddof=1) / math.sqrt(n)
    res["se"] = r4(se); res["z"] = r4(m / se) if se > 0 else None
    # race-clustered SE of the mean
    dev = r5 - m
    s = pd.Series(dev).groupby(race).sum().to_numpy()
    cse = math.sqrt((s ** 2).sum()) / n
    res["se_cluster_race"] = r4(cse); res["z_cluster"] = r4(m / cse) if cse > 0 else None
    res["races"] = int(len(np.unique(race)))
    hr, hs, hb = [], [], []
    for mask in (first, ~first):
        k = int(mask.sum())
        hb.append(k)
        if k >= 2:
            hr.append(r4(r5[mask].mean())); hs.append(r4(r5[mask].std(ddof=1) / math.sqrt(k)))
        else:
            hr.append(None); hs.append(None)
    res["halves_roi_c5"] = hr; res["halves_se_c5"] = hs; res["halves_bets"] = hb
    res["both_halves_2se"] = bool(all(a is not None and b and a > 2 * b for a, b in zip(hr, hs)))
    l5 = lay_ret(price, f, C_TEST) / (price - 1.0)
    res["liab_roi_c5"] = r4(l5.mean()); res["liab_se"] = r4(l5.std(ddof=1) / math.sqrt(n))
    res["mean_price"] = r4(price.mean()); res["median_price"] = r4(np.median(price)); res["max_price"] = r4(price.max())
    res["strike"] = r4((f > 0).mean())
    res["implied_mean_1_over_p"] = r4((1.0 / price).mean())
    if extra:
        res.update(extra)
    out(res)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "data/raw/betfair/split")
    ap.add_argument("--set", default="C1")
    ap.add_argument("--lay-threshold", type=float, default=-0.10)
    a = ap.parse_args()
    t0 = time.time()
    eval_months = [str(p) for p in pd.period_range("2017-06", "2020-11", freq="M")]
    out({"section": "config", "set": a.set, "lay_threshold": a.lay_threshold, "eval_months": [eval_months[0], eval_months[-1]],
         "note": "loading, pairing, lambda path and evaluation frame imported from bf_reduced_field_place.py"})

    j, Q, wj = rfp.load_joined(a.dir, "au", False, eval_months)
    # keep the WIN-side market id per place market for the duplicate check
    pl_to_win = wj.drop_duplicates("pl_mid").set_index("pl_mid")["win_mid"]
    del wj
    paths, lamT = rfp.fit_lambdas(j, Q, eval_months, "au")
    e = rfp.build_eval(j, Q, paths, lamT)
    del j, Q
    main_sel = (e["full"] & e["book_ok"]).to_numpy(bool)
    racesT = e[(e["set"] == "T") & main_sel].drop_duplicates("pl_mid")
    mid = pd.Timestamp(racesT["start"].sort_values().to_numpy()[len(racesT) // 2])
    e["first"] = e["start"] < mid
    e["win_mid"] = e["pl_mid"].map(pl_to_win)
    e["event_id"] = e["pl_mid"].map(pd.read_parquet(a.dir / "au_place.parquet", columns=["market_id", "event_id"])
                                   .drop_duplicates("market_id").set_index("market_id")["event_id"])
    out({"section": "halves", "median_start_of_MAIN_T_races": str(mid)})

    g = e[(e["set"] == a.set) & main_sel].copy()
    L = g[g["e"] < a.lay_threshold].copy()
    out({"section": "set_sizes", "set": a.set, "main_markets": int(g["pl_mid"].nunique()), "main_runners": int(len(g)),
         "rule_rows": int(len(L)), "rule_races": int(L["pl_mid"].nunique())})

    # ---------------------------------------------------------------- 2. duplicates / join
    mk = g.drop_duplicates("pl_mid")
    dup_event = mk.groupby(["event_id", "start"]).size()
    dup_win = mk.groupby("win_mid").size()
    out({"section": "duplicates", "set": a.set, "markets": int(len(mk)), "distinct_event_start": int(len(dup_event)),
         "event_start_with_gt1_place_market": int((dup_event > 1).sum()), "distinct_win_markets": int(len(dup_win)),
         "win_markets_with_gt1_place_market": int((dup_win > 1).sum()),
         "rule_rows_in_duplicated_event": int((pd.Series(L.set_index(["event_id", "start"]).index.map(dup_event)) > 1).sum()),
         "runners_per_market_eq_field_size_share": r4((g.groupby("pl_mid").size() == g.groupby("pl_mid")["field_size"].first()).mean()),
         "place_dead_heat_rule_rows": int(((L["f_pl"] > 0) & (L["f_pl"] < 1)).sum()),
         "rule_rows_same_selection_twice": int(L.duplicated(["pl_mid", "selection_id"]).sum())})

    # ---------------------------------------------------------------- 1. REPRO
    first = L["first"].to_numpy(); race = L["pl_mid"].to_numpy(); f = L["f_pl"].to_numpy()
    cell(L["ltp_pl"], f, first, race, "REPRO lay e<-0.10 at place LTP", "ltp")
    cell(rfp.tick_up(L["ltp_pl"].to_numpy()), f, first, race, "lay at LTP one tick up", "ltp+1tick")
    hb = L["bsp_pl"].notna() & (L["bsp_pl"] > 1)
    cell(L.loc[hb, "bsp_pl"], f[hb], first[hb], race[hb], "lay at place BSP (rows with BSP)", "bsp", skipped=int((~hb).sum()))
    cell(L.loc[hb, "ltp_pl"], f[hb], first[hb], race[hb], "lay at LTP on the SAME rows that have a BSP", "ltp")
    cell(L.loc[~hb, "ltp_pl"], f[~hb], first[~hb], race[~hb], "lay at LTP on rows WITHOUT a BSP", "ltp")
    for cap in (50.0, 100.0):
        cell(np.minimum(L.loc[hb, "bsp_pl"], cap), f[hb], first[hb], race[hb], f"lay at BSP capped at {cap:.0f}", "bsp")
    # BSP lay excluding outlier BSPs relative to LTP (ratio > 5)
    ratio = (L["bsp_pl"] / L["ltp_pl"])
    ok = hb & (ratio <= 5)
    cell(L.loc[ok, "bsp_pl"], f[ok], first[ok], race[ok], "lay at BSP, rows with BSP/LTP <= 5", "bsp", skipped=int((~ok).sum()))
    cell(L.loc[ok, "ltp_pl"], f[ok], first[ok], race[ok], "lay at LTP, same rows (BSP/LTP <= 5)", "ltp")

    # ---------------------------------------------------------------- 3. staleness diagnostics
    def q(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
        x = pd.Series(x).dropna()
        return [r4(v) for v in x.quantile(qs)] if len(x) else None
    has30 = L["t30_pl"].notna()
    g30 = g["t30_pl"].notna()
    out({"section": "staleness_t30", "set": a.set,
         "rule_rows_with_place_t30": int(has30.sum()), "rule_share_with_place_t30": r4(has30.mean()),
         "all_rows_share_with_place_t30": r4(g30.mean()),
         "rule_share_ltp_eq_t30_given_t30": r4((L.loc[has30, "ltp_pl"] == L.loc[has30, "t30_pl"]).mean()),
         "all_share_ltp_eq_t30_given_t30": r4((g.loc[g30, "ltp_pl"] == g.loc[g30, "t30_pl"]).mean()),
         "rule_place_ltp_over_t30_quantiles": q(L.loc[has30, "ltp_pl"] / L.loc[has30, "t30_pl"]),
         "all_place_ltp_over_t30_quantiles": q(g.loc[g30, "ltp_pl"] / g.loc[g30, "t30_pl"]),
         "rule_win_ltp_over_t30_quantiles": q(L["ltp_win"] / L["t30_win"]),
         "all_win_ltp_over_t30_quantiles": q(g["ltp_win"] / g["t30_win"]),
         "rule_share_win_drifted_gt_1.1": r4(((L["ltp_win"] / L["t30_win"]) > 1.1).mean()) if L["t30_win"].notna().any() else None,
         "all_share_win_drifted_gt_1.1": r4(((g["ltp_win"] / g["t30_win"]) > 1.1).mean()),
         "rule_place_bsp_over_ltp_quantiles": q(ratio[hb]),
         "all_place_bsp_over_ltp_quantiles": q((g["bsp_pl"] / g["ltp_pl"])[(g["bsp_pl"] > 1)]),
         "rule_win_bsp_over_ltp_quantiles": q(L["bsp_win"] / L["ltp_win"]),
         "all_win_bsp_over_ltp_quantiles": q(g["bsp_win"] / g["ltp_win"]),
         "rule_share_bsp_gt_ltp": r4((ratio[hb] > 1).mean()), "all_share_place_bsp_gt_ltp": r4(((g["bsp_pl"] / g["ltp_pl"])[(g["bsp_pl"] > 1)] > 1).mean())})
    # e recomputed at the place BSP (not pre-race: diagnostic of whether the short price survives to the off)
    e_bsp = L["q_harv"] * L["bsp_pl"] - 1
    out({"section": "edge_at_bsp", "set": a.set, "rule_rows_with_bsp": int(hb.sum()),
         "share_e_bsp_lt_-0.10": r4((e_bsp[hb] < -0.10).mean()), "share_e_bsp_lt_0": r4((e_bsp[hb] < 0).mean()),
         "e_ltp_quantiles": q(L["e"]), "e_bsp_quantiles": q(e_bsp[hb]),
         "mean_q_harv": r4(L["q_harv"].mean()), "mean_q_mkt": r4(L["q_mkt"].mean()), "mean_1_over_ltp": r4((1 / L["ltp_pl"]).mean()),
         "mean_1_over_bsp": r4((1 / L.loc[hb, "bsp_pl"]).mean()), "placed_rate": r4((L["f_pl"] > 0).mean()), "placed_rate_bsp_rows": r4((L.loc[hb, "f_pl"] > 0).mean())})

    # ---------------------------------------------------------------- 4. executability splits
    stale = has30 & (L["ltp_pl"] == L["t30_pl"])
    fresh = has30 & (L["ltp_pl"] != L["t30_pl"])
    absent = ~has30
    for name, m in (("place t30 absent (market thin 30 min out)", absent), ("t30 present, LTP == t30 (no trade in last 30 min)", stale),
                    ("t30 present, LTP != t30 (traded in last 30 min)", fresh),
                    ("traded in last 30 min and LTP shortened vs t30", fresh & (L["ltp_pl"] < L["t30_pl"])),
                    ("traded in last 30 min and LTP lengthened vs t30", fresh & (L["ltp_pl"] > L["t30_pl"]))):
        m = m.to_numpy()
        cell(L.loc[m, "ltp_pl"], f[m], first[m], race[m], f"lay at LTP: {name}", "ltp")
        mb = m & hb.to_numpy()
        cell(L.loc[mb, "bsp_pl"], f[mb], first[mb], race[mb], f"lay at BSP: {name}", "bsp", skipped=int((m & ~hb.to_numpy()).sum()))
    # market-level liquidity: every runner in the place market has a BSP
    full_bsp = g.groupby("pl_mid")["bsp_pl"].apply(lambda s: bool((s > 1).all()))
    mliq = L["pl_mid"].map(full_bsp).fillna(False).astype(bool).to_numpy()
    cell(L.loc[mliq, "ltp_pl"], f[mliq], first[mliq], race[mliq], "lay at LTP: every runner in the place market has a BSP", "ltp")
    cell(L.loc[mliq, "bsp_pl"], f[mliq], first[mliq], race[mliq], "lay at BSP: every runner in the place market has a BSP", "bsp")
    # SP lay with a price limit = LTP * (1 + m): matched only if BSP <= limit; unmatched = 0 return
    for mlim in (0.0, 0.10, 0.25, 0.50, 1.0):
        lim = L["ltp_pl"] * (1 + mlim)
        matched = (hb & (L["bsp_pl"] <= lim)).to_numpy()
        r5 = lay_ret(L.loc[matched, "bsp_pl"], f[matched], C_TEST)
        n_att = int(hb.sum())
        per_att = r5.sum() / n_att if n_att else float("nan")
        se_att = math.sqrt(((np.concatenate([r5, np.zeros(n_att - matched.sum())]) - per_att) ** 2).sum() / (n_att - 1)) / math.sqrt(n_att) if n_att > 1 else float("nan")
        res = cell(L.loc[matched, "bsp_pl"], f[matched], first[matched], race[matched], f"SP lay with limit LTP*{1 + mlim:.2f} (matched bets)", "bsp",
                   extra={"attempted_with_bsp": n_att, "matched_share": r4(matched.sum() / n_att) if n_att else None,
                          "roi_c5_per_attempted": r4(per_att), "se_per_attempted": r4(se_att)})
    # SP-limit neighbourhood on every set: is the matched-at-or-below-LTP sub-result stable, and does it carry to T / C2?
    for set_ in ("C1", "T", "C2"):
        gs = e[(e["set"] == set_) & main_sel]
        Ls = gs[gs["e"] < a.lay_threshold]
        hbs = (Ls["bsp_pl"] > 1).to_numpy()
        fs = Ls["f_pl"].to_numpy(); fi = Ls["first"].to_numpy(); rc = Ls["pl_mid"].to_numpy()
        for mlim in (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20):
            matched = hbs & (Ls["bsp_pl"] <= Ls["ltp_pl"] * (1 + mlim)).to_numpy()
            n_att = int(len(Ls))  # rows without a BSP = unmatched (no SP formed)
            r5 = lay_ret(Ls.loc[matched, "bsp_pl"], fs[matched], C_TEST)
            full = np.zeros(n_att); full[np.flatnonzero(matched)] = r5
            cell(Ls.loc[matched, "bsp_pl"], fs[matched], fi[matched], rc[matched], f"{set_}: SP lay limit LTP*{1 + mlim:.2f}", "bsp",
                 extra={"attempted_all_rule_rows": n_att, "matched_share": r4(matched.mean()),
                        "roi_c5_per_attempted": r4(full.mean()), "se_per_attempted": r4(full.std(ddof=1) / math.sqrt(n_att)),
                        "mean_q_harv_matched": r4(Ls.loc[matched, "q_harv"].mean()),
                        "bsp_implied_mean_matched": r4((1 / Ls.loc[matched, "bsp_pl"]).mean()),
                        "ltp_implied_mean_matched": r4((1 / Ls.loc[matched, "ltp_pl"]).mean())})
    # price bands (by LTP) at LTP and at BSP
    for lo, hi in ((1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 1000.0)):
        m = ((L["ltp_pl"] > lo) & (L["ltp_pl"] <= hi)).to_numpy()
        cell(L.loc[m, "ltp_pl"], f[m], first[m], race[m], f"lay at LTP, LTP in ({lo},{hi}]", "ltp")
        mb = m & hb.to_numpy()
        cell(L.loc[mb, "bsp_pl"], f[mb], first[mb], race[mb], f"lay at BSP, LTP in ({lo},{hi}]", "bsp", skipped=int((m & ~hb.to_numpy()).sum()))
    # thresholds
    for th in (-0.05, -0.20, -0.30):
        Lt = g[g["e"] < th]
        ft = Lt["f_pl"].to_numpy(); fi = Lt["first"].to_numpy(); rc = Lt["pl_mid"].to_numpy()
        cell(Lt["ltp_pl"], ft, fi, rc, f"lay at LTP, e < {th}", "ltp")
        hbt = (Lt["bsp_pl"] > 1).to_numpy()
        cell(Lt.loc[hbt, "bsp_pl"], ft[hbt], fi[hbt], rc[hbt], f"lay at BSP, e < {th}", "bsp", skipped=int((~hbt).sum()))
    # years
    yr = L["start"].dt.year.to_numpy()
    for y in sorted(set(yr)):
        m = yr == y
        cell(L.loc[m, "ltp_pl"], f[m], first[m], race[m], f"lay at LTP, year {y}", "ltp")
        mb = m & hb.to_numpy()
        cell(L.loc[mb, "bsp_pl"], f[mb], first[mb], race[mb], f"lay at BSP, year {y}", "bsp", skipped=int((m & ~hb.to_numpy()).sum()))
    # mirror: back e > +0.10 at LTP and BSP on the same set (is the paper profit two-sided, as a noisy-price artefact predicts?)
    B = g[g["e"] > 0.10]
    for s, col in (("ltp", "ltp_pl"), ("bsp", "bsp_pl")):
        okb = (B[col] > 1).to_numpy()
        r = rfp.back_ret(B.loc[okb, col].to_numpy(), B.loc[okb, "f_pl"].to_numpy(), C_TEST)
        out({"section": "mirror_back_e_gt_0.10", "settlement": s, "bets": int(okb.sum()), "roi_c5": r4(r.mean()),
             "se": r4(r.std(ddof=1) / math.sqrt(len(r))) if len(r) > 1 else None,
             "placed_rate": r4((B.loc[okb, "f_pl"] > 0).mean()), "mean_q_mkt": r4(B.loc[okb, "q_mkt"].mean()), "mean_q_harv": r4(B.loc[okb, "q_harv"].mean())})
    # the T set, same diagnostics in brief (reduced-field hypothesis set)
    for set_ in ("T", "C2"):
        gs = e[(e["set"] == set_) & main_sel]
        Ls = gs[gs["e"] < a.lay_threshold]
        fs = Ls["f_pl"].to_numpy(); fi = Ls["first"].to_numpy(); rc = Ls["pl_mid"].to_numpy()
        hbs = (Ls["bsp_pl"] > 1).to_numpy()
        h30 = Ls["t30_pl"].notna().to_numpy()
        fr = h30 & (Ls["ltp_pl"] != Ls["t30_pl"]).to_numpy()
        cell(Ls["ltp_pl"], fs, fi, rc, f"{set_}: lay at LTP", "ltp")
        cell(Ls.loc[hbs, "bsp_pl"], fs[hbs], fi[hbs], rc[hbs], f"{set_}: lay at BSP", "bsp", skipped=int((~hbs).sum()))
        cell(Ls.loc[hbs, "ltp_pl"], fs[hbs], fi[hbs], rc[hbs], f"{set_}: lay at LTP, same rows with BSP", "ltp")
        cell(Ls.loc[fr, "ltp_pl"], fs[fr], fi[fr], rc[fr], f"{set_}: lay at LTP, traded in last 30 min", "ltp")
        cell(Ls.loc[fr & hbs, "bsp_pl"], fs[fr & hbs], fi[fr & hbs], rc[fr & hbs], f"{set_}: lay at BSP, traded in last 30 min", "bsp")
    out({"section": "done", "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
