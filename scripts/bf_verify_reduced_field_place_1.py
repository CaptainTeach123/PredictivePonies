"""Adversarial verification of the "reduced_field_place" FORMAL positive:
lay C1 (3 places, 8-9 runners) place runners with e = q_harv*ltp_place - 1 < -0.10, settled at place ltp_pre.

    python scripts/bf_verify_reduced_field_place_1.py [--dir data/raw/betfair/split]

Re-uses the ORIGINAL module (imported, not edited) for loading, pairing, the walk-forward lambda path and
the evaluation frame, so the bet rows are identical, then checks:
  1. reproduction of the cell (bets, roi c0/c5/c8/c10, per-bet SE, halves, per-liability)
  2. cluster-robust SE by place market (bets in a race are correlated) for LTP and BSP, overall and per half
  3. the BSP twin: winsorised / trimmed / outlier-excluded versions, median return, cluster SE
  4. matched rows: same rows at LTP vs BSP (paired difference), BSP/LTP distribution, share BSP > LTP, > tick-up
  5. rows WITHOUT a BSP (illiquid) vs with: LTP ROI on each
  6. staleness proxies: ltp_t30 present & unchanged at the off, market had any t30 trade
  7. executable-price settlements: max(LTP, BSP); BSP where present else tick-up LTP; tick-up
  8. BSP/LTP ratio bins (where does the LTP edge live?)
  9. year-by-year at LTP and BSP (clustered)
 10. baseline: lay EVERY C1 runner at LTP / BSP (and by LTP band) to show the "lay anything at a stale LTP" effect
 11. threshold scan (descriptive) at LTP and BSP
 12. multi-lay races vs single-lay races
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

_spec = importlib.util.spec_from_file_location("orig_rfp", ROOT / "scripts" / "bf_reduced_field_place.py")
orig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(orig)

out, r4 = orig.out, orig.r4
lay_stake_ret, lay_liab_ret, back_ret, tick_up = orig.lay_stake_ret, orig.lay_liab_ret, orig.back_ret, orig.tick_up
COMMS = (0.0, 0.05, 0.08, 0.10)


def cl_se(r: np.ndarray, cl: np.ndarray) -> float:
    """cluster-robust SE of the mean of r with clusters cl (G/(G-1) small-sample factor)."""
    r = np.asarray(r, float)
    n = len(r)
    if n < 2:
        return float("nan")
    m = r.mean()
    s = pd.DataFrame({"r": r - m, "c": cl}).groupby("c")["r"].sum().to_numpy()
    G = len(s)
    if G < 2:
        return float("nan")
    return float(math.sqrt((s ** 2).sum() * G / (G - 1)) / n)


def cell(price, f, cl, first=None, side="lay", label="") -> dict:
    price = np.asarray(price, float); f = np.asarray(f, float); cl = np.asarray(cl)
    n = len(price)
    res = {"label": label, "bets": int(n), "races": int(len(np.unique(cl))) if n else 0}
    if n == 0:
        return res
    fn = lay_stake_ret if side == "lay" else back_ret
    for c in COMMS:
        res[f"roi_c{int(c * 100)}"] = r4(fn(price, f, c).mean())
    r5 = fn(price, f, 0.05)
    se_bet = float(r5.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    se_cl = cl_se(r5, cl)
    res["se_per_bet"] = r4(se_bet); res["se_cluster"] = r4(se_cl)
    res["z_per_bet"] = r4(r5.mean() / se_bet) if se_bet > 0 else None
    res["z_cluster"] = r4(r5.mean() / se_cl) if se_cl > 0 else None
    res["median_ret_c5"] = r4(np.median(r5))
    res["strike"] = r4((f > 0).mean())
    res["price_q"] = [r4(v) for v in np.quantile(price, [0.05, 0.25, 0.5, 0.75, 0.95, 1.0])]
    if first is not None:
        first = np.asarray(first, bool)
        hv = []
        for mask in (first, ~first):
            rr = r5[mask]
            if len(rr) > 1:
                hv.append({"bets": int(mask.sum()), "roi_c5": r4(rr.mean()), "se_per_bet": r4(rr.std(ddof=1) / math.sqrt(len(rr))),
                           "se_cluster": r4(cl_se(rr, cl[mask]))})
            else:
                hv.append({"bets": int(mask.sum())})
        res["halves"] = hv
        res["both_halves_2se_cluster"] = bool(all(h.get("roi_c5") is not None and h.get("se_cluster") and h["roi_c5"] > 2 * h["se_cluster"] for h in hv))
    if side == "lay":
        l5 = lay_liab_ret(price, f, 0.05)
        res["liab_roi_c5"] = r4(l5.mean()); res["liab_se_cluster"] = r4(cl_se(l5, cl))
        res["liab_roi_c8"] = r4(lay_liab_ret(price, f, 0.08).mean())
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "data/raw/betfair/split")
    a = ap.parse_args()
    t0 = time.time()
    eval_months = [str(p) for p in pd.period_range("2017-06", "2020-11", freq="M")]

    # ---------------------------------------------------------------- identical construction via the original module
    j, Q, wj = orig.load_joined(a.dir, "au", False, eval_months)
    del wj
    paths, lamT = orig.fit_lambdas(j, Q, eval_months, "au")
    e = orig.build_eval(j, Q, paths, lamT)
    del j, Q
    main_sel = (e["full"] & e["book_ok"]).to_numpy(bool)
    racesT = e[(e["set"] == "T") & main_sel].drop_duplicates("pl_mid")
    mid = pd.Timestamp(racesT["start"].sort_values().to_numpy()[len(racesT) // 2])
    e["first"] = e["start"] < mid
    e["year"] = e["start"].dt.year
    c1 = e[(e["set"] == "C1") & main_sel].copy()
    lay = c1[c1["e"] < -orig.LAY_T].copy()
    lay["has_bsp"] = (lay["bsp_pl"] > 1).fillna(False).to_numpy()
    lay["tick_up"] = tick_up(lay["ltp_pl"].to_numpy())
    out({"section": "setup", "halves_mid": str(mid), "C1_main_runners": int(len(c1)), "C1_main_races": int(c1["pl_mid"].nunique()),
         "lay_rows": int(len(lay)), "lay_races": int(lay["pl_mid"].nunique()), "lay_rows_with_bsp": int(lay["has_bsp"].sum()),
         "lay_rows_no_bsp": int((~lay["has_bsp"]).sum()), "seconds_setup": round(time.time() - t0, 1)})

    P, F, CL, FIRST = lay["ltp_pl"].to_numpy(), lay["f_pl"].to_numpy(), lay["pl_mid"].to_numpy(), lay["first"].to_numpy()

    # ---------------------------------------------------------------- 1+2. reproduction + clustered SE
    out({"section": "1_2_reproduce_ltp_clustered", **cell(P, F, CL, FIRST, "lay", "C1 lay e<-0.10 @ LTP")})
    out({"section": "7_tick_up", **cell(lay["tick_up"].to_numpy(), F, CL, FIRST, "lay", "C1 lay @ one tick above LTP")})

    # ---------------------------------------------------------------- 3. BSP twin, robust versions
    lb = lay[lay["has_bsp"]]
    Pb, Fb, CLb, Fb1 = lb["bsp_pl"].to_numpy(), lb["f_pl"].to_numpy(), lb["pl_mid"].to_numpy(), lb["first"].to_numpy()
    out({"section": "3_bsp_twin", **cell(Pb, Fb, CLb, Fb1, "lay", "C1 lay @ BSP (rows with BSP)")})
    for cap in (100.0, 50.0, 20.0):
        out({"section": "3_bsp_winsorised", "cap": cap, "rows_capped": int((Pb > cap).sum()),
             **cell(np.minimum(Pb, cap), Fb, CLb, Fb1, "lay", f"C1 lay @ min(BSP,{cap:g})")})
    ratio = lb["bsp_pl"].to_numpy() / lb["ltp_pl"].to_numpy()
    keep = ratio <= 5.0
    out({"section": "3_bsp_excl_ratio_gt5", "excluded": int((~keep).sum()), **cell(Pb[keep], Fb[keep], CLb[keep], Fb1[keep], "lay", "C1 lay @ BSP, BSP/LTP<=5")})
    keep = ratio <= 3.0
    out({"section": "3_bsp_excl_ratio_gt3", "excluded": int((~keep).sum()), **cell(Pb[keep], Fb[keep], CLb[keep], Fb1[keep], "lay", "C1 lay @ BSP, BSP/LTP<=3")})
    # trimmed mean of the c5 return (drop top/bottom 1%)
    rb = lay_stake_ret(Pb, Fb, 0.05)
    lo, hi = np.quantile(rb, [0.01, 0.99])
    out({"section": "3_bsp_trimmed_1pct", "trimmed_mean_c5": r4(rb[(rb >= lo) & (rb <= hi)].mean()), "n_kept": int(((rb >= lo) & (rb <= hi)).sum()),
         "bsp_outliers_gt100": int((Pb > 100).sum()), "bsp_outliers_gt100_placed": int(((Pb > 100) & (Fb > 0)).sum()),
         "max_bsp": r4(Pb.max())})

    # ---------------------------------------------------------------- 4. matched rows: LTP vs BSP on the same rows
    rl = lay_stake_ret(lb["ltp_pl"].to_numpy(), Fb, 0.05)
    d = rl - rb
    out({"section": "4_matched_rows", "rows": int(len(lb)),
         "ltp_same_rows": cell(lb["ltp_pl"].to_numpy(), Fb, CLb, Fb1, "lay", "LTP on BSP rows"),
         "bsp_same_rows_roi_c5": r4(rb.mean()),
         "paired_diff_ltp_minus_bsp_c5": r4(d.mean()), "paired_diff_se_cluster": r4(cl_se(d, CLb)),
         "paired_diff_winsor100_c5": r4((rl - lay_stake_ret(np.minimum(Pb, 100.0), Fb, 0.05)).mean()),
         "bsp_over_ltp_quantiles": [r4(v) for v in np.quantile(ratio, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])],
         "share_bsp_gt_ltp": r4((ratio > 1.0).mean()), "share_bsp_gt_tick_up": r4((lb["bsp_pl"].to_numpy() > lb["tick_up"].to_numpy()).mean()),
         "share_bsp_lt_ltp": r4((ratio < 1.0).mean()), "share_bsp_eq_ltp_pm2pct": r4((np.abs(ratio - 1) <= 0.02).mean()),
         "bsp_implied_mean_1/bsp": r4((1.0 / Pb).mean()), "ltp_implied_mean_1/ltp": r4((1.0 / lb["ltp_pl"].to_numpy()).mean()),
         "placed_rate": r4((Fb > 0).mean()), "mean_q_harv": r4(lb["q_harv"].mean())})
    # same comparison for ALL C1 runners (is BSP > LTP a general feature or specific to the lay rows?)
    cb = c1[(c1["bsp_pl"] > 1).fillna(False)]
    rat_all = cb["bsp_pl"].to_numpy() / cb["ltp_pl"].to_numpy()
    out({"section": "4_bsp_over_ltp_all_C1", "rows": int(len(cb)), "quantiles": [r4(v) for v in np.quantile(rat_all, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])],
         "share_bsp_gt_ltp": r4((rat_all > 1).mean()),
         "by_edge_bin": {lab: {"n": int(m.sum()), "median_bsp_over_ltp": r4(np.median(rat_all[m])), "share_bsp_gt_ltp": r4((rat_all[m] > 1).mean())}
                         for lab, m in (("e<-0.10", (cb["e"] < -0.10).to_numpy()), ("[-0.10,0)", ((cb["e"] >= -0.10) & (cb["e"] < 0)).to_numpy()),
                                        ("[0,0.10)", ((cb["e"] >= 0) & (cb["e"] < 0.10)).to_numpy()), ("e>=0.10", (cb["e"] >= 0.10).to_numpy()))}})

    # ---------------------------------------------------------------- 5. rows without BSP vs with
    ln = lay[~lay["has_bsp"]]
    out({"section": "5_rows_without_bsp", **cell(ln["ltp_pl"].to_numpy(), ln["f_pl"].to_numpy(), ln["pl_mid"].to_numpy(), ln["first"].to_numpy(), "lay", "LTP on rows with NO BSP")})
    out({"section": "5_rows_with_bsp_at_ltp", **cell(lb["ltp_pl"].to_numpy(), Fb, CLb, Fb1, "lay", "LTP on rows with BSP")})

    # ---------------------------------------------------------------- 6. staleness proxies
    has_t30 = lay["t30_pl"].notna().to_numpy()
    same = has_t30 & (np.abs(lay["t30_pl"].to_numpy() - lay["ltp_pl"].to_numpy()) < 1e-9)
    out({"section": "6_staleness_t30", "lay_rows_with_t30": int(has_t30.sum()), "share_ltp_pre_equals_ltp_t30": r4(same[has_t30].mean()),
         "C1_all_rows_share_ltp_pre_equals_t30": r4((np.abs(c1["t30_pl"] - c1["ltp_pl"]) < 1e-9)[c1["t30_pl"].notna()].mean()),
         "t30_present_unchanged": cell(P[same], F[same], CL[same], FIRST[same], "lay", "t30 present, ltp_pre == ltp_t30"),
         "t30_present_changed": cell(P[has_t30 & ~same], F[has_t30 & ~same], CL[has_t30 & ~same], FIRST[has_t30 & ~same], "lay", "t30 present, price moved"),
         "t30_absent": cell(P[~has_t30], F[~has_t30], CL[~has_t30], FIRST[~has_t30], "lay", "no t30 trade on this runner")})
    anyt = lay["any_t30"].to_numpy(bool)
    out({"section": "6_market_any_t30", "market_traded_30min_out": cell(P[anyt], F[anyt], CL[anyt], FIRST[anyt], "lay", "market had any t30 trade"),
         "market_no_t30": cell(P[~anyt], F[~anyt], CL[~anyt], FIRST[~anyt], "lay", "market had no t30 trade"),
         "bsp_version_traded_30min_out": cell(Pb[lb["any_t30"].to_numpy(bool)], Fb[lb["any_t30"].to_numpy(bool)], CLb[lb["any_t30"].to_numpy(bool)],
                                              Fb1[lb["any_t30"].to_numpy(bool)], "lay", "BSP, market had any t30 trade")})

    # ---------------------------------------------------------------- 7. executable-price settlements
    worse = np.maximum(lb["ltp_pl"].to_numpy(), Pb)
    out({"section": "7_max_ltp_bsp", **cell(worse, Fb, CLb, Fb1, "lay", "C1 lay @ max(LTP, BSP) on BSP rows")})
    out({"section": "7_max_ltp_bsp_cap100", **cell(np.minimum(worse, 100.0), Fb, CLb, Fb1, "lay", "C1 lay @ min(max(LTP,BSP),100)")})
    mixed = np.where(lay["has_bsp"].to_numpy(), lay["bsp_pl"].to_numpy(), lay["tick_up"].to_numpy())
    out({"section": "7_bsp_else_tickup", **cell(mixed, F, CL, FIRST, "lay", "BSP where present else tick-up LTP (all rows)")})
    out({"section": "7_bsp_else_tickup_cap100", **cell(np.minimum(mixed, 100.0), F, CL, FIRST, "lay", "min(BSP,100) where present else tick-up LTP")})

    # ---------------------------------------------------------------- 8. ratio bins
    bins = [0, 0.8, 0.95, 1.05, 1.2, 1.5, np.inf]; labs = ["<0.8", "[0.8,0.95)", "[0.95,1.05)", "[1.05,1.2)", "[1.2,1.5)", ">=1.5"]
    rb_bin = pd.cut(ratio, bins, labels=labs, right=False).astype(str)
    out({"section": "8_by_bsp_over_ltp_bin", "bins": {lab: {"n": int((rb_bin == lab).sum()),
                                                            "ltp_roi_c5": r4(lay_stake_ret(lb["ltp_pl"].to_numpy()[rb_bin == lab], Fb[rb_bin == lab], 0.05).mean()) if (rb_bin == lab).any() else None,
                                                            "bsp_roi_c5": r4(lay_stake_ret(Pb[rb_bin == lab], Fb[rb_bin == lab], 0.05).mean()) if (rb_bin == lab).any() else None,
                                                            "placed": r4((Fb[rb_bin == lab] > 0).mean()) if (rb_bin == lab).any() else None,
                                                            "mean_1/ltp": r4((1 / lb["ltp_pl"].to_numpy()[rb_bin == lab]).mean()) if (rb_bin == lab).any() else None,
                                                            "mean_1/bsp": r4((1 / Pb[rb_bin == lab]).mean()) if (rb_bin == lab).any() else None}
                                                      for lab in labs}})

    # ---------------------------------------------------------------- 9. year by year
    yrs = {}
    for y in sorted(lay["year"].unique()):
        m = (lay["year"] == y).to_numpy(); mb = (lb["year"] == y).to_numpy()
        yrs[int(y)] = {"ltp": cell(P[m], F[m], CL[m], None, "lay"), "bsp": cell(Pb[mb], Fb[mb], CLb[mb], None, "lay"),
                       "bsp_cap100": cell(np.minimum(Pb[mb], 100.0), Fb[mb], CLb[mb], None, "lay")}
        for k in yrs[int(y)]:
            yrs[int(y)][k] = {kk: vv for kk, vv in yrs[int(y)][k].items() if kk in ("bets", "roi_c5", "se_cluster", "z_cluster", "strike")}
    out({"section": "9_by_year", "years": yrs})

    # ---------------------------------------------------------------- 10. baseline: lay EVERY C1 runner
    out({"section": "10_lay_every_C1_runner_ltp", **cell(c1["ltp_pl"].to_numpy(), c1["f_pl"].to_numpy(), c1["pl_mid"].to_numpy(), c1["first"].to_numpy(), "lay", "lay every C1 runner @ LTP")})
    out({"section": "10_lay_every_C1_runner_bsp", **cell(cb["bsp_pl"].to_numpy(), cb["f_pl"].to_numpy(), cb["pl_mid"].to_numpy(), cb["first"].to_numpy(), "lay", "lay every C1 runner @ BSP")})
    out({"section": "10_lay_every_C1_runner_bsp_cap100", **cell(np.minimum(cb["bsp_pl"].to_numpy(), 100.0), cb["f_pl"].to_numpy(), cb["pl_mid"].to_numpy(), cb["first"].to_numpy(), "lay", "lay every C1 runner @ min(BSP,100)")})
    # by LTP band: rule rows vs all C1 runners (lay at LTP) -> is the rule adding anything beyond the band mix?
    band = {}
    for b in orig.PL_LABELS:
        mr = (lay["pl_band"] == b).to_numpy(); ma = (c1["pl_band"] == b).to_numpy()
        band[b] = {"rule_n": int(mr.sum()), "rule_ltp_roi_c5": r4(lay_stake_ret(P[mr], F[mr], 0.05).mean()) if mr.any() else None,
                   "rule_placed": r4((F[mr] > 0).mean()) if mr.any() else None,
                   "all_n": int(ma.sum()), "all_ltp_roi_c5": r4(lay_stake_ret(c1["ltp_pl"].to_numpy()[ma], c1["f_pl"].to_numpy()[ma], 0.05).mean()) if ma.any() else None,
                   "all_placed": r4((c1["f_pl"].to_numpy()[ma] > 0).mean()) if ma.any() else None}
        mrb = (lb["pl_band"] == b).to_numpy(); mab = (cb["pl_band"] == b).to_numpy()
        band[b]["rule_bsp_roi_c5"] = r4(lay_stake_ret(Pb[mrb], Fb[mrb], 0.05).mean()) if mrb.any() else None
        band[b]["all_bsp_roi_c5"] = r4(lay_stake_ret(cb["bsp_pl"].to_numpy()[mab], cb["f_pl"].to_numpy()[mab], 0.05).mean()) if mab.any() else None
    out({"section": "10_by_ltp_band", "bands": band})
    # band-reweighted baseline: lay every C1 runner at LTP with the rule's LTP-band mix
    w = {b: band[b]["rule_n"] / len(lay) for b in orig.PL_LABELS}
    bw = sum(w[b] * band[b]["all_ltp_roi_c5"] for b in orig.PL_LABELS if band[b]["all_ltp_roi_c5"] is not None)
    out({"section": "10_band_reweighted_baseline", "lay_every_runner_at_ltp_with_rule_band_mix_roi_c5": r4(bw)})

    # ---------------------------------------------------------------- 11. threshold scan (descriptive, not a claim)
    scan = {}
    for th in (-0.05, -0.10, -0.15, -0.20, -0.30):
        m = (c1["e"] < th).to_numpy(); mb = m & (c1["bsp_pl"] > 1).fillna(False).to_numpy()
        scan[str(th)] = {"ltp": {k: v for k, v in cell(c1["ltp_pl"].to_numpy()[m], c1["f_pl"].to_numpy()[m], c1["pl_mid"].to_numpy()[m], None, "lay").items() if k in ("bets", "roi_c5", "se_cluster")},
                         "bsp": {k: v for k, v in cell(c1["bsp_pl"].to_numpy()[mb], c1["f_pl"].to_numpy()[mb], c1["pl_mid"].to_numpy()[mb], None, "lay").items() if k in ("bets", "roi_c5", "se_cluster")},
                         "bsp_cap100": {k: v for k, v in cell(np.minimum(c1["bsp_pl"].to_numpy()[mb], 100.0), c1["f_pl"].to_numpy()[mb], c1["pl_mid"].to_numpy()[mb], None, "lay").items() if k in ("bets", "roi_c5", "se_cluster")}}
    out({"section": "11_threshold_scan", "scan": scan})

    # ---------------------------------------------------------------- 12. multi-lay races
    per_race = lay.groupby("pl_mid").size()
    multi = lay["pl_mid"].map(per_race).to_numpy() >= 2
    out({"section": "12_multi_lay_races", "races_with_1_lay": int((per_race == 1).sum()), "races_with_2plus_lays": int((per_race >= 2).sum()),
         "max_lays_in_a_race": int(per_race.max()),
         "single_lay_rows": cell(P[~multi], F[~multi], CL[~multi], None, "lay"), "multi_lay_rows": cell(P[multi], F[multi], CL[multi], None, "lay")})

    # ---------------------------------------------------------------- T set for the same checks (reduced-field twin)
    tt = e[(e["set"] == "T") & main_sel & (e["e"] < -orig.LAY_T)]
    tb = tt[(tt["bsp_pl"] > 1).fillna(False)]
    out({"section": "T_twin", "ltp": cell(tt["ltp_pl"].to_numpy(), tt["f_pl"].to_numpy(), tt["pl_mid"].to_numpy(), tt["first"].to_numpy(), "lay"),
         "bsp": cell(tb["bsp_pl"].to_numpy(), tb["f_pl"].to_numpy(), tb["pl_mid"].to_numpy(), tb["first"].to_numpy(), "lay"),
         "max_ltp_bsp": cell(np.maximum(tb["ltp_pl"].to_numpy(), tb["bsp_pl"].to_numpy()), tb["f_pl"].to_numpy(), tb["pl_mid"].to_numpy(), tb["first"].to_numpy(), "lay")})
    out({"section": "done", "total_seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
