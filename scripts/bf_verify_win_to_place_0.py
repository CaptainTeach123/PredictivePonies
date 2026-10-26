"""Adversarial verification of the "win_to_place" PROBABILITY claim
(walk-forward blend of the place ltp market and the calibrated WIN-derived
place probability beats the normalised place ltp market by 0.0102 log loss
per race on fully priced place markets). Lens: leakage and data quality.

    python scripts/bf_verify_win_to_place_0.py [--dir data/raw/betfair/split] [--frac 0.1]

Reuses the original script's loader / fitting functions unchanged (imported
from scripts/bf_win_to_place.py) so that the reproduction is exact, then adds:
  1. reproduction with cluster-by-date SE, median / winsorised gain, gain by year
  2. a calibrated-market baseline (same walk-forward cubic calibrator applied to
     the place market itself) to split the gain into "recalibration" vs "WIN info"
  3. timing / staleness: WIN-market in-play leakage test (ltp_pre vs WIN BSP log
     loss), no-trade-in-last-30-min rates, gain by place-book tightness, gain vs
     the place BSP, and a model built from the 30-minute-old WIN price (ltp_t30)
     against the fresher place ltp_pre market.
Prints summary numbers only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import bf_win_to_place as O  # noqa: E402
from ponies.exotics import order_probs  # noqa: E402

r4 = O.r4
BLEND_GRID = O.BLEND_GRID


def out(obj) -> None:
    print(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)), flush=True)


def gain_stats(ll_model: np.ndarray, ll_base: np.ndarray, dates: np.ndarray, mid) -> dict:
    """gain = base - model per race, with plain SE, cluster-by-date SE, halves,
    robustness (median, winsorised, top-1% share) and per-year gains."""
    d = ll_base - ll_model
    N = len(d)
    if N < 2:
        return {"races": int(N)}
    dates = pd.DatetimeIndex(dates)
    fh = np.asarray(dates < mid)
    res = O.paired(ll_model, ll_base, fh)
    s = pd.Series(d).groupby(dates.to_numpy()).sum()
    n = pd.Series(d).groupby(dates.to_numpy()).size()
    m = d.mean()
    res["gain_se_cluster_date"] = r4(math.sqrt(((s - m * n) ** 2).sum()) / N)
    res["median_gain"] = r4(np.median(d))
    res["share_races_gain_pos"] = r4((d > 0).mean())
    lo, hi = np.percentile(d, [1, 99])
    res["winsor_1_99_gain"] = r4(np.clip(d, lo, hi).mean())
    k = max(1, N // 100)
    res["top1pct_share_of_total_gain"] = r4(np.sort(d)[-k:].sum() / d.sum()) if d.sum() != 0 else None
    yr = dates.year.to_numpy()
    res["gain_by_year"] = {int(y): [r4(d[yr == y].mean()), r4(d[yr == y].std(ddof=1) / math.sqrt((yr == y).sum())), int((yr == y).sum())]
                           for y in sorted(set(yr)) if (yr == y).sum() > 1}
    return res


def blend_grid_ll(qa: np.ndarray, qb: np.ndarray, y: np.ndarray, rc: np.ndarray) -> dict:
    """per-race log loss of w*qa + (1-w)*qb for every w on BLEND_GRID (w on qa)."""
    return {round(float(w), 2): O.logloss_race(w * qa + (1 - w) * qb, y, rc) for w in BLEND_GRID}


def walk_forward_blend(ll_by_w: dict, rmonth: np.ndarray, usable: np.ndarray, min_prev=500):
    """Monthly choice of w on all strictly earlier months (as in the original):
    returns the per-race ll of the chosen w (NaN where no choice) and the choices."""
    mcount = pd.Series(rmonth).value_counts()
    msum = {w: pd.Series(v[usable]).groupby(rmonth[usable]).sum() for w, v in ll_by_w.items()}
    chosen, wf = {}, np.full(len(rmonth), np.nan)
    for m in sorted(set(rmonth)):
        prev = [mm for mm in mcount.index if mm < m]
        if sum(mcount[mm] for mm in prev) < min_prev:
            continue
        best = min(BLEND_GRID, key=lambda w: sum(msum[round(float(w), 2)].get(mm, 0) for mm in prev))
        chosen[m] = round(float(best), 2)
        sel = (rmonth == m) & usable
        wf[sel] = ll_by_w[round(float(best), 2)][sel]
    return wf, chosen


def norm_probs(df: pd.DataFrame, price_col: str, race_col: str, k: pd.Series | int) -> pd.Series:
    inv = 1 / df[price_col]
    return inv * k / inv.groupby(df[race_col]).transform("sum")


def race_level(df: pd.DataFrame, col: str, race_col="race") -> np.ndarray:
    return df.drop_duplicates(race_col).set_index(race_col)[col].reindex(pd.unique(df[race_col])).to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data/raw/betfair/split"))
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()

    # ------------------------------------------------------------------ original pipeline, unchanged
    j = O.load(a.dir, a.frac, a.seed)
    P, jw, k, i, ok, nw, race = O.placed_set_stats(j)
    months = race["month"].to_numpy()
    codes = race["harness"].to_numpy()
    months_sorted = sorted(set(months))
    ll = O.grid_loglik(P, jw, k, i, ok, nw)
    lam = O.fit_lambdas_walk_forward(ll, ok, months, codes, months_sorted)
    del ll
    j["p_fair"] = O.derived_place_probs(j, P, nw, lam, months, codes)
    j = j[j["p_fair"].notna()].copy()
    j["race"] = pd.factorize(j["win_mid"])[0]
    j["p_cal"], _ = O.calibrate_walk_forward(j, "p_fair", sorted(j["month"].unique()))
    days = sorted(j["date"].unique())
    mid = days[len(days) // 2]
    grp = j.groupby("race")
    j["all_priced"] = grp["ltp_pl"].transform(lambda s: s.notna().all() and (s > 1).all())
    full = j[j["all_priced"]].copy()
    full["q_mkt"] = norm_probs(full, "ltp_pl", "race", full["n_winners"])
    full["book_over_nw"] = full.groupby("race")["ltp_pl"].transform(lambda s: (1 / s).sum()) / full["n_winners"]
    y = full["placed"].to_numpy(dtype=float)
    rc = pd.factorize(full["race"])[0]
    rdates = race_level(full, "date")
    rmonth = race_level(full, "month")
    rbook = race_level(full, "book_over_nw")
    rharness = race_level(full, "harness").astype(bool)
    ll_mkt = O.logloss_race(full["q_mkt"].to_numpy(), y, rc)
    p_cal_filled = np.nan_to_num(full["p_cal"].to_numpy(), nan=0.5)
    rc_has = np.bincount(rc, weights=full["p_cal"].notna().to_numpy()) == np.bincount(rc)
    ll_cal = O.logloss_race(p_cal_filled, y, rc)
    cal_blend = blend_grid_ll(full["q_mkt"].to_numpy(), p_cal_filled, y, rc)
    wfc, chosen_c = walk_forward_blend(cal_blend, rmonth, rc_has)
    hasc = ~np.isnan(wfc)
    out({"section": "1_reproduction", "runtime_s": round(time.time() - t0, 1),
         "joined_races": int(j["race"].nunique()), "fully_priced_races": int(len(ll_mkt)),
         "evaluated_races_blend": int(hasc.sum()), "mid": str(pd.Timestamp(mid).date()),
         "market_ltp_ll": r4(ll_mkt[hasc].mean()), "blend_ll": r4(wfc[hasc].mean()),
         "median_w_market": float(np.median(list(chosen_c.values()))),
         "blend_wf_cal_vs_market": gain_stats(wfc[hasc], ll_mkt[hasc], rdates[hasc], mid),
         "p_cal_alone_vs_market": gain_stats(ll_cal[rc_has], ll_mkt[rc_has], rdates[rc_has], mid),
         "fixed_w_market_0.25_cal_vs_market": gain_stats(cal_blend[0.25][rc_has], ll_mkt[rc_has], rdates[rc_has], mid)})

    # ------------------------------------------------------------------ 2. calibrated-market baseline
    full["q_cal"], _ = O.calibrate_walk_forward(full, "q_mkt", sorted(full["month"].unique()))
    q_cal_filled = np.nan_to_num(full["q_cal"].to_numpy(), nan=0.5)
    rq_has = (np.bincount(rc, weights=full["q_cal"].notna().to_numpy()) == np.bincount(rc)) & rc_has
    ll_qcal = O.logloss_race(q_cal_filled, y, rc)
    qq_blend = blend_grid_ll(q_cal_filled, p_cal_filled, y, rc)
    wfq, chosen_q = walk_forward_blend(qq_blend, rmonth, rq_has)
    hq = ~np.isnan(wfq) & hasc
    out({"section": "2_calibrated_market_baseline", "races": int(hq.sum()),
         "calibrated_market_vs_market": gain_stats(ll_qcal[hq], ll_mkt[hq], rdates[hq], mid),
         "blend_wf_cal_vs_market_same_races": gain_stats(wfc[hq], ll_mkt[hq], rdates[hq], mid),
         "blend_wf_cal_vs_calibrated_market": gain_stats(wfc[hq], ll_qcal[hq], rdates[hq], mid),
         "blend_qcal_pcal_wf_vs_market": gain_stats(wfq[hq], ll_mkt[hq], rdates[hq], mid),
         "blend_qcal_pcal_wf_vs_calibrated_market": gain_stats(wfq[hq], ll_qcal[hq], rdates[hq], mid),
         "median_w_on_calibrated_market": float(np.median(list(chosen_q.values()))) if chosen_q else None})

    # ------------------------------------------------------------------ 3a. WIN-market in-play leakage test
    # If ltp_pre contained in-play trades it would BEAT the WIN BSP in log loss; a pre-off (slightly stale)
    # price must be slightly WORSE than the BSP.
    jb = j[j.groupby("race")["bsp_win"].transform(lambda s: s.notna().all() and (s > 1).all())].copy()
    jb["p_bsp_win"] = norm_probs(jb, "bsp_win", "race", 1)
    rcw = pd.factorize(jb["race"])[0]
    yw = jb["won_win"].to_numpy(dtype=float)
    llw_ltp = np.bincount(rcw, weights=-yw * np.log(np.clip(jb["p_win"].to_numpy(), 1e-9, 1)))
    llw_bsp = np.bincount(rcw, weights=-yw * np.log(np.clip(jb["p_bsp_win"].to_numpy(), 1e-9, 1)))
    lr = np.log(jb["ltp_win"] / jb["bsp_win"]).abs()
    pl = jb[jb["bsp_pl"].notna() & (jb["bsp_pl"] > 1) & jb["ltp_pl"].notna()]
    lrp = np.log(pl["ltp_pl"] / pl["bsp_pl"]).abs()
    tw = j[j["t30_win"].notna() & (j["t30_win"] > 1)]
    tp = j[j["t30_pl"].notna() & (j["t30_pl"] > 1) & j["ltp_pl"].notna()]
    inplay = {"section": "3a_win_ltp_pre_inplay_leak_test", "races": int(len(llw_ltp)),
              "win_ltp_vs_win_bsp_logloss_gain(positive=ltp better=LEAK)": gain_stats(llw_ltp, llw_bsp, race_level(jb, "date"), mid),
              "share_win_runners_abs_log_ltp_over_bsp_gt_0.3": r4((lr > 0.3).mean()),
              "share_win_runners_abs_log_ltp_over_bsp_gt_0.5": r4((lr > 0.5).mean()),
              "share_place_runners_abs_log_ltp_over_bsp_gt_0.3": r4((lrp > 0.3).mean()),
              "share_place_runners_abs_log_ltp_over_bsp_gt_0.5": r4((lrp > 0.5).mean()),
              "no_trade_last_30min_proxy_win(ltp_pre==ltp_t30|t30)": r4((tw["ltp_win"] == tw["t30_win"]).mean()),
              "no_trade_last_30min_proxy_place(ltp_pre==ltp_t30|t30)": r4((tp["ltp_pl"] == tp["t30_pl"]).mean())}
    gb = a.dir / "gb_win.parquet"
    if gb.exists() and a.frac >= 1:
        g = pd.read_parquet(gb, columns=["market_id", "selection_id", "ltp_pre", "bsp", "won", "date"]).drop_duplicates(["market_id", "selection_id"])
        gg = g.groupby("market_id")
        okm = (gg["won"].transform("sum") == 1) & gg["ltp_pre"].transform(lambda s: s.notna().all() and (s > 1).all()) \
            & gg["bsp"].transform(lambda s: s.notna().all() and (s > 1).all())
        g = g[okm].copy()
        g["race"] = pd.factorize(g["market_id"])[0]
        g["pw"] = norm_probs(g, "ltp_pre", "race", 1)
        g["pb"] = norm_probs(g, "bsp", "race", 1)
        rg = g["race"].to_numpy(); yg = g["won"].to_numpy(dtype=float)
        a_ = np.bincount(rg, weights=-yg * np.log(np.clip(g["pw"].to_numpy(), 1e-9, 1)))
        b_ = np.bincount(rg, weights=-yg * np.log(np.clip(g["pb"].to_numpy(), 1e-9, 1)))
        inplay["gb_win_ltp_vs_bsp_gain_check"] = {"races": int(len(a_)), "gain": r4((b_ - a_).mean()),
                                                  "se": r4((b_ - a_).std(ddof=1) / math.sqrt(len(a_)))}
        del g
    out(inplay)

    # ------------------------------------------------------------------ 3b. gain by pre-race place-book tightness
    rows = []
    for lo, hi in [(0, 0.9), (0.9, 0.97), (0.97, 1.03), (1.03, 1.1), (1.1, 9)]:
        sel = hasc & (rbook >= lo) & (rbook < hi)
        if sel.sum() > 50:
            gs = gain_stats(wfc[sel], ll_mkt[sel], rdates[sel], mid)
            rows.append({"place_book_over_n_winners": f"{lo}-{hi}", "races": int(sel.sum()), "gain": gs["gain"],
                         "se": gs["gain_se"], "halves": gs["halves_gain"], "median_gain": gs["median_gain"]})
    out({"section": "3b_gain_by_place_book_tightness_pre_race", "rows": rows,
         "gallops": gain_stats(wfc[hasc & ~rharness], ll_mkt[hasc & ~rharness], rdates[hasc & ~rharness], mid),
         "harness": gain_stats(wfc[hasc & rharness], ll_mkt[hasc & rharness], rdates[hasc & rharness], mid)})

    # ------------------------------------------------------------------ 3c. against the place BSP, and by staleness
    full["all_bsp"] = full.groupby("race")["bsp_pl"].transform(lambda s: s.notna().all() and (s > 1).all())
    full["all_bsp_win"] = full.groupby("race")["bsp_win"].transform(lambda s: s.notna().all() and (s > 1).all())
    fb_mask_r = (np.bincount(rc, weights=full["all_bsp"].to_numpy()) == np.bincount(rc)) & hasc
    fb = full[full["all_bsp"]].copy()
    fb["q_bsp"] = norm_probs(fb, "bsp_pl", "race", fb["n_winners"])
    fb["stale_pl"] = np.log(fb["bsp_pl"] / fb["ltp_pl"]).abs()
    fb["stale_win"] = np.log(fb["bsp_win"] / fb["ltp_win"]).abs().where(fb["bsp_win"] > 1)
    fb["stale_pl_r"] = fb.groupby("race")["stale_pl"].transform("mean")
    fb["stale_win_r"] = fb.groupby("race")["stale_win"].transform("mean")
    rcb = pd.factorize(fb["race"])[0]
    yb = fb["placed"].to_numpy(dtype=float)
    ll_bsp = O.logloss_race(fb["q_bsp"].to_numpy(), yb, rcb)
    # align the already computed per-race arrays to fb's races
    race_ids = pd.unique(full["race"])
    pos = pd.Series(np.arange(len(race_ids)), index=race_ids)
    idx = pos.reindex(pd.unique(fb["race"])).to_numpy()
    sub = hasc[idx]
    d_b = rdates[idx][sub]
    res = {"section": "3c_vs_place_bsp", "races": int(sub.sum()),
           "market_ltp_vs_bsp": gain_stats(ll_mkt[idx][sub], ll_bsp[sub], d_b, mid),
           "blend_wf_cal_vs_bsp": gain_stats(wfc[idx][sub], ll_bsp[sub], d_b, mid),
           "p_cal_vs_bsp": gain_stats(ll_cal[idx][sub], ll_bsp[sub], d_b, mid)}
    spr = race_level(fb, "stale_pl_r")
    swr = race_level(fb, "stale_win_r")
    rows = []
    qs = np.nanpercentile(spr[sub], [25, 50, 75])
    edges = [-1, qs[0], qs[1], qs[2], 99]
    for lo, hi in zip(edges, edges[1:]):
        s = sub & (spr > lo) & (spr <= hi)
        rows.append({"place_staleness_quartile(mean|log bsp/ltp|)": f"{max(lo,0):.3f}-{hi:.3f}", "races": int(s.sum()),
                     "blend_vs_ltp_market": [r4((ll_mkt[idx][s] - wfc[idx][s]).mean()), r4((ll_mkt[idx][s] - wfc[idx][s]).std(ddof=1) / math.sqrt(s.sum()))],
                     "blend_vs_bsp": [r4((ll_bsp[s] - wfc[idx][s]).mean()), r4((ll_bsp[s] - wfc[idx][s]).std(ddof=1) / math.sqrt(s.sum()))],
                     "ltp_market_vs_bsp": r4((ll_bsp[s] - ll_mkt[idx][s]).mean()),
                     "mean_win_staleness": r4(np.nanmean(swr[s]))})
    res["by_place_staleness"] = rows
    out(res)

    # ------------------------------------------------------------------ 3d. model from the 30-minute-old WIN price vs the fresher place ltp_pre market
    j["all_t30_win"] = j.groupby("race")["t30_win"].transform(lambda s: s.notna().all() and (s > 1).all())
    j2 = j[j["all_t30_win"]].copy()
    j2["race"] = pd.factorize(j2["win_mid"])[0]
    j2 = j2.sort_values(["race", "t30_win", "selection_id"]).reset_index(drop=True)
    j2["pos"] = j2.groupby("race").cumcount()
    j2["p_win_t30"] = norm_probs(j2, "t30_win", "race", 1)
    P2 = O.pad(j2, "p_win_t30")
    r2 = j2.drop_duplicates("race").set_index("race").sort_index()
    j2["p_fair_t30"] = O.derived_place_probs(j2, P2, r2["n_winners"].to_numpy(), lam, r2["month"].to_numpy(), r2["harness"].to_numpy())
    j2 = j2[j2["p_fair_t30"].notna()].copy()
    j2["p_cal_t30"], _ = O.calibrate_walk_forward(j2, "p_fair_t30", sorted(j2["month"].unique()))
    full["all_t30_pl"] = full.groupby("race")["t30_pl"].transform(lambda s: s.notna().all() and (s > 1).all())
    f2 = full.merge(j2[["win_mid", "selection_id", "p_fair_t30", "p_cal_t30"]], on=["win_mid", "selection_id"], how="inner")
    f2 = f2[f2.groupby("race")["p_cal_t30"].transform(lambda s: s.notna().all()) & f2["p_cal"].notna()].copy()
    rc2 = pd.factorize(f2["race"])[0]
    y2 = f2["placed"].to_numpy(dtype=float)
    rm2 = race_level(f2, "month")
    rd2 = race_level(f2, "date")
    ll_m2 = O.logloss_race(f2["q_mkt"].to_numpy(), y2, rc2)
    ll_c2 = O.logloss_race(f2["p_cal"].to_numpy(), y2, rc2)
    ll_t2 = O.logloss_race(f2["p_cal_t30"].to_numpy(), y2, rc2)
    bl_pre = blend_grid_ll(f2["q_mkt"].to_numpy(), f2["p_cal"].to_numpy(), y2, rc2)
    bl_t30 = blend_grid_ll(f2["q_mkt"].to_numpy(), f2["p_cal_t30"].to_numpy(), y2, rc2)
    ones = np.ones(len(ll_m2), dtype=bool)
    wf_pre, ch_pre = walk_forward_blend(bl_pre, rm2, ones)
    wf_t30, ch_t30 = walk_forward_blend(bl_t30, rm2, ones)
    h2 = ~np.isnan(wf_pre) & ~np.isnan(wf_t30)
    # the WIN market's own information loss from t30 to ltp_pre, on the same races
    f2["p_win_t30n"] = norm_probs(f2, "t30_win", "race", 1)
    yw2 = f2["won_win"].to_numpy(dtype=float)
    lw_pre = np.bincount(rc2, weights=-yw2 * np.log(np.clip(f2["p_win"].to_numpy(), 1e-9, 1)))
    lw_t30 = np.bincount(rc2, weights=-yw2 * np.log(np.clip(f2["p_win_t30n"].to_numpy(), 1e-9, 1)))
    out({"section": "3d_win_t30_model_vs_place_ltp_pre_market", "races": int(h2.sum()),
         "note": "same races; q_mkt is the place market at ltp_pre; p_cal_t30 is derived from the WIN price 30 min before the start",
         "p_cal_from_win_ltp_pre_vs_market": gain_stats(ll_c2[h2], ll_m2[h2], rd2[h2], mid),
         "p_cal_from_win_t30_vs_market": gain_stats(ll_t2[h2], ll_m2[h2], rd2[h2], mid),
         "blend_wf_ltp_pre_vs_market": gain_stats(wf_pre[h2], ll_m2[h2], rd2[h2], mid),
         "blend_wf_t30_vs_market": gain_stats(wf_t30[h2], ll_m2[h2], rd2[h2], mid),
         "fixed_w0.25_t30_vs_market": gain_stats(bl_t30[0.25][h2], ll_m2[h2], rd2[h2], mid),
         "fixed_w0.5_t30_vs_market": gain_stats(bl_t30[0.5][h2], ll_m2[h2], rd2[h2], mid),
         "median_w_market_pre": float(np.median(list(ch_pre.values()))) if ch_pre else None,
         "median_w_market_t30": float(np.median(list(ch_t30.values()))) if ch_t30 else None,
         "win_market_ltp_pre_vs_t30_own_gain": gain_stats(lw_pre[h2], lw_t30[h2], rd2[h2], mid)})

    # ------------------------------------------------------------------ 3e. tiny like-for-like at t30 (both markets fully priced 30 min out)
    f3 = f2[f2["all_t30_pl"]].copy()
    if f3["race"].nunique() > 50:
        f3["q_t30"] = norm_probs(f3, "t30_pl", "race", f3["n_winners"])
        rc3 = pd.factorize(f3["race"])[0]
        y3 = f3["placed"].to_numpy(dtype=float)
        rd3 = race_level(f3, "date")
        m3 = O.logloss_race(f3["q_t30"].to_numpy(), y3, rc3)
        mp3 = O.logloss_race(f3["q_mkt"].to_numpy(), y3, rc3)
        t3 = O.logloss_race(f3["p_cal_t30"].to_numpy(), y3, rc3)
        c3 = O.logloss_race(f3["p_cal"].to_numpy(), y3, rc3)
        b3 = O.logloss_race(0.25 * f3["q_t30"].to_numpy() + 0.75 * f3["p_cal_t30"].to_numpy(), y3, rc3)
        out({"section": "3e_like_for_like_t30_small", "races": int(len(m3)),
             "place_t30_market_ll": r4(m3.mean()), "place_ltp_pre_market_ll": r4(mp3.mean()),
             "p_cal_t30_vs_place_t30_market": gain_stats(t3, m3, rd3, mid),
             "blend_0.25_t30_market_0.75_p_cal_t30_vs_place_t30_market": gain_stats(b3, m3, rd3, mid),
             "p_cal_ltp_pre_vs_place_t30_market": gain_stats(c3, m3, rd3, mid),
             "p_cal_t30_vs_place_ltp_pre_market": gain_stats(t3, mp3, rd3, mid)})
    out({"section": "totals", "runtime_s": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
